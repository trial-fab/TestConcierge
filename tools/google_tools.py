import base64
import os
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Any, List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]

class GoogleWorkspaceError(RuntimeError):
    """Raised when Google Workspace tooling is misconfigured or returns an error."""

class GoogleWorkspaceTools:

    def __init__(self):
        self._creds: Optional[Credentials] = None

    def _build_credentials(self) -> Credentials:
        if self._creds and self._creds.valid:
            return self._creds

        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN")
        token_uri = os.getenv("GOOGLE_TOKEN_URI", "https://oauth2.googleapis.com/token")

        if not all([client_id, client_secret, refresh_token]):
            raise GoogleWorkspaceError(
                "Google credentials missing. Please set GOOGLE_CLIENT_ID, "
                "GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN."
            )

        self._creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=token_uri,
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )
        return self._creds

    def _build_service(self, api: str, version: str):
        try:
            return build(api, version, credentials=self._build_credentials(), cache_discovery=False)
        except HttpError as e:
            raise GoogleWorkspaceError(f"Failed to initialise Google {api} client: {e}") from e

    def list_calendar_events(self, max_results: int = 5) -> List[dict[str, Any]]:
        service = self._build_service("calendar", "v3")
        try:
            events_result = (
                service.events()
                .list(
                    calendarId="primary",
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                    timeMin=datetime.utcnow().isoformat() + "Z",
                )
                .execute()
        )
        except HttpError as e:
            raise GoogleWorkspaceError(f"Calendar API error: {e}") from e

        items = events_result.get("items", [])
        events = []
        for evt in items:
            start = evt.get("start", {}).get("dateTime") or evt.get("start", {}).get("date")
            events.append(
                {
                    "summary": evt.get("summary", "Untitled"),
                    "start": start,
                    "location": evt.get("location", ""),
                    "hangoutLink": evt.get("hangoutLink", ""),
                }
            )
        return events

    def list_recent_messages(self, query: str = "", max_results: int = 5) -> List[dict[str, str]]:
        service = self._build_service("gmail", "v1")
        try:
            resp = (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )
        except HttpError as e:
            raise GoogleWorkspaceError(f"Gmail API error: {e}") from e

        message_ids = resp.get("messages", []) or []
        messages: List[dict[str, str]] = []
        for m in message_ids:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=m["id"], format="metadata", metadataHeaders=["From", "Subject", "Date"])
                .execute()
            )
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            messages.append(
                {
                    "snippet": msg.get("snippet", ""),
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                }
            )
        return messages

    def send_email(self, to_address: str, subject: str, body: str) -> str:
        if not to_address or not subject or not body:
            raise GoogleWorkspaceError("To, subject, and body are required to send email.")

        service = self._build_service("gmail", "v1")
        mime_msg = MIMEText(body)
        mime_msg["to"] = to_address
        mime_msg["subject"] = subject
        raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode("utf-8")
        try:
            result = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        except HttpError as e:
            raise GoogleWorkspaceError(f"Gmail send error: {e}") from e
        return result.get("id", "sent")

    def _normalize_iso(self, raw: str) -> str:
        if not raw:
            raise GoogleWorkspaceError("Start date/time is required in ISO format (e.g., 2024-09-01T14:00-04:00)")
        value = raw.strip()
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(value)
        except ValueError as exc:
            raise GoogleWorkspaceError(f"Invalid ISO date/time: {raw}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    def _time_range(self, start_iso: str, duration_minutes: int) -> tuple[str, str]:
        start = datetime.fromisoformat(start_iso)
        end = start + timedelta(minutes=duration_minutes)
        return start_iso, end.isoformat()

    def check_availability(self, start_iso: str, duration_minutes: int) -> bool:
        start_iso = self._normalize_iso(start_iso)
        start_iso, end_iso = self._time_range(start_iso, duration_minutes)
        service = self._build_service("calendar", "v3")
        body = {
            "timeMin": start_iso,
            "timeMax": end_iso,
            "items": [{"id": "primary"}],
        }
        try:
            resp = service.freebusy().query(body=body).execute()
        except HttpError as e:
            raise GoogleWorkspaceError(f"Calendar free/busy error: {e}") from e
        busy = resp.get("calendars", {}).get("primary", {}).get("busy", [])
        return len(busy) == 0

    def find_next_available_slot(self, start_iso: str, duration_minutes: int, windows: int = 6) -> Optional[str]:
        current = datetime.fromisoformat(self._normalize_iso(start_iso))
        for _ in range(windows):
            current += timedelta(minutes=duration_minutes)
            slot = current.isoformat()
            if self.check_availability(slot, duration_minutes):
                return slot
        return None

    def create_event(
        self,
        summary: str,
        start_iso: str,
        duration_minutes: int,
        attendees: Optional[List[str]] = None,
        description: str = "",
        location: str = "",
    ) -> dict[str, str]:
        start_iso = self._normalize_iso(start_iso)
        start_iso, end_iso = self._time_range(start_iso, duration_minutes)
        service = self._build_service("calendar", "v3")
        attendee_entries = (
            [{"email": email} for email in attendees if email]
            if attendees
            else []
        )
        event_body = {
            "summary": summary,
            "description": description,
            "location": location,
            "start": {"dateTime": start_iso},
            "end": {"dateTime": end_iso},
            "attendees": attendee_entries,
            "conferenceData": {
                "createRequest": {
                    "requestId": f"meet-{uuid.uuid4()}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
        }
        try:
            result = (
                service.events()
                .insert(
                    calendarId="primary",
                    body=event_body,
                    sendUpdates="all",
                    conferenceDataVersion=1,
                )
                .execute()
            )
        except HttpError as e:
            raise GoogleWorkspaceError(f"Calendar insert error: {e}") from e
        hangout_link = result.get("hangoutLink")
        if not hangout_link:
            conference = (result.get("conferenceData") or {}).get("entryPoints") or []
            hangout_link = conference[0].get("uri") if conference else ""
        return {
            "event_id": result.get("id", "event-created"),
            "hangout_link": hangout_link,
        }
