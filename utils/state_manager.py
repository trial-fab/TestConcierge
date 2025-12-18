import streamlit as st
from datetime import date, datetime
from typing import Any
import time


RECENT_ACTION_LIMIT = 5
SESSION_CACHE_TTL = 2.0


def initialize_session_state() -> None:
    # Authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None

    # Chat session
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_regen" not in st.session_state:
        st.session_state.pending_regen = False

    # Token budget tracking
    if "token_total" not in st.session_state:
        st.session_state.token_total = 0
    if "limit_reached" not in st.session_state:
        st.session_state.limit_reached = False
    st.session_state.setdefault("current_request_id", None)
    st.session_state.setdefault("request_start_time", None)

    # Assistant UI state
    st.session_state.setdefault("show_tool_picker", False)
    st.session_state.setdefault("show_email_builder", False)
    st.session_state.setdefault("show_meeting_builder", False)

    # Email assistant state
    st.session_state.setdefault("pending_email", None)
    st.session_state.setdefault("pending_email_draft", None)  
    st.session_state.setdefault("pending_email_edit", None) 
    st.session_state.setdefault("email_to_input", "")
    st.session_state.setdefault("email_subject_input", "")
    st.session_state.setdefault("email_student_message", "")
    st.session_state.setdefault("email_draft_text", "")
    st.session_state.setdefault("email_draft_sync_value", None)
    st.session_state.setdefault("email_subject_sync_value", None)
    st.session_state.setdefault("email_edit_instructions", "")
    st.session_state.setdefault("email_fields_reset_pending", False)

    # Meeting assistant state
    st.session_state.setdefault("pending_meeting", None)
    st.session_state.setdefault("pending_meeting_plan", None)  
    st.session_state.setdefault("pending_meeting_edit", None)  
    st.session_state.setdefault("meeting_summary_input", "")
    st.session_state.setdefault("meeting_duration_input", 30)
    st.session_state.setdefault("meeting_attendees_input", "")
    st.session_state.setdefault("meeting_description_input", "")
    st.session_state.setdefault("meeting_location_input", "")
    st.session_state.setdefault("meeting_timezone_input", "US/Eastern (EST)")
    st.session_state.setdefault("meeting_date_input", date.today())
    st.session_state.setdefault(
        "meeting_time_input",
        datetime.now().replace(second=0, microsecond=0).time(),
    )
    st.session_state.setdefault("meeting_fields_reset_pending", False)
    st.session_state.setdefault("meeting_notes_text", "")
    st.session_state.setdefault("meeting_notes_sync_value", None)
    st.session_state.setdefault("meeting_edit_instructions", "")

    # Action tracking
    st.session_state.setdefault("recent_actions", [])
    st.session_state.setdefault("pending_action_collapses", [])

    # Processing state (for blocking all interactions during bot response)
    st.session_state.setdefault("is_processing", False)
    st.session_state.setdefault("pending_user_input", None)

    # Dashboard
    st.session_state.setdefault("show_dashboard", True)
    st.session_state.setdefault("show_observability", False)

    # Login flow
    st.session_state.setdefault("pending_login", None)

    st.session_state.setdefault("_session_cache", {})
    st.session_state.setdefault("_session_cache_time", 0)
    st.session_state.setdefault("_messages_cache", {})
    st.session_state.setdefault("_messages_cache_time", {})


def activate_assistant(kind: str | None, *, rerun: bool = False) -> None:
    st.session_state.show_email_builder = kind == "email"
    st.session_state.show_meeting_builder = kind == "meeting"
    st.session_state.show_tool_picker = False
    if rerun:
        st.rerun()


def queue_action_collapse(action_type: str, data: dict[str, Any]) -> None:
    if action_type == "email":
        st.session_state.show_email_builder = False
    elif action_type == "meeting":
        st.session_state.show_meeting_builder = False
    st.session_state.show_tool_picker = False

    st.session_state.pending_action_collapses.append(
        {
            "type": action_type,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "data": data,
        }
    )


def handle_pending_action_collapses() -> None:
    pending = st.session_state.pending_action_collapses
    if not pending:
        return

    for entry in pending:
        if entry["type"] == "email":
            st.session_state.show_email_builder = False
        elif entry["type"] == "meeting":
            st.session_state.show_meeting_builder = False

    updated = pending + st.session_state.recent_actions
    st.session_state.recent_actions = updated[:RECENT_ACTION_LIMIT]
    st.session_state.pending_action_collapses = []


def maybe_auto_open_assistant(response_text: str | None) -> None:
    if not response_text:
        return
    lowered = response_text.lower()
    email_cues = (
        "email assistant",
        "draft an email",
        "compose an email",
        "send an email",
    )
    meeting_cues = (
        "meeting assistant",
        "schedule a meeting",
        "calendar invite",
        "book a meeting",
    )
    if any(cue in lowered for cue in email_cues):
        activate_assistant("email")
    elif any(cue in lowered for cue in meeting_cues):
        activate_assistant("meeting")


def get_cached_sessions(user_id: str) -> list[dict] | None:
    if st.session_state._session_cache.get("user_id") != user_id:
        return None
    if time.time() - st.session_state._session_cache_time > SESSION_CACHE_TTL:
        return None
    return st.session_state._session_cache.get("sessions")


def set_cached_sessions(user_id: str, sessions: list[dict]) -> None:
    st.session_state._session_cache = {"user_id": user_id, "sessions": sessions}
    st.session_state._session_cache_time = time.time()


def invalidate_session_cache() -> None:
    st.session_state._session_cache = {}
    st.session_state._session_cache_time = 0


def get_cached_messages(session_id: str) -> list[dict] | None:
    if session_id not in st.session_state._messages_cache:
        return None
    cache_time = st.session_state._messages_cache_time.get(session_id, 0)
    if time.time() - cache_time > SESSION_CACHE_TTL:
        return None
    return st.session_state._messages_cache[session_id]


def set_cached_messages(session_id: str, messages: list[dict]) -> None:
    st.session_state._messages_cache[session_id] = messages
    st.session_state._messages_cache_time[session_id] = time.time()


def invalidate_messages_cache(session_id: str) -> None:
    if session_id in st.session_state._messages_cache:
        del st.session_state._messages_cache[session_id]
    if session_id in st.session_state._messages_cache_time:
        del st.session_state._messages_cache_time[session_id]
