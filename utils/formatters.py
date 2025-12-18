import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")

MEETING_TIMEZONE_OFFSETS = {
    "US/Eastern (EST)": "-04:00", 
    "US/Central (CST)": "-05:00", 
    "US/Mountain (MST)": "-06:00",  
    "US/Pacific (PST)": "-07:00",  
}

_SUBJECT_PREFIX = re.compile(
    r"^\s*(?:\*\*|__|\*|_|\-)?\s*subject\s*(?:\*\*|__|\*|_)?\s*[:\-–—]\s*(.+)$",
    re.I,
)

def format_est_timestamp(raw: str | None) -> str:
    """Convert ISO timestamp to EST format like 'Jan 15, 03:45 PM EST'."""
    if not raw:
        return "Unknown"
    text = raw.strip()
    if not text:
        return "Unknown"
    normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        est_dt = dt.astimezone(EASTERN)
        return est_dt.strftime("%b %d, %I:%M %p EST")
    except ValueError:
        return text


def split_subject_from_body(text: str) -> tuple[Optional[str], str]:
    """Detect leading 'Subject: ...' lines and return (subject, body_without_line)."""
    if not text:
        return None, ""
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return None, ""
    match = _SUBJECT_PREFIX.match(lines[0])
    if not match:
        return None, text
    subject = match.group(1).strip()
    remaining = "\n".join(lines[1:]).lstrip("\n")
    return subject or None, remaining


def build_start_iso(selected_date, selected_time, tz_label: str) -> str:
    """Build ISO timestamp from date, time, and timezone label."""
    offset = MEETING_TIMEZONE_OFFSETS.get(tz_label, "-04:00")
    combined = datetime.combine(selected_date, selected_time)
    return combined.strftime("%Y-%m-%dT%H:%M") + offset