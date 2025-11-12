"""Session state initialization and action management."""
import streamlit as st
from datetime import date, datetime
from typing import Any


RECENT_ACTION_LIMIT = 5


def initialize_session_state() -> None:
    """Initialize all session state variables with defaults."""
    # Authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "session_token" not in st.session_state:
        st.session_state.session_token = None

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

    # Assistant UI state
    st.session_state.setdefault("show_tool_picker", False)
    st.session_state.setdefault("show_email_builder", False)
    st.session_state.setdefault("show_meeting_builder", False)

    # Email assistant state
    st.session_state.setdefault("pending_email", None)
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

    # Processing state (for disabling input during bot response)
    st.session_state.setdefault("is_processing", False)
    st.session_state.setdefault("pending_user_input", None)
    st.session_state.setdefault("processing_input_id", None)
    st.session_state.setdefault("processing_query", None)
    st.session_state.setdefault("processing_tokens_in", None)
    st.session_state.setdefault("processing_cached_result", None)
    st.session_state.setdefault("processing_response_saved", False)
    st.session_state.setdefault("processing_regen_id", None)

    # Dashboard
    st.session_state.setdefault("show_dashboard", True)

    # Login flow
    st.session_state.setdefault("pending_login", None)


def activate_assistant(kind: str | None, *, rerun: bool = False) -> None:
    """
    Activate a specific assistant (email, meeting, or None).
    When kind is None, closes all assistants.
    """
    st.session_state.show_email_builder = kind == "email"
    st.session_state.show_meeting_builder = kind == "meeting"
    st.session_state.show_tool_picker = False
    if rerun:
        st.rerun()


def queue_action_collapse(action_type: str, data: dict[str, Any]) -> None:
    """Queue an action to be collapsed and added to recent actions."""
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
    """Process pending action collapses and add to recent actions."""
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
    """Auto-open assistant based on keywords in model response."""
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
