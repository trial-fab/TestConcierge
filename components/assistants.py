import streamlit as st
from datetime import date, datetime
from utils.formatters import MEETING_TIMEZONE_OFFSETS, build_start_iso
from agents.email_assistant import (
    start_email_draft,
    apply_email_edit,
    save_manual_email_edit,
    send_email_draft,
)
from agents.meeting_assistant import plan_meeting, create_meeting_event
from utils.splunk_logger import get_splunk_logger

logger = get_splunk_logger()

def _log_assistant_event(assistant_type: str, action: str, extra: dict | None = None):
    session_id = st.session_state.get("current_session_id")
    payload = {"assistant_type": assistant_type, "action": action}
    if extra:
        payload.update(extra)
    logger.log_event(
        category="agent",
        event_type="assistant_action",
        payload=payload,
        session_id=session_id,
        component="ui",
    )

def render_tool_picker() -> None:
    """Render the tool picker with email and meeting assistant options."""
    st.markdown(
        """
        <div class="tool-picker-card">
            <div class="tool-picker-title" role="heading" aria-level="4">Assisted Actions</div>
            <p class="tool-picker-subtitle">Choose an assistant to draft outreach or schedule meetings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tool_col1, tool_col2 = st.columns(2)

    with tool_col1:
        st.markdown("<div class='assistant-card'>", unsafe_allow_html=True)
        if st.button("📧 Email Assistant", key="picker_email", use_container_width=True):
            st.session_state.show_email_builder = True
            st.session_state.show_meeting_builder = False
            st.session_state.show_tool_picker = False
            _log_assistant_event("email", "open_picker")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with tool_col2:
        st.markdown("<div class='assistant-card'>", unsafe_allow_html=True)
        if st.button("📅 Meeting Assistant", key="picker_meeting", use_container_width=True):
            st.session_state.show_meeting_builder = True
            st.session_state.show_email_builder = False
            st.session_state.show_tool_picker = False
            _log_assistant_event("meeting", "open_picker")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def render_email_builder(mcp_client, db) -> None:
    """Render the email builder UI."""
    st.subheader("Email Assistant 🐂")

    if st.session_state.email_fields_reset_pending:
        st.session_state.email_to_input = ""
        st.session_state.email_subject_input = ""
        st.session_state.email_student_message = ""
        st.session_state.email_edit_instructions = ""
        st.session_state.email_draft_text = ""
        st.session_state.email_subject_sync_value = None
        st.session_state.email_draft_sync_value = ""
        st.session_state.pending_email = None
        st.session_state.email_fields_reset_pending = False

    pending = st.session_state.pending_email

    # Show input form only if no draft exists
    if not pending:
        if st.session_state.email_subject_sync_value is not None:
            st.session_state.email_subject_input = st.session_state.email_subject_sync_value
            st.session_state.email_subject_sync_value = None

        st.text_input("Student Email", key="email_to_input")
        st.text_input("Subject", key="email_subject_input")
        st.text_area("Student Inquiry / Notes", key="email_student_message", height=120)

        col_generate, col_reset = st.columns([3, 1])

        if not st.session_state.get("is_processing", False):
            if col_generate.button("Generate Draft", key="btn_email_generate", use_container_width=True):
                st.session_state.pending_email_draft = {
                    "to": st.session_state.email_to_input,
                    "subject": st.session_state.email_subject_input,
                    "message": st.session_state.email_student_message,
                }
                _log_assistant_event(
                    "email",
                    "request_draft",
                    {
                        "recipient_provided": bool(st.session_state.email_to_input.strip()),
                        "subject_provided": bool(st.session_state.email_subject_input.strip()),
                    },
                )
                st.session_state.is_processing = True
                st.session_state.show_email_builder = False
                st.rerun()

            if col_reset.button("Reset Fields", key="btn_email_reset", use_container_width=True):
                st.session_state.email_fields_reset_pending = True
                _log_assistant_event("email", "reset_fields")
                st.rerun()
        else:
            col_generate.button("Generate Draft", key="btn_email_generate", use_container_width=True, disabled=True)
            col_reset.button("Reset Fields", key="btn_email_reset", use_container_width=True, disabled=True)

        st.info("No draft generated yet. Enter details above and click Generate Draft.")

    # Draft editor (only if draft exists)
    else:
        if st.session_state.email_draft_sync_value is not None:
            st.session_state.email_draft_text = st.session_state.email_draft_sync_value
            st.session_state.email_draft_sync_value = None

        st.markdown(f"**To:** {pending.get('to', '')}")
        st.markdown(f"**Subject:** {pending.get('subject', '')}")

        st.text_area("Draft Body", key="email_draft_text", height=220)
        st.text_input("AI edit instructions (optional)", key="email_edit_instructions")

        col1, col2, col3, col4 = st.columns(4)

        if not st.session_state.get("is_processing", False):
            if col1.button("Apply AI Edit", key="btn_email_ai_edit"):
                st.session_state.pending_email_edit = {
                    "instructions": st.session_state.email_edit_instructions,
                }
                _log_assistant_event(
                    "email",
                    "ai_edit_request",
                    {"has_instructions": bool(st.session_state.email_edit_instructions.strip())},
                )
                st.session_state.is_processing = True
                st.session_state.show_email_builder = False
                st.rerun()

            if col2.button("Save Manual Edit", key="btn_email_manual_edit"):
                if save_manual_email_edit(st.session_state.email_draft_text):
                    st.success("Draft updated.")
                    _log_assistant_event("email", "manual_edit_saved")

            if col3.button("Send Email", key="btn_email_send"):
                send_email_draft(mcp_client, db)
                _log_assistant_event("email", "send_draft")
                st.rerun()

            if col4.button("Clear Draft", key="btn_email_clear"):
                st.session_state.pending_email = None
                st.session_state.email_draft_sync_value = ""
                _log_assistant_event("email", "clear_draft")
                st.rerun()
        else:
            col1.button("Apply AI Edit", key="btn_email_ai_edit", disabled=True)
            col2.button("Save Manual Edit", key="btn_email_manual_edit", disabled=True)
            col3.button("Send Email", key="btn_email_send", disabled=True)
            col4.button("Clear Draft", key="btn_email_clear", disabled=True)


def render_meeting_builder(mcp_client, db) -> None:
    """Render the meeting builder UI."""
    st.subheader("Meeting Assistant 🐂")

    if st.session_state.meeting_fields_reset_pending:
        st.session_state.meeting_summary_input = ""
        st.session_state.meeting_duration_input = 30
        st.session_state.meeting_attendees_input = ""
        st.session_state.meeting_description_input = ""
        st.session_state.meeting_location_input = ""
        st.session_state.meeting_timezone_input = "US/Eastern (EST)"
        st.session_state.meeting_date_input = date.today()
        st.session_state.meeting_time_input = datetime.now().replace(second=0, microsecond=0).time()
        st.session_state.pending_meeting = None
        st.session_state.meeting_fields_reset_pending = False

    plan = st.session_state.pending_meeting

    if not plan:
        st.text_input("Meeting Summary", key="meeting_summary_input")

        col_dt, col_tm = st.columns(2)
        col_dt.date_input("Meeting Date", key="meeting_date_input")
        col_tm.time_input("Start Time", key="meeting_time_input", step=300)

        st.selectbox("Timezone", options=list(MEETING_TIMEZONE_OFFSETS.keys()), key="meeting_timezone_input")
        st.number_input("Duration (minutes)", min_value=15, max_value=240, key="meeting_duration_input")
        st.text_input("Attendees (comma-separated)", key="meeting_attendees_input")
        st.text_input("Location (optional)", key="meeting_location_input")
        st.text_area("Description / Notes", key="meeting_description_input", height=120)

        col_check, col_reset = st.columns([3, 1])

        if not st.session_state.get("is_processing", False):
            if col_check.button("Check Availability / Update Plan", key="btn_meeting_plan", use_container_width=True):
                start_iso = build_start_iso(
                    st.session_state.meeting_date_input,
                    st.session_state.meeting_time_input,
                    st.session_state.meeting_timezone_input,
                )
                st.session_state.pending_meeting_plan = {
                    "summary": st.session_state.meeting_summary_input,
                    "start_iso": start_iso,
                    "duration": int(st.session_state.meeting_duration_input),
                    "attendees": st.session_state.meeting_attendees_input,
                    "description": st.session_state.meeting_description_input,
                    "location": st.session_state.meeting_location_input,
                }
                _log_assistant_event(
                    "meeting",
                    "plan_request",
                    {
                        "duration": int(st.session_state.meeting_duration_input),
                        "attendee_count": len([a.strip() for a in st.session_state.meeting_attendees_input.split(',') if a.strip()]),
                    },
                )
                st.session_state.is_processing = True
                st.session_state.show_meeting_builder = False
                st.rerun()

            if col_reset.button("Reset Fields", key="btn_meeting_reset", use_container_width=True):
                st.session_state.meeting_fields_reset_pending = True
                _log_assistant_event("meeting", "reset_fields")
                st.rerun()
        else:
            col_check.button("Check Availability / Update Plan", key="btn_meeting_plan", use_container_width=True, disabled=True)
            col_reset.button("Reset Fields", key="btn_meeting_reset", use_container_width=True, disabled=True)

        st.info("No meeting plan yet. Enter details above and click Check Availability.")

    else:
        status = "✅ Slot is free" if plan.get("slot_free") else "⚠️ Slot is busy"
        st.info(status)

        attendees_display = ", ".join(plan.get("attendees", [])) or "None provided"
        st.markdown(
            f"**When:** {plan['start']}  \n"
            f"**Duration:** {plan['duration']} minutes  \n"
            f"**Attendees:** {attendees_display}  \n"
            f"**Location:** {plan.get('location') or 'TBD'}"
        )

        if plan.get("suggested"):
            st.caption(f"Suggested alternative: {plan['suggested']}")

        if st.session_state.meeting_notes_sync_value is not None:
            st.session_state.meeting_notes_text = st.session_state.meeting_notes_sync_value
            st.session_state.meeting_notes_sync_value = None

        st.text_area("Meeting Description / Notes (editable)", key="meeting_notes_text", height=180)
        st.text_input("AI edit instructions (optional)", key="meeting_edit_instructions")

        col1, col2, col3, col4 = st.columns(4)

        if not st.session_state.get("is_processing", False):
            if col1.button("Apply AI Edit", key="btn_meeting_ai_edit"):
                st.session_state.pending_meeting_edit = {
                    "instructions": st.session_state.meeting_edit_instructions,
                }
                _log_assistant_event(
                    "meeting",
                    "ai_edit_request",
                    {"has_instructions": bool(st.session_state.meeting_edit_instructions.strip())},
                )
                st.session_state.is_processing = True
                st.session_state.show_meeting_builder = False
                st.rerun()

            if col2.button("Save Manual Edit", key="btn_meeting_manual_edit"):
                from agents.meeting_assistant import save_manual_meeting_edit
                if save_manual_meeting_edit(st.session_state.meeting_notes_text):
                    st.success("Meeting notes updated.")
                    _log_assistant_event("meeting", "manual_edit_saved")

            if col3.button("Create Event", key="btn_meeting_create", use_container_width=True):
                create_meeting_event(mcp_client, db)
                _log_assistant_event("meeting", "create_event")
                st.rerun()

            if col4.button("Clear Plan", key="btn_meeting_clear", use_container_width=True):
                st.session_state.pending_meeting = None
                st.session_state.meeting_notes_sync_value = None
                _log_assistant_event("meeting", "clear_plan")
                st.rerun()
        else:
            col1.button("Apply AI Edit", key="btn_meeting_ai_edit", disabled=True)
            col2.button("Save Manual Edit", key="btn_meeting_manual_edit", disabled=True)
            col3.button("Create Event", key="btn_meeting_create", use_container_width=True, disabled=True)
            col4.button("Clear Plan", key="btn_meeting_clear", use_container_width=True, disabled=True)