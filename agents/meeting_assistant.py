import streamlit as st
from typing import Any
from utils.rag import estimate_tokens
from utils.state_manager import queue_action_collapse

def plan_meeting(
    mcp_client,
    db,
    summary: str,
    start_raw: str,
    duration: int,
    attendee_raw: str,
    description: str,
    location: str,
) -> None:

    summary = (summary or "Student Meeting").strip()
    start_raw = (start_raw or "").strip()
    attendee_raw = attendee_raw or ""
    description = (description or "").strip()
    location = (location or "").strip()
    duration = int(duration or 30)

    if not start_raw:
        st.warning("Enter a start date/time (ISO format) to check availability.")
        return

    attendees = [email.strip() for email in attendee_raw.split(",") if email.strip()]

    try:
        plan = mcp_client.plan_meeting(
            summary,
            start_raw,
            duration,
            attendees=attendees,
            agenda=description,
            location=location,
            session_id=st.session_state.current_session_id,
        )
    except RuntimeError as e:
        with st.chat_message("assistant"):
            st.error(str(e))
        return

    st.session_state.pending_meeting = plan
    slot_free = plan.get("slot_free", False)
    start_iso = plan.get("start", start_raw)
    suggested = plan.get("suggested")

    # Sync AI notes to editable text area
    if plan.get("ai_notes"):
        st.session_state.meeting_notes_sync_value = plan["ai_notes"]
    elif slot_free:
        assistant_msg = f"The {duration}-minute slot starting {start_iso} is free. Use Create Event when ready."
        st.session_state.meeting_notes_sync_value = assistant_msg
    else:
        suggestion_text = f" Suggested alternative: {suggested}" if suggested else ""
        assistant_msg = "The requested slot is busy." + suggestion_text
        st.session_state.meeting_notes_sync_value = assistant_msg

    mcp_client.log_interaction(
        st.session_state.current_session_id,
        "meeting_plan",
        {
            "summary": plan.get("summary", summary),
            "start": plan.get("start", start_raw),
            "duration": plan.get("duration", duration),
            "attendees": plan.get("attendees", attendees),
            "location": plan.get("location", location),
            "slot_free": slot_free,
            "suggested": suggested,
        },
    )

def create_meeting_event(mcp_client, db) -> None:
    """Create a calendar event from the current meeting plan."""
    plan = st.session_state.pending_meeting
    if not plan:
        st.warning("No meeting plan to create. Check availability first.")
        return

    description = st.session_state.meeting_notes_text or plan.get("description", "")

    try:
        event_info = mcp_client.create_event(
            plan["summary"],
            plan["start"],
            plan["duration"],
            attendees=plan.get("attendees"),
            description=description,
            location=plan.get("location", ""),
        )
    except RuntimeError as e:
        with st.chat_message("assistant"):
            st.error(str(e))
        return

    event_id = event_info.get("event_id", "event-created")
    meet_link = event_info.get("hangout_link", "")
    confirmation = f"Calendar event created for {plan['summary']} (id: {event_id})."
    if meet_link:
        confirmation += f"\nGoogle Meet: {meet_link}"

    with st.chat_message("assistant"):
        st.success(confirmation)

    out_toks = estimate_tokens(confirmation)
    st.session_state.messages.append({"role": "assistant", "content": confirmation})
    db.add_message(
        st.session_state.current_session_id,
        "assistant",
        confirmation,
        tokens_out=out_toks,
    )
    mcp_client.log_interaction(
        st.session_state.current_session_id,
        "meeting_created",
        {
            "summary": plan["summary"],
            "start": plan["start"],
            "duration": plan["duration"],
            "event_id": event_id,
            "hangout_link": meet_link,
        },
    )

    meeting_action = {
        "summary": plan["summary"],
        "start": plan["start"],
        "duration": plan["duration"],
        "attendees": plan.get("attendees", []),
        "location": plan.get("location", ""),
        "event_id": event_id,
        "meeting_link": meet_link,
        "ai_notes": plan.get("ai_notes", ""),
    }

    queue_action_collapse("meeting", meeting_action)
    st.session_state.pending_meeting = None
    st.session_state.meeting_notes_sync_value = None


def apply_meeting_edit(mcp_client, db, instructions: str) -> None:
    """Apply AI-powered edits to meeting notes."""
    plan = st.session_state.pending_meeting
    if not plan:
        st.warning("No meeting plan available. Generate a plan first.")
        return

    instructions = (instructions or "").strip()
    if not instructions:
        st.warning("Enter edit instructions before applying an AI edit.")
        return

    # Use mcp_client to regenerate meeting notes with instructions
    current_notes = st.session_state.meeting_notes_text or plan.get("ai_notes", "")

    try:
        updated_plan = mcp_client.plan_meeting(
            plan.get("summary", ""),
            plan.get("start", ""),
            plan.get("duration", 30),
            attendees=plan.get("attendees"),
            agenda=f"{current_notes}\n\nEdit instructions: {instructions}",
            location=plan.get("location", ""),
            session_id=st.session_state.current_session_id,
        )

        if updated_plan and updated_plan.get("ai_notes"):
            revised_notes = updated_plan["ai_notes"]
        else:
            st.warning("Could not generate updated notes.")
            return
    except RuntimeError as e:
        st.error(f"Failed to update notes: {e}")
        return

    plan["ai_notes"] = revised_notes
    plan["description"] = revised_notes
    st.session_state.pending_meeting = plan
    st.session_state.meeting_notes_sync_value = revised_notes

    mcp_client.log_interaction(
        st.session_state.current_session_id,
        "meeting_edit",
        {"instructions": instructions, "notes": revised_notes},
    )

def save_manual_meeting_edit(text: str) -> bool:
    """Save manual edits to meeting notes."""
    plan = st.session_state.pending_meeting
    if not plan:
        st.warning("No meeting plan to update.")
        return False

    plan["description"] = text
    plan["ai_notes"] = text
    st.session_state.pending_meeting = plan
    st.session_state.meeting_notes_sync_value = None
    return True