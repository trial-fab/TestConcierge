import streamlit as st
from typing import Any, Optional
from utils.formatters import split_subject_from_body
from utils.rag import estimate_tokens
from utils.state_manager import queue_action_collapse
from tools.google_tools import GoogleWorkspaceError

def draft_email_via_mcp(
    mcp_client,
    db,
    student_message: str,
    *,
    subject: str | None = None,
    instructions: str | None = None,
    previous_draft: str | None = None,
    placeholder=None,
) -> dict[str, Any] | None:
    try:
        draft = mcp_client.draft_email(
            student_message,
            subject=subject,
            instructions=instructions,
            previous_draft=previous_draft,
            session_id=st.session_state.current_session_id,
        )
    except RuntimeError as exc:
        if placeholder is not None:
            placeholder.error(f"Email drafting failed: {exc}")
        else:
            st.error(f"Email drafting failed: {exc}")
        return None

    body = draft.get("body", "")
    if placeholder is not None and body:
        placeholder.markdown(body)
    return draft

def start_email_draft(mcp_client, db, to_addr: str, subject: str, student_msg: str) -> None:
    to_addr = (to_addr or "").strip()
    subject = (subject or "USF Follow-up").strip()
    student_msg = (student_msg or "").strip()

    if not to_addr or not student_msg:
        st.warning("Please enter both the student email and their inquiry to generate a draft.")
        return

    in_toks = estimate_tokens(student_msg)

    # Draft email without showing in chat history
    drafted = draft_email_via_mcp(
        mcp_client,
        db,
        student_msg,
        subject=subject,
        placeholder=None,
    )

    if not drafted:
        return

    cleaned_draft = drafted.get("body", "")
    sources = drafted.get("sources", "")
    matched_chunks = drafted.get("context_hits", [])
    inline_subject, cleaned_draft = split_subject_from_body(cleaned_draft)
    subject = drafted.get("subject") or inline_subject or subject

    if subject:
        st.session_state.email_subject_sync_value = subject

    # Log interaction but don't add to chat history
    mcp_client.log_interaction(
        st.session_state.current_session_id,
        "email_draft",
        {"to": to_addr, "subject": subject, "draft": cleaned_draft, "chunks": matched_chunks},
    )

    st.session_state.token_total += in_toks
    st.session_state.pending_email = {
        "to": to_addr,
        "subject": subject,
        "body": cleaned_draft,
        "student_msg": student_msg,
    }
    st.session_state.email_draft_sync_value = cleaned_draft

def apply_email_edit(mcp_client, db, instructions: str) -> None:
    """Apply AI-powered edits to an existing email draft."""
    pending = st.session_state.pending_email
    if not pending:
        st.warning("No email draft is available. Generate a draft first.")
        return

    instructions = (instructions or "").strip()
    if not instructions:
        st.warning("Enter edit instructions before applying an AI edit.")
        return

    # Update email draft without showing in chat history
    drafted = draft_email_via_mcp(
        mcp_client,
        db,
        pending.get("student_msg", ""),
        subject=pending.get("subject"),
        instructions=instructions,
        previous_draft=pending.get("body", ""),
        placeholder=None,
    )

    if not drafted:
        return

    revised = drafted.get("body", pending.get("body", ""))
    sources = drafted.get("sources", "")
    inline_subject, revised = split_subject_from_body(revised)
    new_subject = drafted.get("subject") or inline_subject

    if new_subject:
        pending["subject"] = new_subject
        st.session_state.email_subject_sync_value = new_subject

    # Log interaction but don't add to chat history
    mcp_client.log_interaction(
        st.session_state.current_session_id,
        "email_edit",
        {
            "instructions": instructions,
            "subject": pending["subject"],
            "draft": revised,
            "chunks": drafted.get("context_hits", []),
        },
    )
    pending["body"] = revised
    st.session_state.pending_email = pending
    st.session_state.email_draft_sync_value = revised

def save_manual_email_edit(text: str) -> bool:
    """Save manual edits to the email draft."""
    pending = st.session_state.pending_email
    if not pending:
        st.warning("No email draft to update.")
        return False
    pending["body"] = text
    st.session_state.email_draft_sync_value = None
    return True

def send_email_draft(mcp_client, db) -> None:
    """Send the current email draft."""
    pending = st.session_state.pending_email
    if not pending:
        st.warning("No email draft to send.")
        return

    try:
        message_id = mcp_client.send_email(pending["to"], pending["subject"], pending["body"])
    except (GoogleWorkspaceError, RuntimeError) as e:
        error_text = f"Email delivery failed: {e}"
        with st.chat_message("assistant"):
            st.error(error_text)
        mcp_client.log_interaction(
            st.session_state.current_session_id,
            "email_send_failed",
            {"to": pending.get("to"), "subject": pending.get("subject"), "error": str(e)},
        )
        return

    confirmation = f"Email sent to {pending['to']} (id: {message_id})."
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
        "email_sent",
        {"to": pending["to"], "subject": pending["subject"], "message_id": message_id},
    )

    sent_action = {
        "to": pending["to"],
        "subject": pending["subject"],
        "body": pending["body"],
        "message_id": message_id,
    }

    st.session_state.token_total += out_toks
    st.session_state.pending_email = None
    st.session_state.email_draft_sync_value = None
    queue_action_collapse("email", sent_action)