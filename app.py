## app.py - Streamlined main application
import os
from pathlib import Path

# Load config before importing Streamlit
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / ".streamlit" / "config.toml"
if CONFIG_PATH.exists():
    os.environ.setdefault("STREAMLIT_CONFIG_FILE", str(CONFIG_PATH))

import streamlit as st
from dotenv import load_dotenv

# Prevent transformers from importing heavy backends
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

# Import utility modules
from utils.database import ChatDatabase
from utils.rag import generate_with_rag, estimate_tokens
from utils.security import sanitize_user_input, is_injection, AuthManager
from agents.mcp import SimpleMCPClient
from tools.google_tools import GoogleWorkspaceTools
from utils.streaming import SmoothStreamer
from utils.ui_helpers import inject_global_styles, scroll_chat_to_bottom
from utils.session_manager import (
    get_session_token,
    get_session_from_token,
    issue_session_token,
    revoke_session,
)
from utils.state_manager import (
    initialize_session_state,
    handle_pending_action_collapses,
    maybe_auto_open_assistant,
)
from utils.formatters import format_est_timestamp
from components.assistants import (
    render_tool_picker,
    render_email_builder,
    render_meeting_builder,
)

# Load environment variables
load_dotenv()

SESSION_TOKEN_LIMIT = int(os.environ.get("SESSION_TOKEN_LIMIT", "1500"))

# Initialize services
db = ChatDatabase()
google_tools = GoogleWorkspaceTools()
mcp_client = SimpleMCPClient(chat_db=db, google_tools=google_tools)

# Persist auth across reruns
if "auth" not in st.session_state:
    st.session_state.auth = AuthManager()
auth = st.session_state.auth

# Page configuration
st.set_page_config(
    page_title="USF Campus Concierge",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_global_styles()

# Initialize session state
initialize_session_state()


def recompute_token_total(msgs: list[dict]) -> int:
    """Count only user+assistant tokens for the session budget."""
    return sum(
        estimate_tokens(m.get("content", ""))
        for m in msgs
        if m.get("role") in ("user", "assistant")
    )


# Handle simple token-based authentication
# Try to restore session from query parameter
if not st.session_state.authenticated:
    session_token = get_session_token()
    if session_token:
        session_payload = get_session_from_token(session_token)
        if session_payload:
            st.session_state.authenticated = True
            st.session_state.user_id = session_payload["user_id"]
            st.session_state.username = session_payload["username"]

# Handle pending login (after user submits login form)
pending_login = st.session_state.get("pending_login")
if pending_login:
    # Issue new session token
    user_id = pending_login.get("user_id")
    username = pending_login.get("username")
    issue_session_token(user_id, username)

    # Update session state
    st.session_state.authenticated = True
    st.session_state.user_id = user_id
    st.session_state.username = username
    st.session_state.show_dashboard = True
    st.session_state.pending_login = None
    st.rerun()

# Reduce padding at bottom
st.markdown("""
<style>
[data-testid="stAppViewContainer"] .block-container {
  padding-bottom: 0 !important;
}
[data-testid="stAppViewContainer"] section > div { padding-bottom: 0 !important; }
[data-testid="stAppViewContainer"] section > div > div { padding-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# Login/Register Page
if not st.session_state.authenticated:
    login_shell = st.empty()
    with login_shell.container():
        st.markdown(
            """
            <style>
            header[data-testid="stHeader"],
            div[data-testid="stToolbar"],
            div[data-testid="stDecoration"] {
                display: none !important;
                height: 0 !important;
                padding: 0 !important;
            }
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > .main,
            [data-testid="stAppViewContainer"] > .main > div {
                padding-top: 0 !important;
                margin-top: 0 !important;
            }
            .block-container {
                padding-top: 32vh !important;
                margin-top: 0 !important;
                padding-bottom: 0 !important;
                margin-bottom: 0 !important;
            }
            .hero-fixed {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 35vh;
                z-index: 0;
            }
            </style>
            <div class="usf-hero hero-fixed">
                <h1 class="hero-heading"><span class="emoji">🐂</span>USF Campus Concierge</h1>
                <p>AI Assistant for Registration, Orientation, & Admissions</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            tab1, tab2 = st.tabs(["Login", "Register"])

            with tab1:
                with st.form("login_form"):
                    st.subheader("Welcome Back!")
                    login_username = st.text_input("Username", key="login_username").strip()
                    login_password = st.text_input("Password", type="password", key="login_password")
                    submit = st.form_submit_button("Login", use_container_width=True, type="primary")

                    if submit:
                        success, user_id = auth.authenticate_user(login_username, login_password)

                        if success:
                            st.session_state.pending_login = {
                                "user_id": user_id,
                                "username": login_username,
                            }
                            st.rerun()
                        else:
                            st.error("Invalid username or password")

            with tab2:
                with st.form("register_form"):
                    st.subheader("Create Account")
                    reg_username = st.text_input("Username", key="reg_username").strip()
                    reg_email = st.text_input("Email (optional)", key="reg_email").strip()
                    reg_password = st.text_input("Password", type="password", key="reg_password")
                    reg_password2 = st.text_input("Confirm Password", type="password")
                    submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

                    if submit:
                        if not reg_username or not reg_password:
                            st.error("Username and password are required")
                        elif reg_password != reg_password2:
                            st.error("Passwords don't match")
                        elif len(reg_password) < 6:
                            st.error("Password must be at least 6 characters")
                        else:
                            success, message = auth.create_user(reg_username, reg_password, reg_email)

                            if success:
                                st.success("Account created! Please login.")
                            else:
                                st.error(f"Registration failed: {message}")

    st.stop()

# Main Application
else:
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")

        if st.button("🚪 Logout", use_container_width=True):
            # Revoke session token
            revoke_session()

            # Clear session state
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.session_state.token_total = 0
            st.session_state.limit_reached = False
            st.session_state.show_dashboard = True
            st.rerun()

        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.current_session_id = None
            st.session_state.show_dashboard = True
            st.rerun()

        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            from datetime import datetime
            session_name = f"Chat {datetime.now().strftime('%b %d, %H:%M')}"
            sid = db.create_session(st.session_state.user_id, session_name)
            if not sid:
                st.error("Unable to create a new session. Please try again.")
            else:
                st.session_state.current_session_id = sid
                st.session_state.messages = [
                    {"role": "system", "content": "Assistant configured."}
                ]
                st.session_state.token_total = 0
                st.session_state.limit_reached = False
                st.session_state.show_dashboard = False
                st.rerun()

        # Search
        search_query = st.text_input("🔍 Search sessions", key="search_input")

        # Filter sessions
        sessions = db.search_sessions(st.session_state.user_id, search_query)

        if sessions:
            st.markdown(f"### 📁 Sessions ({len(sessions)})")

            with st.container(height=435, border=True):
                for session in sessions:
                    session_id = session.get("id")
                    is_current = session_id == st.session_state.current_session_id
                    button_type = "primary" if is_current else "secondary"
                    if st.button(
                        f"💬 {session['session_name']}",
                        key=f"session_{session_id}",
                        use_container_width=True,
                        type=button_type,
                    ):
                        st.session_state.current_session_id = session_id
                        db_messages = db.get_session_messages(session_id)
                        st.session_state.messages = [
                            {"role": "system", "content": "Assistant configured."}
                        ] + [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in db_messages
                        ]
                        st.session_state.token_total = recompute_token_total(st.session_state.messages)
                        st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                        st.rerun()
        else:
            st.info("No sessions found")

    # Main Chat Area
    if st.session_state.current_session_id:
        sessions = db.get_user_sessions(st.session_state.user_id)
        current_session = next((s for s in sessions if s.get("id") == st.session_state.current_session_id), None)

        if current_session:
            # Check if user is a demo user
            is_demo_user = st.session_state.username and st.session_state.username.startswith("DemoUser")
            if is_demo_user:
                try:
                    user_num = int(st.session_state.username.replace("DemoUser", ""))
                    is_demo_user = 1 <= user_num <= 20
                except:
                    is_demo_user = False

            if is_demo_user:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            else:
                col1, col2, col3 = st.columns([3, 1, 1])
                col4 = None

            with col1:
                st.title(current_session["session_name"])

            with col2:
                msg_count = len(st.session_state.messages) - 1
                st.metric("Messages", msg_count)

            with col3:
                options_prefix = f"session_options_{current_session['id']}"
                rename_key = f"{options_prefix}_rename"
                with st.popover("✏️ Options", use_container_width=True):
                    default_name = current_session["session_name"]
                    rename_value = st.text_input(
                        "Rename session",
                        value=default_name,
                        key=rename_key,
                    )
                    if st.button("Save name", key=f"{options_prefix}_rename_save", use_container_width=True):
                        final_name = (rename_value or default_name).strip()
                        if final_name != default_name:
                            db.rename_session(st.session_state.current_session_id, final_name)
                        st.rerun()

                    st.divider()

                    export_json = db.export_session_json(
                        st.session_state.user_id,
                        st.session_state.current_session_id,
                    )
                    st.download_button(
                        "⬇️ Export",
                        data=export_json,
                        file_name=f"{current_session['session_name']}.json",
                        mime="application/json",
                        use_container_width=True,
                        key=f"{options_prefix}_export",
                    )

                    if st.button("🗑️ Delete session", key=f"{options_prefix}_delete", use_container_width=True):
                        db.delete_session(st.session_state.current_session_id)
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                        st.session_state.token_total = 0
                        st.session_state.limit_reached = False
                        st.rerun()

            # Demo queries button for demo users
            if col4 is not None:
                with col4:
                    with st.popover("🧪 Demo", use_container_width=True):
                        st.markdown("### Demo Queries")
                        st.markdown("Copy and paste these queries to test the bot:")

                        st.markdown("**Regular Bot:**")
                        query1 = "What are the requirements to transfer to USF?"
                        query2 = "How do I register for orientation?"

                        if st.button("📋 Copy Query 1", key="demo_regular_1", use_container_width=True):
                            st.code(query1, language=None)
                        st.caption(query1)

                        if st.button("📋 Copy Query 2", key="demo_regular_2", use_container_width=True):
                            st.code(query2, language=None)
                        st.caption(query2)

                        st.divider()
                        st.markdown("**Email Bot:**")
                        email_query1 = "Student asking about how to submit transcripts for admission"
                        email_query2 = "Student needs help with financial aid deadlines"

                        st.markdown(f"**To:** student@example.com")
                        st.markdown(f"**Subject:** Re: Your Question")
                        st.markdown(f"**Inquiry 1:**")
                        if st.button("📋 Copy", key="demo_email_1", use_container_width=True):
                            st.code(email_query1, language=None)
                        st.caption(email_query1)

                        st.markdown(f"**Inquiry 2:**")
                        if st.button("📋 Copy", key="demo_email_2", use_container_width=True):
                            st.code(email_query2, language=None)
                        st.caption(email_query2)

                        st.divider()
                        st.markdown("**Meeting Bot:**")
                        st.markdown("**Summary:** Advising Appointment")
                        st.markdown("**Description:**")
                        meeting_query1 = "Schedule a meeting to discuss course selection for next semester"
                        meeting_query2 = "Need to meet with advisor to review degree progress and graduation timeline"

                        if st.button("📋 Copy", key="demo_meeting_1", use_container_width=True):
                            st.code(meeting_query1, language=None)
                        st.caption(meeting_query1)

                        if st.button("📋 Copy", key="demo_meeting_2", use_container_width=True):
                            st.code(meeting_query2, language=None)
                        st.caption(meeting_query2)

        chat_col = st.container()

        with chat_col:
            history = st.session_state.messages[1:]
            show_welcome = (
                not history
                and not st.session_state.show_tool_picker
                and not st.session_state.show_email_builder
                and not st.session_state.show_meeting_builder
                and not st.session_state.recent_actions
            )
            if show_welcome:
                st.markdown(
                    "<div class='chat-welcome'><h2>What can I help with?</h2><p>Start a conversation or open an assistant with the ＋ button.</p></div>",
                    unsafe_allow_html=True,
                )

            for msg in history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # Only show regenerate button for regular query responses, not email/meeting outputs
            if history and history[-1]["role"] == "assistant":
                # Check if this is a regular query response (not email/meeting related)
                is_regular_query = not (
                    st.session_state.show_email_builder or
                    st.session_state.show_meeting_builder or
                    st.session_state.pending_email or
                    st.session_state.pending_meeting
                )
                if is_regular_query and st.button("🔄 Regenerate Last Response", key="regen_button"):
                    st.session_state.messages.pop()
                    st.session_state.pending_regen = True
                    st.rerun()

            # Render assistants
            if st.session_state.show_tool_picker:
                with st.chat_message("assistant"):
                    render_tool_picker()

            if st.session_state.show_email_builder:
                with st.chat_message("assistant"):
                    render_email_builder(mcp_client, db)

            if st.session_state.show_meeting_builder:
                with st.chat_message("assistant"):
                    render_meeting_builder(mcp_client, db)

        scroll_chat_to_bottom()

        # Input area
        # Button shows × if tool picker OR any assistant is open
        any_assistant_open = (
            st.session_state.show_tool_picker
            or st.session_state.show_email_builder
            or st.session_state.show_meeting_builder
        )
        tool_button_label = "×" if any_assistant_open else "＋"
        user_input = None
        toggle_col, input_col = st.columns([0.03, 0.97], gap=None)

        with toggle_col:
            if st.button(
                tool_button_label,
                key="chat_tool_toggle",
                help="Close Bulls assistants" if any_assistant_open else "Open Bulls assistants",
                use_container_width=True,
            ):
                # Close everything if anything is open, otherwise open tool picker
                if any_assistant_open:
                    # Close all
                    st.session_state.show_tool_picker = False
                    st.session_state.show_email_builder = False
                    st.session_state.show_meeting_builder = False
                else:
                    # Open tool picker
                    st.session_state.show_tool_picker = True

                st.rerun()

        with input_col:
            if st.session_state.limit_reached:
                st.warning(
                    f"Session token budget reached "
                    f"({st.session_state.token_total}/{SESSION_TOKEN_LIMIT}). "
                    "Please open a new session to continue."
                )
                user_input = None
            elif st.session_state.is_processing:
                # Show transparent overlay to prevent input while still showing the chat input
                st.markdown(
                    """
                    <div style="
                        background-color: transparent;
                        border: none;
                        border-radius: 0.5rem;
                        padding: 0.75rem 1rem;
                        height: 3rem;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        pointer-events: none;
                    ">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                user_input = None
            else:
                user_input = st.chat_input("Ask the USF Campus Concierge...")

            # Handle regeneration
            if st.session_state.pending_regen and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                if not st.session_state.is_processing:
                    # First time - set processing and rerun to disable input
                    st.session_state.is_processing = True
                    st.rerun()
                else:
                    # Already processing - execute regeneration
                    last_user = st.session_state.messages[-1]["content"]
                    regen_id = hash(("regen", last_user))

                    # Check if we've already started this regeneration
                    if st.session_state.get("processing_regen_id") != regen_id:
                        # First time processing this regeneration
                        st.session_state.processing_regen_id = regen_id

                    with chat_col:
                        # Show thinking indicator
                        with st.chat_message("assistant"):
                            thinking_placeholder = st.empty()
                            thinking_placeholder.markdown("Thinking...")

                            streamer = SmoothStreamer(thinking_placeholder)
                            final_text = None
                            matched_chunks = []
                            last_chunk = ""

                            for kind, payload in generate_with_rag(last_user, mcp_client=mcp_client):
                                text = payload.get("text", "")
                                if not text:
                                    continue
                                last_chunk = text
                                streamer.update(text)
                                if kind != "delta":
                                    final_text = text
                                    matched_chunks = payload.get("hits", [])

                            streamer.finalize(final_text or last_chunk)

                    if final_text is None:
                        final_text = last_chunk

                    out_toks = estimate_tokens(final_text or "")
                    st.session_state.token_total += out_toks
                    st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
                    db.add_message(
                        st.session_state.current_session_id,
                        "assistant",
                        final_text,
                        tokens_out=out_toks,
                    )
                    mcp_client.log_interaction(
                        st.session_state.current_session_id,
                        "assistant_regen",
                        {"query": last_user, "response": final_text, "chunks": matched_chunks},
                    )
                    maybe_auto_open_assistant(final_text)
                    st.session_state.pending_regen = False
                    st.session_state.processing_regen_id = None
                    st.session_state.is_processing = False
                    st.rerun()

        # Handle user input
        if user_input:
            # Ignore input if already processing to prevent duplicates
            if st.session_state.is_processing:
                st.rerun()
            else:
                # Store input and set processing state, then rerun to disable input
                st.session_state.pending_user_input = user_input
                st.session_state.is_processing = True
                st.rerun()

        # Process pending user input
        if st.session_state.is_processing and st.session_state.pending_user_input:
            # Generate a unique ID for this input to prevent duplicate processing
            current_input = st.session_state.pending_user_input
            input_id = hash(current_input)

            # Check if we've already started processing this input
            if st.session_state.get("processing_input_id") == input_id:
                # Already processing this input - rerun interrupted us, so continue
                clean = st.session_state.get("processing_query", "")
                if not clean:
                    # Safety: shouldn't happen, but if it does, clear and restart
                    st.session_state.pending_user_input = None
                    st.session_state.processing_input_id = None
                    st.session_state.processing_query = None
                    st.session_state.processing_tokens_in = None
                    st.session_state.processing_cached_result = None
                    st.session_state.processing_response_saved = False
                    st.session_state.is_processing = False
                    st.rerun()
                # Messages already added and displayed from history
                # Just create the thinking indicator - don't duplicate the user message
                in_toks = st.session_state.get("processing_tokens_in", 0)
                with chat_col:
                    # User message already displayed from history, so only show thinking indicator
                    with st.chat_message("assistant"):
                        thinking_placeholder = st.empty()
                        thinking_placeholder.markdown("Thinking...")
            else:
                # First time processing this input
                handle_pending_action_collapses()
                clean = sanitize_user_input(current_input)
                st.session_state.processing_input_id = input_id
                st.session_state.processing_query = clean

                in_toks = estimate_tokens(clean)
                st.session_state.processing_tokens_in = in_toks
                st.session_state.messages.append({"role": "user", "content": clean})
                db.add_message(
                    st.session_state.current_session_id,
                    "user",
                    clean,
                    tokens_in=in_toks,
                )

                with chat_col:
                    with st.chat_message("user"):
                        st.write(clean)

                    # Show thinking indicator
                    with st.chat_message("assistant"):
                        thinking_placeholder = st.empty()
                        thinking_placeholder.markdown("Thinking...")

            # Check for injection
            if is_injection(clean):
                warn = "That looks like a prompt-injection attempt. For safety, I can't run that. Try a normal question."
                thinking_placeholder.markdown(warn)

                out_toks = estimate_tokens(warn)
                st.session_state.token_total += (in_toks + out_toks)
                st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                st.session_state.messages.append({"role": "assistant", "content": warn})
                db.add_message(
                    st.session_state.current_session_id,
                    "assistant",
                    warn,
                    tokens_out=out_toks,
                )
                mcp_client.log_interaction(
                    st.session_state.current_session_id,
                    "injection_blocked",
                    {"prompt": clean, "response": warn},
                )
                st.session_state.pending_user_input = None
                st.session_state.processing_input_id = None
                st.session_state.processing_query = None
                st.session_state.processing_tokens_in = None
                st.session_state.processing_cached_result = None
                st.session_state.processing_response_saved = False
                st.session_state.is_processing = False
                st.rerun()

            # Generate response with RAG (or use cached result)
            cached_result = st.session_state.get("processing_cached_result")
            response_already_saved = st.session_state.get("processing_response_saved", False)

            if cached_result and cached_result.get("input_id") == st.session_state.processing_input_id:
                # Use cached result from previous completed run
                final_text = cached_result.get("text")
                matched_chunks = cached_result.get("chunks", [])
                thinking_placeholder.markdown(final_text)
            else:
                # Generate new result
                streamer = SmoothStreamer(thinking_placeholder)
                final_text = None
                matched_chunks = []
                last_chunk = ""

                for kind, payload in generate_with_rag(clean, mcp_client=mcp_client):
                    text = payload.get("text", "")
                    if not text:
                        continue
                    last_chunk = text
                    streamer.update(text)
                    if kind != "delta":
                        final_text = text
                        matched_chunks = payload.get("hits", [])

                streamer.finalize(final_text or last_chunk)

                if final_text is None:
                    final_text = last_chunk

                # Cache the result to prevent recomputation on future reruns
                if final_text:
                    st.session_state.processing_cached_result = {
                        "input_id": st.session_state.processing_input_id,
                        "text": final_text,
                        "chunks": matched_chunks,
                    }

            if final_text is None:
                error_msg = "We weren't able to generate a response. Please try again."
                thinking_placeholder.markdown(error_msg)
                mcp_client.log_interaction(
                    st.session_state.current_session_id,
                    "assistant_error",
                    {"prompt": clean, "error": "empty_response"},
                )
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                db.add_message(
                    st.session_state.current_session_id,
                    "assistant",
                    error_msg,
                    tokens_out=estimate_tokens(error_msg),
                )
                st.session_state.pending_user_input = None
                st.session_state.processing_input_id = None
                st.session_state.processing_query = None
                st.session_state.processing_tokens_in = None
                st.session_state.processing_cached_result = None
                st.session_state.processing_response_saved = False
                st.session_state.is_processing = False
                st.rerun()

            # Only save response if we haven't already saved it
            # This prevents duplicates when reruns occur after response generation
            if not response_already_saved:
                out_toks = estimate_tokens(final_text or "")
                st.session_state.token_total += (in_toks + out_toks)
                st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                db.add_message(
                    st.session_state.current_session_id,
                    "assistant",
                    final_text,
                    tokens_out=out_toks,
                )
                mcp_client.log_interaction(
                    st.session_state.current_session_id,
                    "assistant_reply",
                    {
                        "prompt": clean,
                        "response": final_text,
                        "chunks": matched_chunks,
                        "tokens_in": in_toks,
                        "tokens_out": out_toks,
                    },
                )
                maybe_auto_open_assistant(final_text)
                # Mark response as saved to prevent duplicate saves on future reruns
                st.session_state.processing_response_saved = True

            # Clear processing state
            st.session_state.pending_user_input = None
            st.session_state.processing_input_id = None
            st.session_state.processing_query = None
            st.session_state.processing_response_saved = False
            st.session_state.is_processing = False
            st.rerun()

    else:
        # Dashboard
        st.title("🐂 Welcome to USF Campus Concierge")
        st.markdown("### AI Assistant for Registration, Orientation, & Admissions")

        st.divider()

        handle_pending_action_collapses()

        # Stats
        sessions = db.get_user_sessions(st.session_state.user_id)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("📁 Total Sessions", len(sessions))

        with col2:
            # Optimized: Single query instead of N queries
            total_messages = db.get_total_message_count(st.session_state.user_id)
            st.metric("💬 Total Messages", total_messages)

        st.divider()

        # Recent sessions
        if st.session_state.get("show_dashboard", True):
            st.subheader("📌 Recent Sessions")

            if sessions:
                for session in sessions[:5]:
                    session_id = session.get("id")
                    messages = db.get_session_messages(session_id)
                    msg_count = len(messages)
                    created_label = format_est_timestamp(session.get("created_at"))
                    updated_label = format_est_timestamp(session.get("updated_at"))
                    header = f"💬 {session['session_name']}"
                    with st.expander(header, expanded=False):
                        st.markdown(
                            f"**Created:** {created_label}  \n"
                            f"**Updated:** {updated_label}  \n"
                            f"**Messages:** {msg_count}"
                        )

                        if st.button("Open", key=f"open_{session_id}"):
                            st.session_state.current_session_id = session_id
                            st.session_state.messages = [
                                {"role": "system", "content": "Assistant configured."}
                            ] + [
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in messages
                            ]
                            st.session_state.token_total = recompute_token_total(st.session_state.messages)
                            st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                            st.session_state.show_dashboard = False
                            st.rerun()
            else:
                st.info("👈 Create your first session to start chatting!")

            if st.session_state.recent_actions:
                st.subheader("📝 Recent Assisted Actions")
                for idx, action in enumerate(st.session_state.recent_actions):
                    data = action.get("data", {})
                    timestamp_label = format_est_timestamp(action.get("timestamp"))
                    if action.get("type") == "email":
                        label = f"Email to {data.get('to', '(unknown)')} • {timestamp_label}"
                    else:
                        label = f"Meeting: {data.get('summary', 'Untitled')} • {timestamp_label}"

                    with st.expander(label, expanded=False):
                        if action.get("type") == "email":
                            st.write(f"**Subject:** {data.get('subject', '(no subject)')}")
                            st.write(f"**Message ID:** {data.get('message_id', 'pending')}")
                            st.text_area(
                                "Email Body",
                                data.get("body") or "",
                                height=150,
                                disabled=True,
                                key=f"email_log_dash_{idx}",
                            )
                        else:
                            attendees = ", ".join(data.get("attendees", [])) or "None provided"
                            duration_display = f"{data.get('duration')} min" if data.get("duration") else "N/A"
                            st.write(f"**When:** {format_est_timestamp(data.get('start'))} ({duration_display})")
                            st.write(f"**Attendees:** {attendees}")
                            st.write(f"**Location:** {data.get('location') or 'TBD'}")
                            st.write(f"**Event ID:** {data.get('event_id', 'pending')}")
                            st.write(f"**Calendar Summary:** {data.get('summary', 'Meeting')}")
                            if data.get("ai_notes"):
                                st.caption(data["ai_notes"])
                            if data.get("meeting_link"):
                                st.write(f"**Meet Link:** {data.get('meeting_link')}")
        else:
            st.info("Use the sidebar to return to your dashboard and see recent sessions/actions.")
