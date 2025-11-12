"""Simple session-based authentication using query parameters."""
import streamlit as st
from typing import Any, Optional
import secrets
from datetime import datetime, timedelta


def _ensure_session_init() -> None:
    """Initialize session state for authentication if not exists."""
    if "auth_sessions" not in st.session_state:
        st.session_state.auth_sessions = {}


def issue_session_token(user_id: str, username: str) -> str:
    """
    Issue a new session token for a user.
    Returns the token and adds it to query parameters.
    """
    _ensure_session_init()

    # Generate token
    token = secrets.token_urlsafe(32)

    # Store session data keyed by token
    st.session_state.auth_sessions[token] = {
        "user_id": user_id,
        "username": username,
        "issued_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(days=7),
    }

    # Add token to URL query parameters
    st.query_params["session_token"] = token

    return token


def get_session_from_token(token: str) -> dict[str, Any] | None:
    """Retrieve session data for a specific token."""
    _ensure_session_init()

    if not token:
        return None

    session = st.session_state.auth_sessions.get(token)
    if not session:
        return None

    # Check if expired
    if datetime.utcnow() > session.get("expires_at", datetime.min):
        # Token expired, remove it
        st.session_state.auth_sessions.pop(token, None)
        return None

    return session


def get_session_token() -> Optional[str]:
    """Get session token from query parameters."""
    params = st.query_params
    token = params.get("session_token")

    # Handle list values
    if isinstance(token, list):
        token = token[0] if token else None

    return token


def revoke_session(token: str | None = None) -> None:
    """Revoke a session token."""
    _ensure_session_init()

    if not token:
        token = get_session_token()

    if token:
        st.session_state.auth_sessions.pop(token, None)

    # Clear query params
    if "session_token" in st.query_params:
        del st.query_params["session_token"]


def revoke_all_sessions() -> None:
    """Revoke all sessions for current user."""
    _ensure_session_init()
    st.session_state.auth_sessions.clear()

    # Clear query params
    if "session_token" in st.query_params:
        del st.query_params["session_token"]
