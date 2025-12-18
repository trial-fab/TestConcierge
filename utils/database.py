import os
import uuid
import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, Optional

from utils.supabase_client import get_supabase_client
from utils.security import escape_sql_like
from utils.state_manager import (
    get_cached_sessions,
    set_cached_sessions,
    invalidate_session_cache,
    get_cached_messages,
    set_cached_messages,
    invalidate_messages_cache
)
from utils.splunk_logger import get_splunk_logger

logger = logging.getLogger(__name__)
splunk_logger = get_splunk_logger()

class ChatDatabase:
    """
    Supabase-backed chat persistence for sessions, messages, and audit logs.
    """

    def __init__(
        self,
        sessions_table: str | None = None,
        messages_table: str | None = None,
        audit_table: str | None = None,
    ):
        self._client = get_supabase_client()
        self._sessions_table = sessions_table or os.getenv("SUPABASE_SESSIONS_TABLE", "chat_sessions")
        self._messages_table = messages_table or os.getenv("SUPABASE_MESSAGES_TABLE", "messages")
        self._audit_table = audit_table or os.getenv("SUPABASE_AUDIT_TABLE", "audit_logs")

        self._audit_batch = []
        self._audit_lock = threading.Lock()
        self._audit_batch_size = 10
        self._audit_worker_thread = None
        self._audit_shutdown_event = threading.Event()
        self._start_audit_worker()

    def _log_db_event(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        *,
        result_count: int = 0,
        error: Optional[str] = None,
        table: Optional[str] = None,
    ) -> None:
        splunk_logger.log_event(
            category="database",
            event_type="query",
            payload={
                "operation": operation,
                "table": table or self._sessions_table,
                "success": success,
                "error": error,
                "result_count": result_count,
            },
            metrics={"duration_ms": duration_ms},
            component="database",
        )

    def create_session(self, user_id: str, session_name: str) -> Optional[str]:
        start = time.perf_counter()
        success = False
        error_msg = None
        sid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat(timespec="seconds")
        record = {
            "id": sid,
            "user_id": user_id,
            "session_name": session_name,
            "created_at": now,
            "updated_at": now,
        }
        try:
            resp = (
                self._client.table(self._sessions_table)
                .insert(record)
                .execute()
            )
            success = bool(getattr(resp, "data", []))
            if success:
                invalidate_session_cache()
                return sid
            return None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to create session: {e}")
            return None
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_db_event(
                "create_session",
                duration_ms,
                success,
                result_count=1 if success else 0,
                error=error_msg,
                table=self._sessions_table,
            )

    def get_user_sessions(self, user_id: str) -> list[dict]:
        cached = get_cached_sessions(user_id)
        if cached is not None:
            return cached

        start = time.perf_counter()
        success = False
        error_msg = None
        results: list[dict] = []
        try:
            resp = (
                self._client.table(self._sessions_table)
                .select("*")
                .eq("user_id", user_id)
                .order("updated_at", desc=True)
                .execute()
            )
            results = getattr(resp, "data", []) or []
            set_cached_sessions(user_id, results)
            success = True
            return results
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to get user sessions for {user_id}: {e}")
            return []
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_db_event(
                "get_user_sessions",
                duration_ms,
                success,
                result_count=len(results) if success else 0,
                error=error_msg,
                table=self._sessions_table,
            )

    def get_session(self, session_id: str) -> Optional[dict]:
        start = time.perf_counter()
        success = False
        error_msg = None
        result = None
        try:
            resp = (
                self._client.table(self._sessions_table)
                .select("*")
                .eq("id", session_id)
                .limit(1)
                .execute()
            )
            data = getattr(resp, "data", None)
            result = data[0] if data else None
            success = result is not None
            return result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_db_event(
                "get_session",
                duration_ms,
                success,
                result_count=1 if success else 0,
                error=error_msg,
                table=self._sessions_table,
            )

    def rename_session(self, session_id: str, new_name: str) -> bool:
        start = time.perf_counter()
        now = datetime.utcnow().isoformat(timespec="seconds")
        success = False
        error_msg = None
        try:
            self._client.table(self._sessions_table).update(
                {"session_name": new_name, "updated_at": now}
            ).eq("id", session_id).execute()
            invalidate_session_cache()
            success = True
            return True
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to rename session {session_id}: {e}")
            return False
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_db_event(
                "rename_session",
                duration_ms,
                success,
                result_count=1 if success else 0,
                error=error_msg,
                table=self._sessions_table,
            )

    def delete_session(self, session_id: str) -> None:
        start = time.perf_counter()
        success = False
        error_msg = None
        try:
            self._client.table(self._messages_table).delete().eq("session_id", session_id).execute()
            self._client.table(self._sessions_table).delete().eq("id", session_id).execute()
            invalidate_session_cache()
            invalidate_messages_cache(session_id)
            success = True
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to delete session {session_id}: {e}")
            return
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_db_event(
                "delete_session",
                duration_ms,
                success,
                result_count=1 if success else 0,
                error=error_msg,
                table=self._sessions_table,
            )

    # messages
    def get_session_messages(self, session_id: str) -> list[dict]:
        cached = get_cached_messages(session_id)
        if cached is not None:
            return cached

        start = time.perf_counter()
        success = False
        error_msg = None
        results = []
        try:
            resp = (
                self._client.table(self._messages_table)
                .select("*")
                .eq("session_id", session_id)
                .order("created_at", desc=False)
                .execute()
            )
            results = getattr(resp, "data", []) or []
            set_cached_messages(session_id, results)
            success = True
            return results
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to get session messages for {session_id}: {e}")
            return []
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_db_event(
                "get_session_messages",
                duration_ms,
                success,
                result_count=len(results),
                error=error_msg,
                table=self._messages_table,
            )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> Optional[str]:
        start = time.perf_counter()
        success = False
        error_msg = None
        mid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat(timespec="seconds")
        record = {
            "id": mid,
            "session_id": session_id,
            "role": role,
            "content": content,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "created_at": now,
        }
        try:
            self._client.table(self._messages_table).insert(record).execute()
            self._client.table(self._sessions_table).update(
                {"updated_at": now}
            ).eq("id", session_id).execute()
            invalidate_messages_cache(session_id)
            invalidate_session_cache()
            success = True
            return mid
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to add message to session {session_id}: {e}")
            return None
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_db_event(
                "add_message",
                duration_ms,
                success,
                result_count=1 if success else 0,
                error=error_msg,
                table=self._messages_table,
            )

    def get_total_message_count(self, user_id: str) -> int:
        """
        Get total message count across all sessions for a user (optimized).
        Uses a single query instead of N queries.
        """
        start = time.perf_counter()
        success = False
        error_msg = None
        count = 0
        try:
            # Get all session IDs for this user
            sessions_resp = (
                self._client.table(self._sessions_table)
                .select("id")
                .eq("user_id", user_id)
                .execute()
            )
            session_ids = [s["id"] for s in (getattr(sessions_resp, "data", []) or [])]

            if not session_ids:
                duration_ms = (time.perf_counter() - start) * 1000
                return 0

            # Count all messages for these sessions in one query
            messages_resp = (
                self._client.table(self._messages_table)
                .select("id", count="exact")
                .in_("session_id", session_ids)
                .execute()
            )

            # Get count from response
            count = getattr(messages_resp, "count", 0) or 0
            success = True
            return count
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to get total message count for user {user_id}: {e}")
            return 0
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_db_event(
                "get_total_message_count",
                duration_ms,
                success,
                result_count=count if success else 0,
                error=error_msg,
                table=self._messages_table,
            )

    # search/export
    def search_sessions(self, user_id: str, query: str) -> list[dict]:
        start = time.perf_counter()
        error_msg = None
        success = False
        sessions = self.get_user_sessions(user_id)
        if not query:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_db_event(
                "search_sessions",
                duration_ms,
                True,
                result_count=len(sessions),
                table=self._sessions_table,
            )
            return sessions

        q = escape_sql_like((query or "").lower())
        matching_sessions = {}

        # First pass: Find sessions that match by name
        for s in sessions:
            name = s.get("session_name", "")
            if q in name.lower():
                matching_sessions[s.get("id")] = s

        # Second pass: Find sessions that match by message content
        try:
            session_ids = [s.get("id") for s in sessions if s.get("id") not in matching_sessions]
            if session_ids:
                resp = (
                    self._client.table(self._messages_table)
                    .select("session_id")
                    .in_("session_id", session_ids)
                    .ilike("content", f"%{q}%")
                    .execute()
                )
                matched_session_ids = {msg.get("session_id") for msg in (getattr(resp, "data", []) or [])}
                for s in sessions:
                    if s.get("id") in matched_session_ids:
                        matching_sessions[s.get("id")] = s

            success = True
            return list(matching_sessions.values())
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to search messages: {e}")
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_db_event(
                "search_sessions",
                duration_ms,
                success,
                result_count=len(matching_sessions),
                error=error_msg,
                table=self._messages_table,
            )

        return list(matching_sessions.values())

    def export_session_json(self, user_id: str, session_id: str) -> str:
        session = self.get_session(session_id)
        if not session or session.get("user_id") != user_id:
            return json.dumps({"error": "session not found"})
        messages = self.get_session_messages(session_id)
        export_data = {
            "session_name": session.get("session_name"),
            "created_at": session.get("created_at"),
            "messages": messages,
        }
        return json.dumps(export_data, indent=2)

    def _start_audit_worker(self):
        if not self._audit_table:
            return
        self._audit_worker_thread = threading.Thread(target=self._audit_worker_loop, daemon=True)
        self._audit_worker_thread.start()

    def _audit_worker_loop(self):
        while not self._audit_shutdown_event.is_set():
            time.sleep(2)
            with self._audit_lock:
                if len(self._audit_batch) >= self._audit_batch_size:
                    self._flush_audit_batch()

    def _flush_audit_batch(self):
        if not self._audit_batch:
            return
        batch = self._audit_batch.copy()
        self._audit_batch.clear()
        try:
            self._client.table(self._audit_table).insert(batch).execute()
        except Exception as e:
            logger.error(f"Failed to flush audit batch: {e}")

    def log_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        if not self._audit_table:
            return
        record = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "event_type": event_type,
            "payload": payload,
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        with self._audit_lock:
            self._audit_batch.append(record)
            if len(self._audit_batch) >= self._audit_batch_size:
                self._flush_audit_batch()

    def close(self):
        self._audit_shutdown_event.set()
        if self._audit_worker_thread and self._audit_worker_thread.is_alive():
            self._audit_worker_thread.join(timeout=2)
        with self._audit_lock:
            self._flush_audit_batch()
