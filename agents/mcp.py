from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import anyio
except ImportError:
    anyio = None

from dotenv import load_dotenv

from utils.database import ChatDatabase
from tools.google_tools import GoogleWorkspaceTools
from utils.rag import retrieve_matches, format_context, build_sources_block
from utils.azure_llm import complete_chat
from utils.formatters import split_subject_from_body
from utils.splunk_logger import get_splunk_logger

try:
    import mcp.types as mcp_types
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.server import NotificationOptions, Server
    from mcp.server.stdio import stdio_server as mcp_stdio_server

    MCP_AVAILABLE = True
except ImportError:
    mcp_types = None
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    NotificationOptions = None
    Server = None
    mcp_stdio_server = None
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["SimpleMCPClient", "build_mcp_server", "run_mcp_server"]

load_dotenv()

_PYTHON_BIN = sys.executable or "python3"
SERVER_NAME = "usf_workspace_tools"
SERVER_VERSION = "1.0.0"
DEFAULT_SERVER_CMD = [_PYTHON_BIN, "-m", "agents.mcp", "serve"]
DEFAULT_SERVER_CWD = Path(__file__).resolve().parents[1]

PHI4_EMAIL_DEPLOYMENT = os.getenv("AZURE_PHI4_EMAIL") or os.getenv("AZURE_PHI4_ORCHESTRATOR") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
PHI4_MEETING_DEPLOYMENT = os.getenv("AZURE_PHI4_MEETING") or os.getenv("AZURE_PHI4_ORCHESTRATOR") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

EMAIL_SYSTEM_PROMPT = _require_env("EMAIL_SYSTEM_PROMPT")
MEETING_SYSTEM_PROMPT = _require_env("MEETING_SYSTEM_PROMPT")

class _ToolRuntime:
    def __init__(
        self,
        chat_db: Optional[ChatDatabase] = None,
        google_tools: Optional[GoogleWorkspaceTools] = None,
    ):
        self._db = chat_db
        self._google = google_tools

    @property
    def db(self) -> ChatDatabase:
        if self._db is None:
            self._db = ChatDatabase()
        return self._db

    @property
    def google(self) -> GoogleWorkspaceTools:
        if self._google is None:
            self._google = GoogleWorkspaceTools()
        return self._google

    # RAG helpers
    def retrieve_context(
        self,
        query: str,
        match_count: Optional[int] = None,
        extra_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required")
        result = retrieve_matches(query, match_count=match_count, extra_filter=extra_filter)
        return result

    # Audit helpers
    def log_interaction(self, session_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, str]:
        if not session_id or not event_type:
            raise ValueError("session_id and event_type are required to log interactions")
        self.db.log_event(session_id, event_type, payload)
        return {"status": "logged"}

    # Google Workspace helpers
    def list_calendar_events(self, max_results: int = 5) -> list[dict[str, Any]]:
        return self.google.list_calendar_events(max_results=max(1, max_results))

    def list_recent_emails(self, query: str = "", max_results: int = 5) -> list[dict[str, str]]:
        return self.google.list_recent_messages(query=query or "", max_results=max(1, max_results))

    def send_email(self, to_address: str, subject: str, body: str) -> str:
        if not to_address or not subject or not body:
            raise ValueError("To, subject, and body are required to send email.")
        return self.google.send_email(to_address, subject, body)

    def create_event(
        self,
        summary: str,
        start_iso: str,
        duration_minutes: int,
        attendees: Optional[list[str]] = None,
        description: str = "",
        location: str = "",
    ) -> dict[str, str]:
        return self.google.create_event(
            summary,
            start_iso,
            duration_minutes,
            attendees=attendees,
            description=description,
            location=location,
        )

    def draft_email(
        self,
        student_message: str,
        subject: Optional[str] = None,
        instructions: Optional[str] = None,
        previous_draft: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not student_message:
            raise ValueError("student_message is required")
        hits = retrieve_matches(student_message, match_count=6)
        context_block = format_context(hits)
        subject_line = (subject or "USF Follow-up").strip()
        user_sections = [f"Student inquiry:\n{student_message.strip()}\n", f"Subject reference: {subject_line}"]
        if previous_draft:
            user_sections.append(f"Existing draft to refine:\n{previous_draft.strip()}\n")
        if instructions:
            user_sections.append(f"Revision instructions:\n{instructions.strip()}\n")
        user_sections.append(f"Context:\n{context_block}")
        messages = [
            {"role": "system", "content": EMAIL_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ]
        body = complete_chat(
            PHI4_EMAIL_DEPLOYMENT,
            messages,
            temperature=0.25,
            deployment_name="AZURE_PHI4_EMAIL",
            session_id=session_id,
        ).strip()
        extracted_subject, cleaned_body = split_subject_from_body(body)
        if extracted_subject:
            subject_line = extracted_subject
            body = cleaned_body
        sources = build_sources_block(hits)

        draft = {
            "subject": subject_line,
            "body": body,
            "sources": sources,
            "context_hits": hits,
        }
        if session_id:
            self.db.log_event(
                session_id,
                "email_model_draft",
                {
                    "subject": subject_line,
                    "instructions": instructions or "",
                    "had_previous": bool(previous_draft),
                },
            )
        return draft

    def plan_meeting(
        self,
        summary: str,
        start_iso: str,
        duration_minutes: int,
        attendees: Optional[list[str]] = None,
        agenda: str = "",
        location: str = "",
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not start_iso:
            raise ValueError("start_iso is required")
        duration = max(5, int(duration_minutes or 30))
        normalized_start = self.google._normalize_iso(start_iso)
        attendees = attendees or []
        slot_free = self.google.check_availability(normalized_start, duration)
        suggested = None
        if not slot_free:
            try:
                suggested = self.google.find_next_available_slot(normalized_start, duration)
            except Exception:
                suggested = None
        agenda_clean = (agenda or "").strip()
        summary_clean = (summary or "Student Meeting").strip()
        location_clean = (location or "").strip()
        attendee_line = ", ".join(attendees) if attendees else "None provided"
        availability_line = "open" if slot_free else "busy"
        user_prompt = (
            f"Meeting summary: {summary_clean}\n"
            f"Requested start: {normalized_start}\n"
            f"Duration: {duration} minutes\n"
            f"Attendees: {attendee_line}\n"
            f"Availability status: {availability_line}\n"
        )
        if suggested:
            user_prompt += f"Suggested alternative slot: {suggested}\n"
        if agenda_clean:
            user_prompt += f"Agenda / notes:\n{agenda_clean}\n"
        if location_clean:
            user_prompt += f"Preferred location: {location_clean}\n"
        messages = [
            {"role": "system", "content": MEETING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        ai_notes = complete_chat(
            PHI4_MEETING_DEPLOYMENT,
            messages,
            temperature=0.2,
            deployment_name="AZURE_PHI4_MEETING",
            session_id=session_id,
        ).strip()
        plan = {
            "summary": summary_clean,
            "start": normalized_start,
            "duration": duration,
            "attendees": attendees,
            "location": location_clean,
            "slot_free": slot_free,
            "suggested": suggested,
            "ai_notes": ai_notes,
            "description": ai_notes,
        }
        if session_id:
            self.db.log_event(
                session_id,
                "meeting_model_plan",
                {
                    "summary": summary_clean,
                    "start": normalized_start,
                    "slot_free": slot_free,
                    "suggested": suggested,
                },
            )
        return plan

def _tool_definitions() -> list[mcp_types.Tool]:
    """Return the MCP tool catalog."""
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP SDK not installed; run `pip install mcp` to enable tools.")

    annotations_read_only = mcp_types.ToolAnnotations(readOnlyHint=True, idempotentHint=True)
    annotations_mutating = mcp_types.ToolAnnotations(readOnlyHint=False, destructiveHint=False)

    return [
        mcp_types.Tool(
            name="retrieve_context",
            description="Retrieve semantically relevant USF context from the Supabase vector store.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user utterance to embed and search."},
                    "match_count": {"type": "integer", "minimum": 1, "maximum": 20},
                    "extra_filter": {"type": "object", "description": "Optional JSON filter applied server-side."},
                },
                "required": ["query"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "hits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc": {"type": "string"},
                                "meta": {"type": "object"},
                                "score": {"type": ["number", "null"]},
                            },
                            "required": ["doc", "meta"],
                        },
                    }
                },
                "required": ["hits"],
            },
            annotations=annotations_read_only,
        ),
        mcp_types.Tool(
            name="log_interaction",
            description="Persist an audit trail entry for the current chat session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "event_type": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": ["session_id", "event_type", "payload"],
            },
            outputSchema={
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            },
            annotations=annotations_read_only,
        ),
        mcp_types.Tool(
            name="list_calendar_events",
            description="List upcoming Google Calendar events from the primary calendar.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {"type": "object"},
                    }
                },
                "required": ["events"],
            },
            annotations=annotations_read_only,
        ),
        mcp_types.Tool(
            name="list_recent_emails",
            description="List recent Gmail messages matching an optional search query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "default": ""},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "object"},
                    }
                },
                "required": ["messages"],
            },
            annotations=annotations_read_only,
        ),
        mcp_types.Tool(
            name="send_email",
            description="Send an email via Gmail on behalf of the authenticated USF account.",
            inputSchema={
                "type": "object",
                "properties": {
                    "to_address": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to_address", "subject", "body"],
            },
            outputSchema={
                "type": "object",
                "properties": {"message_id": {"type": "string"}},
                "required": ["message_id"],
            },
            annotations=annotations_mutating,
        ),
        mcp_types.Tool(
            name="create_event",
            description="Create a Google Calendar event with optional attendees and description.",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "start_iso": {"type": "string", "description": "ISO-8601 start timestamp."},
                    "duration_minutes": {"type": "integer", "minimum": 5, "maximum": 480, "default": 30},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "description": {"type": "string", "default": ""},
                    "location": {"type": "string", "default": ""},
                },
                "required": ["summary", "start_iso", "duration_minutes"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "hangout_link": {"type": "string"},
                },
                "required": ["event_id"],
            },
            annotations=annotations_mutating,
        ),
        mcp_types.Tool(
            name="draft_email",
            description="Generate or revise a USF policy-aligned email using Phi-4.",
            inputSchema={
                "type": "object",
                "properties": {
                    "student_message": {"type": "string"},
                    "subject": {"type": "string"},
                    "instructions": {"type": "string"},
                    "previous_draft": {"type": "string"},
                    "session_id": {"type": "string"},
                },
                "required": ["student_message"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "draft": {"type": "object"},
                },
                "required": ["draft"],
            },
            annotations=annotations_mutating,
        ),
        mcp_types.Tool(
            name="plan_meeting",
            description="Check availability, summarize details, and prepare a meeting plan via Phi-4.",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "start_iso": {"type": "string"},
                    "duration_minutes": {"type": "integer", "minimum": 5, "maximum": 480, "default": 30},
                    "attendees": {"type": "array", "items": {"type": "string"}, "default": []},
                    "agenda": {"type": "string"},
                    "location": {"type": "string"},
                    "session_id": {"type": "string"},
                },
                "required": ["summary", "start_iso", "duration_minutes"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "plan": {"type": "object"},
                },
                "required": ["plan"],
            },
            annotations=annotations_mutating,
        ),
    ]

def build_mcp_server(runtime: Optional[_ToolRuntime] = None) -> Server:
    """
    Build a fully-compliant MCP server that exposes the USF RAG + Google tools.
    """
    if not MCP_AVAILABLE:
        raise RuntimeError("The `mcp` package is required to run the MCP server.")
    if anyio is None:
        raise RuntimeError("anyio is required to run the MCP server transport.")

    runtime = runtime or _ToolRuntime()
    server = Server(SERVER_NAME, version=SERVER_VERSION)
    tools = _tool_definitions()

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        # Return deep copies to avoid accidental mutation between requests
        return [tool.model_copy(deep=True) for tool in tools]

    @server.call_tool()
    async def _call_tool(tool_name: str, arguments: Optional[dict[str, Any]]):
        args = arguments or {}
        try:
            return await _execute_tool(runtime, tool_name, args)
        except Exception as exc:  # pragma: no cover - defensive net
            logger.exception("Tool %s failed", tool_name)
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=f"{tool_name} failed: {exc}")],
                isError=True,
            )

    return server

async def _run_blocking(func, *args, **kwargs):
    if anyio is None:
        return func(*args, **kwargs)
    return await anyio.to_thread.run_sync(func, *args, **kwargs)

async def _execute_tool(runtime: _ToolRuntime, tool_name: str, args: dict[str, Any]):
    start = time.perf_counter()
    success = False
    error_msg = None
    result: Any = None
    try:
        if tool_name == "retrieve_context":
            hits = await _run_blocking(
                runtime.retrieve_context,
                args.get("query"),
                args.get("match_count"),
                args.get("extra_filter"),
            )
            result = {"hits": hits}
        elif tool_name == "log_interaction":
            result = await _run_blocking(
                runtime.log_interaction,
                args.get("session_id", ""),
                args.get("event_type", ""),
                args.get("payload") or {},
            )
        elif tool_name == "list_calendar_events":
            events = await _run_blocking(runtime.list_calendar_events, args.get("max_results", 5))
            result = {"events": events}
        elif tool_name == "list_recent_emails":
            messages = await _run_blocking(
                runtime.list_recent_emails,
                args.get("query", ""),
                args.get("max_results", 5),
            )
            result = {"messages": messages}
        elif tool_name == "send_email":
            message_id = await _run_blocking(
                runtime.send_email,
                args.get("to_address", ""),
                args.get("subject", ""),
                args.get("body", ""),
            )
            result = {"message_id": message_id}
        elif tool_name == "draft_email":
            draft = await _run_blocking(
                runtime.draft_email,
                args.get("student_message", ""),
                args.get("subject"),
                args.get("instructions"),
                args.get("previous_draft"),
                args.get("session_id"),
            )
            result = {"draft": draft}
        elif tool_name == "plan_meeting":
            plan = await _run_blocking(
                runtime.plan_meeting,
                args.get("summary", ""),
                args.get("start_iso", ""),
                int(args.get("duration_minutes", 30)),
                args.get("attendees"),
                args.get("agenda", ""),
                args.get("location", ""),
                args.get("session_id"),
            )
            result = {"plan": plan}
        elif tool_name == "create_event":
            event_info = await _run_blocking(
                runtime.create_event,
                args.get("summary", ""),
                args.get("start_iso", ""),
                int(args.get("duration_minutes", 30)),
                args.get("attendees"),
                args.get("description", ""),
                args.get("location", ""),
            )
            if isinstance(event_info, dict):
                result = {
                    "event_id": event_info.get("event_id", ""),
                    "hangout_link": event_info.get("hangout_link", ""),
                }
            else:
                result = {"event_id": event_info or "", "hangout_link": ""}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        success = True
        return result
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        get_splunk_logger().log_event(
            category="mcp",
            event_type="server_tool",
            payload={
                "tool_name": tool_name,
                "success": success,
                "error": error_msg,
            },
            metrics={"duration_ms": duration_ms},
            component="mcp_server",
        )

async def run_mcp_server(
    chat_db: Optional[ChatDatabase] = None,
    google_tools: Optional[GoogleWorkspaceTools] = None,
) -> None:
    """Entry point for `python -m utils.mcp serve`."""
    server = build_mcp_server(_ToolRuntime(chat_db=chat_db, google_tools=google_tools))
    init_options = server.create_initialization_options(
        notification_options=NotificationOptions(tools_changed=True),
        experimental_capabilities={},
    )
    async with mcp_stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            init_options,
        )

class SimpleMCPClient:
    """
    MCP client with optimized single-session calls.

    Each tool call reuses the same subprocess session for the duration
    of that call, avoiding repeated subprocess spawning overhead.
    """

    def __init__(
        self,
        chat_db: Optional[ChatDatabase] = None,
        google_tools: Optional[GoogleWorkspaceTools] = None,
        server_command: Optional[Sequence[str]] = None,
        server_cwd: Optional[Path | str] = None,
        server_env: Optional[dict[str, str]] = None,
    ):
        if not MCP_AVAILABLE:
            raise RuntimeError("The `mcp` package is required. Run `pip install -r requirements.txt`.")
        if anyio is None:
            raise RuntimeError("The `anyio` package is required for MCP stdio transport.")
        if os.getenv("USF_DISABLE_MCP", "0") == "1":
            raise RuntimeError("MCP has been disabled via USF_DISABLE_MCP=1.")

        self._runtime = _ToolRuntime(chat_db=chat_db, google_tools=google_tools)
        self._server_command = list(server_command or DEFAULT_SERVER_CMD)
        self._server_cwd = Path(server_cwd or DEFAULT_SERVER_CWD)
        env = dict(os.environ)
        if server_env:
            env.update(server_env)
        self._server_env = env

        self._stdio_params = StdioServerParameters(
            command=self._server_command[0],
            args=self._server_command[1:],
            env=self._server_env,
            cwd=str(self._server_cwd),
        )

        # Initialize Splunk logger for MCP tool call tracking
        self._splunk_logger = get_splunk_logger()

    def _call_tool(self, tool_name: str, arguments: dict[str, Any], timeout: float = 120.0):
        """
        Call MCP tool via stdio transport.
        Optimized to skip unnecessary list_tools() call.
        """
        if not self._stdio_params:
            raise RuntimeError("MCP transport has not been initialised.")

        # Extract session_id from arguments for logging (if available)
        session_id = arguments.get("session_id", "unknown")
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        error_message = None
        error_type = None
        error_traceback = None
        success = True

        async def _call():
            try:
                with anyio.fail_after(timeout):
                    async with stdio_client(self._stdio_params) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            return await session.call_tool(tool_name, arguments)
            except TimeoutError as exc:
                raise RuntimeError(
                    f"MCP tool '{tool_name}' timed out after {timeout}s. "
                    "The subprocess may be unresponsive or stuck."
                ) from exc

        try:
            result = anyio.run(_call)
            if result.isError:
                success = False
                error_message = _extract_error(result)
                error_type = "tool_error"
                raise RuntimeError(error_message)
            return result
        except TimeoutError as e:
            success = False
            error_message = str(e)
            error_type = "timeout"
            error_traceback = traceback.format_exc()
            raise
        except RuntimeError as e:
            success = False
            error_message = str(e)
            if "timed out" in error_message.lower():
                error_type = "timeout"
            else:
                error_type = "runtime_error"
            error_traceback = traceback.format_exc()
            raise
        except Exception as e:
            success = False
            error_message = str(e)
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            payload = {
                "timeout": timeout,
                "arguments_keys": list(arguments.keys()) if arguments else [],
                "subprocess_command": " ".join(self._server_command)
            }

            if not success:
                payload["error_type"] = error_type
                if error_traceback:
                    payload["error_traceback"] = error_traceback[-2000:]

            self._splunk_logger.log_mcp_tool_call(
                request_id=request_id,
                session_id=session_id,
                tool_name=tool_name,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                payload=payload
            )

    @staticmethod
    def _structured(result, key: str, default: Any):
        """Extract structured data from MCP result."""
        data = result.structuredContent or {}
        return data.get(key, default)

    def retrieve_context(
        self,
        query: str,
        match_count: Optional[int] = None,
        extra_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve context via MCP retrieve_context tool."""
        if not query:
            raise ValueError("query is required")
        payload = {"query": query}
        if match_count is not None:
            payload["match_count"] = match_count
        if extra_filter:
            payload["extra_filter"] = extra_filter
        result = self._call_tool("retrieve_context", payload)
        return self._structured(result, "hits", [])

    def log_interaction(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        """
        Log interaction directly to database (bypasses MCP for performance).
        Logging is a side-effect, not a tool, so direct DB access is appropriate.
        """
        if not session_id or not event_type:
            return
        try:
            self._runtime.log_interaction(session_id, event_type, payload)
        except Exception as exc:
            logger.warning("log_interaction failed: %s", exc)

    def list_calendar_events(self, max_results: int = 5) -> list[dict[str, Any]]:
        """List calendar events via MCP list_calendar_events tool."""
        result = self._call_tool("list_calendar_events", {"max_results": max_results})
        return self._structured(result, "events", [])

    def list_recent_emails(self, query: str = "", max_results: int = 5) -> list[dict[str, str]]:
        """List recent emails via MCP list_recent_emails tool."""
        result = self._call_tool(
            "list_recent_emails",
            {"query": query or "", "max_results": max_results},
        )
        return self._structured(result, "messages", [])

    def send_email(self, to_address: str, subject: str, body: str) -> str:
        """Send email via MCP send_email tool."""
        if not to_address or not subject or not body:
            raise ValueError("To, subject, and body are required to send email.")
        result = self._call_tool(
            "send_email",
            {"to_address": to_address, "subject": subject, "body": body},
        )
        return self._structured(result, "message_id", "")

    def draft_email(
        self,
        student_message: str,
        *,
        subject: str | None = None,
        instructions: str | None = None,
        previous_draft: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Draft email via MCP draft_email tool."""
        payload = {"student_message": student_message}
        if subject is not None:
            payload["subject"] = subject
        if instructions is not None:
            payload["instructions"] = instructions
        if previous_draft is not None:
            payload["previous_draft"] = previous_draft
        if session_id:
            payload["session_id"] = session_id
        result = self._call_tool("draft_email", payload)
        return self._structured(result, "draft", {})

    def create_event(
        self,
        summary: str,
        start_iso: str,
        duration_minutes: int,
        attendees: Optional[list[str]] = None,
        description: str = "",
        location: str = "",
    ) -> dict[str, str]:
        """Create calendar event via MCP create_event tool."""
        payload = {
            "summary": summary,
            "start_iso": start_iso,
            "duration_minutes": duration_minutes,
            "attendees": attendees or [],
            "description": description,
            "location": location,
        }
        result = self._call_tool("create_event", payload)
        return {
            "event_id": self._structured(result, "event_id", ""),
            "hangout_link": self._structured(result, "hangout_link", ""),
        }

    def plan_meeting(
        self,
        summary: str,
        start_iso: str,
        duration_minutes: int,
        attendees: Optional[list[str]] = None,
        agenda: str = "",
        location: str = "",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Plan meeting via MCP plan_meeting tool."""
        payload = {
            "summary": summary,
            "start_iso": start_iso,
            "duration_minutes": duration_minutes,
            "attendees": attendees or [],
            "agenda": agenda,
            "location": location,
        }
        if session_id:
            payload["session_id"] = session_id
        result = self._call_tool("plan_meeting", payload)
        return self._structured(result, "plan", {})

def _extract_error(result: mcp_types.CallToolResult) -> str:
    for block in result.content:
        if getattr(block, "type", "") == "text":
            return block.text
    return "Tool call failed"

def _main() -> None:
    parser = argparse.ArgumentParser(description="USF MCP server utilities.")
    parser.add_argument("command", choices=["serve"], help="Run the MCP stdio server.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.command == "serve":
        if not MCP_AVAILABLE:
            parser.error("The `mcp` package is not installed.")
        if anyio is None:
            parser.error("The `anyio` package is required to run the MCP server.")
        anyio.run(run_mcp_server)

if __name__ == "__main__":
    _main()