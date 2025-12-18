import os
import json
import time
import uuid
import socket
import threading
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from queue import Queue, Empty
from contextlib import contextmanager
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
from config.splunk_config import get_splunk_settings

# Disable SSL warnings for Splunk Cloud trial instances
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SplunkLogger:

    def __init__(
        self,
        hec_url: str,
        hec_token: str,
        index: str = "usf_concierge",
        sourcetype_prefix: str = "usf_concierge",
        batch_size: int = 10,
        flush_interval: float = 5.0,
        enabled: bool = True
    ):
        self.hec_url = hec_url
        self.hec_token = hec_token
        self.index = index
        self.sourcetype_prefix = sourcetype_prefix
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.enabled = enabled

        # Get hostname for event metadata
        self.hostname = socket.gethostname()

        # Get deployment environment
        self.deployment_env = os.getenv("DEPLOYMENT_ENV", "development")
        self.version = "1.0.0"

        # Thread safety
        self._lock = threading.Lock()
        self._batch: List[Dict[str, Any]] = []
        self._last_flush_time = time.time()

        # Background processing
        self._queue: Queue = Queue()
        self._shutdown_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Fallback file logging
        self._fallback_file = "logs/splunk_fallback.log"
        self._ensure_fallback_dir()

        # HTTP session with retry logic
        self._session = self._create_session()

        # Track if we've logged a warning about disabled state
        self._warned_disabled = False

        # Start background worker if enabled
        if self.enabled:
            self._start_worker()
            logger.info(f"SplunkLogger initialized: index={index}, batch_size={batch_size}")
        else:
            if not self._warned_disabled:
                logger.warning("SplunkLogger is disabled (missing HEC_URL or HEC_TOKEN)")
                self._warned_disabled = True

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic and connection pooling."""
        session = requests.Session()

        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,  # 1s, 2s, 4s delays
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update({
            "Authorization": f"Splunk {self.hec_token}",
            "Content-Type": "application/json"
        })

        # Disable SSL verification for Splunk Cloud trial instances with self-signed certs
        session.verify = False

        return session

    def _ensure_fallback_dir(self):
        try:
            os.makedirs(os.path.dirname(self._fallback_file), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create fallback log directory: {e}")

    def _start_worker(self):
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        while not self._shutdown_event.is_set():
            try:
                # Process queued events with timeout
                try:
                    event = self._queue.get(timeout=0.5)
                    self._add_to_batch(event)
                except Empty:
                    pass

                # Check if we need to flush based on time
                with self._lock:
                    if (time.time() - self._last_flush_time) >= self.flush_interval:
                        if self._batch:
                            self._flush_batch()

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")

    def _add_to_batch(self, event: Dict[str, Any]):
        with self._lock:
            self._batch.append(event)

            if len(self._batch) >= self.batch_size:
                self._flush_batch()

    def _flush_batch(self):
        if not self._batch:
            return

        batch = self._batch.copy()
        self._batch.clear()
        self._last_flush_time = time.time()

        # Release lock before making HTTP request
        try:
            self._lock.release()
            self._send_to_splunk(batch)
        finally:
            self._lock.acquire()

    def _send_to_splunk(self, events: List[Dict[str, Any]]):
        if not events:
            return

        try:
            # Format as newline-delimited JSON for batch mode
            payload = "\n".join(json.dumps(event) for event in events)

            response = self._session.post(
                self.hec_url,
                data=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.debug(f"Successfully sent {len(events)} events to Splunk")
            else:
                logger.error(
                    f"Failed to send events to Splunk: {response.status_code} - {response.text}"
                )
                self._fallback_log(events, f"HTTP {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send events to Splunk: {e}")
            self._fallback_log(events, str(e))
        except Exception as e:
            logger.error(f"Unexpected error sending to Splunk: {e}")
            self._fallback_log(events, str(e))

    def _fallback_log(self, events: List[Dict[str, Any]], error: str):
        try:
            with open(self._fallback_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Fallback Log - {datetime.now().isoformat()}\n")
                f.write(f"Error: {error}\n")
                f.write(f"{'='*80}\n")
                for event in events:
                    f.write(json.dumps(event, indent=2) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to fallback log: {e}")

    def _create_event(
        self,
        category: str,
        event_type: str,
        payload: Dict[str, Any],
        severity: str = "info",
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        component: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        now = time.time()

        event = {
            "time": now,
            "host": self.hostname,
            "source": "usf_concierge",
            "sourcetype": f"{self.sourcetype_prefix}:{category}",
            "index": self.index,
            "event": {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id or str(uuid.uuid4()),
                "session_id": session_id,
                "event_category": category,
                "event_type": event_type,
                "severity": severity,
                "component": component or "unknown",
                "payload": payload,
                "context": {
                    "deployment": self.deployment_env,
                    "version": self.version
                }
            }
        }

        if metrics:
            event["event"]["metrics"] = metrics

        return event

    def log_event(
        self,
        category: str,
        event_type: str,
        payload: Dict[str, Any],
        severity: str = "info",
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        component: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        if not self.enabled:
            return

        try:
            event = self._create_event(
                category=category,
                event_type=event_type,
                payload=payload,
                severity=severity,
                request_id=request_id,
                session_id=session_id,
                component=component,
                metrics=metrics
            )

            self._queue.put(event, block=False)
        except Exception as e:
            # Never let logging errors break the application
            logger.error(f"Failed to queue Splunk event: {e}")

    def log_security_event(
        self,
        request_id: str,
        session_id: str,
        event_type: str,
        blocked: bool,
        score: float,
        matched_rules: List[str],
        user_input_preview: str,
        severity: str = "warning"
    ):

        payload = {
            "blocked": blocked,
            "score": score,
            "matched_rules": matched_rules,
            "user_input_preview": user_input_preview[:200],
            "rules_count": len(matched_rules)
        }

        self.log_event(
            category="security",
            event_type=event_type,
            payload=payload,
            severity=severity,
            request_id=request_id,
            session_id=session_id,
            component="security_filter"
        )

    def log_mcp_tool_call(
        self,
        request_id: str,
        session_id: str,
        tool_name: str,
        duration_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None
    ):

        metrics = {
            "duration_ms": duration_ms,
            "success": success
        }

        event_payload = {
            "tool_name": tool_name,
            "error_message": error_message
        }

        if payload:
            event_payload.update(payload)

        severity = "error" if not success else "info"

        self.log_event(
            category="mcp",
            event_type="tool_call",
            payload=event_payload,
            severity=severity,
            request_id=request_id,
            session_id=session_id,
            component="mcp_client",
            metrics=metrics
        )

    def log_api_call(
        self,
        request_id: str,
        session_id: str,
        api_name: str,
        operation: str,
        duration_ms: float,
        status_code: int,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):

        metrics = {
            "duration_ms": duration_ms,
            "status_code": status_code
        }

        payload = {
            "api_name": api_name,
            "operation": operation,
            "error": error,
            "success": 200 <= status_code < 300
        }

        # Merge metadata into payload if provided
        if metadata:
            payload.update(metadata)

        severity = "error" if error else "info"

        self.log_event(
            category="api",
            event_type="external_call",
            payload=payload,
            severity=severity,
            request_id=request_id,
            session_id=session_id,
            component=api_name,
            metrics=metrics
        )

    def log_llm_call(
        self,
        request_id: str,
        session_id: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: float,
        temperature: float,
        success: bool = True,
        deployment_name: Optional[str] = None,
        prompt_preview: Optional[str] = None,
        response_preview: Optional[str] = None
    ):
        metrics = {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": tokens_in + tokens_out,
            "duration_ms": duration_ms,
            "tokens_per_second": tokens_out / (duration_ms / 1000) if duration_ms > 0 else 0
        }

        payload = {
            "model": model,
            "temperature": temperature,
            "success": success,
            "prompt_preview": prompt_preview[:200] if prompt_preview else None,
            "response_preview": response_preview[:200] if response_preview else None
        }

        if deployment_name:
            payload["deployment_name"] = deployment_name

        severity = "error" if not success else "info"

        self.log_event(
            category="llm",
            event_type="completion",
            payload=payload,
            severity=severity,
            request_id=request_id,
            session_id=session_id,
            component="llm_client",
            metrics=metrics
        )

    @contextmanager
    def timed_operation(
        self,
        category: str,
        event_type: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        component: Optional[str] = None
    ):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            extras = dict(payload or {})
            self.log_event(
                category=category,
                event_type=event_type,
                payload=extras,
                request_id=request_id,
                session_id=session_id,
                component=component,
                metrics={"duration_ms": duration_ms},
            )

    def flush(self):
        """Force flush all pending events immediately."""
        if not self.enabled:
            return

        with self._lock:
            if self._batch:
                self._flush_batch()

    def close(self):
        """Shutdown the logger with final flush."""
        if not self.enabled:
            return

        logger.info("Shutting down SplunkLogger...")

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

        # Flush any remaining events
        self.flush()

        # Close HTTP session
        self._session.close()

        logger.info("SplunkLogger shutdown complete")


# Singleton instance
_splunk_logger_instance: Optional[SplunkLogger] = None
_instance_lock = threading.Lock()


def get_splunk_logger() -> SplunkLogger:

    global _splunk_logger_instance

    with _instance_lock:
        if _splunk_logger_instance is None:
            settings = get_splunk_settings()
            _splunk_logger_instance = SplunkLogger(
                hec_url=settings["hec_url"],
                hec_token=settings["hec_token"],
                index=settings["index"],
                sourcetype_prefix=settings["sourcetype_prefix"],
                batch_size=int(settings["batch_size"]),
                flush_interval=float(settings["flush_interval"]),
                enabled=bool(settings["enabled"]),
            )

    return _splunk_logger_instance


# Cleanup on module exit
import atexit

def _cleanup_logger():
    """Cleanup function called on program exit."""
    global _splunk_logger_instance
    if _splunk_logger_instance is not None:
        _splunk_logger_instance.close()

atexit.register(_cleanup_logger)
