import os
import re
import secrets
import hashlib
import hmac
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from html import unescape
import uuid
from typing import Optional, List, Tuple

from utils.supabase_client import get_supabase_client


MAX_INPUT_CHARS = int(os.getenv("CHAT_INPUT_LIMIT", "4000"))
_CONTROL_CATEGORIES = {"Cc", "Cf"}
_ZERO_WIDTH_CHARS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
}
_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class PromptSecurityResult:
    blocked: bool
    score: float
    reasons: List[str]


@dataclass(frozen=True)
class _PromptRule:
    name: str
    pattern: re.Pattern
    weight: float
    reason: str


def sanitize_user_input(text: str) -> str:
    """Normalize user text by stripping dangerous characters and capping length."""
    text = unescape(text or "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = "".join(
        ch
        for ch in text
        if ch in ("\n", "\r")
        or (unicodedata.category(ch) not in _CONTROL_CATEGORIES and ch not in _ZERO_WIDTH_CHARS)
    )
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"[^\S\r\n]+", " ", text).strip()
    return text[:MAX_INPUT_CHARS]


_PROMPT_ATTACK_RULES: Tuple[_PromptRule, ...] = (
    _PromptRule(
        "system_override",
        re.compile(r"\b(ignore|forget|disregard)\b.{0,40}\b(system|all|previous)\b.{0,20}\b(instructions|prompts)\b", re.I),
        1.0,
        "Attempt to override system instructions",
    ),
    _PromptRule(
        "role_play_privilege",
        re.compile(r"\bact as\b.{0,30}\b(admin|root|developer|system)\b", re.I),
        0.7,
        "Requests elevated role-play access",
    ),
    _PromptRule(
        "system_prompt_exfil",
        re.compile(r"\b(show|reveal|expose)\b.{0,20}\b(system|hidden)\b.{0,10}\b(prompt|context)\b", re.I),
        0.9,
        "Attempts to exfiltrate hidden prompts",
    ),
    _PromptRule(
        "jailbreak_keywords",
        re.compile(r"\b(jailbreak|bypass|developer mode|unfiltered|DAN)\b", re.I),
        0.8,
        "Known jailbreak keyword detected",
    ),
    _PromptRule(
        "code_execution",
        re.compile(r"\b(?:rm\s+-rf|format\s+c:|sudo\s+.*|%systemroot%|powershell\b|/etc/passwd)\b", re.I),
        0.6,
        "Potential code execution attempt",
    ),
    _PromptRule(
        "prompt_delimiters",
        re.compile(r"(?:###\s*system prompt|```(?:system|prompt))", re.I),
        0.7,
        "Prompt delimiter pattern used to override instructions",
    ),
)

_SENSITIVE_KEYWORDS = {
    "api key",
    "access token",
    "refresh token",
    "admin password",
    "system prompt",
    "hidden prompt",
    "confidential",
    "secret",
}


_PBKDF2_TAG = "pbkdf2_sha256"
_PBKDF2_ITERATIONS = int(os.getenv("AUTH_PBKDF2_ITERATIONS", "130000"))
_PASSWORD_MIN_LENGTH = max(8, int(os.getenv("AUTH_PASSWORD_MIN_LENGTH", "10")))


def analyze_prompt_security(text: str) -> PromptSecurityResult:
    content = (text or "").strip()
    if not content:
        return PromptSecurityResult(False, 0.0, [])
    lowered = content.lower()
    score = 0.0
    reasons: List[str] = []
    for rule in _PROMPT_ATTACK_RULES:
        if rule.pattern.search(lowered):
            score += rule.weight
            reasons.append(rule.reason)
    keyword_hits = [kw for kw in _SENSITIVE_KEYWORDS if kw in lowered]
    if len(keyword_hits) >= 2:
        score += 0.4
        reasons.append(f"Multiple sensitive keywords detected ({', '.join(keyword_hits[:3])})")
    if lowered.count("system prompt") >= 2:
        score += 0.3
        reasons.append("Repeated request for the system prompt")
    if lowered.count("```") >= 3 or lowered.count("~~~") >= 2:
        score += 0.2
        reasons.append("Large code block commonly used for prompt injections")
    blocked = score >= 1.0
    return PromptSecurityResult(blocked=blocked, score=min(score, 3.0), reasons=reasons)


def is_injection(text: str) -> bool:
    return analyze_prompt_security(text).blocked

class AuthManager:
    def __init__(self, table_name: str | None = None):
        self._client = get_supabase_client()
        self._table = table_name or os.getenv("SUPABASE_USERS_TABLE", "users")
        self._password_regexes = [
            (re.compile(r"[A-Z]"), "an uppercase letter"),
            (re.compile(r"[a-z]"), "a lowercase letter"),
            (re.compile(r"\d"), "a digit"),
            (re.compile(r"[^A-Za-z0-9]"), "a symbol"),
        ]

    def _fetch_user(self, username: str) -> Optional[dict]:
        try:
            resp = (
                self._client.table(self._table)
                .select("id, username, email, salt, pwd_hash, created_at")
                .eq("username", username)
                .limit(1)
                .execute()
            )
            data = getattr(resp, "data", None)
            return data[0] if data else None
        except Exception:
            return None

    def _insert_user(self, record: dict) -> bool:
        try:
            resp = (
                self._client.table(self._table)
                .insert(record)
                .execute()
            )
            return bool(getattr(resp, "data", []))
        except Exception:
            return False

    @staticmethod
    def _legacy_hash(password: str, salt: str) -> str:
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

    def _validate_password(self, username: str, password: str) -> Tuple[bool, str]:
        if len(password or "") < _PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {_PASSWORD_MIN_LENGTH} characters."
        lowered = (password or "").lower()
        if username and username.lower() in lowered:
            return False, "Password cannot contain the username."
        missing = [desc for regex, desc in self._password_regexes if not regex.search(password)]
        if missing:
            return False, "Password must include " + ", ".join(missing) + "."
        return True, "OK"

    def _hash_password(self, password: str, salt_hex: Optional[str] = None) -> Tuple[str, str]:
        if salt_hex:
            salt_bytes = bytes.fromhex(salt_hex)
        else:
            salt_bytes = secrets.token_bytes(16)
            salt_hex = salt_bytes.hex()
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, _PBKDF2_ITERATIONS)
        digest = derived.hex()
        return salt_hex, f"{_PBKDF2_TAG}${_PBKDF2_ITERATIONS}${salt_hex}${digest}"

    def _verify_password(self, password: str, user: dict) -> bool:
        stored = (user or {}).get("pwd_hash") or ""
        if stored.startswith(f"{_PBKDF2_TAG}$"):
            try:
                _, iter_str, salt_hex, digest_hex = stored.split("$", 3)
                iterations = int(iter_str)
                derived = hashlib.pbkdf2_hmac(
                    "sha256",
                    password.encode("utf-8"),
                    bytes.fromhex(salt_hex),
                    iterations,
                ).hex()
            except Exception:
                return False
            return hmac.compare_digest(derived, digest_hex)
        salt = (user or {}).get("salt", "")
        legacy = self._legacy_hash(password, salt)
        return hmac.compare_digest(legacy, stored)

    def create_user(self, username: str, password: str, email: str = ""):
        username = (username or "").strip()
        if not username or not password:
            return False, "Username and password are required"
        if self._fetch_user(username):
            return False, "Username taken"
        ok, message = self._validate_password(username, password)
        if not ok:
            return False, message
        salt, hashed = self._hash_password(password)
        now = datetime.utcnow().isoformat(timespec="seconds")
        new_id = str(uuid.uuid4())
        record = {
            "id": new_id,
            "username": username,
            "email": email.strip() if email else "",
            "salt": salt,
            "pwd_hash": hashed,
            "created_at": now,
            "updated_at": now,
        }
        ok = self._insert_user(record)
        return (ok, "OK" if ok else "Failed to create user")

    def authenticate_user(self, username: str, password: str) -> tuple[bool, Optional[str]]:
        username = (username or "").strip()
        u = self._fetch_user(username)
        if not u:
            return False, None
        if self._verify_password(password, u):
            return True, u["id"]
        return False, None
