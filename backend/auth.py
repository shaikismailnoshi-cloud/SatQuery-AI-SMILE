"""
SatQuery AI — Authentication & Persistent History Module
Provides:
  - SQLite-backed user registration and login
  - bcrypt password hashing (stdlib-only fallback available)
  - JWT-style signed session tokens (HMAC-SHA256, no third-party jwt needed)
  - Per-user analysis history persistence
"""

import os
import sqlite3
import hashlib
import hmac
import json
import time
import base64
import secrets
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("satquery.auth")

# ── DB Configuration ───────────────────────────────────────────────────────────
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "satquery_users.db")

# Secret key for token signing — use a fixed env-var or derive one per-instance
_SECRET = os.environ.get("SATQUERY_SECRET", secrets.token_hex(32))
TOKEN_TTL_SECONDS = 7 * 24 * 3600  # 7 days


# ── Database Initialisation ────────────────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all required tables if they do not exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email       TEXT    UNIQUE NOT NULL,
                username    TEXT    NOT NULL,
                password_hash TEXT  NOT NULL,
                created_at  REAL    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                session_id  TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                created_at  REAL    NOT NULL,
                mode        TEXT    NOT NULL,
                query       TEXT    NOT NULL,
                answer      TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                full_data   TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
    logger.info(f"SatQuery auth DB ready at {DB_PATH}")


# ── Password Utilities ─────────────────────────────────────────────────────────
def _hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 with a random per-user salt."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
    return f"{salt}:{base64.b64encode(dk).decode()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, encoded = stored.split(":", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
        expected = base64.b64encode(dk).decode()
        return hmac.compare_digest(expected, encoded)
    except Exception:
        return False


# ── Token Utilities ────────────────────────────────────────────────────────────
def _make_token(user_id: int, email: str) -> str:
    """Create a signed HMAC-SHA256 token: base64(payload).base64(sig)"""
    payload = json.dumps({"uid": user_id, "email": email, "exp": time.time() + TOKEN_TTL_SECONDS})
    b64_payload = base64.urlsafe_b64encode(payload.encode()).decode()
    sig = hmac.new(_SECRET.encode(), b64_payload.encode(), hashlib.sha256).hexdigest()
    return f"{b64_payload}.{sig}"


def _verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify token signature and expiry. Returns payload dict or None."""
    try:
        b64_payload, sig = token.rsplit(".", 1)
        expected_sig = hmac.new(_SECRET.encode(), b64_payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, sig):
            return None
        payload = json.loads(base64.urlsafe_b64decode(b64_payload + "==").decode())
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ── User Management ────────────────────────────────────────────────────────────
def register_user(email: str, username: str, password: str) -> Dict[str, Any]:
    """Register a new user. Returns {success, token, user} or {success:False, error}."""
    email = email.strip().lower()
    username = username.strip()

    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}
    if not email or "@" not in email:
        return {"success": False, "error": "Invalid email address."}
    if not username:
        return {"success": False, "error": "Username is required."}

    pw_hash = _hash_password(password)
    now = time.time()

    try:
        with _get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (email, username, pw_hash, now)
            )
            user_id = cur.lastrowid
            conn.commit()

        token = _make_token(user_id, email)
        logger.info(f"Registered new user: {email} (id={user_id})")
        return {
            "success": True,
            "token": token,
            "user": {"id": user_id, "email": email, "username": username}
        }
    except sqlite3.IntegrityError:
        return {"success": False, "error": "An account with this email already exists."}


def login_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticate and return token+user or error."""
    email = email.strip().lower()

    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not row or not _verify_password(password, row["password_hash"]):
        return {"success": False, "error": "Invalid email or password."}

    token = _make_token(row["id"], email)
    logger.info(f"Login: {email} (id={row['id']})")
    return {
        "success": True,
        "token": token,
        "user": {"id": row["id"], "email": email, "username": row["username"]}
    }


def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Validate token and return user dict or None."""
    payload = _verify_token(token)
    if not payload:
        return None

    with _get_conn() as conn:
        row = conn.execute("SELECT id, email, username FROM users WHERE id = ?", (payload["uid"],)).fetchone()

    if not row:
        return None
    return {"id": row["id"], "email": row["email"], "username": row["username"]}


# ── History Management ─────────────────────────────────────────────────────────
def save_history_entry(user_id: int, entry: Dict[str, Any]) -> int:
    """Persist one analysis history record. Returns its DB id."""
    full_data_json = json.dumps(entry.get("fullData", {}))
    now = time.time()
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO analysis_history
               (user_id, session_id, timestamp, created_at, mode, query, answer, confidence, full_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                entry.get("id", str(int(now * 1000))),
                entry.get("timestamp", ""),
                now,
                entry.get("mode", "ANALYSIS"),
                entry.get("query", ""),
                entry.get("answer", ""),
                float(entry.get("confidence", 0.0)),
                full_data_json,
            )
        )
        conn.commit()
        return cur.lastrowid


def get_user_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent history entries for a user."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()

    result = []
    for row in rows:
        try:
            full_data = json.loads(row["full_data"])
        except Exception:
            full_data = {}
        result.append({
            "id": row["session_id"],
            "db_id": row["id"],
            "timestamp": row["timestamp"],
            "mode": row["mode"],
            "query": row["query"],
            "answer": row["answer"],
            "confidence": row["confidence"],
            "fullData": full_data,
        })
    return result


def delete_history_entry(user_id: int, db_id: int) -> bool:
    """Delete a specific history entry. Returns True if deleted."""
    with _get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM analysis_history WHERE id = ? AND user_id = ?",
            (db_id, user_id)
        )
        conn.commit()
        return cur.rowcount > 0


def clear_user_history(user_id: int) -> int:
    """Delete ALL history for a user. Returns count deleted."""
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM analysis_history WHERE user_id = ?", (user_id,))
        conn.commit()
        return cur.rowcount


# ── Init on import ─────────────────────────────────────────────────────────────
init_db()
