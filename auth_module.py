import os
import secrets
import requests
import sqlite3

from flask import Blueprint, request, jsonify, redirect, make_response
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Load environment (matches your backend)
load_dotenv(dotenv_path="/home/epistemiq/mysite/.env")

# -------------------------------------------------------------------
# DB helper – same DB as app.py
# -------------------------------------------------------------------
DB_PATH = "/home/epistemiq/mysite/sessions.db"


def get_conn():
    """
    Open SQLite connection with WAL mode and sane timeouts.
    Mirrors app.py so both modules talk to the same DB.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    return conn


auth_bp = Blueprint("auth", __name__)

# -------------------------------------------------------------------
# CONFIG (SendGrid)
# -------------------------------------------------------------------
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
SENDGRID_SENDER = os.environ.get("SENDGRID_SENDER", "epistemiq.ai@gmail.com")


# -------------------------------------------------------------------
# EMAIL SENDER (SendGrid HTTP API — works on PythonAnywhere free tier)
# -------------------------------------------------------------------
def send_magic_link(email: str, link_url: str) -> bool:
    """
    Send magic login link via SendGrid (HTTPS API).
    """
    if not SENDGRID_API_KEY:
        print("ERROR: SENDGRID_API_KEY is missing")
        return False

    subject = "Your Epistemiq Login Link"
    body = f"""
Click below to log in to Epistemiq:

{link_url}

This link expires in 20 minutes.

If you did not request this, please ignore it.
"""

    payload = {
        "personalizations": [{
            "to": [{"email": email}],
            "subject": subject
        }],
        "from": {"email": SENDGRID_SENDER},
        "content": [{
            "type": "text/plain",
            "value": body
        }]
    }

    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            json=payload,
            headers=headers,
            timeout=10
        )

        if r.status_code in (200, 202):
            return True

        print("SendGrid error:", r.status_code, r.text)
        return False

    except Exception as e:
        print("SendGrid exception:", e)
        return False


# -------------------------------------------------------------------
# TOKEN HELPERS (Magic link + sessions)
# -------------------------------------------------------------------
def create_magic_token():
    """Returns (plain_token, hash_for_db)."""
    token = secrets.token_urlsafe(32)
    hashed = generate_password_hash(token)
    return token, hashed


def verify_magic_token(hashed_token, plain_token):
    return check_password_hash(hashed_token, plain_token)


def create_session_token():
    return secrets.token_urlsafe(48)


# -------------------------------------------------------------------
# Timestamp helper (fixes datetime vs string issue)
# -------------------------------------------------------------------
def _coerce_timestamp(value):
    """
    SQLite with PARSE_DECLTYPES may return datetime objects or strings.
    Normalize to a datetime or return None if invalid.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        # strip microseconds if present
        v = value.split(".")[0]
        try:
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    return None


# -------------------------------------------------------------------
# AUTH HELPERS (session verification)
# -------------------------------------------------------------------
def get_current_user():
    """
    Returns dict {user_id, email} or None.
    """
    token = request.cookies.get("ep_session")
    if not token:
        return None

    conn = get_conn()
    c = conn.cursor()

    c.execute(
        """
        SELECT users.id AS user_id,
               users.email AS email,
               sessions.expires_at AS expires_at
        FROM sessions
        JOIN users ON sessions.user_id = users.id
        WHERE sessions.session_token=?
        """,
        (token,)
    )
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    expires_at = _coerce_timestamp(row["expires_at"])
    if not expires_at or expires_at < datetime.now():
        return None

    return {"user_id": row["user_id"], "email": row["email"]}


def require_user(fn):
    """Decorator for authenticated routes."""
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        return fn(user, *args, **kwargs)

    return wrapper


# -------------------------------------------------------------------
# ROUTE: POST /auth/request-link → send magic login email
# -------------------------------------------------------------------
@auth_bp.route("/auth/request-link", methods=["POST"])
def request_magic_link():
    data = request.json or {}
    email = (data.get("email") or "").strip().lower()

    if not email or "@" not in email:
        return jsonify({"error": "Invalid email"}), 400

    conn = get_conn()
    c = conn.cursor()

    # Ensure user exists
    c.execute("INSERT OR IGNORE INTO users (email) VALUES (?)", (email,))
    conn.commit()

    # Fetch user id
    c.execute("SELECT id FROM users WHERE email=?", (email,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Failed to create user"}), 500

    user_id = row["id"]

    # Create magic token
    token, token_hash = create_magic_token()
    expires = datetime.now() + timedelta(minutes=20)

    c.execute(
        """
        INSERT INTO magic_links (user_id, token_hash, expires_at)
        VALUES (?, ?, ?)
        """,
        (user_id, token_hash, expires)
    )
    conn.commit()
    conn.close()

    # Magic link URL: go directly to backend
    link_url = (
        f"https://epistemiq.vercel.app"
        f"/auth/verify?token={token}&email={email}"
    )


    if not send_magic_link(email, link_url):
        return jsonify({"error": "Failed to send email"}), 500

    return jsonify({"message": "Magic link sent"}), 200


# -------------------------------------------------------------------
# ROUTE: GET /auth/verify → verify magic link + create session cookie
# -------------------------------------------------------------------
@auth_bp.route("/auth/verify", methods=["GET"])
def verify_magic_link():
    token = request.args.get("token", "")
    email = (request.args.get("email", "") or "").lower()

    if not token or not email:
        return "Invalid link", 400

    conn = get_conn()
    c = conn.cursor()

    # Get user
    c.execute("SELECT id FROM users WHERE email=?", (email,))
    row = c.fetchone()
    if not row:
        conn.close()
        return "Invalid user", 400

    user_id = row["id"]

    # Get last magic link for this user
    c.execute(
        """
        SELECT id, token_hash, expires_at, used_at
        FROM magic_links
        WHERE user_id=?
        ORDER BY id DESC LIMIT 1
        """,
        (user_id,),
    )
    link = c.fetchone()

    if not link:
        conn.close()
        return "Invalid or expired link", 400

    # Validate token
    if not verify_magic_token(link["token_hash"], token):
        conn.close()
        return "Invalid or expired token", 400

    # Validate expiry (handle string/datetime safely)
    expires_at = _coerce_timestamp(link["expires_at"])
    if not expires_at or expires_at < datetime.now():
        conn.close()
        return "Link expired", 400

    # Already used?
    if link["used_at"]:
        conn.close()
        return "Link already used", 400

    # Mark used
    c.execute("UPDATE magic_links SET used_at=? WHERE id=?", (datetime.now(), link["id"]))

    # Create session
    session_token = create_session_token()
    session_expires = datetime.now() + timedelta(days=30)

    c.execute(
        """
        INSERT INTO sessions (user_id, session_token, expires_at)
        VALUES (?, ?, ?)
        """,
        (user_id, session_token, session_expires),
    )
    conn.commit()
    conn.close()

    # Set cookie for cross-site use
    resp = make_response(redirect("https://epistemiq.vercel.app/"))
    resp.set_cookie(
        "ep_session",
        session_token,
        httponly=True,
        secure=True,
        samesite="None",
        expires=session_expires,
    )
    return resp

# -------------------------------------------------------------------
# ROUTE: GET /auth/me → returns login state
# Sliding session refresh + persistent cookie
# -------------------------------------------------------------------
@auth_bp.route("/auth/me", methods=["GET"])
def auth_me():
    user = get_current_user()

    # -----------------------------
    # NOT LOGGED IN
    # -----------------------------
    if not user:
        return jsonify({"authenticated": False})

    # -----------------------------
    # LOGGED IN → refresh session
    # -----------------------------
    session_token = request.cookies.get("ep_session")
    if session_token:
        new_expires = datetime.now() + timedelta(days=30)

        # Refresh DB session expiry
        conn = get_conn()
        c = conn.cursor()
        c.execute(
            "UPDATE sessions SET expires_at=? WHERE session_token=?",
            (new_expires, session_token)
        )
        conn.commit()
        conn.close()

        # Refresh cookie expiry
        resp = jsonify({
            "authenticated": True,
            "email": user["email"],
            "user_id": user["user_id"],
        })
        resp.set_cookie(
            "ep_session",
            session_token,
            httponly=True,
            secure=True,
            samesite="None",
            expires=new_expires,
        )
        return resp

    # -----------------------------
    # FALLBACK (should not happen)
    # -----------------------------
    return jsonify({
        "authenticated": True,
        "email": user["email"],
        "user_id": user["user_id"],
    })



# -------------------------------------------------------------------
# ROUTE: POST /auth/logout → destroys session
# -------------------------------------------------------------------
@auth_bp.route("/auth/logout", methods=["POST"])
def logout():
    token = request.cookies.get("ep_session")

    if token:
        conn = get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM sessions WHERE session_token=?", (token,))
        conn.commit()
        conn.close()

    resp = jsonify({"message": "Logged out"})
    resp.delete_cookie(
        "ep_session",
        domain=".epistemiq.pythonanywhere.com",
        path="/",
    )
    return resp

