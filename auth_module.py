import os
import secrets
import requests
import psycopg2
import psycopg2.extras
from flask import Blueprint, request, jsonify, redirect, make_response
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

auth_bp = Blueprint("auth", __name__)

# --- DATABASE CONNECTION ---
def get_conn():
    """Connects SQL (Postgres)"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise Exception("DATABASE_URL not set")
    
    conn = psycopg2.connect(db_url)
    return conn

# --- CONFIG ---
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
SENDGRID_SENDER = os.environ.get("SENDGRID_SENDER", "epistemiq.ai@gmail.com")

def send_magic_link(email: str, link_url: str) -> bool:
    if not SENDGRID_API_KEY:
        print(f"DEBUG MAGIC LINK: {link_url}")
        return True

    payload = {
        "personalizations": [{"to": [{"email": email}], "subject": "Your Epistemiq Login Link"}],
        "from": {"email": SENDGRID_SENDER},
        "content": [{"type": "text/plain", "value": f"Click below to log in:\n\n{link_url}\n\nExpires in 20 minutes."}]
    }
    headers = {"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"}
    
    try:
        r = requests.post("https://api.sendgrid.com/v3/mail/send", json=payload, headers=headers, timeout=10)
        return r.status_code in (200, 202)
    except Exception as e:
        print("SendGrid Error:", e)
        return False

# --- TOKENS ---
def create_magic_token():
    token = secrets.token_urlsafe(32)
    return token, generate_password_hash(token)

def verify_magic_token(hashed, plain):
    return check_password_hash(hashed, plain)

def create_session_token():
    return secrets.token_urlsafe(48)

# --- AUTH HELPERS ---
def get_current_user():
    token = request.cookies.get("ep_session")
    if not token: return None

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            c.execute("""
                SELECT u.id AS user_id, u.email, s.expires_at
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = %s
            """, (token,))
            row = c.fetchone()
            
            if not row: return None
            if row['expires_at'] < datetime.now(): return None
            
            return {"user_id": row['user_id'], "email": row['email']}
    finally:
        conn.close()

def require_user(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not get_current_user():
            return jsonify({"error": "Authentication required"}), 401
        return fn(get_current_user(), *args, **kwargs)
    return wrapper

# --- ROUTES ---
@auth_bp.route("/auth/request-link", methods=["POST"])
def request_magic_link():
    data = request.json or {}
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email: return jsonify({"error": "Invalid email"}), 400

    conn = get_conn()
    try:
        with conn.cursor() as c:
            # Postgres: ON CONFLICT DO NOTHING
            c.execute("INSERT INTO users (email) VALUES (%s) ON CONFLICT (email) DO NOTHING", (email,))
            c.execute("SELECT id FROM users WHERE email=%s", (email,))
            user_id = c.fetchone()[0]

            token, token_hash = create_magic_token()
            expires = datetime.now() + timedelta(minutes=20)

            c.execute("INSERT INTO magic_links (user_id, token_hash, expires_at) VALUES (%s, %s, %s)", 
                      (user_id, token_hash, expires))
            conn.commit()
    finally:
        conn.close()

    frontend_url = os.getenv("FRONTEND_URL", request.host_url.rstrip('/'))
    link_url = f"{frontend_url}/auth/verify?token={token}&email={email}"
    
    if not send_magic_link(email, link_url):
        return jsonify({"error": "Failed to send email"}), 500
    return jsonify({"message": "Magic link sent"}), 200

@auth_bp.route("/auth/verify", methods=["GET"])
def verify_magic_link():
    token = request.args.get("token")
    email = request.args.get("email", "").lower()
    if not token or not email: return "Invalid link", 400

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            c.execute("SELECT id FROM users WHERE email=%s", (email,))
            row = c.fetchone()
            if not row: return "Invalid user", 400
            user_id = row['id']

            c.execute("""
                SELECT id, token_hash, expires_at, used_at 
                FROM magic_links WHERE user_id=%s ORDER BY id DESC LIMIT 1
            """, (user_id,))
            link = c.fetchone()

            if not link or not verify_magic_token(link['token_hash'], token): return "Invalid token", 400
            if link['expires_at'] < datetime.now(): return "Expired", 400
            if link['used_at']: return "Used", 400

            c.execute("UPDATE magic_links SET used_at=%s WHERE id=%s", (datetime.now(), link['id']))
            
            session_token = create_session_token()
            expires = datetime.now() + timedelta(days=30)
            c.execute("INSERT INTO sessions (user_id, session_token, expires_at) VALUES (%s, %s, %s)", 
                      (user_id, session_token, expires))
            conn.commit()
    finally:
        conn.close()

    frontend_url = os.getenv("FRONTEND_URL", "https://epistemiq.vercel.app")
    resp = make_response(redirect(f"{frontend_url}/"))
    resp.set_cookie("ep_session", session_token, httponly=True, secure=True, samesite="None", expires=expires)
    return resp

@auth_bp.route("/auth/me", methods=["GET"])
def auth_me():
    user = get_current_user()
    return jsonify({"authenticated": bool(user), "email": user['email'] if user else None})

@auth_bp.route("/auth/logout", methods=["POST"])
def logout():
    token = request.cookies.get("ep_session")
    if token:
        conn = get_conn()
        try:
            with conn.cursor() as c:
                c.execute("DELETE FROM sessions WHERE session_token=%s", (token,))
                conn.commit()
        finally:
            conn.close()
    
    resp = jsonify({"message": "Logged out"})
    resp.delete_cookie("ep_session", path="/", samesite="None", secure=True)
    return resp
