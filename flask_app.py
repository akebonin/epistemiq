from flask import Flask, request, render_template, send_file, Response, jsonify, redirect, url_for
from dotenv import load_dotenv
import os
import requests
import io
import json
from urllib.parse import quote_plus
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import logging
import re
import uuid
import time
from bs4 import BeautifulSoup
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import base64
from PIL import Image
import pytesseract
from moviepy.editor import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import yt_dlp
import unicodedata
import hashlib
from flask_cors import CORS, cross_origin
import subprocess
from auth_module import auth_bp, get_current_user, require_user
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from admin_routes import admin_bp
import random
from reportlab.lib.utils import ImageReader
import fcntl
from whitenoise import WhiteNoise
import math

# ✅ NEW: Google GenAI Import
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='templates/static')

# ✅ FIX: Use absolute path for WhiteNoise
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_root = os.path.join(BASE_DIR, 'templates', 'static')

# Only enable WhiteNoise if the directory actually exists (prevents crash)
if os.path.exists(static_root):
    app.wsgi_app = WhiteNoise(app.wsgi_app, root=static_root, prefix='static/')
else:
    logging.warning(f"Static folder not found at {static_root}")


app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    app.secret_key = os.urandom(24)

app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp, url_prefix='/api/admin')

CORS(app, supports_credentials=True, origins=["https://epistemiq.vercel.app", "http://localhost:8080"])

# ----------------------------------------------------------------
# ✅ AI CONFIGURATION & MODEL LISTS
# ----------------------------------------------------------------

# 1. Google AI Studio (Native)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logging.error("GOOGLE_API_KEY is missing.")

# 2. OpenRouter
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Model Lists ---

# A. Google Native Rotation (High throughput, large context)
GOOGLE_ROTATION_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-2.0-pro-exp-02-05",
]

# B. OpenRouter Community Rotation
# Fallback models if Google fails or for user preference.
OPENROUTER_ROTATION_MODELS = [
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "allenai/olmo-3-32b-think:free",
    "mistralai/mistral-7b-instruct:free",
    "amazon/nova-2-lite-v1:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "tngtech/deepseek-r1t2-chimera:free",
]

# Extraction Default (Google Native ID)
# ✅ Updated to Gemma 3 27B Instruction Tuned
EXTRACTION_DEFAULT_MODEL = "gemma-3-27b-it"

# ----------------------------------------------------------------
#  LLM WRAPPERS & HYBRID CONTROLLER
# ----------------------------------------------------------------

class UnifiedResponse:
    """Standardizes responses from Google Native and OpenRouter."""
    def __init__(self, content=None, stream_generator=None, model_used="unknown"):
        self.content = content
        self.stream_generator = stream_generator
        self.model_used = model_used

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}

    def iter_content(self, chunk_size=None, decode_unicode=True):
        if self.stream_generator:
            yield from self.stream_generator

def call_google_native(prompt, model_name, stream=False, temperature=0.0, json_mode=False):
    """Native Google AI Studio Call"""
    try:
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain"
        )

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, stream=stream, generation_config=generation_config)

        if stream:
            def google_stream_adapter():
                for chunk in response:
                    if chunk.text:
                        data = json.dumps({"choices": [{"delta": {"content": chunk.text}}]})
                        yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"

            return UnifiedResponse(stream_generator=google_stream_adapter(), model_used=model_name)
        else:
            return UnifiedResponse(content=response.text, model_used=model_name)

    except Exception as e:
        raise Exception(f"Google Native Error ({model_name}): {str(e)}")

def call_openrouter(prompt, model_name, stream=False, temperature=0.0, json_mode=False, timeout=60):
    """OpenRouter API Call"""
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://epistemiq.pythonanywhere.com/",
        "X-Title": "Epistemiq"
    }

    if "70b" in model_name or "pro" in model_name:
        timeout = max(timeout, 90)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": temperature
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(OR_URL, headers=headers, json=payload, stream=stream, timeout=timeout)

        if response.status_code in [429, 502, 503, 504]:
            raise Exception(f"OpenRouter Provider Error {response.status_code}")

        if response.status_code != 200:
            raise Exception(f"OpenRouter Status {response.status_code}: {response.text[:200]}")

        if stream:
            return UnifiedResponse(stream_generator=response.iter_content(chunk_size=1024, decode_unicode=True), model_used=model_name)
        else:
            return UnifiedResponse(content=response.json()['choices'][0]['message']['content'], model_used=model_name)

    except Exception as e:
        raise Exception(f"OpenRouter Error ({model_name}): {str(e)}")

def call_hybrid_flow(prompt, stream=False, json_mode=False, temperature=0.0, preferred_model=None, task_type="verification", timeout=60):
    """
    Iterates through models based on strategy.

    task_type="extraction":
       1. Gemma-3-27b (Google Native) -> Google Rotation -> OpenRouter Fallback

    task_type="verification" / "report":
       1. User Preferred Model (if set)
       2. Google Native Rotation (Default)
       3. OpenRouter Fallback
    """
    queue = []

    # 1. Build the Execution Queue
    if task_type == "extraction":
        # Force Gemma 3 (Native) first
        queue.append(EXTRACTION_DEFAULT_MODEL)
        queue.extend(GOOGLE_ROTATION_MODELS)
        queue.extend(OPENROUTER_ROTATION_MODELS)

    else: # verification or report
        if preferred_model and preferred_model.strip():
            # User wants a specific OpenRouter model
            queue.append(preferred_model)
            queue.extend(GOOGLE_ROTATION_MODELS)
            queue.extend(OPENROUTER_ROTATION_MODELS)
        else:
            # User wants Google Default
            queue.extend(GOOGLE_ROTATION_MODELS)
            queue.extend(OPENROUTER_ROTATION_MODELS)

    # 2. Deduplicate Queue
    seen = set()
    final_queue = [x for x in queue if not (x in seen or seen.add(x))]

    last_error = None

    # 3. Execute Rotation
    for model_name in final_queue:
        try:
            # Determine provider:
            # Google Native models don't have slashes (e.g. "gemma-3-27b-it", "gemini-2.0-flash")
            # OpenRouter models have slashes (e.g. "google/gemma-3-27b-it:free")
            is_google_native = "/" not in model_name

            logging.info(f"🔄 Attempting model: {model_name} ({'Google Native' if is_google_native else 'OpenRouter'})")

            if is_google_native:
                return call_google_native(prompt, model_name, stream, temperature, json_mode), model_name
            else:
                return call_openrouter(prompt, model_name, stream, temperature, json_mode, timeout), model_name

        except Exception as e:
            logging.warning(f"❌ Model {model_name} failed: {e}")
            last_error = e
            continue

    logging.error("All models in rotation failed.")
    raise last_error or Exception("All hybrid models failed.")


# ==============================================================
#  Utility functions
# ==============================================================

def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def json_loads(s: str, fallback):
    try:
        return json.loads(s) if s else fallback
    except Exception:
        return fallback

def new_analysis_id() -> str:
    return str(uuid.uuid4())

# ==============================================================
#  DB HELPERS (PostgreSQL)
# ==============================================================

def get_conn():
    """Connect to PostgreSQL using DATABASE_URL env var."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise Exception("DATABASE_URL environment variable is not set")
    conn = psycopg2.connect(db_url)
    return conn

def with_retry_db(fn):
    """Retry decorator adapted for Postgres."""
    def wrapper(*args, **kwargs):
        attempts = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except psycopg2.OperationalError as e:
                if attempts < 3:
                    attempts += 1
                    time.sleep(0.2 * attempts)
                else:
                    raise e
            except Exception as e:
                raise e
    return wrapper

# ==============================================================
#  QUOTA MANAGEMENT
# ==============================================================

QUOTA_LIMITS = {
    "analysis": 1,      # Extractions per day
    "verification": 1,  # Claims verified per day (Strict cost saving)
    "report": 1         # Deep dives per day
}

def check_and_increment_quota(user_id, quota_type):
    """
    Checks if user has quota left.
    Admin (epistemiq.ai@gmail.com) is EXEMPT.
    """
    limit = QUOTA_LIMITS.get(quota_type, 0)
    col_name = f"{quota_type}_count"

    conn = get_conn()
    try:
        with conn.cursor() as c:
            # 1. CHECK FOR ADMIN EXEMPTION
            c.execute("SELECT email FROM users WHERE id = %s", (user_id,))
            user_row = c.fetchone()

            if user_row and user_row[0] == "epistemiq.ai@gmail.com":
                return True, 0, 99999

            # 2. Standard Logic
            today = datetime.now().date()

            # Ensure record exists
            c.execute("""
                INSERT INTO user_quotas (user_id, usage_date, analysis_count, verification_count, report_count)
                VALUES (%s, %s, 0, 0, 0)
                ON CONFLICT (user_id, usage_date) DO NOTHING
            """, (user_id, today))

            # Check usage
            c.execute(f"SELECT {col_name} FROM user_quotas WHERE user_id=%s AND usage_date=%s", (user_id, today))
            current_count = c.fetchone()[0]

            if current_count >= limit:
                return False, current_count, limit

            # Increment
            c.execute(f"UPDATE user_quotas SET {col_name} = {col_name} + 1 WHERE user_id=%s AND usage_date=%s", (user_id, today))
            conn.commit()

            return True, current_count + 1, limit
    finally:
        conn.close()

def get_todays_spotlight_id():
    """Calculates the Analysis ID for the Daily Spotlight based on the date."""
    conn = get_conn()
    try:
        with conn.cursor() as c:
            # Pick a "random" one based on the date (stable for 24h) from recent 50
            # Filtering for ordinal=0 ensures we get unique analysis IDs
            c.execute("""
                SELECT a.analysis_id
                FROM analyses a
                JOIN claims c ON a.analysis_id = c.analysis_id
                WHERE c.ordinal = 0
                ORDER BY a.created_at DESC
                LIMIT 50
            """)
            rows = c.fetchall()

            if not rows: return None

            today_int = int(datetime.now().strftime("%Y%m%d"))
            selected_index = today_int % len(rows)
            return rows[selected_index][0]
    except Exception as e:
        logging.error(f"Spotlight ID error: {e}")
        return None
    finally:
        conn.close()

# ==============================================================
#  EMBEDDING FUNCTION (Google AI)
# ==============================================================

def get_google_embedding(text):
    """Generates a 768-dimensional embedding using Google Gemini."""
    if not text: return None
    try:
        # text-embedding-004 returns 768 dimensions
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text[:9000], # Google limit approx 10k chars
            task_type="SEMANTIC_SIMILARITY"
        )
        return result['embedding']
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        return None

# ==============================================================
#  Initialization: POSTGRES SCHEMA
# ==============================================================

def init_db():
    """Initialize Database Tables (PostgreSQL Syntax)"""
    conn = get_conn()
    try:
        with conn.cursor() as c:
            # 1. Enable Vector Extension
            try:
                c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as e:
                conn.rollback()
                logging.warning(f"Vector extension creation failed (might already exist): {e}")

            # Users & Auth
            c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )""")

            # Quota System
            c.execute("""
            CREATE TABLE IF NOT EXISTS user_quotas (
                user_id INTEGER REFERENCES users(id),
                usage_date DATE DEFAULT CURRENT_DATE,
                analysis_count INTEGER DEFAULT 0,
                verification_count INTEGER DEFAULT 0,
                report_count INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, usage_date)
            )""")

            c.execute("""
            CREATE TABLE IF NOT EXISTS magic_links (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                token_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                used_at TIMESTAMP
            )""")

            c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                session_token TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                user_agent TEXT,
                ip_hash TEXT
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token)")

            # Core Analyses (Updated for 768 dim)
            c.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                text_hash TEXT,
                canonical_text TEXT,
                mode TEXT,
                source_type TEXT,
                source_meta TEXT,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_published BOOLEAN DEFAULT FALSE,
                published_slug TEXT UNIQUE,
                published_title TEXT,
                published_at TIMESTAMP,
                published_summary TEXT,
                published_image_url TEXT
            )""")
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_analyses_embedding
                ON analyses USING hnsw (embedding vector_cosine_ops)
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_analyses_hash_mode ON analyses(text_hash, mode)")

            # User Analyses Mapping
            c.execute("""
            CREATE TABLE IF NOT EXISTS user_analyses (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                analysis_id TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, analysis_id)
            )""")

            # Input Caches
            c.execute("""
            CREATE TABLE IF NOT EXISTS pasted_texts (
                text_hash TEXT PRIMARY KEY,
                text_content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

            c.execute("""
            CREATE TABLE IF NOT EXISTS article_cache (
                url_hash TEXT PRIMARY KEY,
                url TEXT,
                raw_html TEXT,
                article_text TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_article_cache_url ON article_cache(url)")

            c.execute("""
            CREATE TABLE IF NOT EXISTS media_cache (
                file_hash TEXT PRIMARY KEY,
                media_type TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

            # Claims Table
            c.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                analysis_id TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                claim_text TEXT NOT NULL,
                claim_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                final_verdict TEXT,
                synthesis_summary TEXT,
                category TEXT,
                tags TEXT,
                search_keywords TEXT
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_claims_analysis ON claims(analysis_id, ordinal)")

            # Verdict Caches (For the 'Deep Dive' Accordion - 768 dim for Google embeddings)
            c.execute("""
            CREATE TABLE IF NOT EXISTS model_cache (
                claim_hash TEXT PRIMARY KEY,
                verdict TEXT,
                questions_json TEXT,
                keywords_json TEXT,
                used_model TEXT,
                claim_embedding vector(768),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

            c.execute("""
            CREATE TABLE IF NOT EXISTS external_cache (
                claim_hash TEXT PRIMARY KEY,
                verdict TEXT,
                sources_json TEXT,
                used_model TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

            c.execute("""
            CREATE TABLE IF NOT EXISTS report_cache (
                rq_hash TEXT PRIMARY KEY,
                question_text TEXT,
                report_text TEXT,
                used_model TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

            # ✅ PUBLISHING ARCHIVE (Snapshot Table)
            c.execute("""
            CREATE TABLE IF NOT EXISTS published_articles (
                id SERIAL PRIMARY KEY,
                slug TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                image_url TEXT,
                published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags JSONB,
                categories JSONB,
                content_snapshot JSONB NOT NULL
            )""")

            # ✅ COMMENTS (Linked to Archive)
            c.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id SERIAL PRIMARY KEY,
                article_id INTEGER REFERENCES published_articles(id) ON DELETE CASCADE,
                user_id INTEGER REFERENCES users(id),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_comments_article ON comments(article_id)")

            conn.commit()
    except Exception as e:
        print(f"❌ Database Init Failed: {e}")
    finally:
        conn.close()


# ==============================================================
#           CACHE + CLAIM HELPERS (UPDATED FOR UNIFIED FLOW)
# ==============================================================

def sha256_str(s: str):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def canonicalize_text(text: str) -> str:
    if not text: return ""
    txt = text.replace("\r\n", "\n").replace("\r", "\n")
    txt = " ".join(txt.split())
    return txt.strip()

def text_hash(text: str) -> str:
    canon = canonicalize_text(text)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()

@with_retry_db
def save_claims_for_analysis(analysis_id: str, claims_data_list: list):
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("DELETE FROM claims WHERE analysis_id=%s", (analysis_id,))

            for idx, item in enumerate(claims_data_list):
                claim_text = item.get("claim", "").strip()
                if not claim_text: continue

                keywords_raw = item.get("keywords", [])
                keywords_json = json_dumps(keywords_raw) if keywords_raw else None

                claim_text_clean = claim_text
                claim_hash = sha256_str(claim_text_clean.lower())
                claim_id = sha256_str(f"{analysis_id}|{idx}|{claim_text_clean}")

                c.execute("""
                INSERT INTO claims (
                    claim_id, analysis_id, ordinal, claim_text,
                    claim_hash, search_keywords, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (claim_id) DO UPDATE SET
                    claim_text = EXCLUDED.claim_text,
                    claim_hash = EXCLUDED.claim_hash,
                    search_keywords = EXCLUDED.search_keywords,
                    created_at = CURRENT_TIMESTAMP
                """, (claim_id, analysis_id, idx, claim_text_clean, claim_hash, keywords_json))
        conn.commit()
    finally:
        conn.close()

def get_claims_for_analysis(analysis_id: str):
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("SELECT claim_text FROM claims WHERE analysis_id=%s ORDER BY ordinal", (analysis_id,))
            rows = c.fetchall()
            return [row[0] for row in rows]
    finally:
        conn.close()

@with_retry_db
def save_pasted_text_to_db(text_content):
    t_hash = text_hash(text_content)
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""
            INSERT INTO pasted_texts (text_hash, text_content, created_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (text_hash) DO NOTHING
            """, (t_hash, text_content))
        conn.commit()
    finally:
        conn.close()

@with_retry_db
def save_article_to_cache_db(url, text):
    url_hash = sha256_str(url)
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""
            INSERT INTO article_cache (url_hash, url, raw_html, article_text, fetched_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (url_hash) DO UPDATE SET
                article_text = EXCLUDED.article_text,
                fetched_at = CURRENT_TIMESTAMP
            """, (url_hash, url, "", text))
        conn.commit()
    finally:
        conn.close()

# ==============================================================
#               MEDIA CACHE HELPERS (POSTGRESQL)
# ==============================================================

def compute_file_hash(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def get_cached_media(file_hash):
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute('SELECT extracted_text FROM media_cache WHERE file_hash = %s', (file_hash,))
            result = c.fetchone()
            return result[0] if result else None
    finally:
        conn.close()

@with_retry_db
def store_media_cache(file_hash, media_type, extracted_text):
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""
            INSERT INTO media_cache (file_hash, media_type, extracted_text, created_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (file_hash) DO UPDATE SET
                extracted_text = EXCLUDED.extracted_text
            """, (file_hash, media_type, extracted_text))
        conn.commit()
    finally:
        conn.close()

def save_uploaded_file(file, upload_folder=None):
    """Save uploaded file and return path"""
    if upload_folder is None:
        if os.getenv("DB_PATH"): # Docker
            upload_folder = "/app/uploads"
        else: # Local
            base = os.path.dirname(os.path.abspath(__file__))
            upload_folder = os.path.join(base, "uploads")
    try:
        os.makedirs(upload_folder, exist_ok=True)
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return filepath
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        return None

# ==============================================================
#                       CLEANUP (POSTGRESQL)
# ==============================================================

@with_retry_db
def cleanup_old_cache():
    conn = get_conn()
    try:
        with conn.cursor() as c:
            cutoff = datetime.now() - timedelta(days=30)

            # 1. DELETE UNUSED MEDIA
            c.execute('DELETE FROM media_cache WHERE created_at < %s', (cutoff,))
            media_deleted = c.rowcount

            # 2. DELETE OLD ANALYSES (Aggressive cleanup - Published content is safe in Archive)
            c.execute("""
                DELETE FROM analyses
                WHERE last_accessed < %s
            """, (cutoff,))
            analyses_deleted = c.rowcount

            # 3. DELETE INPUT CACHES
            c.execute("""
                DELETE FROM pasted_texts
                WHERE created_at < %s
                AND text_hash NOT IN (
                    SELECT text_hash FROM analyses
                )
            """, (cutoff,))
            texts_deleted = c.rowcount

            # 4. DELETE ORPHANED VERDICTS
            # Only keep verdicts that are linked to currently active claims
            active_claims_query = "SELECT claim_hash FROM claims"

            c.execute(f"""
                DELETE FROM model_cache
                WHERE updated_at < %s
                AND claim_hash NOT IN ({active_claims_query})
            """, (cutoff,))
            model_deleted = c.rowcount

            c.execute(f"""
                DELETE FROM external_cache
                WHERE updated_at < %s
                AND claim_hash NOT IN ({active_claims_query})
            """, (cutoff,))
            external_deleted = c.rowcount

            # 5. DELETE ORPHANED REPORTS
            # Reports are harder to join directly, so we delete based on time for now
            # (Or implement a strict join if rq_hash logic allows)
            c.execute('DELETE FROM report_cache WHERE updated_at < %s', (cutoff,))
            report_deleted = c.rowcount

            # 6. DELETE ARTICLE CACHE
            c.execute('DELETE FROM article_cache WHERE fetched_at < %s', (cutoff,))
            articles_deleted = c.rowcount

        conn.commit()
        logging.info(
            f"Cache cleanup: {media_deleted} media, {analyses_deleted} analyses, "
            f"{texts_deleted} texts, {articles_deleted} articles, "
            f"{model_deleted} model, {external_deleted} external, "
            f"{report_deleted} reports removed."
        )
    except Exception as e:
        logging.error(f"Cleanup error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


# ==============================================================
#  UNIFIED PROMPT CONFIGURATION (STRICT & ORIGINAL LOGIC)
# ==============================================================

# 1. EXTRACTION CONFIGURATION (Robust Text Mode)
# We use the pipe format '|' because it is reliable for free models.
BASE_TEXT_INSTRUCTION = '''
**Strict rules:**
- ONLY include claims that appear EXPLICITLY in the text.
- Each claim must be explicitly stated.
- If no explicit, complete, testable claims exist, output exactly: "No explicit claims found."
- Absolutely DO NOT infer, paraphrase, generalize, or introduce external knowledge.
- NEVER include incomplete sentences, headings, summaries, conclusions, speculations, questions, or introductory remarks.
- **FORMAT:** Use a numbered list. Separate the claim and keywords with a pipe symbol "|".
- **Keywords:** 3-5 specific search phrases (2-4 words each). Include specific chemical names, physical laws, project names, or specific authors.

**STRICT OUTPUT FORMAT:**
1. The claim text goes here | Keywords: keyword1, keyword2, keyword3
2. Another claim text here | Keywords: keyword1, keyword2, keyword3
'''

UNIFIED_EXTRACTION_TEMPLATES = {
    "General Analysis of Testable Claims": f'''
    You will be given a text. Extract a **numbered list** of the **top up to 7** most scientifically significant and testable claims.
    Prioritize controversial, specific, or verifiable assertions over general statements.

    {{text}}

    {BASE_TEXT_INSTRUCTION}
    ''',

    "Specific Focus on Scientific Claims": f'''
    You will be given a text. Extract a **numbered list** of the **top up to 7** most significant, scientifically testable claims related to science.
    Prioritize controversial, data-driven or specific experimental assertions.

    {{text}}

    {BASE_TEXT_INSTRUCTION}
    ''',

    "Technology-Focused Extraction": f'''
    You will be given a text. Extract a **numbered list** of the **top up to 7** most significant, testable claims related to technology.
    Prioritize specific capabilities, benchmarks, or innovation claims.

    {{text}}

    {BASE_TEXT_INSTRUCTION}
    '''
}

# 2. VERIFICATION MODES (Exact Original Logic Tables)
VERIFICATION_MODES = {
    "General Analysis of Testable Claims": '''
    **ROLE:** You are a rigorous scientific fact-checker.

    **VERDICT LOGIC TABLE (Follow strictly):**
    - If the claim is a known scientific fact -> **VERIFIED**
    - If the claim is plausible but lacks proof -> **POSSIBLE_BUT_UNPROVEN**
    - If the claim contradicts known science (e.g. "Earth is flat", "CERN opened portal") -> **NONSENSE**
    - If the claim is from a fictional/viral story and not real science -> **NONSENSE**
    - If the claim is nowhere to be found in real science -> **NOT_SUPPORTED**

    **CRITICAL RULES:**
    - Use the Source Text ONLY for definition. If the claim says "The team", check the Source Text to know it refers to CERN.
    - Do NOT treat the Source Text as evidence. The Source Text is the material we are questioning.
    - **REALITY CHECK:** Does the claim entity (event, fact, phenomenon, state, discovery, breakthrough) exist in the real world outside of this text? If not, the verdict is NOT_SUPPORTED or NONSENSE.
    ''',

    "Specific Focus on Scientific Claims": '''
    **ROLE:** You are a rigorous scientific fact-checker.

    **VERDICT LOGIC TABLE (Follow strictly):**
    - If the claim is a known scientific fact -> **VERIFIED**
    - If the claim contradicts standard models (e.g. "CERN simulation became conscious") -> **NONSENSE**
    - If the claim is a misinterpretation of real science -> **UNLIKELY**
    - If the claim exists only in viral posts -> **NOT_SUPPORTED**

    **CRITICAL RULES:**
    - Use the Source Text ONLY for definition.
    - **Consensus:** Judge validity against established mathematics, chemistry, physics, and biology.
    - **REALITY CHECK:** Does the claim entity exist in the real world?
    ''',

    "Technology-Focused Extraction": '''
    **ROLE:** You are a technology fact-checker.

    **VERDICT LOGIC TABLE (Follow strictly):**
    - If technology exists and works -> **VERIFIED**
    - If technology is theoretical -> **FEASIBLE**
    - If technology is scientifically impossible -> **NONSENSE**
    - If claim is a hoax -> **NOT_SUPPORTED**

    **CRITICAL RULES:**
    - **Identify Entities:** Use the Source Text to define vague terms.
    - **Reality Check:** Do not blindly believe the Source Text. Evaluate technical feasibility based on real-world engineering standards.
    '''
}

# 3. THE MASTER UNIFIED PROMPT
UNIFIED_VERIFICATION_PROMPT = '''
You are a rigorous scientific analyst and the Executive Editor of "The Epistemiq Compass".
Your goal is to write a **Definitive Editorial Verdict** that synthesizes both your internal AI analysis and the external scientific literature.

---
### INPUT DATA
1. **Context of Claim:** "{short_context}"
2. **The Claim:** "{claim_text}"
3. **Scientific Papers Found:**
"""{paper_abstracts}"""

---
### STEP 1: INTERNAL ANALYSIS (Blind Test)
**CRITICAL RULE:** Do NOT look at the "Scientific Papers Found" section yet.
Use ONLY your internal pre-trained knowledge to evaluate the scientific validity of the claim.
- If the claim refers to specific data (e.g. "recorded by Perseverance"), verify if this is a known event/discovery/phenomenon in your training data.
- {mode_specific_instructions}
- **Mandatory:** You MUST provide a detailed explanation (approx 300 words) justifying your verdict. Do NOT just output the verdict label.

---
### STEP 2: EXTERNAL ANALYSIS (Paper Review)
Now, look *only* at the "Scientific Papers Found".
1. **Relevance Check:** Do these papers explicitly confirm the specific event/discovery/phenomenon described in the claim?
2. **Verdict:** Analyze the papers. Do they Support, Refute, or are they Irrelevant?
3. If the papers discuss similar topics but do NOT mention the specific breakthrough claimed, point that out.
3. **Citation:** You MUST cite specific papers from the list by Title/Year in your text.
- **Mandatory:** You MUST provide a detailed explanation (approx 300 words) justifying your verdict. Do NOT just output the verdict label.

---
### STEP 3: FUTURE RESEARCH
Generate 3 specific, open-ended research questions that would help a user validate this claim further (e.g., "What is the specific mechanism...?" or "Has this been replicated in humans?").

---
### STEP 4: THE SYNTHESIS (CRITICAL THINKING PROCESS)
**CRITICAL THINKING PROCESS:**
- **Compare Sources:** Does the External Search (Papers) support the Internal AI's theory?
    - If they agree, reinforce the conclusion.
    - If they DISAGREE, **you must reconcile them.**
    - Prioritize External Verification if it cites papers.
    - Explain *why* there is a mismatch (e.g. "External search likely failed due to keywords").
- **Detect Gaps:** If the Internal AI says "Possible" but External Search says "False/Unsupported", check the External Sources. Are they relevant? Or did they miss the topic?
- **Judge the Discrepancy:**
    - If papers exist but disprove the claim -> "Explicitly Debunked".
    - If no relevant papers were found -> "Theoretical but Unproven".
    - If papers confirm it -> "Scientific Consensus".
- **Weigh Evidence:** Prioritize peer-reviewed studies over preprints, and larger studies over smaller ones.
- **Nuance:** If evidence is mixed, highlight the complexity rather than oversimplifying.

**CATEGORY SELECTION GUIDE (CRITICAL):**
- **"The Hype Check"**: Use when a claim exaggerates a real scientific finding (e.g., "New study cures cancer" when it was just mice).
- **"The Scam Bust"**: Use when the claim is pseudoscience, a hoax, or scientifically impossible (e.g., "Perpetual motion machine", "Aliens confirmed").
- **"The Nuance"**: Use when the claim is partially true but lacks context, or if scientific consensus is mixed/debated.
- **"Fact Check"**: Use ONLY for dry, binary historical or data verification (e.g., "Apollo 11 landed in 1969"). **Avoid using this if others apply.**

---
### OUTPUT FORMAT (Strict JSON)
{{
  "compass": {{
    "category": "The Hype Check" | "The Scam Bust" | "The Nuance" | "Fact Check",
    "verdict_label": "Short Badge (e.g. Feasible but Unproven, Debunked, Widely Accepted)",
    "summary": "The Editorial (max 150 words). **The 'Report Style' Reconciliation:** If the AI said 'True' but Papers said 'Unsupported', your summary must start by identifying this gap. Example: 'Initial AI analysis suggests [Consensus], yet external verification found no direct experimental support in the provided literature. This mismatch likely stems from [Keyword failure / Narrow search / Theoretical nature of the claim]. Consequently, the verdict is [Verdict].'",
    "tags": ["#tag1", "#tag2", "#tag3"]
  }},
  "evidence": {{
    "model_verdict": "Verdict: **[VERDICT]**\\n\\n[Detailed Internal Analysis Paragraph - REQUIRED. Start with the **VERDICT** from the Logic Table (e.g., 'Verdict: **POSSIBLE**'). Then provide a detailed internal assessment based on the Logic Table defined above in up to 300 words. **CITATION RULE:** Explicitly cite the sources that you internally have acces to and that you used (provide a numbered list of the sources you used with url links if available), and name the scientific principles, laws, or consensus you are using (e.g., 'According to the Standard Model...', 'Based on known thermodynamics...').",
    "external_verdict": "Verdict: **[VERDICT]**\\n\\n[Detailed External Analysis Paragraph - REQUIRED. Start with a **VERDICT** based strictly on the papers (e.g., 'Verdict: **SUPPORTED**'). Then provide a detailed analysis of the retrieved papers in up to 300 words. **CITATION RULE:** Explicitly cite the papers by title and authors (this is especially important if the authors are mentioned in the source text/context of the claim), and explain how they support or refute the claim.",
    "questions": ["Question 1", "Question 2", "Question 3"]
  }}
}}
'''


# Helper functions

# ==============================================================
#  FACT VERIFICATION RERANKING (GOOGLE AI)
# ==============================================================

def calculate_cosine_similarity(vec_a, vec_b):
    """Pure Python Cosine Similarity for vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

def rank_papers_with_google(claim_text, papers):
    """
    Reranks papers using Google's FACT_VERIFICATION embeddings.
    Optimized to use BATCH embedding to avoid 429 Rate Limits.
    """
    if not papers: return []

    try:
        # 1. Embed the Claim (The "Query")
        claim_result = genai.embed_content(
            model="models/text-embedding-004",
            content=claim_text[:2000],
            task_type="FACT_VERIFICATION"
        )
        claim_vector = claim_result['embedding']

        # 2. Prepare Batch for Papers (The "Corpus")
        valid_papers = []
        texts_to_embed = []

        for p in papers:
            # Combine Title + Abstract
            text_content = f"{p.get('title', '')} {p.get('abstract', '')}"
            if len(text_content) < 20: continue

            valid_papers.append(p)
            texts_to_embed.append(text_content[:4000]) # Stay within char limits

        if not texts_to_embed: return []

        # 3. Call Batch Embed API
        # task_type="RETRIEVAL_DOCUMENT" is applied to all
        batch_result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts_to_embed,
            task_type="RETRIEVAL_DOCUMENT"
        )

        # Google returns a dict with 'embedding' key which is a list of vectors
        doc_vectors = batch_result['embedding']

        scored_papers = []
        for i, doc_vector in enumerate(doc_vectors):
            score = calculate_cosine_similarity(claim_vector, doc_vector)
            scored_papers.append({**valid_papers[i], "score": score})

        # 4. Sort
        scored_papers.sort(key=lambda x: x['score'], reverse=True)
        logging.info(f"Reranked {len(scored_papers)} papers in batch. Top: {scored_papers[0]['score']:.4f}")

        return scored_papers

    except Exception as e:
        logging.error(f"Reranking failed: {e}")
        return papers # Fallback

def fetch_crossref(keywords):
    if not keywords:
        return []

    search_query = ' '.join(keywords)
    url = f"https://api.crossref.org/works?query={quote_plus(search_query)}&rows=3&select=title,URL,author,abstract,published-print,published-online,is-referenced-by-count"
    headers = {"User-Agent": "Epistemiq/1.0 (mailto:epistemiq.ai@gmail.com)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        results = []
        for item in items:
            year = ""
            if item.get("published-print"):
                year = item["published-print"]["date-parts"][0][0]
            elif item.get("published-online"):
                year = item["published-online"]["date-parts"][0][0]

            authors_list = [f"{a.get('given','')} {a.get('family','')}".strip() for a in item.get("author", [])]
            authors_str = ", ".join(authors_list[:3])

            results.append({
                "title": item.get("title", ["No title"])[0] if item.get("title") else "No title",
                "abstract": item.get("abstract", "Abstract not available"),
                "url": item.get("URL", ""),
                "authors": authors_str,
                "year": str(year),
                "citation_count": item.get("is-referenced-by-count", 0),

                # ✅ ENSURE THIS IS PRESENT
                "source": "Crossref"
            })
        return results
    except requests.exceptions.RequestException as e:
        logging.warning(f"CrossRef API call failed: {e}")
        return []

def fetch_core(keywords, max_results=10):
    """Fetch research papers from CORE API (Internal Python Call)"""
    api_key = os.getenv("CORE_API_KEY")
    if not api_key:
        logging.warning("CORE_API_KEY not set")
        return []

    # Construct query
    query = ' '.join(keywords)
    url = "https://api.core.ac.uk/v3/search/works"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"q": query, "limit": max_results}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code != 200:
            logging.warning(f"CORE API Error: {response.status_code}")
            return []

        data = response.json()
        results = []

        for item in data.get("results", []):
            # Extract Abstract
            abstract = item.get("abstract") or item.get("description") or "No abstract available."

            # Extract URL (prefer download, then fullText, then landing)
            link = item.get("downloadUrl")
            if not link and item.get("links"):
                link = item.get("links")[0].get("url")
            if not link:
                link = f"https://core.ac.uk/outputs/{item.get('id')}"

            # Extract Authors
            authors_list = [a.get("name") for a in item.get("authors", [])]
            authors_str = ", ".join(authors_list[:3])

            results.append({
                "title": item.get("title", "No title"),
                "abstract": abstract,
                "url": link,
                "authors": authors_str,
                "year": str(item.get("yearPublished") or ""),
                "citation_count": item.get("citationCount", 0),
                "source": "CORE"
            })
        return results
    except Exception as e:
        logging.warning(f"CORE Fetch failed: {e}")
        return []

def fetch_semantic_scholar(keywords, max_results=3):
    """Fetch research papers from Semantic Scholar API"""
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not SEMANTIC_SCHOLAR_API_KEY:
        logging.warning("SEMANTIC_SCHOLAR_API_KEY not set")
        return []

    search_query = ' '.join(keywords)
    headers = {
        "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
        "User-Agent": "Epistemiq/1.0 (mailto:epistemiq.ai@gmail.com)"
    }
    params = {
        "query": search_query,
        "limit": max_results,
        "fields": "title,abstract,url,authors,year,citationCount,venue,publicationTypes,externalIds"
    }

    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params=params,
            timeout=10
        )
        if response.status_code == 429:
            logging.warning("Semantic Scholar rate limit exceeded")
            return []
        response.raise_for_status()
        data = response.json()
        results = []
        if "data" in data and data["data"]:
            for paper in data["data"]:
                authors_list = []
                if paper.get("authors"):
                    authors_list = [author.get("name", "") for author in paper["authors"]]
                authors_str = ", ".join(authors_list[:3])
                if len(authors_list) > 3:
                    authors_str += " et al."

                paper_url = paper.get("url", "")
                if not paper_url and paper.get("externalIds", {}).get("DOI"):
                    paper_url = f"https://doi.org/{paper['externalIds']['DOI']}"
                elif not paper_url and paper.get("externalIds", {}).get("ArXiv"):
                    paper_url = f"https://arxiv.org/abs/{paper['externalIds']['ArXiv']}"

                result = {
                    "title": paper.get("title", "No title"),
                    "abstract": paper.get("abstract", "Abstract not available"),
                    "url": paper_url,
                    "authors": authors_str,
                    "year": paper.get("year", ""),
                    "citation_count": paper.get("citationCount", 0),
                    "venue": paper.get("venue", ""),
                    "publication_types": paper.get("publicationTypes", []),
                    "source": "Semantic Scholar"
                }
                results.append(result)
        return results
    except requests.exceptions.RequestException as e:
        logging.warning(f"Semantic Scholar API call failed for query '{search_query}': {e}")
        return []
    except Exception as e:
        logging.warning(f"Unexpected error in Semantic Scholar search: {e}")
        return []

def fetch_pubmed(keywords):
    """Fetch medical literature from PubMed"""
    if not keywords:
        return []

    search_query = '+'.join(keywords)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_query}&retmode=json&retmax=3"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        id_list = data.get('esearchresult', {}).get('idlist', [])
        if not id_list:
            return []

        details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(id_list)}&retmode=json"
        details_response = requests.get(details_url, timeout=10)
        details_data = details_response.json()

        results = []
        for pubmed_id in id_list:
            article_data = details_data.get('result', {}).get(pubmed_id, {})

            pubdate = article_data.get('pubdate', '')
            year = pubdate.split(' ')[0] if pubdate else ""

            authors_list = [a.get('name', '') for a in article_data.get('authors', [])]
            authors_str = ", ".join(authors_list[:3])

            results.append({
                "title": article_data.get('title', 'No title'),
                "abstract": article_data.get('abstract', 'Abstract not available on PubMed API'),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                "year": year,
                "authors": authors_str,
                "citation_count": 0,

                # ✅ ENSURE THIS IS PRESENT
                "source": "PubMed"
            })
        return results
    except Exception as e:
        logging.warning(f"PubMed API call failed: {e}")
        return []

def analyze_image_with_ocr(image_path):
    """Extract text from image using OCR"""
    try:
        extracted_text = pytesseract.image_to_string(Image.open(image_path))
        return extracted_text.strip()
    except Exception as e:
        logging.error(f"OCR processing failed: {e}")
        return ""

def analyze_image_with_openrouter(image_path):
    """
    Uses OpenRouter (Free Tier) for OCR to avoid Google Direct Rate Limits.
    Tries Gemini 2.0 Flash first, then Llama 3.2 Vision.
    """
    if not OPENROUTER_API_KEY:
        return None

    # 1. Encode Image to Base64
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Image encoding failed: {e}")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://epistemiq.vercel.app",
        "X-Title": "Epistemiq"
    }

    # Rotation of Free Vision Models
    models = [
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free"
    ]

    prompt = "Transcribe all the text visible in this image. Do not describe the image, just return the text found."

    for model in models:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }

        try:
            logging.info(f"Attempting OCR with OpenRouter: {model}")
            response = requests.post(OR_URL, headers=headers, json=payload, timeout=45)

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                if content:
                    return content.strip()
            else:
                logging.warning(f"OpenRouter OCR failed ({model}): {response.status_code} - {response.text}")

        except Exception as e:
            logging.error(f"OpenRouter OCR error: {e}")
            continue

    return None

def analyze_image_with_gemini(image_path):
    """
    Uses Google Gemini to transcribe text from an image.
    Rotates through SPECIFIC STABLE model versions to avoid 404s and 429s.
    """
    # Use exact version numbers. Aliases like 'gemini-1.5-flash' sometimes fail
    # depending on region/library version.
    vision_models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-pro-exp-02-05"
        "gemma-3-27b-it"
    ]

    try:
        img = Image.open(image_path)
        prompt = "Transcribe all the text visible in this image. Do not describe the image, just return the text found."

        for model_name in vision_models:
            try:
                logging.info(f"Attempting OCR with {model_name}...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, img])

                if response.text:
                    return response.text.strip()
            except Exception as e:
                # Log specific error but keep trying
                if "429" in str(e):
                    logging.warning(f"Quota exceeded for {model_name}. Rotating...")
                    time.sleep(1) # Short pause before next try
                elif "404" in str(e):
                    logging.warning(f"Model {model_name} not found/supported. Rotating...")
                else:
                    logging.warning(f"OCR error with {model_name}: {e}")
                continue

        return None

    except Exception as e:
        logging.error(f"Gemini Vision fatal error: {e}")
        return None

def save_uploaded_file(file, upload_folder=None):
    """Save uploaded file and return path"""
    # Logic to handle Docker vs Local paths automatically
    if upload_folder is None:
        if os.getenv("DB_PATH"):
            # We are likely in Docker if this env var is set
            upload_folder = "/app/uploads"
        else:
            # We are running locally without Docker
            base = os.path.dirname(os.path.abspath(__file__))
            upload_folder = os.path.join(base, "uploads")

    try:
        os.makedirs(upload_folder, exist_ok=True)
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return filepath
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        return None

def normalize_text_for_display(text):
    """Normalize text for HTML display with comprehensive character handling"""
    if not text:
        return text

    # Normalize Unicode first
    normalized = unicodedata.normalize('NFKC', text)  # Changed to NFKC for more aggressive normalization

    # Comprehensive replacements for display
    replacements = {
        # Dashes and hyphens - comprehensive coverage
        '–': '-', '—': '-', '―': '-', '‒': '-', '‐': '-', '‑': '-',
        '−': '-', '–': '-', '—': '-', '―': '-', '': '-', '': '-',

        # Smart quotes and apostrophes
        '‘': "'", '’': "'", '‚': "'", '‛': "'",
        '“': '"', '”': '"', '„': '"', '‟': '"',
        '´': "'", '`': "'", 'ʻ': "'", 'ʼ': "'",
        '«': '"', '»': '"', '‹': "'", '›': "'",

        # Mathematical symbols and special characters
        '°': ' degrees ', '±': '+/-', '×': 'x', '÷': '/',
        '≈': '~', '≠': '!=', '≤': '<=', '≥': '>=',
        'µ': 'u', 'α': 'alpha', 'β': 'beta', 'γ': 'gamma',
        'δ': 'delta', 'ε': 'epsilon', 'θ': 'theta',

        # Common problematic encodings
        '': '', '': '', '': '', '': '', '': '',
        '': '', '': '', '': '', '': '', '': '',
        '': '', '': '', '': '', '': '', '': '',
        '': '', '': '', '': '', '': '', '': '',
        '': '', '': '', '': '', '': '', '': '',
        '': '', '': '', '': '', '': '', '': '',
        '': '', '': '',

        # Spaces and invisible characters
        '\u200b': '', '\ufeff': '', '\u202a': '', '\u202c': '',
        '\u200e': '', '\u200f': '', ' ': ' ', ' ': ' ',
        ' ': ' ', '': '', '': '', '\xa0': ' ',
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    # Remove any remaining problematic control characters
    normalized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', normalized)

    return normalized

# --- TEXT HELPERS ---

def normalize_text(text):
    """Cleans up encoding artifacts and controls."""
    if not text: return ""
    replacements = {
        "â€“": "-", "â€”": "-", "â€": "'", "â€œ": '"', "â€": '"',
        "â€™": "'", "â€˜": "'", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "&nbsp;": " "
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = "".join(ch for ch in text if ch == '\n' or ch == '\r' or ch == '\t' or ord(ch) >= 32)
    return text.strip()

def clean_xml(text):
    """Escapes characters for ReportLab XML parser."""
    if not text: return ""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def clean_verdict_text(text):
    """Removes 'Verdict:' labels to avoid duplication in PDF."""
    if not text: return ""
    # Strip markdown bold/italics markers
    text = text.replace('**', '').replace('*', '')
    # Strip prefixes case-insensitive
    text = re.sub(r'^Verdict:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def parse_markdown_to_reportlab(text):
    """Converts basic Markdown to ReportLab XML tags."""
    if not text: return ""
    text = clean_xml(text) # Escape first

    # Verdict cleanup inside reports
    text = re.sub(r'\*\*Verdict:\*\*', '<b>Verdict:</b>', text, flags=re.IGNORECASE)
    text = re.sub(r'Verdict:', '<b>Verdict:</b>', text, flags=re.IGNORECASE)

    lines = text.split('\n')
    processed_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append("")
            continue

        # Headers
        if line.startswith('###'):
            line = f"<b>{line[3:].strip()}</b><br/>"
        elif line.startswith('##'):
            line = f"<b><font size='13' color='#0d6efd'>{line[2:].strip()}</font></b><br/>"
        elif line.startswith('#'):
            line = f"<b><font size='14' color='#0d6efd'>{line[1:].strip()}</font></b><br/>"

        # Bullets
        if line.startswith('* ') or line.startswith('- '):
            line = f"<bullet>&bull;</bullet> {line[2:].strip()}"

        processed_lines.append(line)

    text = '\n'.join(processed_lines)
    # Inline Bold/Italic
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = text.replace('\n', '<br/>')

    return text

def render_report_html(text):
    """
    Converts Report Markdown (Headers, Tables, Lists) to HTML for the Compass View.
    """
    if not text: return ""

    # 1. Escape HTML first for safety
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines = text.split('\n')
    html_output = []

    table_buffer = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        # --- Table Detection ---
        if "|" in stripped:
            table_buffer.append(stripped)
            in_table = True
            continue
        else:
            if in_table:
                # Flush Table
                html_output.append(_render_table(table_buffer))
                table_buffer = []
                in_table = False

        # --- Normal Markdown ---
        if stripped.startswith("##"):
            # Header
            content = stripped.replace("#", "").strip()
            html_output.append(f'<h4 class="mt-4 mb-3 fw-bold text-dark">{content}</h4>')

        elif stripped.startswith("- ") or stripped.startswith("* "):
            # List Item
            content = stripped[1:].strip()
            # Handle Bold within list
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            html_output.append(f'<li class="ms-3 mb-1">{content}</li>')

        elif stripped == "":
            html_output.append("")

        else:
            # Standard Paragraph
            # ✅ FIX: Initialize 'content' from 'stripped' before regex
            content = stripped
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            html_output.append(f'<p>{content}</p>')

    # Flush pending table at end
    if in_table:
        html_output.append(_render_table(table_buffer))

    return "\n".join(html_output)

def _render_table(buffer):
    """Helper to turn a list of pipe-separated lines into a Bootstrap table."""
    if len(buffer) < 2: return "\n".join(buffer) # Not a valid table

    html = '<div class="table-responsive mb-3"><table class="table table-bordered table-sm table-striped table-hover">'

    # Header (First line)
    html += '<thead class="table-light"><tr>'
    headers = [h.strip() for h in buffer[0].strip('|').split('|')]
    for h in headers:
        html += f'<th>{h}</th>'
    html += '</tr></thead><tbody>'

    # Body (Skip separator line if it exists e.g. ---|---)
    start_idx = 2 if '---' in buffer[1] else 1

    for row in buffer[start_idx:]:
        cols = [c.strip() for c in row.strip('|').split('|')]
        html += '<tr>'
        for c in cols:
            html += f'<td>{c}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    return html

def normalize_text_for_pdf(text):
    """
    Aggressively cleans text for PDF generation (ReportLab Helvetica).
    Converts ALL fancy unicode characters to ASCII equivalents.
    """
    if not text: return ""

    # 1. Normalize unicode
    text = unicodedata.normalize('NFKD', text)

    # 2. Manual Map for common offenders
    replacements = {
        "â€“": "-", "â€”": "-", "â€": "'", "â€œ": '"', "â€": '"',
        "â€™": "'", "â€˜": "'", "â-": "-", "â": "-",
        "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
        "…": "...", "\u2022": "*", "\u25a0": "-" # The square block
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # 3. ASCII Fallback: Remove anything that isn't basic Latin-1
    # This prevents '■' and other mojibake from crashing the font renderer
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text.strip()

# --- STYLES ---

def get_pro_styles():
    styles = getSampleStyleSheet()
    brand_blue = HexColor("#0d6efd")
    text_dark = HexColor("#212529")
    text_grey = HexColor("#495057")

    styles.add(ParagraphStyle(
        name='ClaimHeading', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=14, textColor=brand_blue,
        spaceBefore=15, spaceAfter=8, leading=18, wordWrap='CJK'
    ))
    styles.add(ParagraphStyle(
        name='SectionHeading', parent=styles['Heading3'],
        fontName='Helvetica-Bold', fontSize=11, textColor=text_dark,
        spaceBefore=10, spaceAfter=4, leading=14
    ))
    styles.add(ParagraphStyle(
        name='ProBody', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=14,
        textColor=text_grey, spaceAfter=6, alignment=TA_LEFT,
        wordWrap='CJK'
    ))
    styles.add(ParagraphStyle(
        name='ProList', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=14,
        textColor=text_grey, leftIndent=12, firstLineIndent=0,
        spaceAfter=3, bulletIndent=0, wordWrap='CJK'
    ))
    styles.add(ParagraphStyle(
        name='ProLink', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=12,
        textColor=brand_blue, spaceAfter=2, wordWrap='CJK'
    ))
    return styles

# --- DRAWING HELPERS ---

def draw_page_footer(pdf_canvas, page_width):
    """Draws Page Number and Branding Footer"""
    pdf_canvas.saveState()
    pdf_canvas.setFont("Helvetica", 8)
    pdf_canvas.setFillColor(colors.grey)

    # Line
    pdf_canvas.setLineWidth(0.5)
    pdf_canvas.setStrokeColor(colors.lightgrey)
    pdf_canvas.line(0.75 * inch, 0.75 * inch, page_width - 0.75 * inch, 0.75 * inch)

    # Text
    page_num = pdf_canvas.getPageNumber()
    pdf_canvas.drawString(0.75 * inch, 0.5 * inch, f"Page {page_num}")
    pdf_canvas.drawRightString(page_width - 0.75 * inch, 0.5 * inch, "http://epistemiq.pythonanywhere.com/")

    pdf_canvas.restoreState()

def draw_paragraph(pdf_canvas, xml_content, style, y_pos, page_width):
    """Draws text with page break handling."""
    left_margin = 0.75 * inch
    right_margin = 0.75 * inch
    available_width = page_width - left_margin - right_margin

    try:
        para = Paragraph(xml_content, style)
        w, h = para.wrapOn(pdf_canvas, available_width, 0)

        # Page Break
        if y_pos - h < 1.0 * inch:
            draw_page_footer(pdf_canvas, page_width)
            pdf_canvas.showPage()
            y_pos = A4[1] - 1.0 * inch

        para.drawOn(pdf_canvas, left_margin, y_pos - h)
        return y_pos - h - style.spaceAfter

    except Exception as e:
        logging.warning(f"Paragraph draw error: {e}")
        # Fallback
        raw = re.sub(r'<[^>]+>', '', xml_content)
        pdf_canvas.setFont("Helvetica", 9)
        pdf_canvas.drawString(left_margin, y_pos, raw[:80])
        return y_pos - 12

def parse_markdown_table(markdown_text):
    """
    Robust markdown table parser. Handles tables with/without outer pipes
    and cleans up whitespace/bold markers.
    """
    lines = [line.strip() for line in markdown_text.split('\n') if line.strip()]
    if len(lines) < 2: return None

    # 1. Filter only lines that look like table rows (contain pipes)
    table_lines = [l for l in lines if '|' in l]
    if len(table_lines) < 2: return None

    # 2. Determine separator line (usually ---|---)
    # It's usually the second line
    separator_idx = -1
    for i, line in enumerate(table_lines[:3]): # Check first 3 lines
        # Check if line consists mostly of - | : and whitespace
        if re.match(r'^[\s:|\|\-]+$', line) and '|' in line and '-' in line:
            separator_idx = i
            break

    if separator_idx == -1: return None # No valid table structure found

    # 3. Extract Headers and Data
    # Determine if we need to strip outer pipes based on the separator line
    has_outer_pipes = table_lines[separator_idx].strip().startswith('|') and table_lines[separator_idx].strip().endswith('|')

    def split_row(row_str):
        # Remove markdown bold/italic
        row_str = row_str.replace('**', '').replace('*', '')

        if has_outer_pipes:
            # Remove first and last char if they are pipes
            if row_str.startswith('|'): row_str = row_str[1:]
            if row_str.endswith('|'): row_str = row_str[:-1]

        return [cell.strip() for cell in row_str.split('|')]

    # Get data lines (excluding separator)
    data_rows = []

    # Header is usually the line before separator
    header_row = split_row(table_lines[separator_idx - 1])
    data_rows.append(header_row)

    # Body lines are after separator
    for line in table_lines[separator_idx + 1:]:
        if not line.strip(): continue
        row = split_row(line)
        # Only add if column count matches roughly (allow variance of 1)
        if abs(len(row) - len(header_row)) <= 1:
            data_rows.append(row)

    return data_rows if len(data_rows) > 1 else None

def create_table_from_data(table_data, available_width):
    if not table_data: return None
    cell_style = ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=8, leading=10, wordWrap="CJK")
    header_style = ParagraphStyle(name="CellHeader", parent=cell_style, fontName="Helvetica-Bold")

    max_cols = max(len(r) for r in table_data)
    norm_rows = []
    for r_i, row in enumerate(table_data):
        row = list(row) + [""] * (max_cols - len(row))
        cells = []
        for c_i, cell in enumerate(row):
            txt = clean_xml(cell)
            p = Paragraph(txt, header_style if r_i == 0 else cell_style)
            cells.append(p)
        norm_rows.append(cells)

    col_w = available_width / max_cols
    table = Table(norm_rows, colWidths=[col_w]*max_cols, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('padding', (0,0), (-1,-1), 4)
    ]))
    return table

def process_full_report(report_text):
    report_text = normalize_text(report_text)
    lines = report_text.split('\n')
    blocks = []
    table_buffer = []
    in_table = False

    for line in lines:
        line = line.strip()
        if not line: continue

        if line.startswith('|'):
            table_buffer.append(line)
            in_table = True
            continue
        elif in_table:
            blocks.append(('TABLE', '\n'.join(table_buffer)))
            table_buffer = []
            in_table = False

        if line.startswith('#'):
            header_text = line.lstrip('#').strip()
            xml = parse_markdown_to_reportlab(header_text)
            blocks.append(('ReportHeader', xml))
            continue

        if line.startswith('* ') or line.startswith('- '):
            list_text = line[2:].strip()
            xml = f"<bullet>&bull;</bullet> {parse_markdown_to_reportlab(list_text)}"
            blocks.append(('ProList', xml))
            continue

        xml = parse_markdown_to_reportlab(line)
        blocks.append(('ProBody', xml))

    if table_buffer:
        blocks.append(('TABLE', '\n'.join(table_buffer)))
    return blocks

# API Endpoints

@app.route("/api/test", methods=["GET"])
def api_test():
    return jsonify({"message": "Hello from Dockerized PostgreSQL backend!"})


@app.route("/")
def home_redirect():
    return redirect(url_for('analyze_page'))

@app.route("/analyze")
def analyze_page():
    prefill_claim = request.args.get("claim", "")
    return render_template("index.html", prefill_claim=prefill_claim)

@app.route('/share-target', methods=['POST'])
def share_target():
    shared_text = request.form.get('text', '')
    shared_title = request.form.get('title', '')
    shared_url = request.form.get('url', '')
    prefill_content = ""

    if shared_text:
        prefill_content = shared_text
    if shared_url and not (shared_url in shared_text or "http" in shared_text or "www" in shared_text):
        prefill_content += f"\n\n(Shared from: {shared_url})"
    elif shared_title and not shared_url:
        prefill_content = f"{shared_title}\n\n{shared_text}"
    elif shared_url:
        prefill_content = f"Shared URL: {shared_url}"
        if shared_title:
            prefill_content = f"{shared_title}\n\n{prefill_content}"
    elif shared_title:
        prefill_content = shared_title

    return render_template('index.html', prefill_claim=prefill_content)

@app.route("/api/fetch-article", methods=["POST"])
def fetch_article():
    """
    Python replacement for the Vercel/Node.js article fetcher.
    Uses BeautifulSoup instead of Cheerio.
    """
    data = request.json or {}
    url = data.get("url")

    if not url:
        return jsonify({"error": "Missing URL"}), 400

    try:
        # 1. Fetch raw HTML
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; EpistemiqBot/1.0; +http://epistemiq.pythonanywhere.com/)",
            "Accept": "text/html,application/xhtml+xml"
        }
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            return jsonify({
                "error": f"Failed to fetch URL: {response.reason}",
                "status": response.status_code
            }), 500

        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")

        # 2. Clean HTML (Remove junk tags)
        for tag in soup(["script", "style", "noscript", "iframe", "svg", "button", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        # Also remove common ad/junk classes (Basic attempt)
        for junk in soup.find_all(attrs={"class": re.compile(r"(ad|ads|social-share|menu|sidebar)", re.I)}):
            junk.decompose()

        # 3. Extract Text
        # Strategy: Prefer <article> tag, else <body>
        article_tag = soup.find("article")
        if article_tag:
            clean_text = article_tag.get_text(separator="\n", strip=True)
        else:
            clean_text = soup.body.get_text(separator="\n", strip=True) if soup.body else soup.get_text(separator="\n", strip=True)

        # 4. RUN VALIDATION (Ported from your JS)
        is_valid, reason = is_valid_content(clean_text)

        if not is_valid:
            return jsonify({
                "error": f"Content Rejected: {reason}. Please try pasting the text manually."
            }), 400

        return jsonify({"content": clean_text})

    except Exception as e:
        logging.error(f"fetch-article error: {e}")
        return jsonify({"error": "Internal server error during fetch."}), 500

def is_valid_content(text):
    """
    Helper to validate text quality.
    Returns (bool, reason_string)
    """
    if not text or len(text) < 300:
        return False, "Text is too short (under 300 chars)"

    # A. Check for Code/CSS Garbage
    # Count braces { } and semicolons ;
    code_chars = len(re.findall(r'[{};]', text))
    if code_chars > 15 and (code_chars / len(text)) > 0.02:
        return False, "Source appears to be raw code or CSS"

    # B. Check for Menu/Landing Page Density
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if not lines:
        return False, "Empty content"

    # Count "Short Lines" (typical of menus, footers, link lists)
    short_line_count = sum(1 for line in lines if len(line) < 60)
    ratio_short = short_line_count / len(lines)

    # If >80% of the lines are short snippets, it's likely a homepage/nav menu
    if len(lines) > 10 and ratio_short > 0.8:
        return False, "Source appears to be a navigation menu or landing page"

    return True, ""

# ==============================================================
#  ELEVENLABS AUDIO GENERATION (TTS)
# ==============================================================

@app.route("/api/speak-verdict", methods=["POST"])
def speak_verdict():
    """
    Converts the analysis summary into speech using ElevenLabs.
    """
    data = request.json or {}
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "ElevenLabs API Key missing"}), 500

    # Voice ID: "Brian" (Standard English Male)
    VOICE_ID = "IKne3meq5aSn9XLyUdCD"

    # ✅ FIX: Remove '/stream' from URL, use query param instead
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?stream=true"


    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }

    payload = {
        "text": text,
        # ✅ FIX: Use Flash v2.5 (Faster, Cheaper, Stable)
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)

        # Log error body if status is not 200
        if response.status_code != 200:
            logging.error(f"ElevenLabs Error {response.status_code}: {response.text}")
            return jsonify({"error": f"Provider Error: {response.text}"}), response.status_code

        return Response(
            response.iter_content(chunk_size=1024),
            mimetype="audio/mpeg"
        )
    except Exception as e:
        logging.error(f"ElevenLabs TTS failed: {e}")
        return jsonify({"error": str(e)}), 500

# ==============================================================
#  UNIFIED API ENDPOINTS
# ==============================================================

@app.route("/api/analyze", methods=["POST"])
@require_user
def analyze(user):
    """
    Unified Extraction: Extracts Claims AND Keywords.
    Uses 'call_hybrid_flow' (Gemma-2-27b Native) for generation.
    Uses Snapshot Rehydration logic for maximum UI compatibility on cache hit.
    """
    data = request.json or {}
    text = data.get("text")
    mode = data.get("mode") or "General Analysis of Testable Claims"
    source_url = data.get("source_url")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    canonical = canonicalize_text(text)
    txt_hash = text_hash(text)
    user_id = user["user_id"]
    current_source_type = "url" if source_url else "pasted_text"

    # 1. Generate Google Embedding (768 dim)
    try:
        vector_embedding = get_google_embedding(canonical)
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        vector_embedding = None

    save_pasted_text_to_db(text)
    if source_url:
        save_article_to_cache_db(source_url, text)

    conn = get_conn()
    analysis_id = None
    claims_payload = []

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            # 2. CACHE LOOKUP (Semantic + Exact)
            if vector_embedding:
                try:
                    c.execute("""
                        SELECT analysis_id, (embedding <=> %s::vector) as distance
                        FROM analyses WHERE mode=%s ORDER BY distance ASC LIMIT 1
                    """, (str(vector_embedding), mode))
                    match = c.fetchone()
                    if match and match['distance'] < 0.16:
                        analysis_id = match['analysis_id']
                        logging.info(f"✅ Semantic Cache Hit! Distance: {match['distance']}")
                except Exception as e:
                    logging.warning(f"Vector search failed: {e}")
                    conn.rollback()

            if not analysis_id:
                c.execute("SELECT analysis_id FROM analyses WHERE text_hash=%s AND mode=%s", (txt_hash, mode))
                row = c.fetchone()
                if row:
                    analysis_id = row['analysis_id']

            # 3. REHYDRATION (Snapshot Style - Restores Full Frontend State)
            if analysis_id:
                c.execute("UPDATE analyses SET last_accessed=CURRENT_TIMESTAMP WHERE analysis_id=%s", (analysis_id,))

                # Get basic claim rows
                c.execute("""
                    SELECT ordinal, claim_text, final_verdict, synthesis_summary, category, tags
                    FROM claims
                    WHERE analysis_id = %s
                    ORDER BY ordinal
                """, (analysis_id,))
                claim_rows = c.fetchall()

                if claim_rows:
                    for row in claim_rows:
                        claim_text = row["claim_text"]
                        # Logic matches original snapshot: hash the stripped/lower text
                        ch = sha256_str(claim_text.strip().lower())

                        # Fetch Model Cache (Internal Analysis)
                        c.execute("SELECT verdict, questions_json, used_model FROM model_cache WHERE claim_hash=%s", (ch,))
                        m_row = c.fetchone()

                        # Fetch External Cache (Scientific Verification)
                        c.execute("SELECT verdict, sources_json, used_model FROM external_cache WHERE claim_hash=%s", (ch,))
                        e_row = c.fetchone()

                        claims_payload.append({
                            "ordinal": row["ordinal"],
                            "claim_text": claim_text,
                            "claim_hash": ch,
                            "final_verdict": row["final_verdict"],
                            "synthesis_summary": row["synthesis_summary"],
                            "category": row["category"],
                            "tags": json_loads(row["tags"], []) if row["tags"] else [],
                            # Replicating the exact keys the UI expects
                            "model_verdict": m_row["verdict"] if m_row else None,
                            "used_model": m_row["used_model"] if m_row else None,
                            "questions": json_loads(m_row["questions_json"], []) if (m_row and m_row['questions_json']) else [],
                            "external_verdict": e_row["verdict"] if e_row else None,
                            "external_sources": json_loads(e_row["sources_json"], []) if (e_row and e_row['sources_json']) else []
                        })

                    conn.commit()
                    return jsonify({
                        "claims": claims_payload,
                        "analysis_id": analysis_id,
                        "cached": True,
                        "mode": mode
                    })

            # 4. IF NO CACHE: INITIALIZE NEW RECORD
            analysis_id = new_analysis_id()
            c.execute("""
                INSERT INTO analyses (analysis_id, user_id, text_hash, canonical_text, mode, source_type, source_meta, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (analysis_id, user_id, txt_hash, canonical, mode, current_source_type, source_url, str(vector_embedding) if vector_embedding else None))

            if user_id:
                c.execute("INSERT INTO user_analyses (user_id, analysis_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", (user_id, analysis_id))

            conn.commit()

    finally:
        conn.close()

    # 5. QUOTA & EXTRACTION
    allowed, used, limit = check_and_increment_quota(user_id, "analysis")
    if not allowed:
        return jsonify({"error": f"Daily limit reached ({used}/{limit})."}), 429

    template = UNIFIED_EXTRACTION_TEMPLATES.get(mode, UNIFIED_EXTRACTION_TEMPLATES["General Analysis of Testable Claims"])
    prompt = template.format(text=text[:15000])

    try:
        # ✅ NEW: Use Hybrid Flow (Task: Extraction = Gemma Native First)
        res, _ = call_hybrid_flow(prompt, json_mode=False, task_type="extraction")

        raw_content = res.json()["choices"][0]["message"]["content"]

        extracted_data = []
        for line in raw_content.splitlines():
            line = line.strip()
            if not line or not line[0].isdigit(): continue
            match = re.match(r'^\d+\.\s*(.*?)\s*\|\s*(?:Keywords:?)?\s*(.*)', line, re.IGNORECASE)
            if match:
                # Basic cleanup
                c_txt = match.group(1).strip()
                k_txt = match.group(2)
                k_list = [k.strip() for k in re.split(r'[,;]', k_txt) if k.strip()]
                extracted_data.append({"claim": c_txt, "keywords": k_list})

        if extracted_data:
            save_claims_for_analysis(analysis_id, extracted_data)

        # Return clean structure for frontend (fresh analysis has no verdicts yet)
        return jsonify({
            "claims": [{"claim_text": i["claim"], "final_verdict": None, "questions": [], "tags": [], "category": "General"} for i in extracted_data],
            "analysis_id": analysis_id,
            "cached": False
        })
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/verify-claim", methods=["POST"])
@require_user
def verify_claim_unified(user):
    data = request.json or {}
    analysis_id = data.get("analysis_id")
    ordinal = data.get("claim_idx")
    use_papers = data.get("use_papers", True)
    preferred_model = data.get("preferred_model")
    user_id = user["user_id"]

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            c.execute("""
                SELECT c.claim_text, c.search_keywords, a.canonical_text, c.final_verdict, a.mode
                FROM claims c
                JOIN analyses a ON c.analysis_id = a.analysis_id
                WHERE c.analysis_id=%s AND c.ordinal=%s
            """, (analysis_id, int(ordinal)))
            row = c.fetchone()

            if not row: return jsonify({"error": "Not found"}), 404

            claim_text = row['claim_text']
            keywords = json_loads(row['search_keywords'], [])
            full_text = row['canonical_text'] or ""
            analysis_mode = row['mode'] or "General Analysis of Testable Claims"

            if not keywords:
                keywords = [claim_text[:50]]
    finally:
        conn.close()

    # --- 1. QUOTA CHECK ---
    allowed, used, limit = check_and_increment_quota(user_id, "verification")
    if not allowed:
        return jsonify({"error": f"Daily verification limit reached ({used}/{limit})."}), 429

    # --- 2. Run Verification ---
    paper_abstracts = "No external papers requested."
    unique_sources = []

    if use_papers:
        all_sources = []

        # ✅ 1. Aggressive Fetching (Wider Net)
        # We fetch 10 from major sources to give the Reranker enough candidates
        try: all_sources.extend(fetch_semantic_scholar(keywords, max_results=20))
        except: pass

        try: all_sources.extend(fetch_core(keywords, max_results=20))
        except: pass

        try: all_sources.extend(fetch_pubmed(keywords)) # Returns ~5 default
        except: pass

        try: all_sources.extend(fetch_crossref(keywords)) # Returns ~3 default
        except: pass

        # ✅ 2. Deduplication by Fuzzy Title
        seen_titles = set()
        deduped_sources = []
        for s in all_sources:
            raw_title = s.get('title', '')
            clean_title = re.sub(r'[^a-z0-9]', '', raw_title.lower())

            if len(clean_title) > 10 and clean_title not in seen_titles:
                deduped_sources.append(s)
                seen_titles.add(clean_title)

        # ✅ 3. Semantic Reranking (Fact Verification)
        if deduped_sources:
            logging.info(f"--- RERANKING {len(deduped_sources)} PAPERS VIA EMBEDDINGS ---")
            try:
                # Uses the helper function defined earlier in the file
                ranked_sources = rank_papers_with_google(claim_text, deduped_sources)

                # If ranking worked, use it. Otherwise fall back to raw list.
                if ranked_sources:
                    unique_sources = ranked_sources[:10] # Top 10 most relevant
                else:
                    unique_sources = deduped_sources[:10]
            except Exception as e:
                logging.error(f"Ranking failed, using deduplicated list: {e}")
                unique_sources = deduped_sources[:10]

            paper_abstracts = "\n\n".join([
                f"SOURCE [{i+1}]: {s.get('title', 'Unknown')}\n"
                f"YEAR: {s.get('year', 'N/A')}\n"
                f"ABSTRACT: {(s.get('abstract') or 'No abstract')[:800]}"
                for i, s in enumerate(unique_sources)
            ])
        else:
            paper_abstracts = "Search performed. No relevant papers found."

    short_context = full_text[:4000].replace("{","(").replace("}",")")
    specific_instructions = VERIFICATION_MODES.get(analysis_mode, VERIFICATION_MODES["General Analysis of Testable Claims"])

    prompt = UNIFIED_VERIFICATION_PROMPT.format(
        short_context=short_context,
        claim_text=claim_text,
        paper_abstracts=paper_abstracts,
        mode_specific_instructions=specific_instructions
    )

    try:
        # ✅ CALL HYBRID FLOW
        res, used_model = call_hybrid_flow(
            prompt,
            json_mode=True,
            preferred_model=preferred_model,
            task_type="verification",
            timeout=120
        )

        raw_json = res.json()["choices"][0]["message"]["content"]
        result = json.loads(raw_json)

        compass = result.get("compass", {})
        evidence = result.get("evidence", {})

        conn = get_conn()
        try:
            with conn.cursor() as c:
                c.execute("""
                    UPDATE claims
                    SET final_verdict=%s, synthesis_summary=%s, category=%s, tags=%s
                    WHERE analysis_id=%s AND ordinal=%s
                """, (compass.get("verdict_label"), compass.get("summary"), compass.get("category"), json.dumps(compass.get("tags", [])), analysis_id, int(ordinal)))

                ch = sha256_str(claim_text.strip().lower())
                questions_list = evidence.get("questions", [])

                c.execute("""
                    INSERT INTO model_cache (claim_hash, verdict, questions_json, used_model, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT(claim_hash) DO UPDATE SET verdict=EXCLUDED.verdict, questions_json=EXCLUDED.questions_json, used_model=EXCLUDED.used_model
                """, (ch, evidence.get("model_verdict"), json.dumps(questions_list), used_model))

                # Save top 10 unique sources (Semantically Ranked)
                c.execute("""
                    INSERT INTO external_cache (claim_hash, verdict, sources_json, used_model, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT(claim_hash) DO UPDATE SET verdict=EXCLUDED.verdict, sources_json=EXCLUDED.sources_json, used_model=EXCLUDED.used_model
                """, (ch, evidence.get("external_verdict"), json.dumps(unique_sources), used_model))
            conn.commit()
        finally:
            conn.close()

        return jsonify({
            "compass": compass,
            "evidence": {
                "model_verdict": evidence.get("model_verdict"),
                "external_verdict": evidence.get("external_verdict"),
                "questions": questions_list,
                "sources": unique_sources,
                "used_model": used_model
            }
        })

    except Exception as e:
        logging.error(f"Unified Verification Failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/my-analyses", methods=["GET"])
@require_user
def my_analyses(user):
    """
    Return the last N DISTINCT analyses for the logged-in user.
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            c.execute(
                """
                SELECT
                    a.analysis_id,
                    a.mode,
                    a.source_type,
                    a.canonical_text,
                    a.created_at,
                    a.last_accessed
                FROM analyses a
                JOIN user_analyses ua ON ua.analysis_id = a.analysis_id
                WHERE ua.user_id = %s
                GROUP BY a.analysis_id, a.mode, a.source_type, a.canonical_text, a.created_at, a.last_accessed
                ORDER BY a.last_accessed DESC
                LIMIT 100
                """,
                (user["user_id"],),
            )
            rows = c.fetchall()
    finally:
        conn.close()

    def _ts(val):
        # Convert datetime objects to ISO strings if needed
        if hasattr(val, "isoformat"):
            return val.isoformat()
        return val

    def _title_from_row(r):
        txt = ""
        # Access by key thanks to DictCursor
        if "canonical_text" in r:
            txt = (r["canonical_text"] or "").strip()
        if not txt:
            return r["mode"] or "Untitled analysis"

        # Take first non-empty line
        first_line = txt.splitlines()[0].strip()
        if len(first_line) > 120:
            return first_line[:117] + "..."
        return first_line

    items = []
    for r in rows:
        items.append(
            {
                "analysis_id": r["analysis_id"],
                "mode": r["mode"],
                "source_type": r["source_type"],
                "title": _title_from_row(r),
                "created_at": _ts(r["created_at"]),
                "last_accessed": _ts(r["last_accessed"]),
            }
        )

    return jsonify(
        {
            "authenticated": True,
            "email": user["email"],
            "items": items,
        }
    )

@app.route("/api/analysis-snapshot", methods=["GET"])
def analysis_snapshot():
    """
    Returns data for the Tool View.
    Handles LIVE (UUID) and ARCHIVED (Integer) IDs.
    Allows public access IF the ID matches the Daily Spotlight.
    """
    analysis_id_str = request.args.get("analysis_id")
    if not analysis_id_str: return jsonify({"error": "Missing analysis_id"}), 400

    user = get_current_user()

    # 1. Determine Source: Archive vs Live
    is_archive = analysis_id_str.isdigit()

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:

            # --- PATH A: ARCHIVED ARTICLE (Integer ID) ---
            if is_archive:
                c.execute("SELECT title, published_at, content_snapshot FROM published_articles WHERE id = %s", (int(analysis_id_str),))
                row = c.fetchone()
                if not row: return jsonify({"error": "Archived analysis not found"}), 404

                # Map Archive JSON -> Tool View JSON
                snapshot = row['content_snapshot']
                mapped_claims = []
                for idx, item in enumerate(snapshot):
                    mapped_claims.append({
                        "ordinal": idx,
                        "claim_text": item.get('text'),
                        "final_verdict": item.get('verdict'),
                        "synthesis_summary": item.get('summary'),
                        "category": item.get('category'),
                        "tags": item.get('tags', []),
                        "model_verdict": item.get('internal_verdict'),
                        "used_model": item.get('internal_model', 'Archived'),
                        "external_verdict": item.get('external_verdict'),
                        "external_sources": item.get('sources', []),
                        "questions": [d['question'] for d in item.get('deep_dives', [])] if item.get('deep_dives') else []
                    })

                return jsonify({
                    "analysis_id": analysis_id_str,
                    "mode": "Archived Spotlight",
                    "title": row['title'],
                    "created_at": row['published_at'],
                    "claims": mapped_claims
                })

            # --- PATH B: LIVE ANALYSIS (UUID) ---

            # 1. Access Control
            is_authorized = False

            # Check 1: Is user owner?
            if user:
                c.execute("SELECT 1 FROM user_analyses WHERE user_id=%s AND analysis_id=%s", (user['user_id'], analysis_id_str))
                if c.fetchone(): is_authorized = True

            # Check 2: Is this the Daily Spotlight? (Public Access)
            if not is_authorized:
                # We calculate the spotlight ID again to check permission
                # (Optimization: You could cache this, but calculating it is fast enough)
                spotlight_id = get_todays_spotlight_id()
                if analysis_id_str == spotlight_id:
                    is_authorized = True

            if not is_authorized:
                 return jsonify({"error": "Forbidden: You do not have access to this analysis."}), 403

            # 2. Fetch Live Data
            c.execute("SELECT analysis_id, mode, canonical_text, created_at FROM analyses WHERE analysis_id = %s", (analysis_id_str,))
            meta = c.fetchone()
            if not meta: return jsonify({"error": "Analysis not found"}), 404

            c.execute("""
                SELECT ordinal, claim_text, final_verdict, synthesis_summary, category, tags
                FROM claims
                WHERE analysis_id = %s
                ORDER BY ordinal
            """, (analysis_id_str,))
            claim_rows = c.fetchall()

            claims_payload = []
            for row in claim_rows:
                ordinal = row["ordinal"]
                claim_text = row["claim_text"]
                ch = sha256_str(claim_text.strip().lower())

                c.execute("SELECT verdict, questions_json, used_model FROM model_cache WHERE claim_hash=%s", (ch,))
                model_row = c.fetchone()

                c.execute("SELECT verdict, sources_json, used_model FROM external_cache WHERE claim_hash=%s", (ch,))
                ext_row = c.fetchone()

                claims_payload.append({
                    "ordinal": ordinal,
                    "claim_text": claim_text,
                    "final_verdict": row["final_verdict"],
                    "synthesis_summary": row["synthesis_summary"],
                    "category": row["category"],
                    "tags": json_loads(row["tags"], []) if row["tags"] else [],
                    "model_verdict": model_row["verdict"] if model_row else None,
                    "used_model": model_row["used_model"] if model_row else None,
                    "questions": json_loads(model_row["questions_json"], []) if (model_row and model_row['questions_json']) else [],
                    "external_verdict": ext_row["verdict"] if ext_row else None,
                    "external_model": ext_row["used_model"] if ext_row else None,
                    "external_sources": json_loads(ext_row["sources_json"], []) if (ext_row and ext_row['sources_json']) else []
                })

            def _title_from_text(txt, fallback_mode):
                txt = (txt or "").strip()
                if not txt: return fallback_mode or "Untitled analysis"
                first_line = txt.splitlines()[0].strip()
                return first_line[:117] + "..." if len(first_line) > 120 else first_line

            return jsonify({
                "analysis_id": meta["analysis_id"],
                "mode": meta["mode"],
                "title": _title_from_text(meta["canonical_text"], meta["mode"]),
                "created_at": meta["created_at"],
                "claims": claims_payload,
            })

    finally:
        conn.close()


@app.route("/api/delete-analysis/<analysis_id>", methods=["DELETE"])
@require_user
def delete_analysis(user, analysis_id):
    """
    Delete an analysis belonging to the logged-in user.
    """
    user_id = user["user_id"]

    conn = get_conn()
    try:
        with conn.cursor() as c:
            # 1) Verify ownership
            c.execute(
                """
                SELECT 1 FROM user_analyses
                WHERE user_id = %s AND analysis_id = %s
                """,
                (user_id, analysis_id),
            )
            row = c.fetchone()
            if not row:
                return jsonify({"error": "Not found or not allowed"}), 404

            # 2) Delete from analyses (this cascades claims)
            c.execute(
                "DELETE FROM analyses WHERE analysis_id = %s",
                (analysis_id,),
            )

            # 3) Delete mapping entry
            c.execute(
                "DELETE FROM user_analyses WHERE analysis_id = %s",
                (analysis_id,),
            )
            conn.commit()

    finally:
        conn.close()

    return jsonify({"status": "deleted"})


@app.route("/api/core-proxy", methods=["POST"])
def core_proxy():
    """
    Internal Proxy for CORE API.
    Replaces the Vercel Serverless Function since Vultr/Docker allows outbound traffic.
    """
    data = request.json or {}
    query = data.get("query") or data.get("q") # Handle both formats just in case

    if not query:
        return jsonify({"error": "Missing query"}), 400

    api_key = os.getenv("CORE_API_KEY")
    if not api_key:
        return jsonify({"error": "Server misconfiguration: Missing CORE API Key"}), 500

    # CORE API Endpoint
    url = "https://api.core.ac.uk/v3/search/works"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "limit": 3
    }

    # Retry Logic (similar to what we did in JS)
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)

            # If success, return data
            if resp.status_code == 200:
                return jsonify(resp.json())

            # If server busy (429 or 5xx), wait and retry
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries:
                    time.sleep(1)
                    continue

            # If other error, return it
            return jsonify({"error": f"CORE API Error: {resp.text}"}), resp.status_code

        except Exception as e:
            logging.error(f"CORE Proxy failed (Attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(1)
                continue
            return jsonify({"error": str(e)}), 500


@app.route("/api/process-image", methods=["POST"])
def process_image():
    """Process uploaded image and extract text using OpenRouter Vision (Fallback to Tesseract)"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400

        # Save temporarily
        image_path = save_uploaded_file(image_file)
        if not image_path:
            return jsonify({"error": "Failed to save image"}), 500

        # Check Cache
        file_hash = compute_file_hash(image_path)
        cached_text = get_cached_media(file_hash)
        if cached_text:
            try: os.remove(image_path)
            except: pass
            return jsonify({"extracted_text": cached_text, "cached": True})

        # ✅ NEW: Use OpenRouter Vision (Better Rate Limits than Direct Google)
        extracted_text = analyze_image_with_openrouter(image_path)

        # Fallback to Tesseract
        if not extracted_text:
             logging.info("AI Vision failed/empty, falling back to Tesseract")
             extracted_text = analyze_image_with_ocr(image_path)

        # Store in cache
        if extracted_text:
            store_media_cache(file_hash, 'image', extracted_text)

        # Cleanup
        try: os.remove(image_path)
        except: pass

        if not extracted_text:
            return jsonify({"error": "Could not extract text. Image might be unclear."}), 400

        return jsonify({"extracted_text": extracted_text, "cached": False})

    except Exception as e:
        logging.error(f"Error in process_image endpoint: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500



MAX_VIDEO_SIZE_MB = 50

@app.route("/api/extract-audio", methods=["POST"])
def extract_audio():
    if 'file' not in request.files:
        return "No file", 400

    file = request.files['file']
    input_path = f"/tmp/input_{uuid.uuid4().hex}"
    output_path = f"/tmp/output_{uuid.uuid4().hex}.wav"
    file.save(input_path)

    # ✅ Force 16 kHz mono WAV with SOXR
    cmd = [
        "ffmpeg",
        "-y",                     # overwrite
        "-i", input_path,         # input video
        "-vn",                    # strip video
        "-ac", "1",               # mono
        "-ar", "16000",           # 16kHz request
        "-af", "aresample=resampler=soxr",  # ✅ force high-quality resample
        "-c:a", "pcm_s16le",      # signed 16-bit PCM
        "-f", "wav",              # WAV file
        output_path
    ]

    try:
        print("FFMPEG CMD:", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # ✅ Verify sample rate
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                output_path
            ],
            capture_output=True, text=True
        )

        sample_rate = probe.stdout.strip()
        print("✔️ Extracted audio sample rate:", sample_rate)

        if sample_rate != "16000":
            print("❌ Unexpected sample rate, forcing failure")
            return f"Error: output audio is {sample_rate} Hz, expected 16000 Hz", 500

        # ✅ Return raw WAV bytes to frontend
        with open(output_path, "rb") as f:
            wav_data = f.read()

        return Response(
            wav_data,
            mimetype="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"}
        )

    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:", e.stderr)
        return f"FFmpeg error: {e.stderr}", 500
    except Exception as e:
        print("Server error:", str(e))
        return f"Error: {str(e)}", 500
    finally:
        # ✅ Always clean temp files
        for p in [input_path, output_path]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except:
                pass

@app.route("/api/transcribe-url", methods=["POST"])
def transcribe_url_route():
    data = request.json or {}
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Configure yt-dlp to extract metadata only (don't download video)
        ydl_opts = {
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
        }

        text_content = ""

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # 1. Check for Subtitles (Manual or Auto)
            subtitles = info.get('subtitles', {})
            auto_subs = info.get('automatic_captions', {})

            # Prioritize manual English subs, then auto English
            sub_url = None

            # Helper to find english in a list of langs
            def find_en(subs_dict):
                for lang in ['en', 'en-US', 'en-orig', 'en-GB']:
                    if lang in subs_dict:
                        # return the JSON format if available, else VTT/SRT
                        for fmt in subs_dict[lang]:
                            if fmt['ext'] in ['vtt', 'srv3', 'json3']:
                                return fmt['url']
                return None

            sub_url = find_en(subtitles)
            if not sub_url:
                sub_url = find_en(auto_subs)

            if sub_url:
                # 2. Fetch the subtitle file content
                res = requests.get(sub_url)
                res.raise_for_status()
                raw_text = res.text

                # 3. Clean VTT/XML tags to get pure text
                # (Simple regex to strip timestamps and tags)
                # Remove header
                raw_text = re.sub(r'WEBVTT.*', '', raw_text)
                # Remove timestamps like 00:00:00.000 --> ...
                raw_text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*', '', raw_text)
                # Remove tags like <c.colorE5E5E5>
                raw_text = re.sub(r'<[^>]+>', '', raw_text)
                # Remove timestamps like 00:00:01
                raw_text = re.sub(r'\d{2}:\d{2}:\d{2}', '', raw_text)
                # Remove empty lines and extra spaces
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

                # Deduplicate repeating lines (common in auto-caps)
                unique_lines = []
                prev = ""
                for line in lines:
                    if line != prev:
                        unique_lines.append(line)
                        prev = line

                text_content = " ".join(unique_lines)

            else:
                # Fallback: If it's a site without exposed captions (like TikTok/IG sometimes),
                # You would ideally download audio here and send to Whisper.
                # For now, we return a specific error.
                return jsonify({
                    "error": "No captions found for this video. Please download it and upload the file instead."
                }), 404

        return jsonify({"text": text_content})

    except Exception as e:
        logging.error(f"Video URL fetch failed: {e}")
        return jsonify({"error": f"Could not process video URL: {str(e)}"}), 500

@app.route("/api/generate-report", methods=["GET", "POST"])
def generate_report():
    # 1. Auth Check (Manual because it's a stream)
    user = get_current_user()
    if not user: return Response(json.dumps({"error": "Unauthorized"}), mimetype="application/json", status=401)

    # Setup
    if request.method == "GET":
        claim_idx = request.args.get("claim_idx", type=int)
        question_idx = request.args.get("question_idx", type=int)
        analysis_id = request.args.get("analysis_id")
        preferred_model = request.args.get("preferred_model")
    else:
        data = request.get_json(silent=True) or {}
        claim_idx = data.get("claim_idx")
        question_idx = data.get("question_idx")
        analysis_id = data.get("analysis_id")
        preferred_model = data.get("preferred_model")

    if analysis_id is None or claim_idx is None or question_idx is None:
        return Response(json.dumps({"error": "Missing inputs"}), mimetype="application/json", status=400)

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            # Fetch Context
            c.execute("""
                SELECT c.claim_text, c.final_verdict, c.synthesis_summary, a.canonical_text
                FROM claims c
                JOIN analyses a ON c.analysis_id = a.analysis_id
                WHERE c.analysis_id=%s AND c.ordinal=%s
            """, (analysis_id, int(claim_idx)))
            row = c.fetchone()
            if not row: return Response(json.dumps({"error": "Claim not found"}), mimetype="application/json", status=404)

            claim_text = row["claim_text"]
            full_context = row["canonical_text"]
            compass_verdict = row["final_verdict"] or "Pending"
            claim_hash = sha256_str(claim_text.strip().lower())

            # Fetch Questions
            c.execute("SELECT questions_json FROM model_cache WHERE claim_hash=%s", (claim_hash,))
            questions_row = c.fetchone()
            if not questions_row: return Response(json.dumps({"error": "Questions not found."}), mimetype="application/json", status=400)

            questions = json_loads(questions_row["questions_json"], [])
            if question_idx >= len(questions): return Response(json.dumps({"error": "Invalid Question Index"}), mimetype="application/json", status=400)

            question_text = questions[question_idx]
            rq_hash = sha256_str(claim_text.strip().lower() + "||" + question_text.strip().lower())

            # --- 2. CACHE CHECK (Free) ---
            c.execute("SELECT report_text, used_model FROM report_cache WHERE rq_hash=%s", (rq_hash,))
            hit = c.fetchone()
            if hit and hit['report_text']:
                def stream_cached():
                    yield f"data: {json.dumps({'content': hit['report_text'], 'used_model': hit['used_model'] or 'Unknown Model'})}\n\n"
                    yield "data: [DONE]\n\n"
                return Response(stream_cached(), mimetype="text/event-stream")

            # Fetch Verdicts for prompt
            c.execute("SELECT verdict FROM model_cache WHERE claim_hash=%s", (claim_hash,))
            row_m = c.fetchone()
            model_verdict_content = row_m['verdict'] if row_m else "N/A"
            c.execute("SELECT verdict FROM external_cache WHERE claim_hash=%s", (claim_hash,))
            row_e = c.fetchone()
            external_verdict_content = row_e['verdict'] if row_e else "N/A"

    finally:
        conn.close()

    # --- 3. QUOTA CHECK (Before Stream Starts) ---
    allowed, used, limit = check_and_increment_quota(user["user_id"], "report")
    if not allowed:
        return Response(
            json.dumps({"error": f"Daily report limit reached ({used}/{limit})."}),
            mimetype="application/json",
            status=429
        )

    # --- 4. Run Generation ---
    short_context = full_context[:25000].replace("{","(").replace("}",")")
    prompt = f'''
You are an AI assistant producing a structured research report (3000 words max).

**CRITICAL FORMATTING REQUIREMENTS:**
- Use ONLY plain text with basic formatting
- NO HTML tags of any kind
- Use simple dash "-" for ranges
- Use simple quotes "'"
- Use * bullets; headings with "##"; keep it readable.
- **IMPORTANT:** Put a blank line before every header.

**Structure:**
## 1. Introduction
## 2. Analysis
## 3. Conclusion
## 4. Sources

**INPUT DATA:**
- **Context:** "{short_context}..."
- **Claim:** {claim_text}
- **Target Research Question:** {question_text}

**PREVIOUS FINDINGS:**
- **Internal Analysis:** {model_verdict_content}
- **External Analysis:** {external_verdict_content}
- **FINAL EDITORIAL VERDICT:** "{compass_verdict}"

**INSTRUCTIONS:**
1. Answer the Research Question in depth.
2. Use the **Final Editorial Verdict** as the guiding conclusion.
3. Compare the Internal vs External analysis to explain nuances.
4. Use the "Context" to understand specific entities.
5. Cite any external sources used.

Generate the research report now.
'''

    def stream_response():
        full_report = ""
        used_model_name = "Unknown Model"
        try:
            # ✅ CALL HYBRID FLOW
            # task_type="report" -> Respects user preference -> Falls back to Google Native
            res, actual_model = call_hybrid_flow(
                prompt,
                stream=True,
                json_mode=False,
                preferred_model=preferred_model,
                task_type="report",
                timeout=290
            )
            used_model_name = actual_model

            # Use iter_content to handle both Google (SSE wrapped) and OpenRouter (Raw)
            for chunk in res.iter_content(chunk_size=1024, decode_unicode=True):
                if not chunk: continue
                for line in chunk.split("\n"):
                    line = line.strip()
                    if not line.startswith("data:"): continue
                    data_part = line[5:].strip()
                    if data_part == "[DONE]": continue
                    try:
                        json_data = json.loads(data_part)
                        if "choices" in json_data and json_data["choices"]:
                            content = json_data["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                normalized = normalize_text_for_display(content)
                                full_report += normalized
                                yield f"data: {json.dumps({'content': normalized, 'used_model': actual_model})}\n\n"
                    except: continue
        except Exception as e:
            logging.error(f"Report Gen Error: {e}")
            yield f"data: {json.dumps({'error': 'Report generation failed.'})}\n\n"

        if full_report.strip():
            conn = get_conn()
            try:
                with conn.cursor() as c:
                    c.execute("""
                        INSERT INTO report_cache (rq_hash, question_text, report_text, used_model, updated_at)
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (rq_hash) DO UPDATE SET report_text=EXCLUDED.report_text, used_model=EXCLUDED.used_model, updated_at=CURRENT_TIMESTAMP
                    """, (rq_hash, question_text, full_report, used_model_name))
                conn.commit()
            except: pass
            finally: conn.close()
        yield "data: [DONE]\n\n"

    # 1. Create the response object
    response = Response(stream_response(), mimetype="text/event-stream")

    # 2. Add the header
    response.headers["X-Accel-Buffering"] = "no"

    # 3. Return it
    return response

@app.route("/api/available-reports", methods=["GET"])
def get_available_reports():
    analysis_id = request.args.get("analysis_id")
    if not analysis_id:
        try: analysis_id = request.get_json(silent=True).get("analysis_id")
        except: pass

    if not analysis_id: return jsonify({"error": "Missing analysis_id"}), 400

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            c.execute(
                "SELECT ordinal, claim_text FROM claims WHERE analysis_id=%s ORDER BY ordinal",
                (analysis_id,)
            )
            claim_rows = c.fetchall()

            available_reports = []

            for row in claim_rows:
                ordinal = row['ordinal']
                claim_text = row['claim_text']
                preview = claim_text[:80] + "..." if len(claim_text) > 80 else claim_text
                ch = sha256_str(claim_text.strip().lower())

                c.execute("SELECT verdict, questions_json FROM model_cache WHERE claim_hash=%s", (ch,))
                model_row = c.fetchone()

                if not model_row: continue

                verdict = model_row['verdict']
                questions = json_loads(model_row['questions_json'], [])

                if verdict:
                    available_reports.append({
                        "id": f"claim-{ordinal}-summary",
                        "type": f"Claim {ordinal + 1} - Model Verdict & External Verification",
                        "description": f"Model analysis and external sources for: {preview}"
                    })

                for q_idx, question in enumerate(questions):
                    rq_hash = sha256_str(claim_text.strip().lower() + "||" + question.strip().lower())
                    c.execute("SELECT report_text FROM report_cache WHERE rq_hash=%s", (rq_hash,))
                    if c.fetchone():
                        available_reports.append({
                            "id": f"claim-{ordinal}-question-{q_idx}",
                            "type": f"Claim {ordinal + 1} - Question Report {q_idx + 1}",
                            "description": f"Research report for: {question[:100]}..."
                        })
    finally:
        conn.close()

    return jsonify(available_reports)


@app.route("/api/daily-spotlight", methods=["GET"])
def daily_spotlight():
    target_id = get_todays_spotlight_id()
    if not target_id:
        return jsonify({"error": "No spotlight found"}), 404

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            c.execute("""
                SELECT a.analysis_id, a.source_meta, a.created_at, c.claim_text
                FROM analyses a
                JOIN claims c ON a.analysis_id = c.analysis_id
                WHERE a.analysis_id = %s AND c.ordinal = 0
            """, (target_id,))
            row = c.fetchone()

            if not row: return jsonify({"error": "Spotlight data missing"}), 404

            # Determine source label
            source_url = row["source_meta"]
            domain = "External Source"
            if source_url and "//" in source_url:
                try:
                    domain = source_url.split('/')[2]
                except: pass

            return jsonify({
                "analysis_id": row["analysis_id"],
                "title": row["claim_text"][:100] + "...",
                "source_domain": domain,
                "date": row["created_at"]
            })
    finally:
        conn.close()

# ==============================================================
#  THE COMPASS (PUBLIC MEDIA OUTLET)
# ==============================================================

@app.route('/compass')
def compass_feed():
    search_query = request.args.get('q', '').strip()
    category_filter = request.args.get('cat', '').strip()

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            # 1. Get Categories for Filter
            # (We perform a quick aggregation on the JSONB column)
            c.execute("SELECT DISTINCT jsonb_array_elements_text(categories) FROM published_articles")
            categories = sorted([row[0] for row in c.fetchall()])

            # 2. Build Query
            sql = "SELECT id, title, slug, summary, image_url, published_at, categories, tags, content_snapshot FROM published_articles WHERE 1=1"
            params = []

            if category_filter:
                # JSONB contains check
                sql += " AND categories @> %s"
                params.append(json.dumps([category_filter]))

            if search_query:
                sql += " AND (title ILIKE %s OR summary ILIKE %s)"
                params.extend([f"%{search_query}%", f"%{search_query}%"])

            sql += " ORDER BY published_at DESC"

            c.execute(sql, tuple(params))
            rows = c.fetchall()

            articles = []
            for r in rows:
                snapshot = r['content_snapshot']

                # ✅ FIX: Handle empty snapshot or None verdict safely
                if snapshot and len(snapshot) > 0:
                    # Get verdict, default to "Pending" if None/Null
                    raw_verdict = snapshot[0].get('verdict')
                    first_verdict = raw_verdict if raw_verdict else "Pending"
                else:
                    first_verdict = "Pending"

                # Safety check for categories
                cat_display = "Analysis"
                if r['categories'] and len(r['categories']) > 0:
                    cat_display = r['categories'][0]

                articles.append({
                    "title": r['title'],
                    "slug": r['slug'],
                    "date": r['published_at'].strftime("%b %d, %Y"),
                    "image": r['image_url'],
                    "verdict": first_verdict, # ✅ Now guaranteed to be a string
                    "category": cat_display,
                    "summary": r['summary'],
                    "tags": r['tags'][:3] if r['tags'] else []
                })

            return render_template('compass_feed.html',
                                   articles=articles,
                                   categories=categories,
                                   active_cat=category_filter,
                                   active_search=search_query)
    finally:
        conn.close()

# Helper to clean text for HTML display (Basic Markdown to HTML)
def simple_markdown(text):
    if not text: return ""
    text = text.replace("\n", "<br>")
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'## (.*?)(<br>|$)', r'<h5 class="mt-3 fw-bold">\1</h5>', text)
    return text

@app.route('/compass/<slug>')
def compass_article(slug):
    user = get_current_user()
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            # 1. Fetch Article from Archive
            c.execute("""
                SELECT id, title, published_at, summary, image_url, content_snapshot
                FROM published_articles
                WHERE slug = %s
            """, (slug,))
            row = c.fetchone()

            if not row: return "Article not found", 404

            article_id = row['id']
            content_data = row['content_snapshot'] # This is already the JSON list we need!

            # Helper to process text (Markdown -> HTML) inside the JSON data
            # We do this at render time to keep the DB clean
            final_content = []
            for item in content_data:
                # We recreate the object to process HTML fields
                # Note: deep_dives loop is handled in template usually, but we ensure structure here
                processed_item = item.copy()
                processed_item['internal_verdict'] = render_report_html(item.get('internal_verdict'))
                processed_item['external_verdict'] = render_report_html(item.get('external_verdict'))

                # Process reports if they exist
                if 'deep_dives' in item:
                    for dd in item['deep_dives']:
                        dd['content'] = render_report_html(dd['content'])

                final_content.append(processed_item)

            # 2. Fetch Comments (Now linked to article_id)
            c.execute("""
                SELECT c.content, c.created_at, u.email
                FROM comments c
                JOIN users u ON c.user_id = u.id
                WHERE c.article_id = %s
                ORDER BY c.created_at DESC
            """, (article_id,))
            comments = c.fetchall()

            clean_comments = []
            for cm in comments:
                email_parts = cm['email'].split('@')
                name = email_parts[0]
                clean_comments.append({
                    "user": name[:3] + "***" if len(name) > 3 else "User",
                    "date": cm['created_at'].strftime("%b %d, %Y"),
                    "content": cm['content']
                })

            return render_template('compass_article.html',
                                   title=row['title'],
                                   date=row['published_at'].strftime("%B %d, %Y"),
                                   summary=render_report_html(row['summary']),
                                   image=row['image_url'],
                                   content=final_content,
                                   comments=clean_comments,
                                   user=user,
                                   article_id=article_id) # Pass ID for comments
    finally:
        conn.close()


@app.route('/api/post-comment', methods=['POST'])
@require_user
def post_comment(user):
    data = request.json or {}
    article_id = data.get('article_id') # Changed from analysis_id
    content = data.get('content', '').strip()

    if not article_id or not content:
        return jsonify({"error": "Missing data"}), 400

    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO comments (article_id, user_id, content, created_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            """, (article_id, user['user_id'], content))
        conn.commit()
        return jsonify({"status": "success", "user": user['email'].split('@')[0][:3]+"***"})
    finally:
        conn.close()


@app.route("/api/export-pdf", methods=["POST", "OPTIONS"])
@cross_origin(origins=["http://epistemiq.pythonanywhere.com/"], supports_credentials=True, methods=["POST", "OPTIONS"], allow_headers=["Content-Type"])
def export_pdf():
    payload = request.json or {}
    selected_reports = payload.get("selected_reports", [])
    analysis_id = payload.get("analysis_id")

    if not analysis_id: return "Missing analysis_id", 400

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            # 1. Fetch Metadata
            c.execute("SELECT canonical_text, source_type, source_meta, created_at, mode FROM analyses WHERE analysis_id=%s", (analysis_id,))
            analysis_meta = c.fetchone()
            if not analysis_meta: return "Analysis not found", 404

            # 2. Fetch Claims + Compass Data
            c.execute("""
                SELECT ordinal, claim_text, final_verdict, synthesis_summary, category
                FROM claims
                WHERE analysis_id=%s
                ORDER BY ordinal
            """, (analysis_id,))
            claim_rows = c.fetchall()

            claims_data = {}
            for row in claim_rows:
                ordinal = row['ordinal']
                text = row['claim_text']
                ch = sha256_str(text.strip().lower())

                # Fetch Evidence Caches
                c.execute("SELECT verdict, sources_json, used_model FROM external_cache WHERE claim_hash=%s", (ch,))
                ev = c.fetchone()
                c.execute("SELECT verdict, used_model, questions_json FROM model_cache WHERE claim_hash=%s", (ch,))
                mv = c.fetchone()

                q_list = json_loads(mv['questions_json'], []) if mv else []

                claims_data[ordinal] = {
                    "text": text,
                    # Compass Data
                    "final_verdict": row['final_verdict'],
                    "synthesis_summary": row['synthesis_summary'],
                    "category": row['category'],
                    # Evidence Data
                    "model_verdict": mv['verdict'] if mv else "",
                    "used_model": mv['used_model'] if mv and mv['used_model'] else "Unknown Model",
                    "external_verdict": ev['verdict'] if ev else "",
                    "external_model": ev['used_model'] if ev and ev['used_model'] else "Unknown Model",
                    "sources": json_loads(ev['sources_json'], []) if ev else [],
                    "questions": q_list,
                    "reports": {}
                }

                # Fetch Deep Dive Reports
                for idx, q_text in enumerate(q_list):
                    rq_hash = sha256_str(text.strip().lower() + "||" + q_text.strip().lower())
                    c.execute("SELECT report_text, used_model FROM report_cache WHERE rq_hash=%s", (rq_hash,))
                    rr = c.fetchone()
                    if rr:
                        claims_data[ordinal]["reports"][idx] = {
                            "text": rr['report_text'],
                            "model": rr['used_model'] or "Unknown Model"
                        }
    finally:
        conn.close()

    if not claims_data: return "No claims found.", 400

    # 3. Setup PDF
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    p.setTitle("Epistemiq Report")
    width, height = A4
    styles = get_pro_styles()

    # --- HEADER LOGO ---
    header_x = 0.75 * inch
    logo_url = "https://raw.githubusercontent.com/akebonin/G20techsprint2025/refs/heads/main/logo.png"
    try:
        res = requests.get(logo_url, timeout=5)
        if res.status_code == 200:
            img_data = io.BytesIO(res.content)
            logo_img = ImageReader(img_data)
            p.drawImage(logo_img, 0.75*inch, height-1.25*inch, width=50, height=50, mask='auto', preserveAspectRatio=True)
            header_x += 60
    except: pass

    p.setFont("Helvetica-Bold", 22)
    p.setFillColor(colors.black)
    p.drawString(header_x, height - 0.8*inch, "Epistemiq Analysis Report")
    p.setFont("Helvetica", 10)
    p.setFillColor(colors.darkgrey)
    p.drawString(header_x, height - 0.8*inch - 18, "Your Compass in the Epistemic Fog")
    p.drawRightString(width-0.75*inch, height-0.8*inch, "epistemiq.pythonanywhere.com")
    p.drawRightString(width-0.75*inch, height-0.8*inch - 14, "epistemiq.ai@gmail.com")
    p.setStrokeColor(colors.lightgrey)
    p.setLineWidth(1)
    p.line(0.75*inch, height-1.4*inch, width-0.75*inch, height-1.4*inch)

    # Metadata Box
    y = height - 1.8 * inch
    source_val = analysis_meta["source_meta"] if analysis_meta["source_meta"] else "Pasted Text"
    source_label = "URL/Source" if analysis_meta["source_type"] == "url" else "Input Method"

    display_source_val = source_val[:57] + "..." if len(source_val) > 60 else source_val
    date_val = str(analysis_meta["created_at"]).split('.')[0]

    p.setFillColor(colors.aliceblue)
    p.setStrokeColor(colors.lightgrey)
    p.roundRect(0.75*inch, y - 55, width - 1.5*inch, 50, 6, fill=1, stroke=1)

    p.setFont("Helvetica-Bold", 9)
    p.setFillColor(colors.black)
    p.drawString(0.9*inch, y - 20, "Analysis Metadata")
    p.setFont("Helvetica", 9)
    p.setFillColor(colors.darkgrey)
    p.drawString(0.9*inch, y - 35, f"Date: {date_val}")
    p.drawString(0.9*inch, y - 48, f"Mode: {analysis_meta['mode']}")

    meta_link_style = ParagraphStyle(name='MetaLink', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=11, textColor=colors.darkgrey)
    safe_url = source_val.replace('&', '&amp;').replace('"', '&quot;') if source_val else ""
    xml_content = f'{source_label}: <link href="{safe_url}" color="blue"><u>{clean_xml(display_source_val)}</u></link>' if (analysis_meta["source_type"] == "url" and source_val) else f'{source_label}: {clean_xml(display_source_val)}'

    p_obj = Paragraph(xml_content, meta_link_style)
    p_obj.wrapOn(p, width - 4.0 * inch, 50)
    p_obj.drawOn(p, 3.0 * inch, y - 37)
    y -= 80

    # Content Loop
    claims_to_print = sorted(list(claims_data.keys()))
    for claim_idx in claims_to_print:
        claim_data = claims_data[claim_idx]
        has_summary = f"claim-{claim_idx}-summary" in selected_reports
        selected_q_indices = []
        for i in range(len(claim_data["questions"])):
            if f"claim-{claim_idx}-question-{i}" in selected_reports:
                selected_q_indices.append(i)

        if not has_summary and not selected_q_indices: continue

        if y < 1.5 * inch:
            draw_page_footer(p, width)
            p.showPage()
            y = height - 1.0 * inch

        # Clean Claim Text (Chained correctly)
        raw_claim_text = normalize_text_for_pdf(claim_data['text'])
        raw_claim_text = raw_claim_text.replace("â-", "-").replace("â", "")

        y = draw_paragraph(p, f"Claim {claim_idx + 1}: {raw_claim_text}", styles['ClaimHeading'], y, width)

        # --- COMPASS SYNTHESIS BOX ---
        if has_summary and claim_data['final_verdict']:
            avail_width = width - 1.5*inch

            # Category
            cat_text = f"<b>{claim_data['category'] or 'ANALYSIS'}</b>"
            p_cat = Paragraph(cat_text, ParagraphStyle('Cat', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor("#6c757d")))
            w_c, h_c = p_cat.wrap(avail_width, 0)

            # Verdict (Color Logic)
            v_str = str(claim_data['final_verdict']).lower()
            if any(x in v_str for x in ['true', 'verified', 'supported', 'feasible', 'accepted']):
                v_color = "#198754" # Green
            elif any(x in v_str for x in ['debunked', 'nonsense', 'false', 'scam', 'unlikely']):
                v_color = "#dc3545" # Red
            else:
                v_color = "#fd7e14" # Orange

            verdict_html = f"<font color='{v_color}' size='12'><b>{claim_data['final_verdict']}</b></font>"
            p_verdict = Paragraph(verdict_html, styles['Normal'])
            w_v, h_v = p_verdict.wrap(avail_width, 0)

            # Summary (Cleaned & Chained)
            raw_summary = normalize_text_for_pdf(claim_data['synthesis_summary'])
            raw_summary = raw_summary.replace("â-", "-").replace("â", "")

            summary_html = f"<i>{clean_xml(raw_summary)}</i>"
            p_summary = Paragraph(summary_html, styles['ProBody'])
            w_s, h_s = p_summary.wrap(avail_width, 0)

            # 2. Dynamic Line
            total_box_height = h_c + h_v + h_s + 35

            # Page Break Check
            if y - total_box_height < 1.0 * inch:
                draw_page_footer(p, width)
                p.showPage()
                y = height - 1.0 * inch
                y = draw_paragraph(p, f"Claim {claim_idx + 1} (Cont.):", styles['ClaimHeading'], y, width)

            p.saveState()
            p.setStrokeColor(colors.HexColor("#0d6efd"))
            p.setLineWidth(2)
            p.line(0.75*inch, y, 0.75*inch, y - total_box_height)
            p.restoreState()

            # Draw
            p_cat.drawOn(p, 0.85*inch, y - h_c)
            y -= (h_c + 6)
            p_verdict.drawOn(p, 0.85*inch, y - h_v)
            y -= (h_v + 8)
            p_summary.drawOn(p, 0.85*inch, y - h_s)
            y -= (h_s + 20)

        if has_summary:
            # Internal
            if claim_data['model_verdict']:
                clean_model = claim_data['used_model'].replace(":free", "").replace("/", " / ")
                y = draw_paragraph(p, f"<b>Internal Analysis (AI):</b> <font color='#666666' size='8'>{clean_model}</font>", styles['ProBody'], y, width)

                raw_mv = normalize_text_for_pdf(claim_data['model_verdict'])
                raw_mv = raw_mv.replace("â-", "-").replace("â", "")

                clean_mv = clean_verdict_text(raw_mv)
                y = draw_paragraph(p, clean_mv, styles['ProBody'], y, width)

            # External
            if claim_data['external_verdict']:
                clean_ext_model = claim_data['external_model'].replace(":free", "").replace("/", " / ")
                y = draw_paragraph(p, f"<b>External Verification (Papers):</b> <font color='#666666' size='8'>{clean_ext_model}</font>", styles['ProBody'], y, width)

                raw_ev = normalize_text_for_pdf(claim_data['external_verdict'])
                raw_ev = raw_ev.replace("â-", "-").replace("â", "")

                clean_ev = clean_verdict_text(raw_ev)
                y = draw_paragraph(p, clean_ev, styles['ProBody'], y, width)

            # Sources
            if claim_data['sources']:
                y = draw_paragraph(p, "<b>Scientific Sources Found:</b>", styles['SectionHeading'], y, width)
                for src in claim_data['sources']:
                    title = normalize_text_for_pdf(clean_xml(src.get("title", "Source Link")))
                    title = title.replace("â-", "-").replace("â", "")

                    url = src.get("url", "")
                    tag = f"<b>[{clean_xml(src.get('source', 'Source'))}]</b>"
                    if len(title) > 100: title = title[:97] + "..."
                    if url:
                        safe_url = url.replace('&', '&amp;').replace('"', '&quot;')
                        link_html = f'{tag} <link href="{safe_url}"><u><font color="blue">{title}</font></u></link>'
                    else:
                        link_html = f"{tag} {title}"
                    y = draw_paragraph(p, link_html, styles['ProLink'], y, width)
            y -= 10

        # Deep Dive Reports
        for q_idx in selected_q_indices:
            q_text = normalize_text_for_pdf(claim_data["questions"][q_idx])
            q_text = q_text.replace("â-", "-").replace("â", "")

            report_data = claim_data["reports"].get(q_idx)
            if not report_data: continue

            y = draw_paragraph(p, f"<b>Deep Dive: {clean_xml(q_text)}</b>", styles['SectionHeading'], y, width)

            # Report Model
            clean_rep_model = report_data['model'].replace(':free','').replace('/',' / ')
            y = draw_paragraph(p, f"<b>Report Model:</b> <font color='#666666' size='8'>{clean_rep_model}</font>", styles['ProBody'], y, width)

            # Report Body
            clean_body = normalize_text_for_pdf(report_data['text'])
            clean_body = clean_body.replace("â-", "-").replace("â", "")

            blocks = process_full_report(clean_body)

            for block_type, content in blocks:
                if block_type == 'TABLE':
                    tdata = parse_markdown_table(content)
                    if tdata:
                        tbl = create_table_from_data(tdata, width - 1.5 * inch)
                        if tbl:
                            w, h = tbl.wrapOn(p, width - 1.5 * inch, y)
                            if y - h < 1.0 * inch:
                                draw_page_footer(p, width)
                                p.showPage()
                                y = height - 1.0 * inch
                            try:
                                tbl.drawOn(p, 0.75 * inch, y - h)
                                y -= h + 12
                            except: pass
                else:
                    style = styles.get(block_type, styles['ProBody'])
                    y = draw_paragraph(p, content, style, y, width)
            y -= 10

        y -= 5
        p.setStrokeColor(colors.lightgrey)
        p.setLineWidth(0.5)
        p.line(0.75*inch, y, width-0.75*inch, y)
        y -= 15

    # Appendix: Source Doc
    p.showPage()
    y = height - 1.0 * inch
    draw_page_footer(p, width)
    y = draw_paragraph(p, "Appendix: Source Document", styles['ClaimHeading'], y, width)

    if analysis_meta["source_type"] == "url" and source_val:
        safe_url = source_val.replace('&', '&amp;').replace('"', '&quot;')
        source_html = f'<b>Source:</b> <link href="{safe_url}" color="#0d6efd"><u>{safe_url}</u></link>'
        y = draw_paragraph(p, source_html, meta_link_style, y, width)
    else:
        y = draw_paragraph(p, f"<b>Source:</b> {clean_xml(source_val)}", meta_link_style, y, width)

    y -= 15
    full_text = analysis_meta["canonical_text"] or "No text content available."
    if len(full_text) > 60000: full_text = full_text[:60000] + "\n\n[Text truncated...]"

    source_style = ParagraphStyle(name='SourceText', parent=styles['ProBody'], fontSize=8, leading=10, textColor=colors.HexColor("#333333"), spaceAfter=6)

    clean_source = normalize_text_for_pdf(full_text)
    clean_source = clean_source.replace("â-", "-").replace("â", "")

    for raw_para in clean_xml(clean_source).split('\n'):
        if not raw_para.strip(): continue
        para_obj = Paragraph(raw_para, source_style)
        w, h = para_obj.wrapOn(p, width - 1.5*inch, 0)
        if y - h < 1.0*inch:
            draw_page_footer(p, width)
            p.showPage()
            y = height - 1.0*inch
        para_obj.drawOn(p, 0.75*inch, y - h)
        y -= (h + 6)

    p.save()
    buffer.seek(0)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name="Epistemiq_AI_Report.pdf")


@app.route("/api/global-stats", methods=["GET"])
def global_stats():
    """
    Returns GLOBAL KPI metrics: Total Analyses, Claims, and Top Model across the entire platform.
    Public endpoint.
    """
    conn = get_conn()
    try:
        with conn.cursor() as c:
            # 1. Total Analyses (All sessions, anonymous + logged in)
            c.execute("SELECT COUNT(*) FROM analyses")
            total_analyses = c.fetchone()[0]

            # 2. Total Claims (All claims ever extracted)
            c.execute("SELECT COUNT(*) FROM claims")
            total_claims = c.fetchone()[0]

            # 3. Top Used Model (Global popularity contest)
            c.execute("""
                SELECT used_model, COUNT(used_model) as usage_count
                FROM model_cache
                WHERE used_model IS NOT NULL AND used_model != 'Unknown Model'
                GROUP BY used_model
                ORDER BY usage_count DESC
                LIMIT 1
            """)

            row = c.fetchone()
            top_model = row[0] if row else "N/A"

            # Clean up model name for display
            if top_model != "N/A":
                # Removes ":free" and "vendor/" (e.g. "google/gemini" -> "gemini")
                top_model = top_model.replace(":free", "").split("/")[-1]

            return jsonify({
                "total_analyses": total_analyses,
                "total_claims": total_claims,
                "favorite_model": top_model
            })

    except Exception as e:
        logging.error(f"Global stats error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/cleanup-cache", methods=["POST"])
@require_user
def api_cleanup_cache(user):
    # TODO: later you can add a simple check for admin emails here
    try:
        cleanup_old_cache()
        return jsonify({"status": "ok"})
    except Exception as e:
        logging.error(f"Cleanup error: {e}")
        return jsonify({"error": "Cleanup failed"}), 500


# ==============================================================
# SAFE INITIALIZATION (Prevents Deadlocks in Docker)
# ==============================================================

def safe_init_db():
    """
    Ensures that only ONE Gunicorn worker initializes the database.
    It uses a file lock (/tmp/epistemiq_db_init.lock).
    The first worker gets the lock and runs init_db().
    The others wait until it is finished.
    """
    lock_file = "/tmp/epistemiq_db_init.lock"

    try:
        # Open the lock file (create if doesn't exist)
        with open(lock_file, "w") as f:
            try:
                # Attempt to acquire an EXCLUSIVE, NON-BLOCKING lock
                # If another process has it, this raises IOError immediately
                fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

                print("🔒 Acquired lock. Initializing database schema...")
                init_db()
                print("✅ Database initialization complete.")

                # Keep the lock held for a split second to ensure commits finish
                time.sleep(0.5)

            except IOError:
                # If we get here, another worker has the lock.
                print("⏳ Database is locked by another worker. Waiting for it to finish...")
                # Wait for the lock to be released (blocking wait)
                fcntl.lockf(f, fcntl.LOCK_EX)
                print("🔓 Lock released. Database is ready. Proceeding.")

    except Exception as e:
        print(f"⚠️ Initialization Warning: {e}")

# ==============================================================
# MAIN ENTRY POINT
# ==============================================================

if __name__ != "__main__":
    # This block runs when Gunicorn loads the file (Docker)
    safe_init_db()

elif __name__ == "__main__":
    # This block runs when you run 'python backend.py' manually
    init_db()
    app.run(debug=True, port=8080)
