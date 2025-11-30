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
import sqlite3
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



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv(dotenv_path="/home/epistemiq/mysite/.env")


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    app.secret_key = os.urandom(24)
    logging.warning("FLASK_SECRET_KEY not set. Using a random key for development.")

app.register_blueprint(auth_bp)

CORS(app, supports_credentials=True, origins=["https://epistemiq.vercel.app"])


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
#  Concurrency-safe DB helpers
# ==============================================================

DB_PATH = "/home/epistemiq/mysite/sessions.db"

def get_conn():
    """Open SQLite connection with WAL mode and busy timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=10, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row

    # Hardening for concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=10000;")  # 10 seconds

    return conn


def with_retry_db(fn):
    """Retry decorator to handle 'database is locked' errors."""
    def wrapper(*args, **kwargs):
        attempts = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempts < 5:
                    attempts += 1
                    time.sleep(0.2 * attempts)
                else:
                    raise
    return wrapper


# ==============================================================
#  Initialization: NEW AUTH TABLES + Extended analyses table
# ==============================================================

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # -------------------------------
    # USERS TABLE (passwordless login)
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    """)

    # -------------------------------
    # MAGIC LINKS (email tokens)
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS magic_links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        used_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # -------------------------------
    # SESSIONS (persistent login)
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_token TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        user_agent TEXT,
        ip_hash TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")

    # -------------------------------
    # ANALYSES (EXTENDED)
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        analysis_id TEXT PRIMARY KEY,
        user_id INTEGER,
        text_hash TEXT,
        canonical_text TEXT,
        mode TEXT,
        source_type TEXT,
        source_meta TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    # ✅ NEW: Prevent Duplicate Analyses (Race Condition Fix)
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_analyses_hash_mode ON analyses(text_hash, mode)")

    # -------------------------------
    # USER ↔ ANALYSES mapping
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS user_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        analysis_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id)
    )
    """)
    # ✅ NEW: Prevent Duplicate User Links
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_user_analyses_unique ON user_analyses(user_id, analysis_id)")

    # -------------------------------
    # PASTED TEXT CACHE
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS pasted_texts (
        text_hash TEXT PRIMARY KEY,
        text_content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # -------------------------------
    # ARTICLE CACHE
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS article_cache (
        url_hash TEXT PRIMARY KEY,
        url TEXT,
        raw_html TEXT,
        article_text TEXT,
        etag TEXT,
        last_modified TEXT,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_article_cache_url ON article_cache(url)")

    # -------------------------------
    # MEDIA CACHE
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS media_cache (
        file_hash TEXT PRIMARY KEY,
        media_type TEXT NOT NULL,
        extracted_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # -------------------------------
    # CLAIMS TABLE
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        claim_id TEXT PRIMARY KEY,
        analysis_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        claim_text TEXT NOT NULL,
        claim_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
    )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_claims_analysis ON claims(analysis_id, ordinal)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_claims_hash ON claims(claim_hash)")

    # -------------------------------
    # MODEL CACHE
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_cache (
        claim_hash TEXT PRIMARY KEY,
        verdict TEXT,
        questions_json TEXT,
        keywords_json TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_model_cache_hash ON model_cache(claim_hash)")

    # -------------------------------
    # EXTERNAL CACHE
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS external_cache (
        claim_hash TEXT PRIMARY KEY,
        verdict TEXT,
        sources_json TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_external_cache_hash ON external_cache(claim_hash)")

    # -------------------------------
    # REPORT CACHE
    # -------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS report_cache (
        rq_hash TEXT PRIMARY KEY,
        question_text TEXT,
        report_text TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_report_cache_rq ON report_cache(rq_hash)")

    conn.commit()
    conn.close()


# ==============================================================
#           CACHE + CLAIM HELPERS (UPDATED)
# ==============================================================

def sha256_str(s: str):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def canonicalize_text(text: str) -> str:
    """
    Normalize text so that trivial differences (newlines, spacing)
    do not create different hashes. This keeps raw vs Chrome-cleaned
    versions aligned when the content is effectively the same.
    """
    if not text:
        return ""

    # Normalize line endings
    txt = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse all runs of whitespace (spaces, tabs, newlines) to single spaces
    txt = " ".join(txt.split())

    # Trim edges
    txt = txt.strip()

    return txt


def text_hash(text: str) -> str:
    """
    Hash for full-input texts (pasted, OCR, article, transcription).
    Always use canonicalized text so Chrome AI preprocessing doesn't
    cause duplicate analyses.
    """
    canon = canonicalize_text(text)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()

@with_retry_db
def save_claims_for_analysis(analysis_id: str, claims_list: list):
    conn = get_conn()
    c = conn.cursor()

    c.execute("DELETE FROM claims WHERE analysis_id=?", (analysis_id,))

    for idx, claim_text in enumerate(claims_list):
        claim_text_clean = claim_text.strip()
        claim_hash = sha256_str(claim_text_clean.lower())
        claim_id = sha256_str(f"{analysis_id}|{idx}|{claim_text_clean}")

        c.execute("""
        INSERT OR REPLACE INTO claims (claim_id, analysis_id, ordinal, claim_text, claim_hash)
        VALUES (?, ?, ?, ?, ?)
        """, (claim_id, analysis_id, idx, claim_text_clean, claim_hash))

    conn.commit()
    conn.close()


def get_claims_for_analysis(analysis_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT claim_text FROM claims WHERE analysis_id=? ORDER BY ordinal", (analysis_id,))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]


# ==============================================================
#               MEDIA CACHE HELPERS
# ==============================================================

def compute_file_hash(file_path):
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_cached_media(file_hash):
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT extracted_text FROM media_cache WHERE file_hash = ?', (file_hash,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None


@with_retry_db
def store_media_cache(file_hash, media_type, extracted_text):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT OR REPLACE INTO media_cache (file_hash, media_type, extracted_text)
    VALUES (?, ?, ?)
    """, (file_hash, media_type, extracted_text))
    conn.commit()
    conn.close()

# ==============================================================
#                       CLEANUP
# ==============================================================

@with_retry_db
def cleanup_old_cache():
    """Clean up old cache entries to prevent database bloat."""
    conn = get_conn()
    c = conn.cursor()

    try:
        # 30-day media
        c.execute('DELETE FROM media_cache WHERE created_at < ?', (datetime.now() - timedelta(days=30),))
        media_deleted = c.rowcount

        # 7-day analyses
        c.execute('DELETE FROM analyses WHERE last_accessed < ?', (datetime.now() - timedelta(days=7),))
        analyses_deleted = c.rowcount

        # pasted texts
        c.execute('DELETE FROM pasted_texts WHERE created_at < ?', (datetime.now() - timedelta(days=30),))
        texts_deleted = c.rowcount

        # article cache
        c.execute('DELETE FROM article_cache WHERE fetched_at < ?', (datetime.now() - timedelta(days=30),))
        articles_deleted = c.rowcount

        # model cache
        c.execute('DELETE FROM model_cache WHERE updated_at < ?', (datetime.now() - timedelta(days=30),))
        model_deleted = c.rowcount

        # external cache
        c.execute('DELETE FROM external_cache WHERE updated_at < ?', (datetime.now() - timedelta(days=30),))
        external_deleted = c.rowcount

        # report cache
        c.execute('DELETE FROM report_cache WHERE updated_at < ?', (datetime.now() - timedelta(days=30),))
        report_deleted = c.rowcount

        conn.commit()
        logging.info(
            f"Cache cleanup: {media_deleted} media, {analyses_deleted} analyses, "
            f"{texts_deleted} texts, {articles_deleted} articles, "
            f"{model_deleted} model, {external_deleted} external, {report_deleted} reports removed"
        )

        if (media_deleted + analyses_deleted + texts_deleted +
                articles_deleted + model_deleted + external_deleted +
                report_deleted) > 50:
            c.execute('VACUUM')
            logging.info("Database vacuum performed")

    except Exception as e:
        logging.error(f"Cleanup error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


# ==============================================================
#                    INIT ON STARTUP
# ==============================================================

init_db()


# API Configuration
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Base prompt templates
BASE_EXTRACTION_RULES = '''
**Strict rules:**
- ONLY include claims that appear EXPLICITLY in the text.
- Each claim must be explicitly stated.
- If no explicit, complete, testable claims exist, output exactly: "No explicit claims found."
- Absolutely DO NOT infer, paraphrase, generalize, or introduce external knowledge.
- NEVER include incomplete sentences, headings, summaries, conclusions, speculations, questions, or introductory remarks.
- Output ONLY the claims formatted as a numbered list, or "No explicit claims found."
'''

BASE_JSON_STRUCTURE = '''
Output a structured text response with the following format. Do NOT use code fences (```), JSON, or extra text outside this structure. Use exact labels and colons.

Verdict: VERIFIED
Justification: Concise explanation of up to 300 words.
Sources: None
Keywords: term1, term2, term3, term4, term5

STRICT RULES:
- Verdict: Exactly one of VERIFIED, PARTIALLY_SUPPORTED, INCONCLUSIVE, CONTRADICTED, SUPPORTED, NOT_SUPPORTED, FEASIBLE, POSSIBLE_BUT_UNPROVEN, UNLIKELY, NONSENSE
- Justification: String, max 300 words
- Sources: 0-2 valid URLs, comma-separated, or "None" if none
- Keywords: 3-5 scientific/technical terms, comma-separated, each 3-20 characters
- Output ONLY the structured text, nothing else
'''

# Prompt templates
extraction_templates = {
    "General Analysis of Testable Claims": f'''
You will be given a text. Extract a **numbered list** of explicit, scientifically testable claims.

{BASE_EXTRACTION_RULES}

TEXT:

{{text}}

OUTPUT:
''',
    "Specific Focus on Scientific Claims": f'''
You will be given a text. Extract a **numbered list** of explicit, scientifically testable claims related to science.

{BASE_EXTRACTION_RULES}

TEXT:

{{text}}

OUTPUT:
''',
    "Technology-Focused Extraction": f'''
You will be given a text. Extract a **numbered list** of explicit, testable claims related to technology.

{BASE_EXTRACTION_RULES}

TEXT:

{{text}}

OUTPUT:
'''
}

# Updated verification prompts with STRICT LOGIC TABLE
verification_prompts = {
    "General Analysis of Testable Claims": f'''
You are a rigorous scientific fact-checker.

INPUT DATA:
1. **SOURCE TEXT TO ANALYZE** (The document containing the claims):
"""{{context}}"""

2. **CLAIM TO CHECK**:
"{{claim}}"

YOUR TASK:
Determine if the Claim is true in the real world, based on your internal training data and established scientific consensus.

VERDICT LOGIC TABLE (Follow strictly):
- If the claim is a known scientific fact -> VERIFIED
- If the claim is plausible but lacks proof -> POSSIBLE_BUT_UNPROVEN
- If the claim contradicts known science (e.g. "Earth is flat", "CERN opened portal") -> NONSENSE
- If the claim is from a fictional/viral story and not real science -> NONSENSE
- If the claim is nowhere to be found in real science -> NOT_SUPPORTED

CRITICAL RULES:
1. **Use the Source Text ONLY for definition.** If the claim says "The team", check the Source Text to know it refers to CERN.
2. **Do NOT treat the Source Text as evidence.** The Source Text is the material we are questioning.
3. **REALITY CHECK:** Does this event exist in the real world outside of this text? If not, the verdict is NOT_SUPPORTED or NONSENSE.

{BASE_JSON_STRUCTURE}
''',

    "Specific Focus on Scientific Claims": f'''
You are a rigorous scientific fact-checker.

INPUT DATA:
1. **SOURCE TEXT TO ANALYZE** (The document containing the claims):
"""{{context}}"""

2. **CLAIM TO CHECK**:
"{{claim}}"

YOUR TASK:
Determine if this scientific claim is true in the real world.

VERDICT LOGIC TABLE (Follow strictly):
- If the claim is a known scientific fact -> VERIFIED
- If the claim contradicts standard models (e.g. "CERN simulation became conscious") -> NONSENSE
- If the claim is a misinterpretation of real science -> UNLIKELY
- If the claim exists only in viral posts -> NOT_SUPPORTED

CRITICAL RULES:
1. **Context Usage:** Use the Source Text only to understand specific entities.
2. **No Hallucinated Support:** Do NOT verify the claim just because it appears in the Source Text.
3. **Consensus:** Judge validity against established physics and biology.

{BASE_JSON_STRUCTURE}
''',

    "Technology-Focused Extraction": f'''
You are a technology fact-checker.

INPUT DATA:
1. **SOURCE TEXT TO ANALYZE**:
"""{{context}}"""

2. **CLAIM TO CHECK**:
"{{claim}}"

YOUR TASK:
Evaluate the technical feasibility and truth of this claim based on real-world engineering standards.

VERDICT LOGIC TABLE:
- If technology exists and works -> VERIFIED
- If technology is theoretical -> FEASIBLE
- If technology is scientifically impossible -> NONSENSE
- If claim is a hoax -> NOT_SUPPORTED

CRITICAL RULES:
1. **Identify Entities:** Use the Source Text to define vague terms.
2. **Reality Check:** Do not blindly believe the Source Text.

{BASE_JSON_STRUCTURE}
'''
}

# Helper functions


def call_openrouter(prompt, stream=False, temperature=0.0, json_mode=False, model="x-ai/grok-4.1-fast:free"):
    """Calls the OpenRouter API, supports streaming, JSON mode, and dynamic models."""
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://epistemiq.vercel.app", # Recommended by OpenRouter
        "X-Title": "Epistemiq"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": temperature
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        # Reduced timeout to fail fast and try next model
        response = requests.post(OR_URL, headers=headers, json=payload, stream=stream, timeout=25)

        # Handle 429 specifically
        if response.status_code == 429:
            raise Exception("Rate Limit Hit")

        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            # Pass the status code up
            if e.response.status_code == 429:
                raise Exception("Rate Limit Hit")
            raise Exception(f"API Error {e.response.status_code}: {e.response.text}") from e
        raise Exception(f"Network error: {e}") from e

def generate_questions_for_claim(claim):
    """Generates up to 3 research questions for a claim."""
    prompt = f"For the following claim, propose up to 3 concise research questions. Only list the questions.\n\nClaim: {claim}"
    try:
        res = call_openrouter(prompt, temperature=0.5)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        questions = [q.strip("-•* ") for q in content.splitlines() if q.strip() and len(q.strip()) > 5]
        return questions[:3]
    except Exception as e:
        logging.error(f"Failed to generate questions for claim '{claim}': {e}")
        return []

def generate_model_verdict_and_questions(prompt, claim_text, preferred_model=None):
    """
    Generate verdict with robust fallback strategies.
    1. Rotates models if 429 (Rate Limit) occurs.
    2. Uses 'soft' regex parsing if strict parsing fails.
    3. Prioritizes preferred_model if provided.
    """

    # Priority list of free models
    default_models = [
        "x-ai/grok-4.1-fast:free",
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-2.0-flash-lite-preview-02-05:free",
        "openai/gpt-oss-20b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-7b-instruct:free"
    ]

    # Reorder based on preference
    models = list(default_models)
    # Ensure preferred_model is a valid string before inserting
    if preferred_model and isinstance(preferred_model, str) and preferred_model.strip():
        if preferred_model in models:
            models.remove(preferred_model)
        models.insert(0, preferred_model)

    model_verdict_content = "Could not generate model verdict."
    questions = []
    search_keywords = []

    # Valid verdict keywords for soft parsing
    VALID_VERDICTS = [
        "VERIFIED", "PARTIALLY_SUPPORTED", "INCONCLUSIVE", "CONTRADICTED",
        "SUPPORTED", "NOT_SUPPORTED", "FEASIBLE", "POSSIBLE_BUT_UNPROVEN",
        "UNLIKELY", "NONSENSE"
    ]

    for model in models:
        try:
            logging.info(f"Attempting verdict generation with model: {model}")

            res = call_openrouter(prompt, json_mode=False, temperature=0.0, model=model)

            try:
                raw_llm_response = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            except:
                raw_llm_response = ""

            raw_llm_response = normalize_text_for_display(raw_llm_response)

            if not raw_llm_response.strip():
                logging.warning(f"Model {model} returned empty response.")
                continue # Try next model

            # --- 1. STRICT PARSING ---
            verdict_match = re.search(
                r'Verdict:\s*(VERIFIED|PARTIALLY_SUPPORTED|INCONCLUSIVE|CONTRADICTED|SUPPORTED|NOT_SUPPORTED|FEASIBLE|POSSIBLE_BUT_UNPROVEN|UNLIKELY|NONSENSE)',
                raw_llm_response,
                re.IGNORECASE
            )

            # --- 2. SOFT PARSING (Fallback) ---
            verdict = None
            if verdict_match:
                verdict = verdict_match.group(1).upper()
            else:
                first_lines = raw_llm_response[:200].upper()
                for v in VALID_VERDICTS:
                    # Look for distinct word
                    if re.search(rf'\b{v}\b', first_lines):
                        verdict = v
                        raw_llm_response = f"Verdict: {v}\n{raw_llm_response}"
                        break

            if not verdict:
                logging.warning(f"Model {model} failed format check.")
                continue # Try next model

            # Parse Justification
            justification_match = re.search(r'Justification:\s*([\s\S]{20,1000}?(?=\n\s*(?:Sources|Keywords|$)))', raw_llm_response, re.IGNORECASE | re.DOTALL)
            if justification_match:
                justification = justification_match.group(1).strip()
            else:
                parts = raw_llm_response.split(verdict, 1)
                if len(parts) > 1:
                    raw_remains = parts[1].replace("Verdict:", "").strip()
                    justification = raw_remains.split("Sources:")[0].split("Keywords:")[0].strip()[:1000]
                else:
                    justification = "Justification parsed via fallback."

            # Parse Sources
            sources_match = re.search(r'Sources:\s*([\s\S]*?)(?=\n\s*(?:Keywords|$))', raw_llm_response, re.IGNORECASE | re.DOTALL)
            sources = []
            if sources_match:
                source_text = sources_match.group(1).strip()
                sources = re.findall(r'(https?://[^\s,)]+)', source_text)[:2] or ['None']

            # Parse Keywords
            keywords_match = re.search(r'Keywords:\s*([\w\s,-]{10,})', raw_llm_response, re.IGNORECASE | re.DOTALL)
            if keywords_match:
                kw_text = keywords_match.group(1).strip()
                search_keywords = [kw.strip().lower() for kw in re.split(r'[,;\s]+', kw_text) if len(kw.strip()) > 3][:5]
            else:
                words = re.findall(r'\b[a-zA-Z]{4,}\b', claim_text.lower())
                search_keywords = list(set(words[:5]))

            # SUCCESS
            model_verdict_content = f"Verdict: **{verdict}**\n\nJustification: {justification}"
            if sources and sources != ['None']:
                model_verdict_content += f"\n\nSources:\n" + "\n".join(f"- {src}" for src in sources)

            try:
                questions = generate_questions_for_claim(claim_text)
            except:
                questions = ["Could not generate research questions"]

            return model_verdict_content, questions, search_keywords

        except Exception as e:
            logging.error(f"Model {model} failed: {e}")
            continue # Loop to next model

    # If all models fail
    logging.error("All models failed to generate verdict.")
    words = re.findall(r'\b[a-zA-Z]{4,}\b', claim_text.lower())
    search_keywords = list(set(words[:5])) or [claim_text.lower()[:50]]

    return f"Error: Analysis timed out or failed. Please try again.", [], search_keywords

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

def save_uploaded_file(file, upload_folder="/home/epistemiq/mysite/uploads"):
    """Save uploaded file and return path"""
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
    pdf_canvas.drawRightString(page_width - 0.75 * inch, 0.5 * inch, "epistemiq.vercel.app")

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
    lines = [line.strip() for line in markdown_text.split('\n') if line.strip().startswith('|')]
    if len(lines) < 2: return None
    table_data = []
    for line in lines:
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        table_data.append(cells)
    if len(table_data) > 1 and all(set(c).issubset({'-', ':', ' '}) for c in table_data[1]):
        table_data.pop(1)
    return table_data if table_data else None

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
    return jsonify({"message": "Hello from PythonAnywhere backend!"})


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

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    API endpoint to extract claims.
    Includes RACE CONDITION FIX: Uses unique DB constraints and try/except
    to safely handle double-clicks/duplicate requests.
    """
    data = request.json or {}
    text = data.get("text")
    mode = data.get("mode") or "General Analysis of Testable Claims"
    source_url = data.get("source_url")

    if not text or not mode:
        return jsonify({"error": "Missing text or analysis mode."}), 400

    # 1) Canonicalize text + compute stable hash
    canonical = canonicalize_text(text)
    if not canonical:
        return jsonify({"error": "Text is empty after normalization."}), 400

    txt_hash = text_hash(text)

    # Logged-in user?
    user = get_current_user()
    user_id = user["user_id"] if user else None


    conn = get_conn()
    analysis_id = None
    cached = False

    try:
        c = conn.cursor()

        # --------------------------------------------------
        # 2) Try to Insert NEW Analysis (Optimistic Locking)
        # --------------------------------------------------
        new_id = new_analysis_id()

        try:
            c.execute(
                """
                INSERT INTO analyses (
                    analysis_id, user_id, text_hash, canonical_text,
                    mode, source_type, source_meta, created_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (new_id, user_id, txt_hash, canonical, mode, "pasted_text", None),
            )
            # Success - it's new
            analysis_id = new_id
            cached = False

        except sqlite3.IntegrityError:
            # --------------------------------------------------
            # 3) Duplicate Detected (Race Condition caught)
            # --------------------------------------------------
            # Reuse existing analysis logic
            c.execute(
                "SELECT analysis_id FROM analyses WHERE text_hash = ? AND mode = ?",
                (txt_hash, mode)
            )
            row = c.fetchone()
            if row:
                analysis_id = row["analysis_id"]
                cached = True
                # Bump timestamp
                c.execute("UPDATE analyses SET last_accessed=CURRENT_TIMESTAMP WHERE analysis_id=?", (analysis_id,))
            else:
                conn.rollback()
                return jsonify({"error": "Database integrity error."}), 500

        # --------------------------------------------------
        # 4) Link to User
        # --------------------------------------------------
        if user_id is not None:
            c.execute(
                """
                INSERT OR IGNORE INTO user_analyses (user_id, analysis_id)
                VALUES (?, ?)
                """,
                (user_id, analysis_id),
            )

        conn.commit()

        # --------------------------------------------------
        # 5) Return immediately if cached
        # --------------------------------------------------
        if cached:
            cached_claims = get_claims_for_analysis(analysis_id)
            if cached_claims:
                return jsonify({
                    "claims": cached_claims,
                    "analysis_id": analysis_id,
                    "cached": True
                })

    finally:
        conn.close()

    # ------------------------------------------------------
    # 6) Run Extraction (OpenRouter) - Only if NOT cached
    # ------------------------------------------------------
    template = extraction_templates.get(
        mode,
        extraction_templates["General Analysis of Testable Claims"]
    )
    extraction_prompt = template.format(text=text)

    try:
        res = call_openrouter(extraction_prompt)
        raw = res.json()["choices"][0]["message"]["content"]

        if "No explicit claims found" in raw or not raw.strip():
            return jsonify({
                "claims": [],
                "analysis_id": analysis_id,
                "cached": False
            })

        claims_list = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped: continue
            if stripped[0].isdigit():
                i = 0
                while i < len(stripped) and (stripped[i].isdigit() or stripped[i] in (".", ")", " ")): i += 1
                if i < len(stripped): claims_list.append(stripped[i:].strip())
            else:
                claims_list.append(stripped)

        claims_list = [
            c for c in claims_list
            if len(c) > 10 and not c.lower().startswith(
                ("output:", "text:", "no explicit claims found")
            )
        ]

        save_claims_for_analysis(analysis_id, claims_list)

        return jsonify({
            "claims": claims_list,
            "analysis_id": analysis_id,
            "cached": False
        })

    except Exception as e:
        logging.error(f"Failed to extract claims: {e}")
        return jsonify({"error": f"Failed to extract claims: {str(e)}"}), 500


@app.route("/api/my-analyses", methods=["GET"])
@require_user
def my_analyses(user):
    """
    Return the last N DISTINCT analyses for the logged-in user.
    - Deduplicates multiple user_analyses rows for the same analysis_id.
    - Adds a human-readable title derived from canonical_text.
    - SORTS BY last_accessed (most recently opened/created first).
    """
    conn = get_conn()
    try:
        c = conn.cursor()
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
            WHERE ua.user_id = ?
            GROUP BY a.analysis_id
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
        """
        Derive a title from the analysed content (canonical_text).
        Fallback to mode if content missing.
        """
        txt = ""
        # Row is sqlite.Row, so use keys()
        if "canonical_text" in r.keys():
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
@require_user
def analysis_snapshot(user):
    """
    Return a read-only snapshot of an existing analysis for UI restore.
    Uses ONLY cached data (no new model/external calls).
    """
    analysis_id = request.args.get("analysis_id")
    if not analysis_id:
        return jsonify({"error": "Missing analysis_id"}), 400

    user_id = user["user_id"]

    conn = get_conn()
    try:
        c = conn.cursor()

        # 1) Verify that this analysis belongs to the user (via mapping)
        c.execute(
            """
            SELECT a.analysis_id,
                   a.mode,
                   a.canonical_text,
                   a.created_at
            FROM analyses a
            JOIN user_analyses ua ON ua.analysis_id = a.analysis_id
            WHERE ua.user_id = ? AND a.analysis_id = ?
            """,
            (user_id, analysis_id),
        )
        meta = c.fetchone()
        if not meta:
            return jsonify({"error": "Not found or not allowed"}), 404

        # 2) Fetch all claims for this analysis
        c.execute(
            """
            SELECT ordinal, claim_text
            FROM claims
            WHERE analysis_id = ?
            ORDER BY ordinal
            """,
            (analysis_id,),
        )
        claim_rows = c.fetchall()

        # We'll re-use the same connection for cache lookups
        claims_payload = []
        for row in claim_rows:
            ordinal = row["ordinal"]
            claim_text = row["claim_text"]
            ch = sha256_str(claim_text.strip().lower())

            # model_cache
            c.execute(
                """
                SELECT verdict, questions_json
                FROM model_cache
                WHERE claim_hash=?
                """,
                (ch,),
            )
            model_row = c.fetchone()

            if model_row:
                model_verdict = model_row["verdict"]
                questions = json_loads(model_row["questions_json"], [])
            else:
                model_verdict = None
                questions = []

            # external_cache
            c.execute(
                """
                SELECT verdict, sources_json
                FROM external_cache
                WHERE claim_hash=?
                """,
                (ch,),
            )
            ext_row = c.fetchone()

            if ext_row:
                external_verdict = ext_row["verdict"]
                external_sources = json_loads(ext_row["sources_json"], [])
            else:
                external_verdict = None
                external_sources = []

            claims_payload.append(
                {
                    "ordinal": ordinal,
                    "claim_text": claim_text,
                    "model_verdict": model_verdict,
                    "questions": questions,
                    "external_verdict": external_verdict,
                    "external_sources": external_sources,
                }
            )

    finally:
        conn.close()

    # Title = derived from canonical_text, same logic as history
    def _title_from_text(txt, fallback_mode):
        txt = (txt or "").strip()
        if not txt:
            return fallback_mode or "Untitled analysis"
        first_line = txt.splitlines()[0].strip()
        if len(first_line) > 120:
            return first_line[:117] + "..."
        return first_line

    title = _title_from_text(meta["canonical_text"], meta["mode"])
    created_at = meta["created_at"]
    if hasattr(created_at, "isoformat"):
        created_at = created_at.isoformat()

    return jsonify(
        {
            "analysis_id": meta["analysis_id"],
            "mode": meta["mode"],
            "title": title,
            "created_at": created_at,
            "claims": claims_payload,
        }
    )


@app.route("/api/delete-analysis/<analysis_id>", methods=["DELETE"])
@require_user
def delete_analysis(user, analysis_id):
    """
    Delete an analysis belonging to the logged-in user.
    Removes entries from analyses, claims (via cascade), and user_analyses.
    """
    user_id = user["user_id"]

    conn = get_conn()
    try:
        c = conn.cursor()

        # 1) Verify ownership
        c.execute(
            """
            SELECT 1 FROM user_analyses
            WHERE user_id = ? AND analysis_id = ?
            """,
            (user_id, analysis_id),
        )
        row = c.fetchone()
        if not row:
            return jsonify({"error": "Not found or not allowed"}), 404

        # 2) Delete from analyses (this cascades claims)
        c.execute(
            "DELETE FROM analyses WHERE analysis_id = ?",
            (analysis_id,),
        )

        # 3) Delete mapping entry
        c.execute(
            "DELETE FROM user_analyses WHERE analysis_id = ?",
            (analysis_id,),
        )

        conn.commit()

    finally:
        conn.close()

    return jsonify({"status": "deleted"})


@app.route("/api/get-claim-details", methods=["POST"])
def get_claim_details():
    payload = request.json or {}
    ordinal = payload.get("claim_idx")
    analysis_id = payload.get("analysis_id")
    mode = payload.get("mode")

    # ✅ Capture user preference
    preferred_model = payload.get("preferred_model")

    if analysis_id is None or ordinal is None:
        return jsonify({"error": "Missing analysis or claim index"}), 400

    conn = get_conn()
    try:
        c = conn.cursor()

        # 1) Fetch claim text AND full context text
        c.execute("""
            SELECT c.claim_text, a.canonical_text
            FROM claims c
            JOIN analyses a ON c.analysis_id = a.analysis_id
            WHERE c.analysis_id=? AND c.ordinal=?
        """, (analysis_id, int(ordinal)))

        row = c.fetchone()
    finally:
        conn.close()

    if not row:
        return jsonify({"error": "Claim not found"}), 404

    claim_text = row["claim_text"]
    full_text_context = row["canonical_text"]
    ch = sha256_str(claim_text.strip().lower())

    # 2) Check model_cache
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT verdict, questions_json, keywords_json
            FROM model_cache
            WHERE claim_hash=?
        """, (ch,))
        hit = c.fetchone()
    finally:
        conn.close()

    if hit:
        return jsonify({
            "model_verdict": hit[0],
            "questions": json_loads(hit[1], []),
            "search_keywords": json_loads(hit[2], []),
            "cached": True
        })

    # 3) Compute verdict + questions
    chosen_mode = mode if mode in verification_prompts else 'General Analysis of Testable Claims'

    short_context = full_text_context[:4000]

    verdict_prompt = verification_prompts[chosen_mode].format(
        claim=claim_text,
        context=short_context
    )

    # ✅ Pass preference to the generator
    model_verdict_content, questions, search_keywords = generate_model_verdict_and_questions(
        verdict_prompt,
        claim_text,
        preferred_model=preferred_model
    )

    # 4) Store results
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO model_cache (claim_hash, verdict, questions_json, keywords_json, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(claim_hash) DO UPDATE SET
                verdict=excluded.verdict,
                questions_json=excluded.questions_json,
                keywords_json=excluded.keywords_json,
                updated_at=CURRENT_TIMESTAMP
        """, (
            ch,
            model_verdict_content,
            json_dumps(questions or []),
            json_dumps(search_keywords or [])
        ))
        conn.commit()
    finally:
        conn.close()

    return jsonify({
        "model_verdict": model_verdict_content,
        "questions": questions or [],
        "search_keywords": search_keywords or [],
        "cached": False
    })

@app.route("/api/verify-external", methods=["POST"])
def verify_external():
    payload = request.json or {}
    ordinal = payload.get("claim_idx")
    analysis_id = payload.get("analysis_id")

    # ✅ 1. Capture User Preference safely
    preferred_model = payload.get("preferred_model")

    # 2. Capture sources sent from Frontend
    client_sources = payload.get("client_sources", [])

    if analysis_id is None or ordinal is None:
        return jsonify({"error": "Missing analysis or claim index"}), 400

    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT c.claim_text, a.canonical_text
            FROM claims c
            JOIN analyses a ON c.analysis_id = a.analysis_id
            WHERE c.analysis_id=? AND c.ordinal=?
        """, (analysis_id, int(ordinal)))

        row = c.fetchone()
    finally:
        conn.close()

    if not row:
        return jsonify({"error": "Claim not found"}), 404

    claim_text = row["claim_text"]
    full_text_context = row["canonical_text"]
    ch = sha256_str(claim_text.strip().lower())

    # --- Check Cache ---
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute(
            "SELECT verdict, sources_json FROM external_cache WHERE claim_hash=?",
            (ch,)
        )
        hit = c.fetchone()
    finally:
        conn.close()

    if hit:
        return jsonify({
            "verdict": hit[0],
            "sources": json_loads(hit[1], []),
            "cached": True
        })

    # --- Build Keywords ---
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT keywords_json FROM model_cache WHERE claim_hash=?", (ch,))
        kw_row = c.fetchone()
    finally:
        conn.close()

    search_keywords = json_loads(kw_row[0], []) if kw_row else []

    if not search_keywords:
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', claim_text.lower())
        search_keywords = list(set(words))[:5] or [claim_text.lower()[:50]]

    # ------------------------------------------------------
    # 4) Fetch external sources
    # ------------------------------------------------------
    all_sources = []

    if isinstance(client_sources, list):
        for s in client_sources:
            if isinstance(s, dict) and s.get('title') and s.get('abstract'):
                all_sources.append(s)

    try:
        all_sources.extend(fetch_semantic_scholar(search_keywords))
    except Exception as e:
        logging.error(f"Semantic Scholar error: {e}")

    time.sleep(1.1)

    try:
        all_sources.extend(fetch_crossref(search_keywords))
    except Exception as e:
        logging.error(f"CrossRef error: {e}")

    time.sleep(0.5)

    try:
        all_sources.extend(fetch_pubmed(search_keywords))
    except Exception as e:
        logging.error(f"PubMed error: {e}")

    seen = set()
    unique_sources = []
    for s in all_sources:
        url = s.get("url") or ""
        key = url if url else s.get("title", "").lower()[:50]
        if key and key not in seen:
            unique_sources.append(s)
            seen.add(key)

    # ------------------------------------------------------
    # 5) Generate external verdict
    # ------------------------------------------------------
    if unique_sources:
        try:
            unique_sources.sort(key=lambda x: int(x.get('year') or 0), reverse=True)
        except:
            pass

        abstracts_and_titles = "\n\n".join(
            f"Title: {s.get('title','No title')}\n"
            f"Abstract: {s.get('abstract','Abstract not available')}\n"
            f"Authors: {s.get('authors','')}\n"
            f"Year: {s.get('year','')}\n"
            f"Source: {s.get('source','Unknown')}"
            for s in unique_sources[:10]
            if s.get('title')
        )

        short_context = full_text_context[:3000]

        prompt = f"""
You are an AI assistant evaluating a claim based on provided scientific paper information.

Context of the claim:
"{short_context}..."

Claim: "{claim_text}"

Scientific Papers found:
{abstracts_and_titles}

Based on the papers provided:
1. Do these papers confirm the specific event/discovery described in the claim?
2. If the papers discuss similar topics (e.g. CERN simulations) but do NOT mention the specific breakthrough claimed (e.g. "parallel universe"), point that out.

Return a justified verdict (TRUE, FALSE, UNCERTAIN, or SUPPORTED/UNSUPPORTED) of up to 300 words.
"""

        # ✅ FIXED: Strict Model List Definition
        default_models = [
            "x-ai/grok-4.1-fast:free",
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-2.0-flash-lite-preview-02-05:free",
            "openai/gpt-oss-20b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free"
        ]

        models = list(default_models)

        # ✅ Validate user preference string before using
        if preferred_model and isinstance(preferred_model, str) and preferred_model.strip():
            if preferred_model in models:
                models.remove(preferred_model)
            models.insert(0, preferred_model)

        external_verdict = "Could not generate external verdict."

        for model in models:
            try:
                # Ensure model passed is a string
                if not isinstance(model, str): continue

                res = call_openrouter(prompt, model=model)
                content = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                if content.strip():
                    external_verdict = normalize_text_for_display(content)
                    break
            except Exception as e:
                logging.error(f"External verdict model {model} failed: {e}")
                continue
    else:
        external_verdict = "No relevant scientific papers found for this claim."

    # --- Cache and Return ---
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO external_cache
            (claim_hash, verdict, sources_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(claim_hash) DO UPDATE SET
                verdict=excluded.verdict,
                sources_json=excluded.sources_json,
                updated_at=CURRENT_TIMESTAMP
        """, (ch, external_verdict, json_dumps(unique_sources)))
        conn.commit()
    finally:
        conn.close()

    return jsonify({
        "verdict": external_verdict,
        "sources": unique_sources,
        "cached": False
    })


@app.route("/api/process-image", methods=["POST"])
def process_image():
    """Process uploaded image and extract text using OCR with caching"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400

        # Save the uploaded image temporarily to compute hash
        image_path = save_uploaded_file(image_file)
        if not image_path:
            return jsonify({"error": "Failed to save image"}), 500

        # Compute file hash and check cache
        file_hash = compute_file_hash(image_path)
        cached_text = get_cached_media(file_hash)
        if cached_text:
            try:
                os.remove(image_path)
            except:
                pass
            return jsonify({"extracted_text": cached_text, "cached": True})

        # Extract text using OCR if not cached
        extracted_text = analyze_image_with_ocr(image_path)

        # Store in cache
        if extracted_text:
            store_media_cache(file_hash, 'image', extracted_text)

        # Clean up the uploaded file
        try:
            os.remove(image_path)
        except:
            pass

        if not extracted_text:
            return jsonify({"error": "Could not extract text from image. Please ensure the image contains clear text."}), 400

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
    # Handle both GET (EventSource) and POST (direct)
    if request.method == "GET":
        claim_idx = request.args.get("claim_idx", type=int)
        question_idx = request.args.get("question_idx", type=int)
        analysis_id = request.args.get("analysis_id")
    else:
        data = request.get_json(silent=True) or {}
        claim_idx = data.get("claim_idx")
        question_idx = data.get("question_idx")
        analysis_id = data.get("analysis_id")

    if analysis_id is None or claim_idx is None or question_idx is None:
        return Response(
            json.dumps({"error": "Missing analysis_id, claim_idx or question_idx"}),
            mimetype="application/json",
            status=400
        )

    # ---------------------------------------------------------
    # 1) Get claim text AND FULL CONTEXT using get_conn()
    # ---------------------------------------------------------
    conn = get_conn()
    try:
        c = conn.cursor()
        # ✅ FETCH CONTEXT
        c.execute("""
            SELECT c.claim_text, a.canonical_text
            FROM claims c
            JOIN analyses a ON c.analysis_id = a.analysis_id
            WHERE c.analysis_id=? AND c.ordinal=?
        """, (analysis_id, int(claim_idx)))
        row = c.fetchone()
    finally:
        conn.close()

    if not row:
        return Response(
            json.dumps({"error": "Claim not found"}),
            mimetype="application/json",
            status=404
        )

    claim_text = row["claim_text"]
    full_context = row["canonical_text"]  # ✅ Context retrieved
    claim_hash = sha256_str(claim_text.strip().lower())

    # ---------------------------------------------------------
    # 2) Get selected research question
    # ---------------------------------------------------------
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT questions_json FROM model_cache WHERE claim_hash=?", (claim_hash,))
        questions_row = c.fetchone()
    finally:
        conn.close()

    if not questions_row:
        return Response(
            json.dumps({"error": "Questions not found. Please generate model verdict first."}),
            mimetype="application/json",
            status=400
        )

    questions = json_loads(questions_row[0], [])
    if question_idx >= len(questions):
        return Response(
            json.dumps({"error": "Question index out of range."}),
            mimetype="application/json",
            status=400
        )

    question_text = questions[question_idx]
    rq_hash = sha256_str(
        (claim_text.strip().lower() + "||" + question_text.strip().lower())
    )

    # ---------------------------------------------------------
    # 3) Check report_cache
    # ---------------------------------------------------------
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT report_text FROM report_cache WHERE rq_hash=?", (rq_hash,))
        hit = c.fetchone()
    finally:
        conn.close()

    if hit and hit[0]:
        def stream_cached():
            yield f"data: {json.dumps({'content': hit[0]})}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_cached(), mimetype="text/event-stream")

    # ---------------------------------------------------------
    # 4) Fetch verdicts
    # ---------------------------------------------------------
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT verdict FROM model_cache WHERE claim_hash=?", (claim_hash,))
        row = c.fetchone()
        model_verdict_content = row[0] if row else "Verdict not yet generated by AI."

        c.execute("SELECT verdict FROM external_cache WHERE claim_hash=?", (claim_hash,))
        row2 = c.fetchone()
        external_verdict_content = row2[0] if row2 else "Not yet externally verified."
    finally:
        conn.close()

    # ---------------------------------------------------------
    # 5) Compose prompt with CONTEXT + RECONCILIATION INSTRUCTIONS
    # ---------------------------------------------------------
    short_context = full_context[:4000] # Limit context size

    prompt = f'''
You are an AI assistant producing a structured research report of up to 3000 words.

**CRITICAL FORMATTING REQUIREMENTS:**
- Use ONLY plain text with basic formatting
- NO HTML tags of any kind
- Use simple dash "-" for ranges
- Use simple quotes "'"
- Use * bullets; headings with "##"; keep it readable.

**Structure:**
## 1. **Introduction**
## 2. **Analysis**
## 3. **Conclusion**
## 4. **Sources**

---
**Context of the Claim:**
"{short_context}..."

**Claim:** {claim_text}

**Research Question:** {question_text}

**AI's Initial Verdict:** {model_verdict_content}
**External Verification Verdict:** {external_verdict_content}

---
**INSTRUCTIONS:**
1. Use the "Context" to understand specific entities (e.g. if the claim mentions "the team", check the context to see if it's CERN, NASA, etc.).
2. **Compare the Initial Verdict vs External Verification.**
   - If they agree, reinforce the conclusion with details.
   - If they DISAGREE (e.g. Model says True, External says Unsupported), **you must reconcile them.**
   - Prioritize the External Verification if it cites specific papers.
   - If External Verification failed to find papers but the claim is famous (e.g. widely known science), you may rely on general scientific consensus but note the lack of direct papers.
   - Explicitly state *why* there is a mismatch (e.g. "External search likely failed due to keywords," or "The claim is a viral hoax not found in literature").

Generate the research report now.
'''

    # ---------------------------------------------------------
    # 6) Streamed SSE response
    # ---------------------------------------------------------
    def stream_response():
        full_report = ""
        try:
            response = call_openrouter(prompt, stream=True)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if not chunk: continue
                for line in chunk.split("\n"):
                    line = line.strip()
                    if not line.startswith("data:"): continue
                    data_part = line[5:].strip()
                    if data_part == "[DONE]": continue
                    try:
                        json_data = json.loads(data_part)
                    except json.JSONDecodeError: continue
                    if "choices" in json_data and json_data["choices"]:
                        delta = json_data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            normalized = normalize_text_for_display(content)
                            full_report += normalized
                            yield f"data: {json.dumps({'content': normalized})}\n\n"

        except Exception as e:
            logging.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': 'Streaming failed'})}\n\n"

        finally:
            if full_report.strip():
                try:
                    conn = get_conn()
                    c = conn.cursor()
                    c.execute("""
                        INSERT OR REPLACE INTO report_cache
                        (rq_hash, question_text, report_text, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, (rq_hash, question_text, full_report))
                    conn.commit()
                except Exception as db_err:
                    logging.error(f"Cache save error: {db_err}")
                finally:
                    conn.close()
            yield "data: [DONE]\n\n"

    return Response(stream_response(), mimetype="text/event-stream")


@app.route("/api/available-reports", methods=["GET"])
def get_available_reports():
    """List all available reports for a given analysis (sessionless)."""
    analysis_id = request.args.get("analysis_id")

    if not analysis_id:
        # Optional alternate POST usage
        try:
            payload = request.get_json(silent=True) or {}
            analysis_id = payload.get("analysis_id")
        except Exception:
            analysis_id = None

    if not analysis_id:
        return jsonify({"error": "Missing analysis_id"}), 400

    # ---------------------------------------------------------
    # 1) Fetch all claims for this analysis
    # ---------------------------------------------------------
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute(
            "SELECT ordinal, claim_text FROM claims WHERE analysis_id=? ORDER BY ordinal",
            (analysis_id,)
        )
        claim_rows = c.fetchall()
    finally:
        conn.close()

    available_reports = []

    # ---------------------------------------------------------
    # 2) For each claim, check model_cache + report_cache
    # ---------------------------------------------------------
    for ordinal, claim_text in claim_rows:
        preview = claim_text[:80] + "..." if len(claim_text) > 80 else claim_text
        claim_hash = sha256_str(claim_text.strip().lower())

        # Get model verdict + questions in one DB call
        conn = get_conn()
        try:
            c = conn.cursor()
            c.execute("""
                SELECT verdict, questions_json
                FROM model_cache
                WHERE claim_hash=?
            """, (claim_hash,))
            model_cache_row = c.fetchone()
        finally:
            conn.close()

        if not model_cache_row:
            continue

        verdict, questions_json = model_cache_row
        questions = json_loads(questions_json, [])

        # ------------------------------------------
        # 2a) Summary report (Model verdict exists)
        # ------------------------------------------
        if verdict:
            available_reports.append({
                "id": f"claim-{ordinal}-summary",
                "type": f"Claim {ordinal + 1} - Model Verdict & External Verification",
                "description": f"Model analysis and external sources for: {preview}"
            })

        # ------------------------------------------
        # 2b) Question-based full reports
        # ------------------------------------------
        for q_idx, question in enumerate(questions):
            rq_hash = sha256_str(
                claim_text.strip().lower() +
                "||" +
                question.strip().lower()
            )

            # Check if report exists
            conn = get_conn()
            try:
                c = conn.cursor()
                c.execute(
                    "SELECT report_text FROM report_cache WHERE rq_hash=?",
                    (rq_hash,)
                )
                report_row = c.fetchone()
            finally:
                conn.close()

            if report_row:
                available_reports.append({
                    "id": f"claim-{ordinal}-question-{q_idx}",
                    "type": f"Claim {ordinal + 1} - Question Report {q_idx + 1}",
                    "description": f"Research report for: {question[:100]}..."
                })

    return jsonify(available_reports)


# ==============================================================
#                 EXPORT PDF ROUTE (GROUPED LOGIC + WEB LOGO)
# ==============================================================

@app.route("/api/export-pdf", methods=["POST", "OPTIONS"])
@cross_origin(origins=["https://epistemiq.vercel.app"], supports_credentials=True, methods=["POST", "OPTIONS"], allow_headers=["Content-Type"])
def export_pdf():
    payload = request.json or {}
    selected_reports = payload.get("selected_reports", [])
    analysis_id = payload.get("analysis_id")

    if not analysis_id: return "Missing analysis_id", 400

    # 1. Fetch ALL data first to avoid DB locking later
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT ordinal, claim_text FROM claims WHERE analysis_id=? ORDER BY ordinal", (analysis_id,))
        claim_rows = c.fetchall()

        # Pre-fetch cache data into a structured dict
        claims_data = {}

        for ordinal, text in claim_rows:
            ch = sha256_str(text.strip().lower())

            # Fetch Verdicts/Sources
            c.execute("SELECT verdict FROM model_cache WHERE claim_hash=?", (ch,))
            mv = c.fetchone()
            c.execute("SELECT verdict, sources_json FROM external_cache WHERE claim_hash=?", (ch,))
            ev = c.fetchone()

            # Fetch Questions List
            c.execute("SELECT questions_json FROM model_cache WHERE claim_hash=?", (ch,))
            qr = c.fetchone()
            q_list = json_loads(qr[0], []) if qr else []

            claims_data[ordinal] = {
                "text": text,
                "model_verdict": mv[0] if mv else "",
                "external_verdict": ev[0] if ev else "",
                "sources": json_loads(ev[1], []) if ev else [],
                "questions": q_list,
                "reports": {}
            }

            # Fetch Cached Reports for this claim
            for idx, q_text in enumerate(q_list):
                rq_hash = sha256_str(text.strip().lower() + "||" + q_text.strip().lower())
                c.execute("SELECT report_text FROM report_cache WHERE rq_hash=?", (rq_hash,))
                rr = c.fetchone()
                if rr:
                    claims_data[ordinal]["reports"][idx] = rr[0]

    finally:
        conn.close()

    if not claims_data: return "No claims found.", 400

    # 2. Determine what to print
    claims_to_print = sorted(list(claims_data.keys()))

    # 3. Setup PDF
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    p.setTitle("Epistemiq Report")
    width, height = A4
    styles = get_pro_styles()

    # --- HEADER LOGO (FIXED: Uses Remote URL) ---
    header_x = 0.75 * inch
    logo_url = "https://raw.githubusercontent.com/akebonin/epistemiq/refs/heads/main/templates/static/icons/logo.png"

    # Import locally to ensure it doesn't break global imports
    from reportlab.lib.utils import ImageReader

    try:
        # Fetch image into memory
        res = requests.get(logo_url, timeout=5)
        if res.status_code == 200:
            img_data = io.BytesIO(res.content)
            logo_img = ImageReader(img_data)

            # Draw Logo: Top Left
            p.drawImage(logo_img, 0.75*inch, height-1.25*inch, width=50, height=50, mask='auto', preserveAspectRatio=True)

            # Shift text to the right to make room for logo
            header_x += 60
    except Exception as e:
        logging.warning(f"Logo fetch failed: {e}")
        # If fetch fails, we just continue without the logo, text stays at margin.

    p.setFont("Helvetica-Bold", 22)
    p.setFillColor(colors.black)
    p.drawString(header_x, height - 0.8*inch, "Epistemiq Analysis Report")
    p.setFont("Helvetica", 10)
    p.setFillColor(colors.darkgrey)
    p.drawString(header_x, height - 0.8*inch - 18, "Your Compass in the Epistemic Fog")
    p.drawRightString(width-0.75*inch, height-0.8*inch, "epistemiq.vercel.app")
    p.drawRightString(width-0.75*inch, height-0.8*inch - 14, "epistemiq.ai@gmail.com")
    p.setStrokeColor(colors.lightgrey)
    p.setLineWidth(1)
    p.line(0.75*inch, height-1.4*inch, width-0.75*inch, height-1.4*inch)

    y = height - 1.8 * inch

    # 4. Render Content Loop (Unchanged)
    for claim_idx in claims_to_print:
        claim_data = claims_data[claim_idx]

        # Check selection
        has_summary = f"claim-{claim_idx}-summary" in selected_reports
        selected_q_indices = []
        for i in range(len(claim_data["questions"])):
            if f"claim-{claim_idx}-question-{i}" in selected_reports:
                selected_q_indices.append(i)

        if not has_summary and not selected_q_indices:
            continue

        # -- Draw Claim Header --
        if y < 1.5 * inch:
            draw_page_footer(p, width)
            p.showPage()
            y = height - 1.0 * inch

        y = draw_paragraph(p, f"Claim {claim_idx + 1}: {claim_data['text']}", styles['ClaimHeading'], y, width)

        # -- Draw Summary --
        if has_summary:
            if claim_data['model_verdict']:
                clean_mv = clean_verdict_text(claim_data['model_verdict'])
                y = draw_paragraph(p, f"<b>Model Verdict:</b> {clean_mv}", styles['ProBody'], y, width)

            if claim_data['external_verdict']:
                clean_ev = clean_verdict_text(claim_data['external_verdict'])
                y = draw_paragraph(p, f"<b>External Verdict:</b> {clean_ev}", styles['ProBody'], y, width)

            if claim_data['sources']:
                y = draw_paragraph(p, "External Sources:", styles['SectionHeading'], y, width)
                for src in claim_data['sources']:
                    title = clean_xml(src.get("title", "Source Link"))
                    url = src.get("url", "")
                    tag = f"<b>[{clean_xml(src.get('source', 'Source'))}]</b>"

                    if len(title) > 120: title = title[:117] + "..."

                    if url:
                        safe_url = url.replace('&', '&amp;').replace('"', '&quot;')
                        link_html = f'{tag} <link href="{safe_url}"><u><font color="blue">{title}</font></u></link>'
                    else:
                        link_html = f"{tag} {title}"

                    y = draw_paragraph(p, link_html, styles['ProLink'], y, width)

            y -= 10

        # -- Draw Question Reports --
        for q_idx in selected_q_indices:
            q_text = claim_data["questions"][q_idx]
            report_text = claim_data["reports"].get(q_idx)

            if not report_text: continue

            y = draw_paragraph(p, f"<b>Research Question {q_idx+1}:</b> {clean_xml(q_text)}", styles['SectionHeading'], y, width)

            blocks = process_full_report(report_text)
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

    # Footer on last page
    draw_page_footer(p, width)
    p.save()
    buffer.seek(0)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name="Epistemiq_AI_Report.pdf")


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


if __name__ == "__main__":
    app.run(debug=True)
