from flask import Flask, request, render_template, send_file, Response, session, jsonify, redirect, url_for
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
import yt_dlp
import unicodedata
import hashlib
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv(dotenv_path="/home/epistemiq/mysite/.env")


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    app.secret_key = os.urandom(24)
    logging.warning("FLASK_SECRET_KEY not set. Using a random key for development.")

CORS(app, resources={
    r"/api/*": {"origins": ["https://epistemiq.vercel.app"]}
})

# Utility functions
def sha256_str(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def json_loads(s: str, fallback):
    try:
        return json.loads(s) if s else fallback
    except Exception:
        return fallback

def new_analysis_id() -> str:
    return str(uuid.uuid4())

# Database setup for normalized storage
def init_db():
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()

    # Workspace pointer only
    c.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        analysis_id TEXT PRIMARY KEY,
        mode TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Pasted text cache
    c.execute("""
    CREATE TABLE IF NOT EXISTS pasted_texts (
        text_hash TEXT PRIMARY KEY,
        text_content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Article cache (URL -> text)
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

    # Media cache
    c.execute("""
    CREATE TABLE IF NOT EXISTS media_cache (
        file_hash TEXT PRIMARY KEY,
        media_type TEXT NOT NULL,
        extracted_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Claims per analysis
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

    # Model verdict + questions + keywords (by claim_hash)
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_cache (
        claim_hash TEXT PRIMARY KEY,
        verdict TEXT,
        questions_json TEXT,
        keywords_json TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # External verdict + sources (by claim_hash)
    c.execute("""
    CREATE TABLE IF NOT EXISTS external_cache (
        claim_hash TEXT PRIMARY KEY,
        verdict TEXT,
        sources_json TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Report cache (claim+question -> report text)
    c.execute("""
    CREATE TABLE IF NOT EXISTS report_cache (
        rq_hash TEXT PRIMARY KEY,
        question_text TEXT,
        report_text TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def save_claims_for_analysis(analysis_id: str, claims_list: list):
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("DELETE FROM claims WHERE analysis_id=?", (analysis_id,))

    for idx, claim_text in enumerate(claims_list):
        claim_hash = sha256_str(claim_text.strip().lower())
        claim_id = sha256_str(f"{analysis_id}|{idx}|{claim_text.strip()}")
        c.execute("""
        INSERT OR REPLACE INTO claims (claim_id, analysis_id, ordinal, claim_text, claim_hash)
        VALUES (?, ?, ?, ?, ?)
        """, (claim_id, analysis_id, idx, claim_text.strip(), claim_hash))

    conn.commit()
    conn.close()

def get_claims_for_analysis(analysis_id: str):
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("SELECT claim_text FROM claims WHERE analysis_id=? ORDER BY ordinal", (analysis_id,))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

def compute_file_hash(file_path):
    """Compute SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def get_cached_media(file_hash):
    """Get cached media extraction result"""
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute('SELECT extracted_text FROM media_cache WHERE file_hash = ?', (file_hash,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def store_media_cache(file_hash, media_type, extracted_text):
    """Store media extraction result in cache"""
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("""
    INSERT OR REPLACE INTO media_cache (file_hash, media_type, extracted_text)
    VALUES (?, ?, ?)
    """, (file_hash, media_type, extracted_text))
    conn.commit()
    conn.close()

def cleanup_old_cache():
    """Clean up old cache entries to prevent database bloat"""
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    try:
        # Clean up media cache older than 30 days
        c.execute('DELETE FROM media_cache WHERE created_at < ?',
                 (datetime.now() - timedelta(days=30),))
        media_deleted = c.rowcount

        # Clean up analyses older than 7 days
        c.execute('DELETE FROM analyses WHERE last_accessed < ?',
                 (datetime.now() - timedelta(days=7),))
        analyses_deleted = c.rowcount

        # Clean up pasted_texts older than 30 days
        c.execute('DELETE FROM pasted_texts WHERE created_at < ?',
                 (datetime.now() - timedelta(days=30),))
        texts_deleted = c.rowcount

        # Clean up article_cache older than 30 days
        c.execute('DELETE FROM article_cache WHERE fetched_at < ?',
                 (datetime.now() - timedelta(days=30),))
        articles_deleted = c.rowcount

        # Clean up model_cache older than 30 days
        c.execute('DELETE FROM model_cache WHERE updated_at < ?',
                 (datetime.now() - timedelta(days=30),))
        model_deleted = c.rowcount

        # Clean up external_cache older than 30 days
        c.execute('DELETE FROM external_cache WHERE updated_at < ?',
                 (datetime.now() - timedelta(days=30),))
        external_deleted = c.rowcount

        # Clean up report_cache older than 30 days
        c.execute('DELETE FROM report_cache WHERE updated_at < ?',
                 (datetime.now() - timedelta(days=30),))
        report_deleted = c.rowcount

        conn.commit()
        logging.info(f"Cache cleanup completed: {media_deleted} media, {analyses_deleted} analyses, {texts_deleted} texts, {articles_deleted} articles, {model_deleted} model, {external_deleted} external, {report_deleted} reports removed")

        # Optional: Run VACUUM if significant space was freed
        if (media_deleted + analyses_deleted + texts_deleted + articles_deleted + model_deleted + external_deleted + report_deleted) > 50:
            c.execute('VACUUM')
            logging.info("Database vacuum performed")

    except Exception as e:
        logging.error(f"Cleanup error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

# Initialize database on startup
init_db()

# API Configuration
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")

if not WHISPER_API_KEY:
    logging.error("WHISPER_API_KEY not set.")
    raise ValueError("WHISPER_API_KEY is not set in environment variables.")

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
Justification: Concise explanation under 1000 characters.
Sources: None
Keywords: term1, term2, term3, term4, term5

STRICT RULES:
- Verdict: Exactly one of VERIFIED, PARTIALLY_SUPPORTED, INCONCLUSIVE, CONTRADICTED, SUPPORTED, NOT_SUPPORTED, FEASIBLE, POSSIBLE_BUT_UNPROVEN, UNLIKELY, NONSENSE
- Justification: String, max 1000 characters
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

verification_prompts = {
    "General Analysis of Testable Claims": f'''
Analyze this claim and return a structured text response. {BASE_JSON_STRUCTURE}

Claim: "{{claim}}"
''',
    "Specific Focus on Scientific Claims": f'''
Analyze this scientific claim and return a structured text response. {BASE_JSON_STRUCTURE}

Claim: "{{claim}}"
''',
    "Technology-Focused Extraction": f'''
Evaluate this technology claim and return a structured text response. {BASE_JSON_STRUCTURE}

Claim: "{{claim}}"
'''
}

# Helper functions
def call_openrouter(prompt, stream=False, temperature=0.0, json_mode=False):
    """Calls the OpenRouter API, supports streaming and JSON mode."""
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemini-2.0-flash-exp:free",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": temperature
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(OR_URL, headers=headers, json=payload, stream=stream, timeout=90)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"OpenRouter API call failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            raise Exception(f"API Error {e.response.status_code}: {e.response.text}") from e
        raise Exception(f"Network or API connection error: {e}") from e

def extract_article_from_url(url):
    """Fetch and extract article content from a URL using direct requests and BeautifulSoup."""
    try:
        headers = {"User-Agent": "Epistemiq/1.0 (mailto:epistemiq.ai@gmail.com)"}
        session = requests.Session()
        logging.info(f"Fetching URL: {url}")
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Prioritize common article content selectors
        content_selectors = [
            'article',
            '.article-body-commercial-selector',
            'main',
            '.article-content',
            '.post-content',
            '.entry-content',
            'div[itemprop="articleBody"]',
            'div[id*="content"]',
            'div[class*="text"]'
        ]

        text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                current_text = ' '.join(elem.get_text(separator=' ', strip=True) for elem in elements)
                if len(current_text) > 200:
                    logging.info(f"BeautifulSoup extracted {len(current_text)} characters using selector: {selector}")
                    return current_text
                elif len(current_text) > len(text):
                    text = current_text

        # Fallback if specific selectors didn't yield much
        if len(text) > 50:
            return text

        logging.info("Falling back to raw HTML body extraction if no specific content found.")
        body = soup.find('body')
        if body:
            for elem in body(['script', 'style', 'nav', 'header', 'footer', 'aside', '.sidebar', '.comments', '#comments']):
                elem.decompose()
            raw_body_text = ' '.join(body.get_text(separator=' ', strip=True).split())
            if len(raw_body_text) > 200:
                return raw_body_text
            elif len(raw_body_text) > 50:
                return raw_body_text

        logging.warning("BeautifulSoup extracted insufficient content from URL.")
        return ""
    except requests.exceptions.RequestException as e:
        logging.error(f"Network or HTTP error fetching URL {url}: {e}")
        return ""
    except Exception as e:
        logging.error(f"General error extracting article from URL {url}: {e}")
        return ""

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

def generate_model_verdict_and_questions(prompt, claim_text):
    """Generate model verdict, questions and keywords from claim text"""
    model_verdict_content = "Could not generate model verdict."
    questions = []
    search_keywords = []

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            res = call_openrouter(prompt, json_mode=False, temperature=0.0)
            raw_llm_response = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            raw_llm_response = normalize_text_for_display(raw_llm_response)

            if not raw_llm_response.strip():
                raise ValueError("Empty response from OpenRouter")

            # Parse text response with regex
            verdict_match = re.search(r'Verdict:\s*(VERIFIED|PARTIALLY_SUPPORTED|INCONCLUSIVE|CONTRADICTED|SUPPORTED|NOT_SUPPORTED|FEASIBLE|POSSIBLE_BUT_UNPROVEN|UNLIKELY|NONSENSE)', raw_llm_response, re.IGNORECASE)
            if not verdict_match:
                logging.warning(f"Invalid verdict format in attempt {retry_count + 1}, retrying...")
                retry_count += 1
                continue

            verdict = verdict_match.group(1).upper()
            justification_match = re.search(r'Justification:\s*([\s\S]{20,1000}?(?=\n\s*(?:Sources|Keywords|$)))', raw_llm_response, re.IGNORECASE | re.DOTALL)
            justification = justification_match.group(1).strip()[:1000] if justification_match else 'Justification could not be parsed from response.'

            sources_match = re.search(r'Sources:\s*([\s\S]*?)(?=\n\s*(?:Keywords|$))', raw_llm_response, re.IGNORECASE | re.DOTALL)
            sources = []
            if sources_match:
                source_text = sources_match.group(1).strip()
                sources = re.findall(r'(https?://[^\s,)]+)', source_text)[:2] or ['None']

            keywords_match = re.search(r'Keywords:\s*([\w\s,-]{10,})', raw_llm_response, re.IGNORECASE | re.DOTALL)
            if keywords_match:
                kw_text = keywords_match.group(1).strip()
                search_keywords = [kw.strip().lower() for kw in re.split(r'[,;\s]+', kw_text) if len(kw.strip()) > 3][:5]
            else:
                words = re.findall(r'\b[a-zA-Z]{4,}\b', claim_text.lower())
                search_keywords = list(set(words[:5])) or [claim_text.lower()[:50]]

            # Format for display
            model_verdict_content = f"Verdict: **{verdict}**\n\nJustification: {justification}"
            if sources and sources != ['None']:
                model_verdict_content += f"\n\nSources:\n" + "\n".join(f"- {src}" for src in sources)

            # Generate questions
            try:
                questions = generate_questions_for_claim(claim_text)
            except Exception as e:
                logging.error(f"Failed to generate questions: {e}")
                questions = ["Could not generate research questions"]

            break  # Successful parse, exit retry loop

        except Exception as e:
            logging.error(f"Failed to process LLM response in attempt {retry_count + 1}: {e}")
            retry_count += 1
            if retry_count == max_retries:
                model_verdict_content = f"Error generating verdict after {max_retries} attempts: {str(e)}"
                words = re.findall(r'\b[a-zA-Z]{4,}\b', claim_text.lower())
                search_keywords = list(set(words[:5])) or [claim_text.lower()[:50]]
                questions = ["Could not generate research questions"]

    return model_verdict_content, questions, search_keywords

def fetch_crossref(keywords):
    if not keywords:
        return []

    search_query = ' AND '.join([f'"{kw}"' if ' ' in kw else kw for kw in keywords])
    url = f"https://api.crossref.org/works?query={quote_plus(search_query)}&rows=3&select=title,URL,author,abstract"
    headers = {"User-Agent": "Epistemiq/1.0 (mailto:epistemiq.ai@gmail.com)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        results = []
        for item in items:
            results.append({
                "title": item.get("title", ["No title"])[0] if item.get("title") else "No title",
                "abstract": item.get("abstract", "Abstract not available"),
                "url": item.get("URL", "")
            })
        return results
    except requests.exceptions.RequestException as e:
        logging.warning(f"CrossRef API call failed for query '{search_query}': {e}")
        return []

def fetch_core(keywords):
    if not keywords:
        return []

    search_query = ' AND '.join([f'"{kw}"' if ' ' in kw else kw for kw in keywords])
    url = f"https://core.ac.uk:443/api-v2/search/{quote_plus(search_query)}?page=1&pageSize=3&metadata=true"
    headers = {"User-Agent": "EpistemiqFallback/1.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        results = []
        if "data" in response.json():
            for item in response.json()["data"]:
                results.append({
                    "title": item.get("title", "No title"),
                    "abstract": item.get("description", "No abstract available"),
                    "url": item.get("downloadUrl", item.get("urls", {}).get("fullText", ""))
                })
        return results
    except requests.exceptions.RequestException as e:
        logging.warning(f"CORE API call failed for query '{search_query}': {e}")
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
        "User-Agent": "Epistemiq/1.0 (mailto:epistemiq.ai@gmail.com"
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
            results.append({
                "title": article_data.get('title', 'No title'),
                "abstract": article_data.get('abstract', 'Abstract not available'),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
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

def transcribe_video(video_path, max_retries=5, retry_delay=5):
    """Transcribe uploaded video using Whisper API"""
    try:
        WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")
        if not WHISPER_API_KEY:
            raise ValueError("WHISPER_API_KEY is not set in environment variables.")

        audio_path = video_path + ".mp3"
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()

        with open(audio_path, "rb") as audio_file:
            files = {"file": audio_file}
            headers = {"X-API-Key": WHISPER_API_KEY}
            data = {
                "format": "text",
                "language": "en",
                "model_size": "base"
            }

            response = requests.post(
                "https://api.whisper-api.com/transcribe",
                files=files,
                headers=headers,
                data=data,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "pending":
                    task_id = result.get("task_id")
                    if not task_id:
                        raise ValueError("No task_id returned for pending transcription")
                    return poll_transcription_status(task_id, WHISPER_API_KEY, max_retries, retry_delay)
                transcription = result.get("result", "")
                if not transcription:
                    raise ValueError("No transcription returned from Whisper API")
                return transcription
            else:
                raise ValueError(f"Whisper API error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error: Failed to connect to transcription service")
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise ValueError(f"Failed to transcribe video: {str(e)}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

def transcribe_from_url(video_url, max_retries=5, retry_delay=5):
    """Transcribe video URL using Whisper API"""
    try:
        WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")
        if not WHISPER_API_KEY:
            raise ValueError("WHISPER_API_KEY is not set in environment variables.")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '/tmp/%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            audio_path = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.mp3'

        with open(audio_path, "rb") as audio_file:
            files = {"file": audio_file}
            headers = {"X-API-Key": WHISPER_API_KEY}
            data = {
                "format": "text",
                "language": "en",
                "model_size": "base"
            }

            response = requests.post(
                "https://api.whisper-api.com/transcribe",
                files=files,
                headers=headers,
                data=data,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "pending":
                    task_id = result.get("task_id")
                    if not task_id:
                        raise ValueError("No task_id returned for pending transcription")
                    return poll_transcription_status(task_id, WHISPER_API_KEY, max_retries, retry_delay)
                transcription = result.get("result", "")
                if not transcription:
                    raise ValueError("No transcription returned from Whisper API")
                return transcription
            else:
                raise ValueError(f"Whisper API error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error: Failed to connect to transcription service")
    except Exception as e:
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        raise ValueError(f"Failed to transcribe video URL: {str(e)}")
    finally:
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)

def poll_transcription_status(task_id, api_key, max_retries, retry_delay):
    """Poll for transcription status until completion or max retries"""
    headers = {"X-API-Key": api_key}
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"https://api.whisper-api.com/transcribe/{task_id}",
                headers=headers,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "completed":
                    transcription = result.get("result", "")
                    if transcription:
                        return transcription
                    else:
                        raise ValueError("Transcription completed but no result returned")
                elif result.get("status") == "processing":
                    time.sleep(retry_delay)
                else:
                    raise ValueError(f"Transcription failed with status: {result.get('status')}")
            else:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise ValueError("Transcription timed out after maximum retries")

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

def normalize_text_for_pdf(text):
    """Normalize text specifically for PDF generation with strict HTML cleaning"""
    if not text:
        return text

    # First apply display normalization
    text = normalize_text_for_display(text)

    # Additional PDF-specific replacements
    pdf_replacements = {
        # Convert Greek letters to text for PDF compatibility
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
        'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
        'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
        'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
        'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
        'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta',
        'Ε': 'Epsilon', 'Ζ': 'Zeta', 'Η': 'Eta', 'Θ': 'Theta',
        'Ι': 'Iota', 'Κ': 'Kappa', 'Λ': 'Lambda', 'Μ': 'Mu',
        'Ν': 'Nu', 'Ξ': 'Xi', 'Ο': 'Omicron', 'Π': 'Pi',
        'Ρ': 'Rho', 'Σ': 'Sigma', 'Τ': 'Tau', 'Υ': 'Upsilon',
        'Φ': 'Phi', 'Χ': 'Chi', 'Ψ': 'Psi', 'Ω': 'Omega',

        # Fix common formatting issues
        '<br>': '\n', '<br/>': '\n', '<br />': '\n',
        '<b>': '**', '</b>': '**', '<i>': '*', '</i>': '*',
        '<em>': '*', '</em>': '*', '<strong>': '**', '</strong>': '**',
        '<p>': '\n', '</p>': '\n', '<para>': '', '</para>': '',
    }

    for old, new in pdf_replacements.items():
        text = text.replace(old, new)

    # Clean up any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Fix multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text

def clean_html_for_reportlab(text):
    """Clean HTML but preserve table structure for PDF"""
    if not text:
        return text

    # First normalize
    text = normalize_text_for_pdf(text)

    # Remove problematic HTML but keep table-related structure
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)  # Remove all other HTML tags

    # Fix common issues
    text = re.sub(r'&[^;]+;', '', text)
    text = re.sub(r'\xa0', ' ', text)

    # Ensure proper spacing
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text.strip()

def process_report_content_for_pdf(content):
    """Process report content, handling mixed text and tables"""
    lines = content.split('\n')
    processed_blocks = []
    current_text_block = []
    current_table_lines = []
    in_table = False

    for line in lines:
        line = line.strip()

        # Check if this line starts a table
        if line.startswith('|') and ('---' in line or any(cell.strip() for cell in line.split('|')[1:-1])):
            if current_text_block:
                # Save the current text block
                processed_blocks.append(('text', '\n'.join(current_text_block)))
                current_text_block = []

            in_table = True
            current_table_lines.append(line)

        elif in_table and line.startswith('|'):
            # Continue collecting table lines
            current_table_lines.append(line)

        elif in_table and not line.startswith('|'):
            # Table ended, process it
            if current_table_lines:
                table_content = '\n'.join(current_table_lines)
                processed_blocks.append(('table', table_content))
                current_table_lines = []
            in_table = False
            current_text_block.append(line)

        else:
            # Regular text line
            current_text_block.append(line)

    # Don't forget any remaining content
    if current_text_block:
        processed_blocks.append(('text', '\n'.join(current_text_block)))
    if current_table_lines:
        processed_blocks.append(('table', '\n'.join(current_table_lines)))

    return processed_blocks

def split_long_report(report_content, max_section_length=1500):
    """Split very long reports into manageable sections"""
    if len(report_content) <= max_section_length:
        return [report_content]

    # Try to split at natural boundaries
    sections = []
    current_section = ""

    # Split by lines and rebuild sections
    lines = report_content.split('\n')
    for line in lines:
        if len(current_section + line) > max_section_length and current_section:
            sections.append(current_section)
            current_section = line + '\n'
        else:
            current_section += line + '\n'

    if current_section:
        sections.append(current_section)

    return sections

def draw_paragraph(pdf_canvas, text_content, style, y_pos, page_width, left_margin=0.75*inch, right_margin=0.75*inch):
    """Safe paragraph drawing with better page management"""
    available_width = page_width - left_margin - right_margin

    # Clean the text but keep it simple - NO TABLE CONVERSION!
    text_content = clean_html_for_reportlab(text_content)

    # Replace newlines with <br/> for Paragraph
    text_content = text_content.replace('\n', '<br/>')

    try:
        para = Paragraph(text_content, style)
        w, h = para.wrapOn(pdf_canvas, available_width, 0)

        # Check if we need a new page - be more conservative
        if y_pos - h < 1.5 * inch:  # Increased bottom margin
            pdf_canvas.showPage()
            y_pos = A4[1] - 0.75 * inch

        para.drawOn(pdf_canvas, left_margin, y_pos - h)
        return y_pos - h - style.spaceAfter

    except Exception as e:
        logging.warning(f"Paragraph drawing failed: {e}")
        # Fallback to simple text
        return draw_simple_text(pdf_canvas, text_content, y_pos, page_width, left_margin)

def draw_simple_text(pdf_canvas, text_content, y_pos, page_width, left_margin):
    """Fallback text drawing that ensures no content is lost"""
    pdf_canvas.setFont("Helvetica", 9)
    line_height = 11
    lines = text_content.replace('<br/>', '\n').split('\n')

    for line in lines:
        # Simple line wrapping
        if len(line) > 120:
            words = line.split()
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                if len(test_line) < 120:
                    current_line = test_line
                else:
                    # Draw current line
                    if y_pos < 1.5 * inch:
                        pdf_canvas.showPage()
                        y_pos = A4[1] - 0.75 * inch
                        pdf_canvas.setFont("Helvetica", 9)
                    pdf_canvas.drawString(left_margin, y_pos, current_line.strip())
                    y_pos -= line_height
                    current_line = word + " "
            # Draw remaining line
            if current_line:
                if y_pos < 1.5 * inch:
                    pdf_canvas.showPage()
                    y_pos = A4[1] - 0.75 * inch
                    pdf_canvas.setFont("Helvetica", 9)
                pdf_canvas.drawString(left_margin, y_pos, current_line.strip())
                y_pos -= line_height
        else:
            # Draw single line
            if y_pos < 1.5 * inch:
                pdf_canvas.showPage()
                y_pos = A4[1] - 0.75 * inch
                pdf_canvas.setFont("Helvetica", 9)
            pdf_canvas.drawString(left_margin, y_pos, line.strip())
            y_pos -= line_height

    return y_pos

def parse_markdown_table(markdown_text):
    """Parse markdown table and return data for ReportLab Table"""
    lines = [line.strip() for line in markdown_text.split('\n') if line.strip().startswith('|')]

    if len(lines) < 2:
        return None

    table_data = []
    for line in lines:
        cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
        table_data.append(cells)

    # Remove separator line if it exists
    if len(table_data) > 1 and all(cell.replace('-', '').replace(':', '').replace(' ', '') == '' for cell in table_data[1]):
        table_data.pop(1)

    return table_data if table_data else None

def create_table_from_data(table_data, available_width):
    """Return a ReportLab Table that never exceeds available_width and wraps cell text."""
    if not table_data:
        return None

    # ---- 1) Normalize rows & convert every cell to a Paragraph with wrapping ----
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib import colors

    # A small cell style: wrap long words/URLs; tight leading; no hyphenation needed
    cell_style = ParagraphStyle(
        name="Cell",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        wordWrap="CJK",     # <- forces wrapping even for long tokens/URLs
    )
    header_style = ParagraphStyle(
        name="CellHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )

    # Make all rows the same length
    max_cols = max(len(r) for r in table_data)
    norm_rows = []
    for r_i, row in enumerate(table_data):
        row = list(row) + [""] * (max_cols - len(row))
        cells = []
        for c_i, cell in enumerate(row):
            txt = (cell or "")
            # basic cleanup: strip markdown stars and convert <br/> to real breaks
            txt = re.sub(r"\*+", "", txt).replace("<br/>", "<br/>")
            # header row = first row
            p = Paragraph(txt, header_style if r_i == 0 else cell_style)
            cells.append(p)
        norm_rows.append(cells)

    # ---- 2) Proportional initial widths, then force exact fit to available_width ----
    # weight columns by rough content length of header (fallback to 1)
    def _wlen(par):  # Paragraph -> text length
        return max(1, len(par.getPlainText() or ""))
    weights = [max(_wlen(norm_rows[r][i]) for r in range(len(norm_rows))) for i in range(max_cols)]
    total_w = sum(weights) or max_cols
    raw_widths = [max(36.0, available_width * (w / total_w)) for w in weights]  # min 36pt per col
    # scale so the sum is EXACTLY available_width (prevents drift from padding/borders)
    sum_raw = sum(raw_widths)
    factor = available_width / sum_raw if sum_raw > 0 else 1.0
    col_widths = [w * factor for w in raw_widths]

    # ---- 3) Build the table and style it ----
    table = Table(norm_rows, colWidths=col_widths, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 8),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
        # slightly tighter padding reduces overflow risk
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ]))

    # ---- 4) Final safety: if wrap still reports wider, shrink columns proportionally ----
    tw, _ = table.wrap(available_width, 0)
    if tw > available_width:
        k = available_width / tw
        table._argW = [w * k for w in table._argW]

    return table

# API Endpoints

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

@app.route("/api/extract-article", methods=["POST"])
def extract_article():
    url = request.json.get("url")
    if not url:
        return jsonify({"error": "URL is required."}), 400

    try:
        text = extract_article_from_url(url)
        if not text:
            return jsonify({"error": "Could not extract content from URL. Please ensure it's a valid public webpage or paste the text manually."}), 400
        return jsonify({"article_text": text})
    except Exception as e:
        logging.error(f"Error in extract_article endpoint: {e}")
        return jsonify({"error": f"Failed to fetch article: {str(e)}"}), 400

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """API endpoint to extract claims ONLY with normalized storage."""
    data = request.json
    text = data.get("text")
    mode = data.get("mode")


    if not text or not mode:
        return jsonify({"error": "Missing text or analysis mode."}), 400

    # Create new analysis
    analysis_id = new_analysis_id()
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("INSERT INTO analyses (analysis_id, mode) VALUES (?, ?)", (analysis_id, mode))
    conn.commit()
    conn.close()

    session['analysis_id'] = analysis_id
    session['mode'] = mode

    extraction_prompt = extraction_templates[mode].format(text=text)

    try:
        res = call_openrouter(extraction_prompt)
        raw_claims = res.json()["choices"][0]["message"]["content"]

        if "No explicit claims found" in raw_claims or not raw_claims.strip():
            return jsonify({"claims": []})

        claims_list = []
        for line in raw_claims.splitlines():
            stripped_line = line.strip()
            if stripped_line and stripped_line[0].isdigit():
                content_start = 0
                while content_start < len(stripped_line) and (stripped_line[content_start].isdigit() or stripped_line[content_start] in ['.', ' ']):
                    content_start += 1
                if content_start < len(stripped_line):
                    claims_list.append(stripped_line[content_start:].strip())
            elif stripped_line:
                claims_list.append(stripped_line)

        claims_list = [c for c in claims_list if len(c) > 10 and not c.lower().startswith(("output:", "text:", "no explicit claims found"))]

        # Save claims to database
        save_claims_for_analysis(analysis_id, claims_list)

        return jsonify({
            "claims": claims_list,
            "analysis_id": analysis_id
        })

    except Exception as e:
        logging.error(f"Failed to extract claims: {e}")
        return jsonify({"error": f"Failed to extract claims: {str(e)}"}), 500

@app.route("/api/get-claim-details", methods=["POST"])
def get_claim_details():
    payload = request.json or {}
    ordinal = payload.get("claim_idx")
    analysis_id = session.get("analysis_id")
    mode = session.get("mode") or "General Analysis of Testable Claims"

    if analysis_id is None or ordinal is None:
        return jsonify({"error": "Missing analysis or claim index"}), 400

    # 1) Get the claim text
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("SELECT claim_text FROM claims WHERE analysis_id=? AND ordinal=?", (analysis_id, int(ordinal)))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Claim not found"}), 404

    claim_text = row[0]
    ch = sha256_str(claim_text.strip().lower())

    # 2) Hit model_cache
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("SELECT verdict, questions_json, keywords_json FROM model_cache WHERE claim_hash=?", (ch,))
    hit = c.fetchone()
    conn.close()

    if hit:
        return jsonify({
            "model_verdict": hit[0],
            "questions": json_loads(hit[1], []),
            "search_keywords": json_loads(hit[2], []),
            "cached": True
        })

    # 3) Compute using existing logic
    verdict_prompt = verification_prompts[(mode if mode in verification_prompts else 'General Analysis of Testable Claims')].format(claim=claim_text)
    model_verdict_content, questions, search_keywords = generate_model_verdict_and_questions(verdict_prompt, claim_text)

    # 4) Store in model_cache
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("""
    INSERT INTO model_cache (claim_hash, verdict, questions_json, keywords_json, updated_at)
    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(claim_hash) DO UPDATE SET
    verdict=excluded.verdict,
    questions_json=excluded.questions_json,
    keywords_json=excluded.keywords_json,
    updated_at=CURRENT_TIMESTAMP
    """, (ch, model_verdict_content, json_dumps(questions or []), json_dumps(search_keywords or [])))
    conn.commit()
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
    analysis_id = session.get("analysis_id")

    if analysis_id is None or ordinal is None:
        return jsonify({"error": "Missing analysis or claim index"}), 400

    # 1) get claim text
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("SELECT claim_text FROM claims WHERE analysis_id=? AND ordinal=?", (analysis_id, int(ordinal)))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Claim not found"}), 404

    claim_text = row[0]
    ch = sha256_str(claim_text.strip().lower())

    # 2) check external_cache
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("SELECT verdict, sources_json FROM external_cache WHERE claim_hash=?", (ch,))
    hit = c.fetchone()
    conn.close()

    if hit:
        return jsonify({"verdict": hit[0], "sources": json_loads(hit[1], []), "cached": True})

    # 3) build keywords; if model_cache has them, reuse
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("SELECT keywords_json FROM model_cache WHERE claim_hash=?", (ch,))
    kw_row = c.fetchone()
    conn.close()

    search_keywords = json_loads(kw_row[0], []) if kw_row else []
    if not search_keywords:
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', claim_text.lower())
        search_keywords = list(set(words[:5])) or [claim_text.lower()[:50]]

    # 4) fetch sources (existing functions)
    all_sources = []
    all_sources.extend(fetch_semantic_scholar(search_keywords))
    time.sleep(1.1)
    all_sources.extend(fetch_crossref(search_keywords))
    time.sleep(0.5)
    all_sources.extend(fetch_core(search_keywords))
    time.sleep(0.5)
    all_sources.extend(fetch_pubmed(search_keywords))

    # de-dup by URL
    seen, unique_sources = set(), []
    for s in all_sources:
        url = s.get("url") or ""
        if url and url not in seen:
            unique_sources.append(s)
            seen.add(url)

    # 5) create external verdict with OpenRouter call
    if unique_sources:
        abstracts_and_titles = "\n\n".join(
            f"Title: {s.get('title','No title')}\n"
            f"Abstract: {s.get('abstract','Abstract not available')}\n"
            f"Authors: {s.get('authors','')}\n"
            f"Year: {s.get('year','')}\n"
            f"Citations: {s.get('citation_count',0)}\n"
            f"Source: {s.get('source','Unknown')}"
            for s in unique_sources if s.get('title')
        )

        prompt = f"""You are an AI assistant evaluating a claim based on provided scientific paper information.

Claim: "{claim_text}"

Papers:
{abstracts_and_titles}

Return a short verdict and concise justification."""

        try:
            verdict_res = call_openrouter(prompt)
            external_verdict = verdict_res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"External verdict failed: {e}")
            external_verdict = "Could not generate external verdict."
    else:
        external_verdict = "No relevant scientific papers found for this claim."

    # 6) store in external_cache
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("""
    INSERT INTO external_cache (claim_hash, verdict, sources_json, updated_at)
    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(claim_hash) DO UPDATE SET
    verdict=excluded.verdict,
    sources_json=excluded.sources_json,
    updated_at=CURRENT_TIMESTAMP
    """, (ch, external_verdict, json_dumps(unique_sources)))
    conn.commit()
    conn.close()

    return jsonify({"verdict": external_verdict, "sources": unique_sources, "cached": False})

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

@app.route("/api/process-video", methods=["POST"])
def process_video():
    """Process uploaded video and extract transcription using Whisper with caching"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400

        # Save the uploaded video temporarily to compute hash
        video_path = save_uploaded_file(video_file)
        if not video_path:
            return jsonify({"error": "Failed to save video"}), 500

        # Compute file hash and check cache
        file_hash = compute_file_hash(video_path)
        cached_transcription = get_cached_media(file_hash)
        if cached_transcription:
            try:
                os.remove(video_path)
            except:
                pass
            return jsonify({"transcription": cached_transcription, "cached": True})

        # Transcribe video if not cached
        transcription = transcribe_video(video_path)

        # Store in cache
        if transcription:
            store_media_cache(file_hash, 'video', transcription)

        # Clean up the uploaded file
        try:
            os.remove(video_path)
        except:
            pass

        return jsonify({"transcription": transcription, "cached": False})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Error in process_video endpoint: {e}")
        return jsonify({"error": f"Failed to process video: {str(e)}"}), 500

@app.route("/api/transcribe-video-url", methods=["POST"])
def transcribe_video_url():
    """Transcribe video from URL using Whisper."""
    try:
        data = request.json
        video_url = data.get("video_url")
        if not video_url:
            return jsonify({"error": "No video URL provided"}), 400

        transcription = transcribe_from_url(video_url)
        return jsonify({"transcription": transcription})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Error in transcribe_video_url endpoint: {e}")
        return jsonify({"error": f"Failed to transcribe video URL: {str(e)}"}), 500


@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    claim_idx = request.json.get("claim_idx")
    question_idx = request.json.get("question_idx")
    analysis_id = session.get("analysis_id")

    if analysis_id is None:
        return Response(json.dumps({"error": "Analysis context missing. Please re-run analysis."}), mimetype='application/json', status=400)

    # Get claim text
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("SELECT claim_text FROM claims WHERE analysis_id=? AND ordinal=?", (analysis_id, int(claim_idx)))
    row = c.fetchone()

    if not row:
        conn.close()
        return Response(json.dumps({"error": "Claim not found"}), mimetype='application/json', status=404)

    claim_text = row[0]

    # Get question_text from model_cache using claim_hash
    claim_hash = sha256_str(claim_text.strip().lower())
    c.execute("SELECT questions_json FROM model_cache WHERE claim_hash=?", (claim_hash,))
    questions_row = c.fetchone()

    if not questions_row:
        conn.close()
        return Response(json.dumps({"error": "Questions not found for this claim. Please generate model verdict first."}), mimetype='application/json', status=400)

    questions = json_loads(questions_row[0], [])

    if question_idx >= len(questions):
        conn.close()
        return Response(json.dumps({"error": f"Question index {question_idx} out of range. Only {len(questions)} questions available."}), mimetype='application/json', status=400)

    question_text = questions[question_idx]
    conn.close()

    rq_hash = sha256_str((claim_text.strip().lower() + "||" + question_text.strip().lower()))

    # Hit cache
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("SELECT report_text FROM report_cache WHERE rq_hash=?", (rq_hash,))
    hit = c.fetchone()
    conn.close()

    if hit and hit[0]:
        def stream_cached():
            yield f"data: {json.dumps({'content': hit[0]})}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_cached(), mimetype="text/event-stream")

    # Generate report content
    article_cache_data = {"text": "", "mode": session.get("mode", "General Analysis of Testable Claims")}

    # Get model verdict and external verdict from their caches
    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()
    c.execute("SELECT verdict FROM model_cache WHERE claim_hash=?", (claim_hash,))
    model_row = c.fetchone()
    model_verdict_content = model_row[0] if model_row else "Verdict not yet generated by AI."

    c.execute("SELECT verdict FROM external_cache WHERE claim_hash=?", (claim_hash,))
    external_row = c.fetchone()
    external_verdict_content = external_row[0] if external_row else "Not yet externally verified."
    conn.close()

    # Define the prompt variable here (this was missing)
    prompt = f'''
You are an AI researcher writing a short, evidence-based report (maximum 1000 words). Your task is to investigate the research question in relation to the claim using verifiable scientific knowledge. Use the article context to ground your analysis where helpful. Clearly explain how the answer to the research question supports, contradicts, or contextualizes the claim. Provide concise reasoning and avoid speculation.

**CRITICAL FORMATTING REQUIREMENTS:**
- Use ONLY plain text with basic formatting
- NO HTML tags of any kind
- Use simple dash "-" for ranges (NOT en-dash or em-dash)
- Use simple quotes "' for apostrophes and quotes
- For tables: use simple text with | separators OR just describe the data
- Use **bold** for emphasis only when necessary
- Use simple bullet points with *
- Separate sections with clear headings using ##

**Structure:**
## 1. **Introduction**
[Content]

## 2. **Analysis**
[Content - use simple text descriptions instead of complex tables when possible]

## 3. **Conclusion**
[Content]

## 4. **Sources**
[Content]

---

**Article Context:**
{article_cache_data.get("text", "")}

**Claim:**
{claim_text}

**AI's Initial Verdict on Claim:**
{model_verdict_content}

**External Verification Verdict (if available):**
{external_verdict_content}

**Research Question:**
{question_text}

---

**AI Research Report**
'''

    def stream_response():
        full_report_content = ""
        try:
            response = call_openrouter(prompt, stream=True)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    lines = chunk.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith("data:"):
                            data_part = line[5:].strip()  # Remove "data:"
                            if data_part == '[DONE]':
                                continue
                            try:
                                json_data = json.loads(data_part)
                                if 'choices' in json_data and json_data['choices']:
                                    content = json_data['choices'][0].get('delta', {}).get('content', '')
                                    if content:
                                        normalized_content = normalize_text_for_display(content)
                                        full_report_content += normalized_content
                                        yield f"data: {json.dumps({'content': normalized_content})}\n\n"
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logging.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': 'Streaming failed'})}\n\n"
        finally:
            # Store in cache only if we have meaningful content
            if full_report_content.strip():
                try:
                    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
                    c = conn.cursor()
                    c.execute("""
                        INSERT OR REPLACE INTO report_cache
                        (rq_hash, question_text, report_text, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, (rq_hash, question_text, full_report_content))  # These variables are now defined in the outer scope
                    conn.commit()
                    conn.close()
                except Exception as db_error:
                    logging.error(f"Cache error: {db_error}")

        yield "data: [DONE]\n\n"

    return Response(stream_response(), mimetype='text/event-stream')


@app.route("/api/available-reports", methods=["GET"])
def get_available_reports():
    """Get list of all available reports for the current session"""
    analysis_id = session.get("analysis_id")
    if not analysis_id:
        return jsonify({"error": "No active analysis session found."}), 400

    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()

    # Get claims for this analysis
    c.execute("SELECT ordinal, claim_text FROM claims WHERE analysis_id=? ORDER BY ordinal", (analysis_id,))
    claim_rows = c.fetchall()

    available_reports = []

    for ordinal, claim_text in claim_rows:
        claim_text_preview = claim_text[:80] + '...' if len(claim_text) > 80 else claim_text

        # Add model verdict if available
        claim_hash = sha256_str(claim_text.strip().lower())
        c.execute("SELECT verdict FROM model_cache WHERE claim_hash=?", (claim_hash,))
        model_row = c.fetchone()
        if model_row:
            available_reports.append({
                "id": f"claim-{ordinal}-summary",
                "type": f"Claim {ordinal + 1} - Model Verdict & External Verification",
                "description": f"Model analysis and external sources for: {claim_text_preview}"
            })

        # Add question reports if available
        c.execute("SELECT questions_json FROM model_cache WHERE claim_hash=?", (claim_hash,))
        questions_row = c.fetchone()
        if questions_row:
            questions = json_loads(questions_row[0], [])
            for q_idx, question in enumerate(questions):
                rq_hash = sha256_str((claim_text.strip().lower()+"||"+question.strip().lower()))
                c.execute("SELECT report_text FROM report_cache WHERE rq_hash=?", (rq_hash,))
                report_row = c.fetchone()
                if report_row:
                    available_reports.append({
                        "id": f"claim-{ordinal}-question-{q_idx}",
                        "type": f"Claim {ordinal + 1} - Question Report {q_idx + 1}",
                        "description": f"Research report for: {question[:100]}..."
                    })

    conn.close()
    return jsonify(available_reports)

@app.route("/export-pdf", methods=["POST"])
def export_pdf():
    selected_reports = request.json.get("selected_reports", [])
    analysis_id = session.get("analysis_id")

    if not analysis_id:
        return "No active analysis session found. Please run an analysis first.", 400

    conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
    c = conn.cursor()

    # Get analysis mode
    c.execute("SELECT mode FROM analyses WHERE analysis_id=?", (analysis_id,))
    analysis_row = c.fetchone()
    if not analysis_row:
        return "Analysis session expired or not found.", 400

    # Get claims
    c.execute("SELECT ordinal, claim_text FROM claims WHERE analysis_id=? ORDER BY ordinal", (analysis_id,))
    claim_rows = c.fetchall()
    conn.close()

    if not claim_rows:
        return "No claims found for this analysis session.", 400

    pdf_reports = []
    added_ids = set()

    # Process selected reports based on their IDs
    for report_id in selected_reports:
        if report_id in added_ids:
            continue

        if report_id.endswith('-summary'):
            try:
                claim_idx = int(report_id.split('-')[1])
                for ordinal, claim_text in claim_rows:
                    if ordinal == claim_idx:
                        claim_hash = sha256_str(claim_text.strip().lower())

                        conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
                        c = conn.cursor()

                        c.execute("SELECT verdict FROM model_cache WHERE claim_hash=?", (claim_hash,))
                        model_row = c.fetchone()
                        model_verdict = model_row[0] if model_row else ""

                        c.execute("SELECT verdict, sources_json FROM external_cache WHERE claim_hash=?", (claim_hash,))
                        external_row = c.fetchone()
                        external_verdict = external_row[0] if external_row else "Not verified externally."
                        sources = json_loads(external_row[1], []) if external_row else []

                        conn.close()

                        pdf_reports.append({
                            "id": report_id,
                            "claim_text": claim_text,
                            "model_verdict": model_verdict,
                            "external_verdict": external_verdict,
                            "sources": sources,
                            "question": "Model verdict + external verification",
                            "report": None
                        })
                        added_ids.add(report_id)
                        break
            except (IndexError, ValueError) as e:
                continue

        elif 'question' in report_id:
            try:
                parts = report_id.split('-')
                claim_idx = int(parts[1])
                q_idx = int(parts[3])

                for ordinal, claim_text in claim_rows:
                    if ordinal == claim_idx:
                        claim_hash = sha256_str(claim_text.strip().lower())

                        conn = sqlite3.connect('/home/epistemiq/mysite/sessions.db')
                        c = conn.cursor()

                        # Get question text
                        c.execute("SELECT questions_json FROM model_cache WHERE claim_hash=?", (claim_hash,))
                        questions_row = c.fetchone()
                        if questions_row:
                            questions = json_loads(questions_row[0], [])
                            if q_idx < len(questions):
                                question_text = questions[q_idx]

                                # Get report
                                rq_hash = sha256_str((claim_text.strip().lower()+"||"+question_text.strip().lower()))
                                c.execute("SELECT report_text FROM report_cache WHERE rq_hash=?", (rq_hash,))
                                report_row = c.fetchone()

                                if report_row:
                                    pdf_reports.append({
                                        "id": report_id,
                                        "claim_text": claim_text,
                                        "model_verdict": "",
                                        "external_verdict": "",
                                        "sources": [],
                                        "question": question_text,
                                        "report": report_row[0]
                                    })
                                    added_ids.add(report_id)
                        conn.close()
                        break
            except (IndexError, ValueError) as e:
                continue

    if not pdf_reports:
        return "No valid reports selected for export.", 400

    # PDF generation
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ClaimHeading', parent=styles['h2'], fontName='Helvetica-Bold', fontSize=14, spaceAfter=6))
    styles.add(ParagraphStyle(name='SectionHeading', parent=styles['h3'], fontName='Helvetica-Bold', fontSize=12, spaceAfter=4, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='NormalParagraph', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=12, spaceAfter=8))
    styles.add(ParagraphStyle(name='SourceLink', parent=styles['NormalParagraph'], textColor=colors.blue, fontName='Helvetica', fontSize=9, leading=10, spaceAfter=4))
    styles.add(ParagraphStyle(name='ReportBody', parent=styles['NormalParagraph'], fontName='Helvetica', fontSize=10, leading=14, spaceAfter=10))

    y = height - inch

    p.setFont("Helvetica-Bold", 20)
    p.drawCentredString(width / 2.0, y, "Epistemiq Analysis Report")
    y -= 40

    for item in pdf_reports:
        if y < 1.5 * inch:
            p.showPage()
            y = height - inch

        y -= 20

        # Claim heading
        y = draw_paragraph(p, f"Claim: {item['claim_text']}", styles['ClaimHeading'], y, width)

        # Model verdict
        if item['model_verdict']:
            y = draw_paragraph(p, f"<b>Model Verdict:</b> {item.get('model_verdict','')}", styles['NormalParagraph'], y, width)

        # External verdict
        if item['external_verdict']:
            y = draw_paragraph(p, f"<b>External Verdict:</b> {item.get('external_verdict','')}", styles['NormalParagraph'], y, width)

        # Sources (if any)
        if item.get('sources'):
            y = draw_paragraph(p, "<b>External Sources:</b>", styles['SectionHeading'], y, width)
            for src in item.get('sources', []):
                link_text = f"{src.get('title','')}"
                if src.get('url'):
                    escaped_url = src['url'].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                    link_text = f'<link href="{escaped_url}">{link_text}</link>'
                y = draw_paragraph(p, f"- {link_text}", styles['SourceLink'], y, width)

        # Question heading
        y = draw_paragraph(p, f"<b>Research Question:</b> {item.get('question','')}", styles['SectionHeading'], y, width)

        # Full report if present
        if item.get('report'):
            y = draw_paragraph(p, "<b>AI Research Report:</b>", styles['SectionHeading'], y, width)

            report_content = item['report']
            sections = split_long_report(report_content)

            for i, section in enumerate(sections):
                blocks = process_report_content_for_pdf(section)

                for block_type, block_content in blocks:
                    if block_type == "text":
                        y = draw_paragraph(p, block_content, styles['ReportBody'], y, width)

                    elif block_type == "table":
                        table_data = parse_markdown_table(block_content)
                        if table_data:
                            table = create_table_from_data(table_data, width - 1.5 * inch)
                            if table:
                                w, h = table.wrapOn(p, width - 1.5 * inch, y)
                                if y - h < 1.5 * inch:
                                    p.showPage()
                                    y = A4[1] - 0.75 * inch
                                try:
                                    table.drawOn(p, 0.75 * inch, y - h)
                                except Exception as e:
                                    logging.error(f"Error drawing table: {e}")
                                y -= h + 10

                # Add spacing between sections
                if i < len(sections) - 1:
                    y -= 10

                # New page if near bottom
                if y < 2 * inch:
                    p.showPage()
                    y = A4[1] - 0.75 * inch

            y -= 20

    p.save()
    buffer.seek(0)

    return send_file(buffer, mimetype='application/pdf', as_attachment=True, download_name= "Epistemiq_AI_Report.pdf")


@app.route("/api/cleanup-cache", methods=["POST"])
def cleanup_cache_endpoint():
    """Manual cache cleanup endpoint"""
    try:
        cleanup_old_cache()
        return jsonify({"message": "Cache cleanup completed successfully"})
    except Exception as e:
        logging.error(f"Manual cleanup error: {e}")
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
