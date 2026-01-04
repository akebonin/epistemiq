from flask import Blueprint, request, jsonify, render_template
import psycopg2
import psycopg2.extras
import os
import json
from datetime import datetime, date
from auth_module import get_current_user
import uuid
from werkzeug.utils import secure_filename
import hashlib


admin_bp = Blueprint('admin', __name__)

def get_db():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    return conn

def get_admin_emails():
    # Attempt to read from Environment Variable
    # Default is an empty string "" (No admins)
    raw = os.getenv("ADMIN_EMAILS", "")

    if not raw:
        return [] # No admins configured

    try:
        # Try parsing as JSON list: ["email1@test.com", "email2@test.com"]
        return json.loads(raw)
    except:
        # Fallback: If user just put "me@test.com" without brackets, treat as single email
        # Or split by comma if they did "me@test.com,you@test.com"
        return [e.strip() for e in raw.split(',') if e.strip()]

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        # ✅ Check against dynamic list
        if not user or user['email'] not in get_admin_emails():
            return jsonify({"error": "Forbidden"}), 403
        return f(*args, **kwargs)
    return decorated_function

# ✅ CONFIGURATION: Defines how to handle each table
# pk: Primary Key column name (needed for delete)
# sort: Default sort column
# search: Which text column to filter when searching
TABLE_SCHEMA = {
    # --- Content & Moderation ---
    "comments":           {"pk": "id", "sort": "created_at", "search": "content"},  # ✅ NEW: Moderate comments
    "published_articles": {"pk": "id", "sort": "published_at", "search": "title"}, # ✅ NEW: Manage articles

    # --- Core Data ---
    "analyses":     {"pk": "analysis_id", "sort": "created_at", "search": "canonical_text"},
    "claims":       {"pk": "claim_id",    "sort": "created_at", "search": "claim_text"},

    # --- Users & Auth ---
    "users":        {"pk": "id",          "sort": "created_at", "search": "email"},
    "user_quotas":  {"pk": "user_id",     "sort": "usage_date", "search": "user_id"}, # (Composite PK handled loosely here)
    "sessions":     {"pk": "id",          "sort": "created_at", "search": "session_token"},
    "magic_links":  {"pk": "id",          "sort": "created_at", "search": "token_hash"},
    "user_analyses":{"pk": "id",          "sort": "created_at", "search": "analysis_id"},

    # --- Caches ---
    "model_cache":  {"pk": "claim_hash",  "sort": "updated_at", "search": "verdict"},
    "external_cache": {"pk": "claim_hash", "sort": "updated_at", "search": "verdict"},
    "report_cache": {"pk": "rq_hash",     "sort": "updated_at", "search": "question_text"},
    "pasted_texts": {"pk": "text_hash",   "sort": "created_at", "search": "text_content"},
    "article_cache":{"pk": "url_hash",    "sort": "fetched_at", "search": "url"},
    "media_cache":  {"pk": "file_hash",   "sort": "created_at", "search": "media_type"}
}

# Helper to serialize dates/json
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

@admin_bp.route('/dashboard', methods=['GET'])
def admin_dashboard():
    return render_template('admin.html')

@admin_bp.route('/get_tables', methods=['GET'])
@admin_required
def get_tables():
    return jsonify({"tables": list(TABLE_SCHEMA.keys())})

@admin_bp.route('/fetch_table', methods=['GET'])
@admin_required
def fetch_table():
    table = request.args.get('table')
    page = int(request.args.get('page', 1))
    search_query = request.args.get('search', '').strip()
    per_page = 20
    offset = (page - 1) * per_page

    if table not in TABLE_SCHEMA:
        return jsonify({"error": "Invalid table"}), 400

    config = TABLE_SCHEMA[table]
    pk_col = config['pk']
    sort_col = config['sort']
    search_col = config.get('search')

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:
            # Build Query
            base_query = f"FROM {table}"
            params = []

            if search_query and search_col:
                base_query += f" WHERE {search_col} ILIKE %s"
                params.append(f"%{search_query}%")

            # Count Total
            c.execute(f"SELECT COUNT(*) {base_query}", tuple(params))
            total = c.fetchone()[0]

            # Fetch Rows
            sql = f"SELECT * {base_query} ORDER BY {sort_col} DESC LIMIT %s OFFSET %s"
            params.extend([per_page, offset])

            c.execute(sql, tuple(params))
            rows = c.fetchall()

            # Convert to dict and inject _pk_val for frontend generic deletion
            data = []
            for row in rows:
                row_dict = dict(row)
                row_dict['_pk_val'] = str(row_dict.get(pk_col)) # Inject PK for frontend
                data.append(row_dict)

            return json.dumps({
                "data": data,
                "page": page,
                "total": total,
                "total_pages": (total // per_page) + 1 if total > 0 else 1
            }, cls=DateEncoder) # Use custom encoder for datetimes
    finally:
        conn.close()

@admin_bp.route('/delete_row', methods=['POST'])
@admin_required
def delete_row():
    data = request.json or {}
    table = data.get('table')
    pk_val = data.get('pk_val')

    if table not in TABLE_SCHEMA:
        return jsonify({"error": "Invalid table"}), 400

    config = TABLE_SCHEMA[table]
    pk_col = config['pk']

    conn = get_db()
    try:
        with conn.cursor() as c:
            # ⚠️ DANGER: Executes delete
            c.execute(f"DELETE FROM {table} WHERE {pk_col} = %s", (pk_val,))
            conn.commit()
            if c.rowcount == 0:
                return jsonify({"error": "Row not found"}), 404
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

    return jsonify({"status": "deleted", "id": pk_val})

@admin_bp.route('/clear_table', methods=['POST'])
@admin_required
def clear_table():
    data = request.json or {}
    table = data.get('table')

    if table not in TABLE_SCHEMA:
        return jsonify({"error": "Invalid table"}), 400

    conn = get_db()
    try:
        with conn.cursor() as c:
            # ⚠️ HIGH DANGER: Truncates table
            c.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
            conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

    return jsonify({"status": "cleared", "table": table})

@admin_bp.route('/publish_analysis', methods=['POST'])
@admin_required
def publish_analysis():
    data = request.json or {}
    analysis_id = data.get('analysis_id')
    title = data.get('title')
    slug = data.get('slug')
    summary = data.get('summary')
    image_url = data.get('image_url')
    action = data.get('action')

    if not analysis_id: return jsonify({"error": "Missing ID"}), 400

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as c:

            # --- 1. CHECK CURRENT STATUS ---
            c.execute("SELECT is_published, published_slug FROM analyses WHERE analysis_id = %s", (analysis_id,))
            current_state = c.fetchone()

            if not current_state:
                return jsonify({"error": "Analysis not found"}), 404

            is_already_published = current_state['is_published']
            old_slug = current_state['published_slug']

            # --- 2. DELETE / UNPUBLISH ---
            if action == 'unpublish':
                if old_slug:
                    c.execute("DELETE FROM published_articles WHERE slug = %s", (old_slug,))
                c.execute("UPDATE analyses SET is_published = FALSE, published_slug = NULL, published_at = NULL WHERE analysis_id = %s", (analysis_id,))
                conn.commit()
                return jsonify({"status": "unpublished"})

            # --- 3. PREPARE DATA ---
            if not title or not slug:
                return jsonify({"error": "Title and Slug required"}), 400

            # Only fetch VERIFIED claims
            c.execute("""
                SELECT claim_text, final_verdict, synthesis_summary, category, tags
                FROM claims
                WHERE analysis_id = %s
                AND final_verdict IS NOT NULL
                AND final_verdict != ''
                ORDER BY ordinal
            """, (analysis_id,))
            rows = c.fetchall()

            if not rows:
                return jsonify({"error": "No verified claims found to publish."}), 400

            full_content = []
            all_tags = set()
            all_cats = set()

            for r in rows:
                if r['tags']:
                    try: all_tags.update(json.loads(r['tags']))
                    except: pass
                if r['category']: all_cats.add(r['category'])

                import hashlib
                ch = hashlib.sha256(r['claim_text'].strip().lower().encode('utf-8')).hexdigest()

                c.execute("SELECT verdict, questions_json, used_model FROM model_cache WHERE claim_hash=%s", (ch,))
                mv = c.fetchone()
                c.execute("SELECT verdict, sources_json FROM external_cache WHERE claim_hash=%s", (ch,))
                ev = c.fetchone()

                deep_dives = []
                if mv and mv['questions_json']:
                    try:
                        q_list = json.loads(mv['questions_json'])
                        for q_txt in q_list:
                            rq_hash = hashlib.sha256((r['claim_text'].strip().lower() + "||" + q_txt.strip().lower()).encode('utf-8')).hexdigest()
                            c.execute("SELECT report_text, used_model FROM report_cache WHERE rq_hash=%s", (rq_hash,))
                            rep = c.fetchone()
                            if rep:
                                deep_dives.append({
                                    "question": q_txt,
                                    "model": rep['used_model'],
                                    "content": rep['report_text']
                                })
                    except: pass

                full_content.append({
                    "text": r['claim_text'],
                    "verdict": r['final_verdict'],
                    "summary": r['synthesis_summary'],
                    "category": r['category'],
                    "tags": json.loads(r['tags']) if r['tags'] else [],
                    "internal_verdict": mv['verdict'] if mv else None,
                    "internal_model": mv['used_model'] if mv else None,
                    "external_verdict": ev['verdict'] if ev else None,
                    "sources": json.loads(ev['sources_json']) if (ev and ev['sources_json']) else [],
                    "deep_dives": deep_dives
                })

            # --- 4. EXECUTE DB UPDATE (Smart Upsert) ---

            if is_already_published and old_slug:
                # UPDATE existing row (PRESERVE ORIGINAL DATE)
                # ✅ FIX: Removed 'published_at = CURRENT_TIMESTAMP' from this query
                c.execute("""
                    UPDATE published_articles SET
                        slug = %s,
                        title = %s,
                        summary = %s,
                        image_url = %s,
                        tags = %s,
                        categories = %s,
                        content_snapshot = %s
                    WHERE slug = %s
                """, (
                    slug, title, summary, image_url,
                    json.dumps(list(all_tags)), json.dumps(list(all_cats)),
                    json.dumps(full_content),
                    old_slug
                ))
            else:
                # INSERT new row (SET DATE NOW)
                c.execute("""
                    INSERT INTO published_articles (
                        slug, title, summary, image_url, tags, categories, content_snapshot, published_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    slug, title, summary, image_url,
                    json.dumps(list(all_tags)), json.dumps(list(all_cats)),
                    json.dumps(full_content)
                ))

            # 5. SYNC LIVE TABLE
            # ✅ FIX: Use COALESCE to keep existing date if present
            c.execute("""
                UPDATE analyses
                SET is_published = TRUE,
                    published_slug = %s,
                    published_title = %s,
                    published_summary = %s,
                    published_image_url = %s,
                    published_at = COALESCE(published_at, CURRENT_TIMESTAMP)
                WHERE analysis_id = %s
            """, (slug, title, summary, image_url, analysis_id))

            conn.commit()
            return jsonify({"status": "success"})

    except Exception as e:
        conn.rollback()
        if "unique constraint" in str(e).lower():
             return jsonify({"error": "Slug already exists on another article."}), 400
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@admin_bp.route('/upload_cover', methods=['POST'])
@admin_required
def upload_cover():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # 1. Check Extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            return jsonify({"error": "Invalid image format"}), 400

        # 2. Generate Hash of Content (Deduplication Logic)
        file_content = file.read() # Read file into memory
        file_hash = hashlib.md5(file_content).hexdigest()

        # Reset file pointer so we can save it if needed (though we write bytes directly below)
        file.seek(0)

        # 3. Create Filename based on Hash
        filename = f"cover_{file_hash}{ext}"

        # 4. Determine Path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upload_folder = os.path.join(current_dir, 'templates', 'static', 'covers')
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, filename)

        # 5. Save ONLY if it doesn't exist (Prevents Duplicates)
        if not os.path.exists(filepath):
            with open(filepath, "wb") as f:
                f.write(file_content)

        # 6. Return URL
        url = f"/static/covers/{filename}"

        return jsonify({"url": url})
