from flask import Blueprint, request, jsonify, render_template
import psycopg2
import psycopg2.extras
import os
import json
from datetime import datetime, date
from auth_module import get_current_user


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
    "analyses":     {"pk": "analysis_id", "sort": "created_at", "search": "canonical_text"},
    "claims":       {"pk": "claim_id",    "sort": "created_at", "search": "claim_text"},
    "model_cache":  {"pk": "claim_hash",  "sort": "updated_at", "search": "verdict"},
    "external_cache": {"pk": "claim_hash", "sort": "updated_at", "search": "verdict"},
    "report_cache": {"pk": "rq_hash",     "sort": "updated_at", "search": "question_text"},
    "users":        {"pk": "id",          "sort": "created_at", "search": "email"},
    "sessions":     {"pk": "id",          "sort": "created_at", "search": "session_token"},
    "magic_links":  {"pk": "id",          "sort": "created_at", "search": "token_hash"},
    "user_analyses":{"pk": "id",          "sort": "created_at", "search": "analysis_id"},
    "pasted_texts": {"pk": "text_hash",   "sort": "created_at", "search": "text_content"},
    "article_cache":{"pk": "url_hash",    "sort": "fetched_at", "search": "url"}
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
    image_url = data.get('image_url') # <--- Capture Image URL
    action = data.get('action') 

    if not analysis_id: return jsonify({"error": "Missing ID"}), 400

    conn = get_db()
    try:
        with conn.cursor() as c:
            if action == 'unpublish':
                c.execute("""
                    UPDATE analyses 
                    SET is_published = FALSE, published_at = NULL 
                    WHERE analysis_id = %s
                """, (analysis_id,))
            else:
                if not title or not slug: 
                    return jsonify({"error": "Title and Slug required"}), 400
                
                c.execute("SELECT 1 FROM analyses WHERE published_slug = %s AND analysis_id != %s", (slug, analysis_id))
                if c.fetchone():
                    return jsonify({"error": "Slug already exists"}), 400

                c.execute("""
                    UPDATE analyses 
                    SET is_published = TRUE, 
                        published_title = %s, 
                        published_slug = %s,
                        published_summary = %s,
                        published_image_url = %s, 
                        published_at = COALESCE(published_at, CURRENT_TIMESTAMP)
                    WHERE analysis_id = %s
                """, (title, slug, summary, image_url, analysis_id)) # <--- Save it
            
            conn.commit()
            return jsonify({"status": "success"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()