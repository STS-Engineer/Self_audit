from fastapi import FastAPI, HTTPException, File, Form, UploadFile, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, date, timezone
from pathlib import Path
import os
import uuid
import base64
import io
import mimetypes
import json
import requests
import psycopg2.extras
from models import (
    AuditeeCreateIn,
    AuditeeCreateOut,
    AuthAuditeeOut,
    today_iso,
    AuditStartIn,
    QuestionsBulkIn,
    AnswerIn,
    NonConformityIn,
    CompleteAuditIn,
    ObjectionOut,
    MatrixOut,
    AuditeePrecheckIn,
    AuditeePrecheckOut,
    FileUploadPayload,
    AuthCheckIn,
    AuthCheckOut,
    ConversationIn,
    ConversationOut,
    ConversationSummary,
    ConversationDetail,
)
from typing import Optional, List
from db import get_connection, get_connection_sales

app = FastAPI()
# Create uploads directory for evidence images
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# =======================================
# EMAIL CONFIG 
# =======================================
SMTP_SERVER = "avocarbon-com.mail.protection.outlook.com"
SMTP_PORT = 25
EMAIL_USER = "administration.STS@avocarbon.com"
EMAIL_PASSWORD = "shnlgdyfbcztbhxn"
# ---------------------------------------------------------------------
# 1. Auth simple: /auth/check (lecture DB name+code)
# ---------------------------------------------------------------------
@app.post("/auth/check", response_model=AuthCheckOut)
def auth_check(payload: AuthCheckIn):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # On ne révèle pas si le name existe : réponse générique
        GENERIC_FAIL = {"ok": False, "reason": "Invalid name or code"}

        # Lecture stricte par name
        cur.execute(
            """
            SELECT code, is_active, expires_at
            FROM access_codes
            WHERE name = %s
            LIMIT 1
            """,
            (payload.name,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        conn = None

        if not row:
            return GENERIC_FAIL

        db_code, is_active, expires_at = row

        # Vérifs état / expiration
        if not is_active:
            return {"ok": False, "reason": "Access disabled"}

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        if expires_at is not None and expires_at <= now:
            return {"ok": False, "reason": "Code expired"}

        # Comparaison en clair (POC)
        if payload.code != db_code:
            return GENERIC_FAIL

        # OK
        return {"ok": True}

    except Exception as e:
        if conn:
            conn.close()
        # On garde 200 pour simplicité côté GPT, mais on peut aussi lever 500
        return {"ok": False, "reason": f"Server error"}
# ------------------------------------------------------------------------------------------------
# 2) POST /auditees/precheck (auth by first_name + email)
# ------------------------------------------------------------------------------------------------
@app.post("/auditees/precheck", response_model=AuditeePrecheckOut, status_code=200)
def auditee_precheck(payload: AuditeePrecheckIn):
    """
    Step A: Profile Pre-Check.
    - Input: first_name + email
    - If auditee exists:
        * Return profile.
        * Flag profile_incomplete if plant/dept are missing.
    - If not exists:
        * exists=false → client should collect full profile and call /auditees.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT id, first_name, email, "function",
                   plant_name, dept_name, manager_email
            FROM auditees
            WHERE lower(email) = lower(%s)
            LIMIT 1
        """, (payload.email,))
        row = cur.fetchone()

        if not row:
            cur.close()
            conn.close()
            return {
                "ok": True,
                "today": today_iso(),
                "exists": False,
                "reason": "No profile was found for this email."
            }

        (
            aid, db_first_name, db_email, db_function,
            plant_name, dept_name, manager_email
        ) = row

        incoming_first = payload.first_name.strip()
        if incoming_first and incoming_first != db_first_name:
            cur.execute("""
                UPDATE auditees
                SET first_name = %s
                WHERE id = %s
            """, (incoming_first, aid))
            conn.commit()
            db_first_name = incoming_first

        cur.close()
        conn.close()

        # Profile completeness check
        profile_incomplete = not (db_first_name and db_email)

        return {
            "ok": True,
            "today": today_iso(),
            "exists": True,
            "profile_incomplete": profile_incomplete,
            "auditee": {
                "id": aid,
                "first_name": db_first_name,
                "email": db_email,
                "function": db_function,
                "plant_name": plant_name,
                "dept_name": dept_name,
                "manager_email": manager_email,
            },
        }

    except Exception as e:
        if conn:
            conn.close()
        return {
            "ok": False,
            "today": today_iso(),
            "exists": False,
            "reason": f"Server error: {e}"
        }

# ------------------------------------------------------------------------------------------------
# 3) GET /auditees/check (auth by first_name + email)
# ------------------------------------------------------------------------------------------------
@app.get("/auditees/check", response_model=AuthAuditeeOut)
def auditee_check(first_name: str, email: EmailStr, code: str):
    """
    Auth: first_name (case-insensitive) + email (case-insensitive) + code (exact)
    - If match: ok=true and return profile
    - If not found or first_name mismatch: ok=false with reason
    - Always returns 'today' (UTC) for assistant to use as audit date
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # 1) Find by email + code
        cur.execute("""
            SELECT id, first_name, email, "function",
                   plant_name, dept_name, manager_email, code
            FROM auditees
            WHERE lower(email) = lower(%s)
              AND code = %s
            LIMIT 1
        """, (email, code))
        row = cur.fetchone()

        if not row:
            cur.close()
            conn.close()
            return {
                "ok": False,
                "today": today_iso(),
                "reason": "Not found for provided email+code"
            }

        (aid, db_first_name, db_email, db_function,
         plant_name, dept_name, manager_email, db_code) = row

        # 2) Verify first_name (case-insensitive)
        incoming_first = (first_name or "").strip()
        if not incoming_first or incoming_first.casefold() != (db_first_name or "").strip().casefold():
            cur.close()
            conn.close()
            return {
                "ok": False,
                "today": today_iso(),
                "reason": "First name does not match this email+code"
            }

        # 3) Optional: cheap sync display of first_name (e.g., capitalization/spacing)
        if incoming_first != db_first_name:
            cur.execute("""
                UPDATE auditees
                SET first_name = %s
                WHERE id = %s
            """, (incoming_first, aid))
            conn.commit()
            db_first_name = incoming_first

        cur.close()
        conn.close()

        return {
            "ok": True,
            "today": today_iso(),
            "auditee": {
                "id": aid,
                "first_name": db_first_name,
                "email": db_email,
                "function": db_function,
                "plant_name": plant_name,
                "dept_name": dept_name,
                "manager_email": manager_email,
                "code": db_code,
            }
        }

    except Exception as e:
        if conn:
            conn.close()
        # Keep 200 so the assistant handles uniformly
        return {"ok": False, "today": today_iso(), "reason": f"Server error: {e}"}

# ----------------------
# 4) POST /auditees (create or update full profile)
# ----------------------
@app.post("/auditees", response_model=AuditeeCreateOut, status_code=200)
def create_or_update_auditee(payload: AuditeeCreateIn):
    """
    Upsert rule:
      - If email exists -> update provided fields (non-null keep existing via COALESCE)
      - If not -> insert new row
    Returns the full profile + today's date for the audit.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Normalize inputs (preserve first_name spelling, just trim spaces)
        email_val = payload.email.strip()
        first_name_val = payload.first_name.strip()
        function_val = payload.function.strip() if payload.function else None
        plant_name_val = payload.plant_name.strip() if payload.plant_name else None
        dept_name_val = payload.dept_name.strip() if payload.dept_name else None
        manager_email_val = payload.manager_email.strip() if payload.manager_email else None

        # Exists?
        cur.execute(
            """
            SELECT id FROM auditees
            WHERE lower(email) = lower(%s)
            LIMIT 1
            """,
            (email_val,),
        )
        hit = cur.fetchone()

        if hit:
            aid = hit[0]
            cur.execute(
                """
                UPDATE auditees
                SET first_name = COALESCE(%s, first_name),
                    "function" = COALESCE(%s, "function"),
                    plant_name = COALESCE(%s, plant_name),
                    dept_name = COALESCE(%s, dept_name),
                    manager_email = COALESCE(%s, manager_email)
                WHERE id = %s
                RETURNING id, first_name, email, "function",
                          plant_name, dept_name, manager_email
                """,
                (
                    first_name_val,
                    function_val,
                    plant_name_val,
                    dept_name_val,
                    manager_email_val,
                    aid,
                ),
            )
            row = cur.fetchone()
        else:
            cur.execute(
                """
                INSERT INTO auditees (
                    first_name, email, "function",
                    plant_name, dept_name, manager_email
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, first_name, email, "function",
                          plant_name, dept_name, manager_email
                """,
                (
                    first_name_val,
                    email_val,
                    function_val,
                    plant_name_val,
                    dept_name_val,
                    manager_email_val,
                ),
            )
            row = cur.fetchone()

        conn.commit()
        cur.close()
        conn.close()

        (
            aid,
            first_name,
            email,
            function,
            plant_name,
            dept_name,
            manager_email,
        ) = row

        return {
            "ok": True,
            "today": today_iso(),
            "auditee": {
                "id": aid,
                "first_name": first_name,
                "email": email,
                "function": function,
                "plant_name": plant_name,
                "dept_name": dept_name,
                "manager_email": manager_email,
            },
        }

    except Exception:
        if conn:
            conn.rollback()
            conn.close()
        raise HTTPException(status_code=500, detail="Failed to upsert auditee.")

# ----------------------
# 5) POST /questions/bulk
# ----------------------
@app.post("/questions/bulk")
def questions_bulk_upsert(payload: QuestionsBulkIn):
    """
    For each question in order, if (version_tag, text) exists => return existing question_id,
    else insert and return the new question_id. Response order matches input order.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        out_items = []

        for idx, q in enumerate(payload.questions):
            # 1) try to find existing
            cur.execute("""
                SELECT question_id FROM questions
                 WHERE version_tag = %s AND text = %s
                 LIMIT 1
            """, (payload.version_tag, q.text))
            hit = cur.fetchone()

            if hit:
                qid = hit[0]
            else:
                # 2) insert
                cur.execute("""
                    INSERT INTO questions (text, category, mandatory, source_doc, version_tag, created_at)
                    VALUES (%s, %s, %s, %s, %s, now())
                    RETURNING question_id
                """, (q.text, q.category, q.mandatory, q.source_doc, payload.version_tag))
                qid = cur.fetchone()[0]

            out_items.append({"index": idx, "question_id": qid})

        conn.commit()
        cur.close()
        conn.close()
        conn = None
        return {"ok": True, "version_tag": payload.version_tag, "items": out_items}

    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to upsert questions: {e}")

# ----------------------
# 6) POST /audits/{audit_id}/answers (UPDATED - with image upload)
# ----------------------
@app.post("/audits/{audit_id}/answers")
async def save_answer(
    audit_id: int,
    question_id: int = Form(...),
    response_text: str = Form(""),
    is_compliant: bool = Form(None),
    attempt_number: int = Form(1),
    evidence_image: UploadFile = File(None)
):
    """
    Save answer with optional evidence image upload.
    Now accepts multipart/form-data instead of JSON.
    If evidence_image is provided, it will be saved to uploads folder.
    """
    conn = None
    evidence_filename = None
    
    try:
        # Handle file upload if provided
        if evidence_image and evidence_image.filename:
            # Validate file type (images only)
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            file_ext = os.path.splitext(evidence_image.filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
                )
            
            # Generate unique filename to avoid conflicts
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = UPLOAD_DIR / unique_filename
            
            # Save file to disk
            with open(file_path, "wb") as f:
                content = await evidence_image.read()
                f.write(content)
            
            evidence_filename = unique_filename
        
        # Save to database
        conn = get_connection()
        cur = conn.cursor()

        # Try update first (unique key: audit_id, question_id, attempt_number)
        cur.execute("""
            UPDATE answers
               SET response_text = %s,
                   is_compliant = %s,
                   evidence_filename = %s
             WHERE audit_id = %s AND question_id = %s AND attempt_number = %s
         RETURNING answer_id
        """, (
            response_text, is_compliant, evidence_filename,
            audit_id, question_id, attempt_number
        ))
        row = cur.fetchone()

        if not row:
            # Insert new answer if doesn't exist
            cur.execute("""
                INSERT INTO answers (
                    audit_id, question_id, response_text, 
                    is_compliant, attempt_number, evidence_filename, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, now())
                RETURNING answer_id
            """, (
                audit_id, question_id, response_text,
                is_compliant, attempt_number, evidence_filename
            ))
            row = cur.fetchone()

        conn.commit()
        answer_id = row[0]
        cur.close()
        conn.close()
        conn = None
        
        return {
            "ok": True, 
            "answer_id": answer_id,
            "evidence_filename": evidence_filename
        }

    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        # Clean up uploaded file if database operation failed
        if evidence_filename:
            file_path = UPLOAD_DIR / evidence_filename
            if file_path.exists():
                file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save answer: {e}")

# ----------------------
# 6b) GET /audits/answers/{answer_id}/evidence (NEW - retrieve evidence image)
# ----------------------
@app.get("/audits/answers/{answer_id}/evidence")
def get_evidence_image(answer_id: int):
    """
    Download the evidence image for a specific answer.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT evidence_filename
            FROM answers
            WHERE answer_id = %s
        """, (answer_id,))
        
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if not row or not row[0]:
            raise HTTPException(status_code=404, detail="No evidence image found for this answer")
        
        filename = row[0]
        file_path = UPLOAD_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Evidence file not found on server")
        
        # Return the image file
        return FileResponse(
            path=file_path,
            media_type=f"image/{os.path.splitext(filename)[1][1:]}",
            filename=filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve evidence: {e}")

# ----------------------
# 7) GET /audits/{audit_id}/answers (UPDATED - includes evidence info)
# ----------------------
@app.get("/audits/{audit_id}/answers")
def get_answers(audit_id: int):
    """
    Get all answers for a given audit_id, linked with the auditee.
    Now includes evidence_filename and evidence_url for retrieval.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # CORRECTION: au.id au lieu de au.audit_id
        cur.execute("""
            SELECT a.answer_id,
                   a.audit_id,
                   q.text AS question_text,
                   a.response_text,
                   a.is_compliant,
                   a.attempt_number,
                   a.evidence_filename,
                   a.created_at,
                   au.auditee_id,
                   CONCAT(aue.first_name, ' ', COALESCE(aue.email, '')) AS auditee_name
            FROM answers a
            JOIN audits au ON a.audit_id = au.id
            JOIN questions q ON a.question_id = q.question_id
            LEFT JOIN auditees aue ON au.auditee_id = aue.id
            WHERE a.audit_id = %s
            ORDER BY q.question_id, a.attempt_number
        """, (audit_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        conn = None

        answers = []
        for r in rows:
            answer_data = {
                "answer_id": r[0],
                "audit_id": r[1],
                "question_text": r[2],
                "response_text": r[3],
                "is_compliant": r[4],
                "attempt_number": r[5],
                "evidence_filename": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
                "auditee_id": r[8],
                "auditee_name": r[9],
            }
            # Add evidence URL if file exists
            if r[6]:  # if evidence_filename exists
                answer_data["evidence_url"] = f"/audits/answers/{r[0]}/evidence"
            answers.append(answer_data)

        return {"ok": True, "audit_id": audit_id, "count": len(answers), "answers": answers}

    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to fetch answers: {e}")

# ----------------------
# 8) POST /audits/{audit_id}/nonconformities
# ----------------------
@app.post("/audits/{audit_id}/nonconformities")
def save_nc(audit_id: int, payload: NonConformityIn):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO non_conformities (
                audit_id, question_id, description, severity, status,
                responsible_id, due_date, evidence_url, closed_at, closure_comment, detected_at
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
            RETURNING nc_id
        """, (
            audit_id, payload.question_id, payload.description, payload.severity, payload.status,
            payload.responsible_id, payload.due_date, payload.evidence_url, payload.closed_at, payload.closure_comment
        ))
        nc_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        conn = None
        return {"ok": True, "nc_id": nc_id}

    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to save non-conformity: {e}")

# ----------------------
# 9) POST /audits/{audit_id}/complete
# ----------------------
@app.post("/audits/{audit_id}/complete")
def complete_audit(audit_id: int, payload: CompleteAuditIn):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        score_value = payload.score_global

        if score_value is None:
            # compute % compliant questions: any attempt true per question
            cur.execute("""
                WITH per_q AS (
                  SELECT question_id, bool_or(is_compliant) AS compliant
                    FROM answers
                   WHERE audit_id = %s
                GROUP BY question_id
                )
                SELECT
                  COALESCE(SUM(CASE WHEN compliant THEN 1 ELSE 0 END),0)::float,
                  COALESCE(COUNT(*),0)::float
                FROM per_q
            """, (audit_id,))
            srow = cur.fetchone()
            numer, denom = (srow or (0.0, 0.0))
            score_value = round((numer / denom) * 100.0, 2) if denom > 0 else 0.0

        cur.execute("""
            UPDATE audits
               SET status = 'completed',
                   ended_at = now(),
                   score_global = %s
             WHERE id = %s
         RETURNING id, status, ended_at, score_global
        """, (score_value, audit_id))
        row = cur.fetchone()
        conn.commit()

        if not row:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Audit not found")

        (aid, status, ended_at, score_global) = row
        cur.close()
        conn.close()
        conn = None
        return {
            "id": aid,
            "status": status,
            "ended_at": ended_at,
            "score_global": float(score_global) if score_global is not None else None
        }

    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to complete audit: {e}")

# ----------------------
# 10) GET /objections
# ----------------------
@app.get("/objections", response_model=list[ObjectionOut])
def get_objections(
    category: str | None = Query(None, description="Filter by category (e.g. 'Lead Time', 'MOQ')"),
    q: str | None = Query(None, description="Full-text search in concern/argument/response"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    conn = None
    try:
        conn = get_connection_sales()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        sql = """
            SELECT id, customer_concern, example_customer_argument, recommended_response, category
            FROM customer_objection_handling
            WHERE 1=1
        """
        params: list = []

        if category:
            sql += " AND category = %s"
            params.append(category)

        if q:
            like = f"%{q}%"
            sql += """
                AND (
                    customer_concern ILIKE %s OR
                    example_customer_argument ILIKE %s OR
                    recommended_response ILIKE %s
                )
            """
            params.extend([like, like, like])

        sql += " ORDER BY id LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows

    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to fetch objections: {e}")

# ----------------------
# 11) GET /matrix
# ----------------------
@app.get("/matrix", response_model=list[MatrixOut])
def get_matrix(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    conn = None
    try:
        conn = get_connection_sales()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT
                id,
                freeze_time_respected,
                demand_vs_moq,
                inventory_vs_demand,
                recommended_strategy
            FROM customer_handling_matrix
            ORDER BY id
            LIMIT %s OFFSET %s
        """, (limit, offset))

        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows

    except psycopg2.Error as db_err:
        if conn:
            conn.close()
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(db_err)}"
        )
    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch matrix: {str(e)}"
        )
# ----------------------
# 12) GET /auditees/audits-by-name (NEW - Get all audits and answers by auditee name)
# ----------------------
@app.get("/auditees/audits-by-name")
def get_audits_by_auditee_name(
    name: str = Query(..., description="Auditee first name (case-insensitive, partial match)")
):
    """
    Get all audits and their answers for a given auditee by name.
    Searches by first_name (case-insensitive, partial match).
    Returns complete audit information with all answers.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Step 1: Find auditees matching the name (case-insensitive, partial match)
        cur.execute("""
            SELECT id, first_name, email, function, plant_name, dept_name, manager_email
            FROM auditees
            WHERE LOWER(first_name) LIKE LOWER(%s)
            ORDER BY first_name
        """, (f"%{name}%",))
        
        auditees = cur.fetchall()
        
        if not auditees:
            cur.close()
            conn.close()
            return {
                "ok": False,
                "auditee_name": name,
                "total_audits": 0,
                "audits": [],
                "message": "No auditee found with that name"
            }

        all_audits = []
        
        # Step 2: For each auditee, get their audits
        for auditee in auditees:
            auditee_id, first_name, email, function, plant_name, dept_name, manager_email = auditee
            
            # Get all audits for this auditee
            cur.execute("""
                SELECT id, auditee_id, type, status, started_at, ended_at, 
                       score_global, questionnaire_version, external_id
                FROM audits
                WHERE auditee_id = %s
                ORDER BY started_at DESC
            """, (auditee_id,))
            
            audits = cur.fetchall()
            
            # Step 3: For each audit, get all answers
            for audit in audits:
                (audit_id, aid, audit_type, status, started_at, ended_at, 
                 score_global, questionnaire_version, external_id) = audit
                
                # Get all answers for this audit
                cur.execute("""
                    SELECT a.answer_id,
                           a.audit_id,
                           a.question_id,
                           q.text AS question_text,
                           q.category AS question_category,
                           q.mandatory AS question_mandatory,
                           a.response_text,
                           a.is_compliant,
                           a.attempt_number,
                           a.evidence_filename,
                           a.created_at
                    FROM answers a
                    JOIN questions q ON a.question_id = q.question_id
                    WHERE a.audit_id = %s
                    ORDER BY q.question_id, a.attempt_number
                """, (audit_id,))
                
                answer_rows = cur.fetchall()
                
                answers_list = []
                for ans in answer_rows:
                    answer_data = {
                        "answer_id": ans[0],
                        "audit_id": ans[1],
                        "question_id": ans[2],
                        "question_text": ans[3],
                        "question_category": ans[4],
                        "question_mandatory": ans[5],
                        "response_text": ans[6],
                        "is_compliant": ans[7],
                        "attempt_number": ans[8],
                        "evidence_filename": ans[9],
                        "created_at": ans[10].isoformat() if ans[10] else None,
                    }
                    # Add evidence URL if file exists
                    if ans[9]:
                        answer_data["evidence_url"] = f"/audits/answers/{ans[0]}/evidence"
                    answers_list.append(answer_data)
                
                # Build audit object with answers
                audit_data = {
                    "audit_id": audit_id,
                    "audit_type": audit_type,
                    "status": status,
                    "started_at": started_at.isoformat() if started_at else None,
                    "ended_at": ended_at.isoformat() if ended_at else None,
                    "score_global": float(score_global) if score_global is not None else None,
                    "questionnaire_version": questionnaire_version,
                    "external_id": external_id,
                    "auditee_id": auditee_id,
                    "auditee_name": first_name,
                    "auditee_email": email,
                    "auditee_function": function,
                    "auditee_plant": plant_name,
                    "auditee_dept": dept_name,
                    "answer_count": len(answers_list),
                    "answers": answers_list
                }
                all_audits.append(audit_data)
        
        cur.close()
        conn.close()
        conn = None
        
        return {
            "ok": True,
            "auditee_name": name,
            "total_audits": len(all_audits),
            "audits": all_audits
        }

    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to fetch audits by name: {e}")
    
    # ---------------------------
# Save conversation
# ---------------------------
@app.post("/save-conversation", response_model=ConversationOut)
def save_conversation(payload: ConversationIn):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        date_conv = payload.date_conversation or datetime.now(timezone.utc)
        cur.execute(
            """
            INSERT INTO conversations (user_name, conversation, date_conversation, assistant_name)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (payload.user_name.strip(), payload.conversation, date_conv, payload.assistant_name),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        conn = None
        return ConversationOut(id=new_id, status="ok")
    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        raise HTTPException(status_code=500, detail=f"Insertion failed: {e}")
# ---------------------------
# List conversations with filters
# ---------------------------
@app.get("/conversations")
def list_conversations(
    date: Optional[str] = Query(None, description="YYYY-MM-DD (UTC)"),
    user_name: Optional[str] = None,
    assistant_name: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        where = []
        params = []
        
        if date:
            where.append("DATE(date_conversation AT TIME ZONE 'UTC') = %s")
            params.append(date)
        if user_name:
            where.append("LOWER(user_name) LIKE %s")
            params.append(f"%{user_name.lower()}%")
        if assistant_name:
            where.append("LOWER(assistant_name) LIKE %s")
            params.append(f"%{assistant_name.lower()}%")
        
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        
        cur.execute(
            f"""
            SELECT id, user_name, date_conversation, conversation, assistant_name
            FROM conversations
            {where_sql}
            ORDER BY date_conversation DESC, id DESC
            LIMIT %s OFFSET %s;
            """,
            (*params, limit, offset),
        )
        rows = cur.fetchall()
        
        cur.execute(f"SELECT COUNT(*) FROM conversations {where_sql};", tuple(params))
        total = cur.fetchone()[0]
        
        items = []
        for (cid, uname, dconv, conv, aname) in rows:
            preview = (conv[:140] + "...") if len(conv) > 140 else conv
            items.append(ConversationSummary(
                id=cid,
                user_name=uname,
                date_conversation=dconv,
                preview=preview,
                assistant_name=aname
            ))
        
        cur.close()
        conn.close()
        conn = None
        return {"items": items, "total": total}
    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
# ---------------------------
# Get conversation by id
# ---------------------------
@app.get("/conversations/{id}", response_model=ConversationDetail)
def get_conversation_by_id(id: int):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, user_name, date_conversation, conversation, assistant_name
            FROM conversations WHERE id=%s;
            """,
            (id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        conn = None
        
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationDetail(
            id=row[0],
            user_name=row[1],
            date_conversation=row[2],
            conversation=row[3],
            assistant_name=row[4]
        )
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
# ---------------------------
# Get all conversations by user_name (case-insensitive LIKE)
# ---------------------------
@app.get("/conversations/user/{user_name}")
def get_conversations_by_user(

    user_name: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            SELECT
                id,
                user_name,
                assistant_name,
                date_conversation,
                conversation,
                COUNT(*) OVER() AS total_count
            FROM conversations
            WHERE LOWER(user_name) LIKE %s
            ORDER BY date_conversation DESC, id DESC
            LIMIT %s OFFSET %s;
            """,
            (f"%{user_name.lower()}%", limit, offset),
        )
        rows = cur.fetchall()
        
        items = []
        total = 0
        for (cid, uname, aname, dconv, conv, tot) in rows:
            total = tot
            preview = (conv[:160] + "…") if isinstance(conv, str) and len(conv) > 160 else conv
            items.append({
                "id": cid,
                "user_name": uname,
                "assistant_name": aname,
                "date_conversation": dconv,
                "preview": preview,
            })
        
        cur.close()
        conn.close()
        conn = None
        
        return {"items": items, "total": total if rows else 0}

    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    
    # ---------------------------
# Get all conversations by user_name AND assistant_name
# ---------------------------
@app.get("/conversations/user/{user_name}/assistant/{assistant_name}")
def get_conversations_by_user_and_assistant(
    user_name: str,
    assistant_name: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            SELECT
                id,
                user_name,
                assistant_name,
                date_conversation,
                conversation,
                COUNT(*) OVER() AS total_count
            FROM conversations
            WHERE LOWER(user_name) LIKE %s
              AND LOWER(assistant_name) LIKE %s
            ORDER BY date_conversation DESC, id DESC
            LIMIT %s OFFSET %s;
            """,
            (f"%{user_name.lower()}%", f"%{assistant_name.lower()}%", limit, offset),
        )
        rows = cur.fetchall()
        
        items = []
        total = 0
        for (cid, uname, aname, dconv, conv, tot) in rows:
            total = tot
            preview = (conv[:160] + "…") if isinstance(conv, str) and len(conv) > 160 else conv
            items.append({
                "id": cid,
                "user_name": uname,
                "assistant_name": aname,
                "date_conversation": dconv,
                "preview": preview,
            })
        
        cur.close()
        conn.close()
        conn = None
        
        return {"items": items, "total": total if rows else 0}

    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
