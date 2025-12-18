from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Literal, Any, Dict
from datetime import datetime, timezone, date 
# ----------------------
# Helper function
# ----------------------
def today_iso():
    return datetime.now(timezone.utc).date().isoformat()
# ----------------------
# Models & helpers for Auditee auth/create
# ----------------------
# Define AuditeeOut FIRST (because other models reference it)
class AuditeeOut(BaseModel):
    id: int
    first_name: str
    email: EmailStr
    function: Optional[str] = None
    plant_name: Optional[str] = None
    dept_name: Optional[str] = None
    manager_email: Optional[EmailStr] = None

class AuditeePrecheckIn(BaseModel):
    first_name: str
    email: EmailStr

class AuditeePrecheckOut(BaseModel):
    ok: bool
    today: str
    exists: bool
    profile_incomplete: Optional[bool] = None
    auditee: Optional[AuditeeOut] = None  # Now AuditeeOut is defined above
    reason: Optional[str] = None

class AuthAuditeeOut(BaseModel):
    ok: bool
    today: str
    auditee: Optional[AuditeeOut] = None
    reason: Optional[str] = None

class AuditeeCreateIn(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=120)
    email: EmailStr
    function: Optional[str] = Field(None, max_length=120)
    plant_name: Optional[str] = Field(None, max_length=120)
    dept_name: Optional[str] = Field(None, max_length=120)
    manager_email: Optional[EmailStr] = None

class AuditeeCreateOut(BaseModel):
    ok: bool
    today: str
    auditee: AuditeeOut

class AuditStartIn(BaseModel):
    auditee_id: int
    type: str
    questionnaire_version: Optional[str] = None
    external_id: Optional[str] = None  # uuid from client for idempotency

class QuestionIn(BaseModel):
    text: str
    category: Optional[str] = None
    mandatory: Optional[bool] = True
    source_doc: Optional[str] = None

class QuestionsBulkIn(BaseModel):
    version_tag: str
    questions: List[QuestionIn]

class AnswerIn(BaseModel):
    question_id: int
    response_text: Optional[str] = ""
    is_compliant: Optional[bool] = None
    attempt_number: int = Field(1, ge=1, le=2)
    evidence_filename: Optional[str] = None  # Changed from evidence_url

class NonConformityIn(BaseModel):
    question_id: int
    description: str
    severity: Literal["minor", "major", "critical"] = "major"
    status: Literal["open", "in_progress", "closed"] = "open"
    responsible_id: Optional[int] = None
    due_date: Optional[date] = None
    evidence_url: Optional[str] = None
    closed_at: Optional[datetime] = None
    closure_comment: Optional[str] = None

class CompleteAuditIn(BaseModel):
    score_global: Optional[float] = None

class FileUploadPayload(BaseModel):
    action_plan_id: int
    filename: str
    filetype: str
    content: str

class AuthCheckIn(BaseModel):
    name: str
    code: str

class AuthCheckOut(BaseModel):
    ok: bool
    reason: Optional[str] = None  # Changed from str | None for Python 3.9 compatibility

class AuditWithAnswersOut(BaseModel):
    audit_id: int
    audit_type: str
    status: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    score_global: Optional[float] = None
    questionnaire_version: Optional[str] = None
    auditee_id: int
    auditee_name: str
    auditee_email: str
    answer_count: int
    answers: List[Dict[str, Any]] = []

class AuditeeAuditsOut(BaseModel):

    ok: bool
    auditee_name: str
    total_audits: int
    audits: List[AuditWithAnswersOut]

# ----------------------
# Models (conversations)
# ----------------------
class ConversationIn(BaseModel):
    user_name: str = Field(..., min_length=1, max_length=200)
    conversation: str = Field(..., min_length=1)
    date_conversation: Optional[datetime] = None
    assistant_name: Optional[str] = None

class ConversationOut(BaseModel):
    id: int
    status: str = "ok"

class ConversationSummary(BaseModel):
    id: int
    user_name: str
    date_conversation: datetime
    preview: str
    assistant_name: Optional[str] = None

class ConversationDetail(BaseModel):
    id: int
    user_name: str
    date_conversation: datetime
    conversation: str
    assistant_name: Optional[str] = None

# -------------------------------------------------
# Models & helpers for Sales
# -------------------------------------------------
class ObjectionOut(BaseModel):
    id: int
    customer_concern: str
    example_customer_argument: str
    recommended_response: str
    category: Optional[str] = None

    class Config:
        orm_mode = True

class MatrixOut(BaseModel):
    id: int
    freeze_time_respected: bool
    demand_vs_moq: str
    inventory_vs_demand: str
    recommended_strategy: str

    class Config:
        orm_mode = True
