"""Employee Attendance & Analytics API (FastAPI + MongoDB)."""
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Literal, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from pymongo import MongoClient, ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError
from bson import ObjectId, json_util

# Load local .env settings. Environment variables still take priority.
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=3000)
db = client[os.getenv("MONGO_DB", "attendance_db")]
# Shift times and attendance dates use Indian Standard Time.
IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
PRESENCE = ("PRESENT", "WFH", "ON_DUTY")
STATUSES = (*PRESENCE, "ABSENT", "LEAVE")
EPOCH_MIN, EPOCH_MAX = 100_000_000_000, 4_102_444_800_000

app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0")


@app.on_event("startup")

def create_indexes():
    # These indexes speed up common queries and prevent duplicate records.
    db.employees.create_index([("emp_code", ASCENDING)], unique=True, name="uq_employee_code")
    db.employees.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="department_employee_code")
    db.employees.create_index([("joined_on", ASCENDING), ("department", ASCENDING)], name="joined_department")
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_attendance_employee_date")
    db.attendance_logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="attendance_date_employee")
    db.attendance_logs.create_index([("date", ASCENDING), ("status", ASCENDING)], name="attendance_date_status")
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING), ("status", ASCENDING)], name="attendance_employee_date_status")


# ------------------------------- validation ---------------------------------

def parse_date(value: str, field: str = "date") -> date:
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
        return parsed
    except (TypeError, ValueError):
        raise HTTPException(422, f"{field} must be YYYY-MM-DD")


def parse_month(value: str) -> tuple[date, date]:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise HTTPException(422, "month must be YYYY-MM")

    start = date.fromisoformat(value + "-01")

    # Find the first day of the next month, then step back one day.
    if start.month == 12:
        next_month = date(start.year + 1, 1, 1)
    else:
        next_month = date(start.year, start.month + 1, 1)

    end = next_month - timedelta(days=1)
    return start, end


def validate_epoch(value: Any, field: str = "punched_at") -> int:
    if type(value) is not int or not EPOCH_MIN <= value <= EPOCH_MAX:
        raise HTTPException(422, f"{field} must be an epoch-millisecond integer in range")
    return value


def epoch_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def from_epoch_ms(value: int) -> datetime:
    validate_epoch(value)
    return datetime.fromtimestamp(value / 1000, tz=UTC).replace(microsecond=0)


def utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(microsecond=0)
    return dt.astimezone(UTC).replace(tzinfo=None, microsecond=0)


def as_utc_aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def local_datetime(dt: datetime) -> datetime:
    return as_utc_aware(dt).astimezone(IST).replace(microsecond=0)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    local = local_datetime(punch_in)
    if shift_end <= shift_start and local.strftime("%H:%M") < shift_end:
        local -= timedelta(days=1)
    return local.date().isoformat()


def shift_start_datetime(punch_in: datetime, shift_start: str, shift_end: str) -> datetime:
    local = local_datetime(punch_in)
    day = local.date()
    if shift_end <= shift_start and local.strftime("%H:%M") < shift_end:
        day -= timedelta(days=1)
    h, m = map(int, shift_start.split(":"))
    return datetime.combine(day, time(h, m), IST)


def shift_end_datetime(punch_in: datetime, shift_start: str, shift_end: str) -> datetime:
    local = local_datetime(punch_in)
    start_day = date.fromisoformat(attendance_date(punch_in, shift_start, shift_end))
    h, m = map(int, shift_end.split(":"))
    end_day = start_day + timedelta(days=1) if shift_end <= shift_start else start_day
    return datetime.combine(end_day, time(h, m), IST)


def compute_late_minutes(punch_in: datetime, shift_start: str, shift_end: str) -> int:
    start = shift_start_datetime(punch_in, shift_start, shift_end)
    actual_time = local_datetime(punch_in)
    late_seconds = (actual_time - start).total_seconds()

    # The grace period is 10 minutes. Count lateness from shift start.
    if late_seconds <= 10 * 60:
        return 0

    return int(late_seconds // 60)


def round_half_up(value: float, places: int = 2) -> float:
    quantum = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    seconds = (as_utc_aware(punch_out) - as_utc_aware(punch_in)).total_seconds()
    return round_half_up(seconds / 3600, 2)


def compute_overtime(punch_out: datetime, punch_in: datetime, shift_start: str, shift_end: str) -> int:
    end = shift_end_datetime(punch_in, shift_start, shift_end).astimezone(UTC)
    seconds = (as_utc_aware(punch_out) - end).total_seconds()
    minutes = int(seconds // 60)
    return minutes if minutes >= 30 else 0


def serialize_value(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return epoch_ms(value)
    if isinstance(value, list):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items() if k != "_id"}
    return value


def public_record(doc: dict) -> dict:
    out = dict(doc)
    out.pop("_id", None)
    out.setdefault("history", [])
    out.setdefault("half_day", False)
    out.setdefault("late_minutes", 0)
    out.setdefault("overtime_minutes", 0)
    return serialize_value(out)


def valid_page(page: int, page_size: int):
    if page < 1 or page_size < 1 or page_size > 100:
        raise HTTPException(422, "page must be >= 1 and page_size must be between 1 and 100")


def working_days(start: date, end: date) -> int:
    if end < start:
        return 0

    count = 0
    current_day = start

    while current_day <= end:
        # weekday(): Monday is 0 and Sunday is 6.
        if current_day.weekday() < 5:
            count += 1
        current_day += timedelta(days=1)

    return count


def month_filter(month: str) -> tuple[str, str, date, date]:
    start, end = parse_month(month)
    return start.isoformat(), end.isoformat(), start, end


# ---------------------------------- models -----------------------------------

class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def check_joined(cls, v):
        date.fromisoformat(v)
        if date.fromisoformat(v).isoformat() != v:
            raise ValueError("joined_on must be YYYY-MM-DD")
        return v

    @model_validator(mode="after")
    def shifts_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str
    punched_at: Optional[StrictInt] = None
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"

    @field_validator("punched_at", mode="before")
    @classmethod
    def reject_null_timestamp(cls, v):
        # Omission means "use the current time"; explicit null is not in the API contract.
        if v is None:
            raise ValueError("punched_at must be omitted or an epoch-millisecond integer")
        return v

    @field_validator("punched_at")
    @classmethod
    def check_ts(cls, v):
        if v is not None:
            validate_epoch(v)
        return v


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str
    punched_at: Optional[StrictInt] = None

    @field_validator("punched_at", mode="before")
    @classmethod
    def reject_null_timestamp(cls, v):
        # Omission means "use the current time"; explicit null is not in the API contract.
        if v is None:
            raise ValueError("punched_at must be omitted or an epoch-millisecond integer")
        return v

    @field_validator("punched_at")
    @classmethod
    def check_ts(cls, v):
        if v is not None:
            validate_epoch(v)
        return v


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Optional[Literal["PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"]] = None
    punch_in: Optional[StrictInt] = None
    punch_out: Optional[StrictInt] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @field_validator("status", "punch_in", "punch_out", mode="before")
    @classmethod
    def reject_explicit_null(cls, v, info):
        # These properties are optional, but not nullable, in openapi.yaml.
        # Omit a property to leave it unchanged; use status=ABSENT/LEAVE to clear punches.
        if v is None:
            raise ValueError(f"{info.field_name} cannot be null; omit it to leave it unchanged")
        return v

    @field_validator("punch_in", "punch_out")
    @classmethod
    def check_ts(cls, v):
        if v is not None:
            validate_epoch(v, "punch timestamp")
        return v


# ---------------------------------- routes -----------------------------------

@app.get("/health")

def health():
    try:
        db.command("ping")
        return {"status": "ok"}
    except PyMongoError:
        raise HTTPException(503, "MongoDB unavailable")


@app.post("/employees", status_code=201)

def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = datetime.now(UTC).replace(microsecond=0)
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return serialize_value(doc)


@app.get("/employees")

def list_employees(department: Optional[str] = None, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    valid_page(page, page_size)
    q = {"department": department} if department is not None else {}
    total = db.employees.count_documents(q)
    items = list(db.employees.find(q, {"_id": 0}).sort("emp_code", ASCENDING).skip((page - 1) * page_size).limit(page_size))
    return {"items": [serialize_value(x) for x in items], "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)

def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "unknown employee")
    ts = from_epoch_ms(body.punched_at) if body.punched_at is not None else datetime.now(UTC).replace(microsecond=0)
    d = attendance_date(ts, emp["shift_start"], emp["shift_end"])
    doc = {"emp_code": body.emp_code, "date": d, "status": body.status, "punch_in": utc_naive(ts), "punch_out": None,
           "work_hours": None, "late_minutes": compute_late_minutes(ts, emp["shift_start"], emp["shift_end"]),
           "overtime_minutes": 0, "half_day": False, "history": []}
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return public_record(doc)


@app.post("/attendance/punch-out")

def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "unknown employee")
    ts = from_epoch_ms(body.punched_at) if body.punched_at is not None else datetime.now(UTC).replace(microsecond=0)
    ts_store = utc_naive(ts)
    # Find the latest punch-in that happened before this punch-out time.
    candidate = db.attendance_logs.find_one({"emp_code": body.emp_code, "punch_in": {"$ne": None, "$lte": ts_store}}, sort=[("punch_in", DESCENDING)])
    if not candidate:
        raise HTTPException(404, "no punch-in found")
    if candidate.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    start = as_utc_aware(candidate["punch_in"])
    elapsed = (ts - start).total_seconds()
    if elapsed <= 0 or elapsed > 24 * 3600:
        raise HTTPException(422, "punched_at must be after punch-in and within 24 hours")
    hours = compute_work_hours(candidate["punch_in"], ts_store)
    overtime = compute_overtime(ts_store, candidate["punch_in"], emp["shift_start"], emp["shift_end"])
    updated = db.attendance_logs.find_one_and_update(
        {"_id": candidate["_id"], "punch_out": None},
        {"$set": {"punch_out": ts_store, "work_hours": hours, "overtime_minutes": overtime, "half_day": hours < 4.50}},
        return_document=ReturnDocument.AFTER)
    if not updated:
        raise HTTPException(409, "already punched out")
    return public_record(updated)


@app.get("/attendance")

def list_attendance(emp_code: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None,
                    status: Optional[Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]] = None,
                    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    valid_page(page, page_size)
    if date_from:
        parse_date(date_from, "date_from")
    if date_to:
        parse_date(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must be <= date_to")
    q = {}
    if emp_code is not None: q["emp_code"] = emp_code
    if status is not None: q["status"] = status
    if date_from or date_to:
        q["date"] = {}
        if date_from: q["date"]["$gte"] = date_from
        if date_to: q["date"]["$lte"] = date_to
    total = db.attendance_logs.count_documents(q)
    docs = list(db.attendance_logs.find(q).sort([("date", DESCENDING), ("emp_code", ASCENDING)]).skip((page - 1) * page_size).limit(page_size))
    return {"items": [public_record(d) for d in docs], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")

def regularize(emp_code: str, date: str, body: RegularizeIn):
    parse_date(date)
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, "unknown employee")
    old = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not old:
        raise HTTPException(404, "attendance record not found")
    new = {k: old.get(k) for k in ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day")}
    # Only update fields the caller actually sent; omitted fields stay unchanged.
    supplied = body.model_dump(exclude={"reason", "regularized_by"}, exclude_unset=True)
    for key, val in supplied.items():
        new[key] = from_epoch_ms(val) if key in ("punch_in", "punch_out") and val is not None else val
    status = new["status"]
    if status in ("ABSENT", "LEAVE"):
        if "punch_in" in supplied and supplied["punch_in"] is not None or "punch_out" in supplied and supplied["punch_out"] is not None:
            raise HTTPException(422, "ABSENT and LEAVE cannot have punch times")
        new.update(punch_in=None, punch_out=None, work_hours=None, late_minutes=0, overtime_minutes=0, half_day=False)
    else:
        if new["punch_in"] is None:
            raise HTTPException(422, "presence status requires punch_in")
        if attendance_date(new["punch_in"], emp["shift_start"], emp["shift_end"]) != date:
            raise HTTPException(422, "punch_in must belong to the attendance date")
        new["late_minutes"] = compute_late_minutes(new["punch_in"], emp["shift_start"], emp["shift_end"])
        if new["punch_out"] is not None:
            elapsed = (as_utc_aware(new["punch_out"]) - as_utc_aware(new["punch_in"])).total_seconds()
            if elapsed <= 0 or elapsed > 86400:
                raise HTTPException(422, "punch_out must be after punch_in and within 24 hours")
            new["work_hours"] = compute_work_hours(new["punch_in"], new["punch_out"])
            new["overtime_minutes"] = compute_overtime(new["punch_out"], new["punch_in"], emp["shift_start"], emp["shift_end"])
            new["half_day"] = new["work_hours"] < 4.50
        else:
            new["work_hours"], new["overtime_minutes"], new["half_day"] = None, 0, False
    changes = {}
    for key, value in new.items():
        before = old.get(key, False if key == "half_day" else 0 if key in ("late_minutes", "overtime_minutes") else None)
        if key in ("punch_in", "punch_out"):
            before_cmp, after_cmp = as_utc_aware(before), as_utc_aware(value)
        else:
            before_cmp, after_cmp = before, value
        if before_cmp != after_cmp:
            changes[key] = {"from": epoch_ms(before) if key in ("punch_in", "punch_out") else before,
                            "to": epoch_ms(value) if key in ("punch_in", "punch_out") else value}
    if not changes:
        raise HTTPException(422, "request does not change the record")
    entry = {"at": datetime.now(UTC).replace(microsecond=0), "by": body.regularized_by, "reason": body.reason, "changes": changes}
    filt = {"_id": old["_id"], "history": old.get("history", [])}
    for key in ("status", "punch_in", "punch_out"):
        filt[key] = old.get(key)
    set_values = {k: v for k, v in new.items()}
    updated = db.attendance_logs.find_one_and_update(filt, {"$set": set_values, "$push": {"history": entry}}, return_document=ReturnDocument.AFTER)
    if not updated:
        raise HTTPException(409, "record changed concurrently; retry")
    return public_record(updated)


# ------------------------------ analytics -----------------------------------

def attendance_month_pipeline(emp_code: str, start_s: str, end_s: str) -> list:
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start_s, "$lte": end_s}}},
        {"$set": {"_weekday": {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}}}},
        {"$group": {"_id": None,
            "present_days": {"$sum": {"$cond": [{"$and": [{"$in": ["$status", list(PRESENCE)]}, {"$in": ["$_weekday", [2,3,4,5,6]]}]}, {"$cond": [{"$eq": [{"$ifNull": ["$half_day", False]}, True]}, 0.5, 1]}, 0]}},
            "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
            "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}}}}
    ]


@app.get("/analytics/employees/{emp_code}/monthly")

def employee_monthly(emp_code: str, month: str):
    start_s, end_s, start, end = month_filter(month)
    emp = db.employees.find_one({"emp_code": emp_code}, {"_id": 0, "joined_on": 1})
    if not emp: raise HTTPException(404, "unknown employee")
    effective_start = max(start, parse_date(emp["joined_on"], "joined_on"))
    wd = working_days(effective_start, end) if effective_start <= end else 0
    rows = list(db.attendance_logs.aggregate(attendance_month_pipeline(emp_code, start_s, end_s)))
    row = rows[0] if rows else {}
    present = float(row.get("present_days", 0))
    pct = round_half_up(present / wd * 100, 4) if wd else None
    return {"emp_code": emp_code, "month": month, "working_days": wd, "present_days": round_half_up(present),
            "leave_days": row.get("leave_days", 0), "late_count": row.get("late_count", 0),
            "total_late_minutes": row.get("total_late_minutes", 0), "total_overtime_minutes": row.get("total_overtime_minutes", 0), "attendance_pct": pct}


def department_summary_pipeline(start_s: str, end_s: str, department: Optional[str]) -> list:
    match = {"joined_on": {"$lte": end_s}}
    if department is not None: match["department"] = department
    return [
        {"$match": match},
        {"$lookup": {"from": "attendance_logs", "let": {"code": "$emp_code"}, "pipeline": [
            {"$match": {"$expr": {"$and": [{"$eq": ["$emp_code", "$$code"]}, {"$gte": ["$date", start_s]}, {"$lte": ["$date", end_s]}]}}}], "as": "logs"}},
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {"$set": {"_weekday": {"$cond": [{"$ne": [{"$ifNull": ["$logs.date", None]}, None]}, {"$dayOfWeek": {"$dateFromString": {"dateString": "$logs.date"}}}, None]} }},
        {"$group": {"_id": {"department": "$department", "emp_code": "$emp_code"}, "headcount": {"$first": 1},
            "present_days": {"$sum": {"$cond": [{"$and": [{"$in": [{"$ifNull": ["$logs.status", ""]}, list(PRESENCE)]}, {"$in": ["$_weekday", [2,3,4,5,6]]}]}, {"$cond": [{"$eq": [{"$ifNull": ["$logs.half_day", False]}, True]}, 0.5, 1]}, 0]}},
            "work_hours_sum": {"$sum": {"$cond": [{"$and": [{"$in": [{"$ifNull": ["$logs.status", ""]}, list(PRESENCE)]}, {"$ne": [{"$ifNull": ["$logs.work_hours", None]}, None]}]}, "$logs.work_hours", 0]}},
            "work_hours_count": {"$sum": {"$cond": [{"$and": [{"$in": [{"$ifNull": ["$logs.status", ""]}, list(PRESENCE)]}, {"$ne": [{"$ifNull": ["$logs.work_hours", None]}, None]}]}, 1, 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$logs.late_minutes", 0]}, 0]}, 1, 0]}},
            "total_late_minutes": {"$sum": {"$ifNull": ["$logs.late_minutes", 0]}},
            "leave_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]}},
            "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]}}}},
        {"$group": {"_id": "$_id.department", "headcount": {"$sum": "$headcount"}, "present_days": {"$sum": "$present_days"},
            "work_hours_sum": {"$sum": "$work_hours_sum"}, "work_hours_count": {"$sum": "$work_hours_count"},
            "late_count": {"$sum": "$late_count"}, "total_late_minutes": {"$sum": "$total_late_minutes"},
            "leave_count": {"$sum": "$leave_count"}, "on_duty_count": {"$sum": "$on_duty_count"}}},
        {"$sort": {"_id": 1}}, {"$project": {"_id": 0, "department": "$_id", "headcount": 1, "present_days": 1,
            "avg_work_hours": {"$cond": [{"$eq": ["$work_hours_count", 0]}, None, {"$divide": ["$work_hours_sum", "$work_hours_count"]}]},
            "late_count": 1, "total_late_minutes": 1, "leave_count": 1, "on_duty_count": 1}}
    ]


@app.get("/analytics/departments/summary")

def department_summary(month: str, department: Optional[str] = None):
    start_s, end_s, _, _ = month_filter(month)
    items = list(db.employees.aggregate(department_summary_pipeline(start_s, end_s, department)))
    return {"month": month, "items": [{**x, "present_days": round_half_up(x.get("present_days", 0)),
        "avg_work_hours": round_half_up(x["avg_work_hours"]) if x.get("avg_work_hours") is not None else None} for x in items if x.get("headcount", 0) > 0]}


def leaderboard_pipeline(start_s: str, end_s: str, limit: int, department: Optional[str]) -> list:
    pipeline = [
        {"$match": {"date": {"$gte": start_s, "$lte": end_s}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                    "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}}}},
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "employee"}},
        {"$unwind": "$employee"},
    ]
    if department is not None: pipeline.append({"$match": {"employee.department": department}})
    pipeline += [
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}}, {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {"_id": 0, "rank": 1, "emp_code": "$_id", "name": "$employee.name", "department": "$employee.department", "total_late_minutes": 1, "late_count": 1}}
    ]
    return pipeline


@app.get("/analytics/leaderboard/late")

def late_leaderboard(month: str, limit: int = Query(10, ge=1, le=50), department: Optional[str] = None):
    start_s, end_s, _, _ = month_filter(month)
    items = list(db.attendance_logs.aggregate(leaderboard_pipeline(start_s, end_s, limit, department)))
    return {"month": month, "items": items}


def trend_pipeline(department: str, from_s: str, to_s: str) -> list:
    start_expr = {"$dateFromString": {"dateString": from_s}}
    end_expr = {"$dateFromString": {"dateString": to_s}}
    day_count = {"$add": [{"$dateDiff": {"startDate": start_expr, "endDate": end_expr, "unit": "day"}}, 1]}
    return [
        {"$match": {"department": department}},
        {"$group": {"_id": None,
                    "employees": {"$push": {"emp_code": "$emp_code", "joined_on": "$joined_on"}},
                    "codes": {"$addToSet": "$emp_code"}}},
        {"$set": {"dates": {"$map": {
            "input": {"$range": [0, day_count]}, "as": "offset",
            "in": {"$dateToString": {"date": {"$dateAdd": {
                "startDate": start_expr, "unit": "day", "amount": "$$offset"}}, "format": "%Y-%m-%d"}}
        }}}},
        {"$unwind": "$dates"},
        {"$set": {"date": "$dates"}},
        {"$lookup": {"from": "attendance_logs", "let": {"d": "$date", "codes": "$codes"},
                      "pipeline": [{"$match": {"$expr": {"$and": [
                          {"$eq": ["$date", "$$d"]}, {"$in": ["$emp_code", "$$codes"]}
                      ]}}}], "as": "logs"}},
        {"$set": {
            "headcount": {"$size": {"$filter": {"input": "$employees", "as": "e",
                                                    "cond": {"$lte": ["$$e.joined_on", "$date"]}}}},
            "present_count": {"$sum": {"$map": {"input": "$logs", "as": "l", "in": {
                "$cond": [{"$in": ["$$l.status", list(PRESENCE)]},
                           {"$cond": [{"$eq": [{"$ifNull": ["$$l.half_day", False]}, True]}, 0.5, 1]}, 0]
            }}}},
            "late_count": {"$size": {"$filter": {"input": "$logs", "as": "l",
                                                   "cond": {"$gt": [{"$ifNull": ["$$l.late_minutes", 0]}, 0]}}}},
            "is_working_day": {"$in": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}}, [2, 3, 4, 5, 6]]}
        }},
        {"$set": {"attendance_rate": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            {"$divide": ["$present_count", "$headcount"]}, None
        ]}}},
        {"$setWindowFields": {"sortBy": {"date": 1}, "output": {
            "_ma": {"$avg": "$attendance_rate", "window": {"documents": [-6, 0]}}
        }}},
        {"$project": {"_id": 0, "date": 1, "is_working_day": 1, "headcount": 1,
                       "present_count": 1, "late_count": 1,
                       "attendance_rate": {"$cond": [{"$eq": ["$attendance_rate", None]}, None, {"$round": [{"$add": ["$attendance_rate", 0.000000001]}, 4]}]},
                       "moving_avg_7d": {"$cond": [{"$eq": ["$_ma", None]}, None, {"$round": [{"$add": ["$_ma", 0.000000001]}, 4]}]}}},
        {"$sort": {"date": 1}}
    ]


@app.get("/analytics/departments/{department}/trend")

def department_trend(department: str, from_: str = Query(..., alias="from"), to: str = Query(...)):
    start = parse_date(from_, "from")
    end = parse_date(to, "to")
    if end < start or (end - start).days + 1 > 92:
        raise HTTPException(422, "to must be >= from and range must not exceed 92 days")
    if not db.employees.find_one({"department": department}, {"_id": 1}):
        raise HTTPException(404, "unknown department")
    pipeline = trend_pipeline(department, start.isoformat(), end.isoformat())
    rows = list(db.employees.aggregate(pipeline))
    for row in rows:
        if row.get("attendance_rate") is not None: row["attendance_rate"] = round_half_up(row["attendance_rate"], 4)
        if row.get("moving_avg_7d") is not None: row["moving_avg_7d"] = round_half_up(row["moving_avg_7d"], 4)
        row["present_count"] = round_half_up(row.get("present_count", 0))
    return {"department": department, "items": rows}


# ------------------------------- explain ------------------------------------

def explain_aggregate(collection, pipeline):
    try:
        return db.command({"explain": {"aggregate": collection, "pipeline": pipeline, "cursor": {}}, "verbosity": "executionStats"})
    except PyMongoError as exc:
        raise HTTPException(422, f"explain failed: {str(exc)}")


@app.get("/admin/explain/{endpoint}")

def admin_explain(endpoint: str, emp_code: Optional[str] = None, month: Optional[str] = None,
                  department: Optional[str] = None, limit: int = Query(10, ge=1, le=50),
                  date_from: Optional[str] = None, date_to: Optional[str] = None,
                  status: Optional[Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]] = None,
                  page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                  from_: Optional[str] = Query(None, alias="from"), to: Optional[str] = None):
    valid_page(page, page_size)
    if endpoint == "attendance_list":
        if date_from: parse_date(date_from, "date_from")
        if date_to: parse_date(date_to, "date_to")
        if date_from and date_to and date_from > date_to: raise HTTPException(422, "date_from must be <= date_to")
        q = {}
        if emp_code is not None: q["emp_code"] = emp_code
        if status is not None: q["status"] = status
        if date_from or date_to:
            q["date"] = {}
            if date_from: q["date"]["$gte"] = date_from
            if date_to: q["date"]["$lte"] = date_to
        explain = db.command({"explain": {"find": "attendance_logs", "filter": q, "sort": {"date": -1, "emp_code": 1}, "skip": (page-1)*page_size, "limit": page_size}, "verbosity": "executionStats"})
        collection = "attendance_logs"
    elif endpoint == "employee_monthly":
        if not emp_code or not month: raise HTTPException(422, "emp_code and month are required")
        a,b,_,_ = month_filter(month)
        explain = explain_aggregate("attendance_logs", attendance_month_pipeline(emp_code,a,b)); collection = "attendance_logs"
    elif endpoint == "department_summary":
        if not month: raise HTTPException(422, "month is required")
        a,b,_,_ = month_filter(month)
        explain = explain_aggregate("employees", department_summary_pipeline(a,b,department)); collection = "employees"
    elif endpoint == "late_leaderboard":
        if not month: raise HTTPException(422, "month is required")
        a,b,_,_ = month_filter(month)
        explain = explain_aggregate("attendance_logs", leaderboard_pipeline(a,b,limit,department)); collection = "attendance_logs"
    elif endpoint == "department_trend":
        if not department or not from_ or not to: raise HTTPException(422, "department, from and to are required")
        a, b = parse_date(from_, "from"), parse_date(to, "to")
        if b < a or (b-a).days+1 > 92: raise HTTPException(422, "invalid date range")
        explain = explain_aggregate("employees", trend_pipeline(department,a.isoformat(),b.isoformat())); collection = "employees"
    else:
        raise HTTPException(422, "endpoint must be one of attendance_list, employee_monthly, department_summary, late_leaderboard, department_trend")
    # Convert MongoDB-specific values so FastAPI can return them as JSON.
    safe = json_util.loads(json_util.dumps(explain))
    return {"endpoint": endpoint, "collection": collection, "explain": safe}
