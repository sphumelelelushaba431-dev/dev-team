"""Persistence for complaints and chat messages.

FirebaseStorage (Cloud Firestore) is used when FIREBASE_CREDENTIALS is set,
as proposed in the paper; otherwise SQLiteStorage keeps everything local.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

from . import config


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class SQLiteStorage:
    kind = "sqlite"

    def __init__(self, path: str | None = None):
        self.path = path or config.SQLITE_PATH
        self.lock = threading.Lock()
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT, reference TEXT UNIQUE, session_id TEXT,
                channel TEXT, text TEXT, department TEXT, confidence REAL, urgency TEXT,
                status TEXT, status_reason TEXT, answer TEXT, sources TEXT, grounded INTEGER,
                latency_ms REAL, created_at TEXT, updated_at TEXT, staff_note TEXT);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, text TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS staff (
                id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
                staff_number TEXT UNIQUE NOT NULL, department TEXT NOT NULL, password_hash TEXT NOT NULL,
                created_at TEXT, last_login TEXT);
            """)

    def _conn(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def next_reference(self) -> str:
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM complaints").fetchone()[0] + 1
        return f"CV-{datetime.now().year}-{n:06d}"

    def add_complaint(self, rec: dict) -> dict:
        with self.lock, self._conn() as c:
            rec = dict(rec, reference=rec.get("reference") or self.next_reference(),
                       created_at=now_iso(), updated_at=now_iso())
            rec["sources"] = json.dumps(rec.get("sources", []))
            rec["grounded"] = int(bool(rec.get("grounded")))
            cols = ["reference", "session_id", "channel", "text", "department", "confidence", "urgency",
                    "status", "status_reason", "answer", "sources", "grounded", "latency_ms",
                    "created_at", "updated_at"]
            c.execute(f"INSERT INTO complaints ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                      [rec.get(k) for k in cols])
        return rec

    def add_message(self, session_id, role, text):
        with self._conn() as c:
            c.execute("INSERT INTO messages (session_id, role, text, created_at) VALUES (?,?,?,?)",
                      (session_id, role, text, now_iso()))

    def list_complaints(self, limit=500):
        with self._conn() as c:
            rows = c.execute("SELECT * FROM complaints ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["sources"] = json.loads(d["sources"] or "[]")
            out.append(d)
        return out

    def get(self, reference):
        with self._conn() as c:
            r = c.execute("SELECT * FROM complaints WHERE reference=?", (reference,)).fetchone()
        return dict(r) if r else None

    def update_status(self, reference, status, note=None):
        with self._conn() as c:
            c.execute("UPDATE complaints SET status=?, staff_note=COALESCE(?, staff_note), updated_at=? "
                      "WHERE reference=?", (status, note, now_iso(), reference))


    # ------------------------------------------------------------- staff
    def add_staff(self, rec: dict) -> dict:
        with self.lock, self._conn() as c:
            cur = c.execute("INSERT INTO staff (full_name, email, staff_number, department, password_hash, created_at) "
                            "VALUES (?,?,?,?,?,?)", (rec["full_name"], rec["email"].lower(), rec["staff_number"],
                                                     rec["department"], rec["password_hash"], now_iso()))
            return dict(rec, id=cur.lastrowid)

    def staff_by_email(self, email: str):
        with self._conn() as c:
            r = c.execute("SELECT * FROM staff WHERE email=?", (email.lower(),)).fetchone()
        return dict(r) if r else None

    def staff_exists(self, email: str, staff_number: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM staff WHERE email=? OR staff_number=?",
                             (email.lower(), staff_number)).fetchone() is not None

    def touch_login(self, staff_id):
        with self._conn() as c:
            c.execute("UPDATE staff SET last_login=? WHERE id=?", (now_iso(), staff_id))


class FirebaseStorage:
    kind = "firebase"

    def __init__(self, cred_path: str):
        # optional dependency: only needed when FIREBASE_CREDENTIALS is set
        import firebase_admin  # type: ignore[import-not-found]
        from firebase_admin import credentials, firestore  # type: ignore[import-not-found]
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        self.db = firestore.client()

    def next_reference(self):
        n = len(list(self.db.collection("complaints").select([]).stream())) + 1
        return f"CV-{datetime.now().year}-{n:06d}"

    def add_complaint(self, rec):
        rec = dict(rec, reference=rec.get("reference") or self.next_reference(),
                   created_at=now_iso(), updated_at=now_iso())
        self.db.collection("complaints").document(rec["reference"]).set(rec)
        return rec

    def add_message(self, session_id, role, text):
        self.db.collection("sessions").document(session_id).collection("messages").add(
            {"role": role, "text": text, "created_at": now_iso()})

    def list_complaints(self, limit=500):
        docs = (self.db.collection("complaints").order_by("created_at", direction="DESCENDING")
                .limit(limit).stream())
        return [d.to_dict() for d in docs]

    def get(self, reference):
        d = self.db.collection("complaints").document(reference).get()
        return d.to_dict() if d.exists else None

    def update_status(self, reference, status, note=None):
        data = {"status": status, "updated_at": now_iso()}
        if note:
            data["staff_note"] = note
        self.db.collection("complaints").document(reference).update(data)


    def add_staff(self, rec):
        rec = dict(rec, email=rec["email"].lower(), created_at=now_iso())
        ref = self.db.collection("staff").document(rec["email"])
        ref.set(rec)
        return dict(rec, id=rec["email"])

    def staff_by_email(self, email):
        d = self.db.collection("staff").document(email.lower()).get()
        return dict(d.to_dict(), id=d.id) if d.exists else None

    def staff_exists(self, email, staff_number):
        if self.staff_by_email(email):
            return True
        return any(True for _ in self.db.collection("staff").where("staff_number", "==", staff_number).limit(1).stream())

    def touch_login(self, staff_id):
        self.db.collection("staff").document(staff_id).update({"last_login": now_iso()})


def make_storage():
    if config.FIREBASE_CREDENTIALS:
        return FirebaseStorage(config.FIREBASE_CREDENTIALS)
    return SQLiteStorage()
