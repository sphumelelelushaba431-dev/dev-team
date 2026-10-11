"""Flask web application.

Students: use the chatbot only (no account needed).
Staff:    sign in, view student reports, and view the model performance dashboard.
          Accounts are created from the terminal: flask --app app create-admin
"""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from collections import Counter
from functools import wraps

from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for
import click
from werkzeug.security import check_password_hash, generate_password_hash

from . import config
from .router import SLA_HOURS
from .service import ComplaintService
from .storage import make_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
_service: ComplaintService | None = None
_eval_state = {"running": False, "message": "idle", "error": None}
DEPARTMENTS = config.CATEGORIES + ["Student Affairs", "Management"]
STATUSES = ["Open", "In progress", "Escalated", "Resolved"]


def service() -> ComplaintService:
    global _service
    if _service is None:
        _service = ComplaintService()
    return _service


def staff_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not session.get("staff"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Staff sign-in required"}), 401
            return redirect(url_for("staff_login", next=request.path))
        return fn(*a, **kw)
    return wrapper


@app.context_processor
def inject_staff():
    return {"staff": session.get("staff")}


# ------------------------------------------------------------- student pages
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/chat")
def chat_page():
    session.setdefault("sid", uuid.uuid4().hex[:12])
    return render_template("chat.html", session_id=session["sid"])


# ---------------------------------------------------------------- staff auth
@app.cli.command("create-admin")
@click.option("--name", prompt="Full name")
@click.option("--email", prompt="Work email")
@click.option("--staff-number", prompt="Staff number")
@click.option("--department", prompt="Department", type=click.Choice(DEPARTMENTS, case_sensitive=False))
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
def create_admin(name, email, staff_number, department, password):
    """Create a staff account (there is no registration page)."""
    name, email, staff_number = name.strip(), email.strip(), staff_number.strip()
    if len(name) < 3:
        raise click.ClickException("Enter the full name.")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise click.ClickException("Enter a valid email address.")
    if not staff_number:
        raise click.ClickException("Enter the staff number.")
    if len(password) < 8:
        raise click.ClickException("Use a password of at least 8 characters.")
    storage = make_storage()
    if storage.staff_exists(email, staff_number):
        raise click.ClickException("An account with this email or staff number already exists.")
    storage.add_staff({"full_name": name, "email": email, "staff_number": staff_number,
                       "department": department, "password_hash": generate_password_hash(password)})
    click.echo(f"Created staff account for {email} ({department}). Sign in at /staff/login")


@app.route("/staff/login", methods=["GET", "POST"])
def staff_login():
    error = None
    if request.method == "POST":
        user = service().storage.staff_by_email((request.form.get("email") or "").strip())
        if user and check_password_hash(user["password_hash"], request.form.get("password") or ""):
            session["staff"] = {"id": user["id"], "name": user["full_name"], "department": user["department"],
                                "email": user["email"], "staff_number": user["staff_number"],
                                "last_login": user.get("last_login")}
            service().storage.touch_login(user["id"])
            nxt = request.args.get("next") or ""
            return redirect(nxt if nxt.startswith("/staff") else url_for("staff_reports"))
        error = "Email or password is incorrect."
    return render_template("staff_login.html", error=error)


@app.get("/staff/logout")
def staff_logout():
    session.pop("staff", None)
    return redirect(url_for("staff_login"))


# --------------------------------------------------------------- staff pages
@app.get("/staff")
@staff_required
def staff_home():
    return redirect(url_for("staff_reports"))


@app.get("/staff/reports")
@staff_required
def staff_reports():
    return render_template("staff_reports.html", info=service().info(), statuses=STATUSES)


@app.get("/staff/models")
@staff_required
def staff_models():
    return render_template("staff_models.html")


# ======================================================== REST API (v1)
# ---- chatbot (public: this is the only thing students can use)
@app.post("/api/v1/chat")
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    sid = data.get("session_id") or session.get("sid") or uuid.uuid4().hex[:12]
    return jsonify(service().handle(message, sid, channel=data.get("channel", "web")))


# ---- reports (staff)
@app.get("/api/v1/reports")
@staff_required
def api_reports():
    return jsonify(service().storage.list_complaints())


@app.patch("/api/v1/reports/<reference>")
@staff_required
def api_update_report(reference):
    data = request.get_json(force=True, silent=True) or {}
    if data.get("status") not in STATUSES:
        return jsonify({"error": f"status must be one of {STATUSES}"}), 400
    if not service().storage.get(reference):
        return jsonify({"error": "report not found"}), 404
    note = (data.get("note") or "").strip() or None
    if note:
        note = f"{note} ({session['staff']['name']})"
    service().storage.update_status(reference, data["status"], note)
    return jsonify({"ok": True})


@app.get("/api/v1/stats")
@staff_required
def api_stats():
    rows = service().storage.list_complaints(limit=5000)
    lat = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    return jsonify({
        "total": len(rows),
        "escalated": sum(r["status"] == "Escalated" for r in rows),
        "resolved": sum(r["status"] == "Resolved" for r in rows),
        "grounded_rate": round(sum(bool(r.get("grounded")) for r in rows) / len(rows), 3) if rows else 0,
        "latency_ms_mean": round(sum(lat) / len(lat), 1) if lat else 0,
        "by_department": Counter(r["department"] for r in rows),
        "by_urgency": Counter(r["urgency"] for r in rows),
        "by_status": Counter(r["status"] for r in rows),
        "sla_hours": SLA_HOURS,
    })


# ---- models (staff)
@app.get("/api/v1/models")
@staff_required
def api_models():
    return jsonify(service().models())


@app.post("/api/v1/models/<model_key>/predict")
@staff_required
def api_predict(model_key):
    if model_key not in service().attitude_models:
        return jsonify({"error": f"model '{model_key}' is not available",
                        "available": list(service().attitude_models)}), 404
    answers = request.get_json(force=True, silent=True) or {}
    return jsonify(service().predict_attitude(model_key, answers))


@app.get("/api/v1/evaluation")
@staff_required
def api_evaluation():
    from .evaluation import load_rag_results
    return jsonify({"results": load_rag_results(), "state": _eval_state})


@app.post("/api/v1/evaluation/run")
@staff_required
def api_run_evaluation():
    from .evaluation import evaluate_rag
    if _eval_state["running"]:
        return jsonify({"error": "An evaluation is already running", "state": _eval_state}), 409

    def job():
        _eval_state.update(running=True, error=None, message="Starting")
        try:
            evaluate_rag(progress=lambda m: _eval_state.update(message=m))
            _eval_state["message"] = "Finished"
        except Exception as exc:  # report the failure on the dashboard
            _eval_state.update(error=f"{type(exc).__name__}: {exc}", message="Failed")
        finally:
            _eval_state["running"] = False

    threading.Thread(target=job, daemon=True).start()
    return jsonify({"started": True, "state": _eval_state}), 202


@app.get("/api/v1")
def api_index():
    return jsonify({"version": "v1", "endpoints": {
        "POST /api/v1/chat": "Public chatbot. Body: {message, session_id?}. Returns the answer, department, "
                             "urgency, status and reference. Students can ask for the status of a reference.",
        "GET /api/v1/reports": "Staff: all student reports",
        "PATCH /api/v1/reports/<reference>": "Staff: {status, note?}",
        "GET /api/v1/stats": "Staff: report statistics",
        "GET /api/v1/models": "Staff: models in use",
        "POST /api/v1/models/rag/predict": "Staff: predict a student's chatbot adoption level from "
                                           "survey answers {Q1, Q2, Q3, Q5.1 ... Q9.5, Q10}",
        "GET /api/v1/evaluation": "Staff: latest RAG accuracy, precision, recall, F1 and response latency "
                                  "on student complaints, against the intent-matching baseline",
        "POST /api/v1/evaluation/run": "Staff: rebuild the RAG index and re-test it (runs in the background)",
    }})


@app.get("/health")
def health():
    return jsonify({"status": "ok", **service().info()})


# ---- LINE OA (optional)
@app.post("/webhook/line")
def line_webhook():
    from .channels import line
    body = request.get_data()
    if not line.valid_signature(body, request.headers.get("X-Line-Signature", "")):
        abort(403)
    for event in json.loads(body).get("events", []):
        if event.get("type") == "message" and event["message"].get("type") == "text":
            result = service().handle(event["message"]["text"], "line-" + event["source"].get("userId", "anon"),
                                      channel="line")
            ref = f"\n\nReference: {result['reference']}" if result.get("reference") else ""
            line.reply(event["replyToken"], result["answer"] + ref)
    return "OK"


def run():
    import os
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)


if __name__ == "__main__":
    run()
