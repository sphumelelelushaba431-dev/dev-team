const log = document.getElementById("log");
const form = document.getElementById("composer");
const input = document.getElementById("message");
const send = document.getElementById("send");

function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function scroll() { log.scrollTop = log.scrollHeight; }

async function submit(text) {
  log.append(el("div", "msg user", text)); scroll();
  const typing = el("div", "msg bot typing", "Checking policy..."); log.append(typing); scroll();
  send.disabled = true;
  try {
    const res = await fetch("/api/v1/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: window.SESSION_ID }) });
    const r = await res.json();
    typing.remove();
    if (!res.ok) throw new Error(r.error || "The server could not process the message.");
    log.append(el("div", "msg bot", r.answer));
  } catch (err) {
    typing.textContent = `Message not sent: ${err.message} Try again.`;
    typing.classList.remove("typing");
  } finally { send.disabled = false; scroll(); input.focus(); }
}

form.addEventListener("submit", e => { e.preventDefault(); const t = input.value.trim(); if (!t) return; input.value = ""; submit(t); });
input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });

