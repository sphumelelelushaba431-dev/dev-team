const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const deptColor = d => css({ Academic: "--academic", Finance: "--finance", ICT: "--ict", Facilities: "--facilities", Hostel: "--hostel" }[d] || "--unknown");
const urgColor = u => css({ Critical: "--critical", High: "--high", Medium: "--medium", Low: "--low" }[u] || "--unknown");
const charts = {};
let rows = [];

function chart(id, type, labels, data, colors, extra = {}) {
  if (typeof Chart === "undefined") return;
  charts[id]?.destroy();
  charts[id] = new Chart(document.getElementById(id), {
    type, data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0 }] },
    options: { plugins: { legend: { display: type !== "bar", position: "bottom" } }, maintainAspectRatio: false, ...extra } });
}
const td = (t, cls) => { const c = document.createElement("td"); if (cls) c.className = cls; c.textContent = t; return c; };
const chip = (cls, attr, v) => { const s = document.createElement("span"); s.className = cls; s.dataset[attr] = v; s.textContent = v; return s; };

async function loadStats() {
  const s = await (await fetch("/api/v1/stats")).json();
  kTotal.textContent = s.total; kEsc.textContent = s.escalated; kRes.textContent = s.resolved;
  kGround.textContent = Math.round(s.grounded_rate * 100) + "%"; kLat.textContent = s.latency_ms_mean + " ms";
  const order = ["Academic", "Finance", "ICT", "Facilities", "Hostel"];
  chart("cDept", "bar", order, order.map(d => s.by_department[d] || 0), order.map(deptColor), { scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } });
  const u = ["Critical", "High", "Medium", "Low"];
  chart("cUrg", "doughnut", u, u.map(x => s.by_urgency[x] || 0), u.map(urgColor));
  chart("cStatus", "doughnut", STATUSES, STATUSES.map(x => s.by_status[x] || 0), [css("--brand"), css("--ict"), css("--critical"), css("--finance")]);
}

function render() {
  const d = fDept.value, s = fStatus.value, u = fUrg.value, q = fText.value.toLowerCase();
  const tb = document.querySelector("#tbl tbody"); tb.innerHTML = "";
  const shown = rows.filter(r => (!d || r.department === d) && (!s || r.status === s) && (!u || r.urgency === u) &&
    (!q || r.text.toLowerCase().includes(q) || r.reference.toLowerCase().includes(q)));
  empty.hidden = shown.length > 0;
  shown.forEach(r => {
    const tr = document.createElement("tr");
    tr.append(td(r.reference), td((r.created_at || "").replace("T", " ").slice(0, 16)));
    const dc = document.createElement("td"); dc.append(chip("dept", "d", r.department), document.createTextNode(` ${Math.round(r.confidence * 100)}%`)); tr.append(dc);
    const uc = document.createElement("td"); uc.append(chip("urg", "u", r.urgency)); tr.append(uc);
    const tc = document.createElement("td"); tc.className = "text-cell";
    const det = document.createElement("details"); const sum = document.createElement("summary"); sum.textContent = r.text;
    const ans = document.createElement("p"); ans.className = "muted"; ans.textContent = `Chatbot reply: ${r.answer}`;
    const why = document.createElement("p"); why.className = "muted"; why.textContent = `Routing: ${r.status_reason}. ${r.latency_ms} ms.` + (r.staff_note ? ` Note: ${r.staff_note}` : "");
    det.append(sum, ans, why); tc.append(det); tr.append(tc);
    const sc = document.createElement("td"); const sel = document.createElement("select");
    STATUSES.forEach(o => { const op = document.createElement("option"); op.textContent = o; op.selected = o === r.status; sel.append(op); });
    sel.setAttribute("aria-label", `Status of ${r.reference}`);
    sel.addEventListener("change", async () => {
      const note = sel.value === "Resolved" ? prompt("Optional note for the student (shown when they ask for the status):", "") : null;
      const res = await fetch(`/api/v1/reports/${r.reference}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: sel.value, note }) });
      if (!res.ok) { alert("Status not saved. Sign in again and retry."); sel.value = r.status; return; }
      r.status = sel.value; loadStats();
    });
    sc.append(sel); tr.append(sc); tb.append(tr);
  });
}

async function loadRows() { rows = await (await fetch("/api/v1/reports")).json(); render(); }
[fDept, fStatus, fUrg].forEach(x => x.addEventListener("change", render)); fText.addEventListener("input", render);
loadStats(); loadRows();
setInterval(() => { loadStats(); loadRows(); }, 15000);
