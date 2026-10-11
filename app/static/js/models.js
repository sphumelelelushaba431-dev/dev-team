const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const METRICS = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1-score"]];
const COLORS = { RAG: () => css("--brand"), "Intent-matching baseline": () => css("--low") };
const charts = {};
const pct = v => (v * 100).toFixed(1) + "%";
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

function tiles(results) {
  const box = document.getElementById("tiles"); box.innerHTML = "";
  ["RAG"].forEach(name => {
    const r = results.find(x => x.model === name);
    const g = el("div", "mtile"); g.style.setProperty("--c", COLORS[name]());
    const head = el("div", "mtile-head"); head.append(el("strong", "", name), el("span", "muted", r ? r.detail : "not evaluated yet"));
    g.append(head);
    const row = el("div", "mtile-row");
    METRICS.forEach(([k, label]) => {
      const cell = el("div", "mcell");
      cell.append(el("div", "v", r ? pct(r[k]) : "n/a"), el("div", "k", label));
      row.append(cell);
    });
    const lat = el("div", "mcell"); lat.append(el("div", "v", r ? `${r.latency_ms_mean} ms` : "n/a"), el("div", "k", "Mean latency"));
    row.append(lat); g.append(row); box.append(g);
  });
}

function bars(results) {
  charts.bars?.destroy();
  charts.bars = new Chart(cBars, {
    type: "bar",
    data: { labels: METRICS.map(m => m[1]),
      datasets: results.map(r => ({ label: r.model, data: METRICS.map(([k]) => +(r[k] * 100).toFixed(1)),
        backgroundColor: COLORS[r.model] ? COLORS[r.model]() : css("--unknown"), borderRadius: 3 })) },
    options: { maintainAspectRatio: false, scales: { y: { min: 0, max: 100, ticks: { callback: v => v + "%" } } },
      plugins: { legend: { position: "bottom" },
        tooltip: { callbacks: { label: c => `${results[c.datasetIndex].model}: ${c.raw}%` } } } } });
}

function radar(results) {
  charts.radar?.destroy();
  charts.radar = new Chart(cRadar, {
    type: "radar",
    data: { labels: METRICS.map(m => m[1]),
      datasets: results.map(r => { const c = COLORS[r.model] ? COLORS[r.model]() : css("--unknown");
        return { label: r.model, data: METRICS.map(([k]) => +(r[k] * 100).toFixed(1)), borderColor: c,
          backgroundColor: c + "33", pointBackgroundColor: c, borderWidth: 2 }; }) },
    options: { maintainAspectRatio: false, scales: { r: { min: 0, max: 100, ticks: { stepSize: 25, callback: v => v + "%", backdropColor: "transparent" } } },
      plugins: { legend: { position: "bottom" } } } });
}

function heatmaps(results, labels) {
  const box = document.getElementById("heatmaps"); box.innerHTML = "";
  results.filter(r => r.model !== "Intent-matching baseline").forEach(r => {
    const wrap = el("figure", "hm"); wrap.append(el("figcaption", "", r.model));
    const grid = el("div", "hm-grid"); grid.style.gridTemplateColumns = `auto repeat(${labels.length}, 1fr)`;
    grid.append(el("div", "hm-corner", ""));
    labels.forEach(l => grid.append(el("div", "hm-col", l)));
    const max = Math.max(1, ...r.confusion_matrix.flat());
    r.confusion_matrix.forEach((row, i) => {
      grid.append(el("div", "hm-row", labels[i]));
      row.forEach((v, j) => {
        const c = el("div", "hm-cell" + (i === j ? " diag" : ""), v);
        const a = v / max; c.style.background = `color-mix(in srgb, ${COLORS[r.model]()} ${Math.round(a * 85)}%, white)`;
        c.style.color = a > 0.55 ? "#fff" : "var(--ink)";
        c.title = `Actual ${labels[i]}, predicted ${labels[j]}: ${v} reports`;
        grid.append(c);
      });
    });
    wrap.append(grid); box.append(wrap);
  });
}

async function load() {
  const res = await (await fetch("/api/v1/evaluation")).json();
  const state = res.state; const btn = document.getElementById("runBtn");
  btn.disabled = state.running;
  runState.textContent = state.running ? `Evaluating: ${state.message}` : (state.error ? `Last run failed: ${state.error}` : "");
  const data = res.results;
  if (!data) { meta.textContent = "No results yet. Select Re-run evaluation to build and test the RAG model."; return state.running; }
  const d = data.dataset;
  meta.textContent = `Routing student complaints to the right department: ${d.complaints} complaints${d.synthetic ? " (synthetic sample data)" : ""}, ` +
    `${d.train} to build the RAG index and ${d.test} held out for testing. ${data.protocol}; macro-averaged metrics. ` +
    `Updated ${data.created_at.replace("T", " ").slice(0, 16)}.`;
  const order = ["RAG", "Intent-matching baseline"];
  const results = data.results.filter(r => order.includes(r.model)).sort((a, b) => order.indexOf(a.model) - order.indexOf(b.model));
  tiles(results); bars(results); radar(results); heatmaps(results, data.labels);
  return state.running;
}

document.getElementById("runBtn").addEventListener("click", async () => {
  const r = await fetch("/api/v1/evaluation/run", { method: "POST" });
  if (r.status === 409) return;
  poll();
});
async function poll() { if (await load()) setTimeout(poll, 2000); }
poll();
