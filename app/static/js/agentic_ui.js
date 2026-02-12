const landingEl = document.getElementById("landing");
const resultsEl = document.getElementById("results");
const inputEl = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const metaEl = document.getElementById("meta");

const panelEl = document.getElementById("detailPanel");
const detailBodyEl = document.getElementById("detailBody");
const toggleBtn = document.getElementById("togglePanelBtn");

function setSending(isSending){
  sendBtn.disabled = isSending;
  inputEl.disabled = isSending;
}

function showResults(){
  if (landingEl) landingEl.style.display = "none";
  if (resultsEl) resultsEl.style.display = "block";
}

function renderMeta(confidence, intent, domain){
  if (!metaEl) return;
  const pct = Math.round((confidence || 0) * 100);
  metaEl.textContent = `Confidence: ${pct}% • Intent: ${intent || "—"} • Domain: ${domain || "—"}`;
}

function openPanel(){
  panelEl.classList.remove("hidden");
  panelEl.classList.remove("collapsed");
  toggleBtn.textContent = "Collapse";
}

function closePanel(){
  panelEl.classList.add("collapsed");
  toggleBtn.textContent = "Expand";
}

toggleBtn.addEventListener("click", () => {
  if (panelEl.classList.contains("collapsed")) openPanel();
  else closePanel();
});

function escapeHtml(str){
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderDetail(item){
  if (!detailBodyEl) return;
  const readUrl = `/article/${encodeURIComponent(item.id || "")}`;
  const extRefs = (item.references || []).slice(0, 2).map(r => {
    const u = escapeHtml(r);
    return `<a class="wp-ref" href="${u}" target="_blank" rel="noreferrer">${u}</a>`;
  }).join("");

  detailBodyEl.innerHTML = `
    <div class="wp-detail-title">${escapeHtml(item.title || "")}</div>
    <div class="wp-detail-meta">${escapeHtml(item.domain || "")} • ${escapeHtml(item.topic || "")}</div>
    <div class="wp-detail-summary">${escapeHtml(item.summary || "")}</div>
    <div class="wp-detail-refs">
      <div class="wp-detail-refs-title">Read more</div>
      <a class="wp-ref" href="${readUrl}" target="_blank" rel="noreferrer">Open full article</a>
      ${extRefs ? `<div style="margin-top:10px">${extRefs}</div>` : ""}
    </div>
  `;
  openPanel();
}

function renderCombined(message, items){
  resultsEl.innerHTML = "";
  const wrapper = document.createElement("div");
  wrapper.className = "wp-answer";

  const card = document.createElement("div");
  card.className = "wp-answer-card";

  const text = document.createElement("div");
  text.className = "wp-answer-text";
  text.textContent = (message || "").trim() || "Here are the most relevant results from the knowledge base and graph.";
  card.appendChild(text);

  const top = (items || []).slice(0, 3);
  if (top.length === 0){
    const empty = document.createElement("div");
    empty.className = "wp-empty";
    empty.textContent = "No matching articles found in the local index.";
    wrapper.appendChild(card);
    wrapper.appendChild(empty);
    resultsEl.appendChild(wrapper);
    return;
  }

  const hint = document.createElement("div");
  hint.className = "wp-morehint";
  hint.textContent = "Want to know more? Click any headline below to view the article summary.";
  card.appendChild(hint);

  const ul = document.createElement("ul");
  ul.className = "wp-article-links";
  top.forEach((it) => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = "#";
    a.className = "wp-article-link";
    a.textContent = it.title || "(untitled)";
    a.addEventListener("click", (e) => {
      e.preventDefault();
      renderDetail(it);
    });
    li.appendChild(a);
    ul.appendChild(li);
  });
  card.appendChild(ul);

  wrapper.appendChild(card);
  resultsEl.appendChild(wrapper);
}

async function queryAgent(text){
  setSending(true);
  try{
    // Reset detail panel for a new search
    if (panelEl){
      panelEl.classList.add("hidden");
      panelEl.classList.add("collapsed");
      toggleBtn.textContent = "Collapse";
      if (detailBodyEl) detailBodyEl.innerHTML = `<div class="wp-right-empty">Select an article title to see the summary.</div>`;
    }
    const res = await fetch("/api/query", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    if (!res.ok){
      throw new Error(data.error || "Request failed");
    }

    showResults();

    if (data.type === "blocked"){
      resultsEl.innerHTML = `<div class="wp-empty">${escapeHtml(data.message || "Blocked.")}</div>`;
      renderMeta(0, "blocked", "—");
      return;
    }

    renderCombined(data.message, data.items || []);
    renderMeta(data.confidence, data.intent, data.domain);

  }catch(e){
    showResults();
    resultsEl.innerHTML = `<div class="wp-empty">Error: ${escapeHtml(e.message || String(e))}</div>`;
  }finally{
    setSending(false);
    inputEl.focus();
  }
}

sendBtn.addEventListener("click", () => {
  const text = (inputEl.value || "").trim();
  if(!text) return;
  inputEl.value = "";
  queryAgent(text);
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter"){
    e.preventDefault();
    sendBtn.click();
  }
});

document.querySelectorAll(".wp-prompt").forEach(btn => {
  btn.addEventListener("click", () => {
    const p = btn.getAttribute("data-prompt") || btn.textContent.trim();
    inputEl.value = p;
    inputEl.focus();
  });
});
