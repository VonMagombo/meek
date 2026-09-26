const API_BASE = "/api/v1";
const MAX_LEN = 5000;
const MAX_BATCH_ITEMS = 100;

const input = document.getElementById("comment-input");
const inputLabel = document.getElementById("input-label");
const charCount = document.getElementById("char-count");
const checkBtn = document.getElementById("check-btn");
const resultPanel = document.getElementById("result-panel");
const errorPanel = document.getElementById("error-panel");
const errorText = document.getElementById("error-text");
const verdictEl = document.getElementById("verdict");
const scoresEl = document.getElementById("scores");
const modelNameEl = document.getElementById("model-name");
const langButtons = document.querySelectorAll(".lang-btn");
const modeButtons = document.querySelectorAll(".mode-btn");

let language = "auto";
let mode = "single";
let healthModels = {};

const PLACEHOLDERS = {
  single: {
    auto: "e.g. You are a wonderful person! OR Uri munhu anoshamisa!",
    en: "e.g. You are a wonderful person, thank you for helping me!",
    sn: "e.g. Uri munhu anoshamisa, unokutendai nokuda kwokundibetsera!",
  },
  batch: "Paste comments, one per line (up to 100):\ne.g.\nThanks for your review!\nShut up you idiot.\nNdichakuuraya iwe mbavha.",
};

function updateCharCount() {
  if (mode === "batch") {
    const lines = input.value.split("\n").filter((l) => l.trim().length > 0);
    charCount.textContent = `${lines.length} / ${MAX_BATCH_ITEMS} comments`;
  } else {
    charCount.textContent = `${input.value.length} / ${MAX_LEN}`;
  }
}

function hide(el) {
  el.hidden = true;
}

function show(el) {
  el.hidden = false;
}

function setMode(newMode) {
  mode = newMode;
  for (const btn of modeButtons) {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  }
  if (mode === "batch") {
    inputLabel.textContent = "Comments (one per line)";
    input.placeholder = PLACEHOLDERS.batch;
    checkBtn.textContent = "Check comments (batch)";
  } else {
    inputLabel.textContent = "Comment text";
    input.placeholder = PLACEHOLDERS.single[language] ?? "";
    checkBtn.textContent = "Check comment";
  }
  hide(resultPanel);
  hide(errorPanel);
  updateCharCount();
}

function setLanguage(lang) {
  language = lang;
  for (const btn of langButtons) {
    const active = btn.dataset.lang === lang;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-pressed", String(active));
  }
  if (mode === "single") {
    input.placeholder = PLACEHOLDERS.single[lang] ?? "";
  }
  hide(resultPanel);
  hide(errorPanel);
  updateModelLabel();
}

function updateModelLabel() {
  if (language === "auto") {
    modelNameEl.textContent = "model: auto (toxic-bert / xlm-roberta-shona)";
  } else {
    const name = healthModels[language];
    modelNameEl.textContent = name ? `model: ${name}` : "model: —";
  }
}

function renderSingleResult(result) {
  hide(errorPanel);
  const langBadge = `[${result.language.toUpperCase()}]`;
  verdictEl.textContent = result.is_toxic
    ? `${langBadge} Flagged: ${result.flagged_labels.join(", ")}`
    : `${langBadge} No toxicity detected`;
  verdictEl.className = `verdict ${result.is_toxic ? "flagged" : "clean"}`;

  const entries = Object.entries(result.scores).sort((a, b) => b[1] - a[1]);
  scoresEl.innerHTML = "";
  const thresholds = result.thresholds_used || {};

  for (const [label, score] of entries) {
    const pct = Math.min(100, Math.round(score * 100));
    const threshPct = Math.round((thresholds[label] ?? 0.5) * 100);
    const row = document.createElement("div");
    row.className = "score-row" + (result.flagged_labels.includes(label) ? " is-flagged" : "");
    row.innerHTML = `
      <span class="score-label">${label.replace("_", " ")}</span>
      <span class="score-track">
        <span class="score-fill" style="width:${pct}%"></span>
        <span class="score-threshold-mark" style="left:${threshPct}%" title="Threshold: ${threshPct}%"></span>
      </span>
      <span class="score-value">${pct}%</span>
    `;
    scoresEl.appendChild(row);
  }
  show(resultPanel);
}

function renderBatchResult(batchResult) {
  hide(errorPanel);
  verdictEl.textContent = `Batch Result: ${batchResult.flagged_count} of ${batchResult.total} comments flagged`;
  verdictEl.className = `verdict ${batchResult.flagged_count > 0 ? "flagged" : "clean"}`;

  scoresEl.innerHTML = "";
  const list = document.createElement("div");
  list.className = "batch-list";

  for (const r of batchResult.results) {
    const item = document.createElement("div");
    item.className = `batch-item ${r.is_toxic ? "is-toxic" : ""}`;
    const badgeClass = r.is_toxic ? "flagged" : "clean";
    const badgeText = r.is_toxic ? "Flagged" : "Clean";

    const tagsHtml = r.flagged_labels.length > 0
      ? r.flagged_labels.map((l) => `<span class="batch-tag">${l.replace("_", " ")}</span>`).join("")
      : '<span class="batch-tag">none</span>';

    item.innerHTML = `
      <div class="batch-item-header">
        <span class="batch-badge ${badgeClass}">${badgeText} [${r.language.toUpperCase()}]</span>
        <div class="batch-tags">${tagsHtml}</div>
      </div>
      <div class="batch-text">${escapeHtml(r.text)}</div>
    `;
    list.appendChild(item);
  }

  scoresEl.appendChild(list);
  show(resultPanel);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderError(message) {
  hide(resultPanel);
  errorText.textContent = message;
  show(errorPanel);
}

async function checkComment() {
  const rawText = input.value.trim();
  if (!rawText) {
    renderError("Enter some text first.");
    return;
  }

  checkBtn.disabled = true;
  checkBtn.textContent = mode === "batch" ? "Checking batch..." : "Checking...";

  try {
    if (mode === "batch") {
      const texts = rawText
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0);

      if (texts.length === 0) {
        renderError("Enter at least one non-empty comment.");
        return;
      }
      if (texts.length > MAX_BATCH_ITEMS) {
        renderError(`Too many comments (${texts.length}). Maximum batch size is ${MAX_BATCH_ITEMS}.`);
        return;
      }

      const res = await fetch(`${API_BASE}/moderate/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts, language }),
      });

      if (!res.ok) {
        handleHttpError(res);
        return;
      }

      const data = await res.json();
      renderBatchResult(data);
    } else {
      const res = await fetch(`${API_BASE}/moderate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: rawText, language }),
      });

      if (!res.ok) {
        handleHttpError(res);
        return;
      }

      const result = await res.json();
      renderSingleResult(result);
    }
  } catch (err) {
    renderError("Could not reach the moderation service. Is the backend running?");
  } finally {
    checkBtn.disabled = false;
    checkBtn.textContent = mode === "batch" ? "Check comments (batch)" : "Check comment";
  }
}

async function handleHttpError(res) {
  const body = await res.json().catch(() => ({}));
  if (res.status === 503) {
    renderError(
      "The Shona model hasn't been fine-tuned on this deployment yet. Run " +
        "ml/translation/translate_jigsaw.py then ml/training/train_shona_classifier.py."
    );
    return;
  }
  const detail = typeof body.detail === "string" ? body.detail : "Request failed.";
  renderError(detail);
}

async function loadHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return;
    const body = await res.json();
    healthModels = body.models || {};
    updateModelLabel();
  } catch {
    modelNameEl.textContent = "model: unavailable";
  }
}

input.addEventListener("input", updateCharCount);
checkBtn.addEventListener("click", checkComment);
input.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    checkComment();
  }
});

for (const btn of langButtons) {
  btn.addEventListener("click", () => setLanguage(btn.dataset.lang));
}

for (const btn of modeButtons) {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
}

updateCharCount();
loadHealth();
