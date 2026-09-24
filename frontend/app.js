const API_BASE = "/api/v1";
const MAX_LEN = 5000;

const input = document.getElementById("comment-input");
const charCount = document.getElementById("char-count");
const checkBtn = document.getElementById("check-btn");
const resultPanel = document.getElementById("result-panel");
const errorPanel = document.getElementById("error-panel");
const errorText = document.getElementById("error-text");
const verdictEl = document.getElementById("verdict");
const scoresEl = document.getElementById("scores");
const modelNameEl = document.getElementById("model-name");
const langButtons = document.querySelectorAll(".lang-btn");

let language = "en";
let healthModels = {};

const PLACEHOLDERS = {
  en: "e.g. You are a wonderful person, thank you for helping me!",
  sn: "e.g. Uri munhu anoshamisa, unokutendai nokuda kwokundibetsera!",
};

function updateCharCount() {
  charCount.textContent = `${input.value.length} / ${MAX_LEN}`;
}

function hide(el) {
  el.hidden = true;
}

function show(el) {
  el.hidden = false;
}

function setLanguage(lang) {
  language = lang;
  for (const btn of langButtons) {
    const active = btn.dataset.lang === lang;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-pressed", String(active));
  }
  input.placeholder = PLACEHOLDERS[lang] ?? "";
  hide(resultPanel);
  hide(errorPanel);
  updateModelLabel();
}

function updateModelLabel() {
  const name = healthModels[language];
  modelNameEl.textContent = name ? `model: ${name}` : "model: —";
}

function renderResult(result) {
  hide(errorPanel);
  verdictEl.textContent = result.is_toxic
    ? `Flagged: ${result.flagged_labels.join(", ")}`
    : "No toxicity detected";
  verdictEl.className = `verdict ${result.is_toxic ? "flagged" : "clean"}`;

  const entries = Object.entries(result.scores).sort((a, b) => b[1] - a[1]);
  scoresEl.innerHTML = "";
  for (const [label, score] of entries) {
    const pct = Math.round(score * 100);
    const row = document.createElement("div");
    row.className = "score-row" + (result.flagged_labels.includes(label) ? " is-flagged" : "");
    row.innerHTML = `
      <span class="score-label">${label.replace("_", " ")}</span>
      <span class="score-track"><span class="score-fill" style="width:${pct}%"></span></span>
      <span class="score-value">${pct}%</span>
    `;
    scoresEl.appendChild(row);
  }
  show(resultPanel);
}

function renderError(message) {
  hide(resultPanel);
  errorText.textContent = message;
  show(errorPanel);
}

async function checkComment() {
  const text = input.value.trim();
  if (!text) {
    renderError("Enter some text first.");
    return;
  }

  checkBtn.disabled = true;
  checkBtn.textContent = "Checking...";

  try {
    const res = await fetch(`${API_BASE}/moderate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language }),
    });

    if (!res.ok) {
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
      return;
    }

    const result = await res.json();
    renderResult(result);
  } catch (err) {
    renderError("Could not reach the moderation service. Is the backend running?");
  } finally {
    checkBtn.disabled = false;
    checkBtn.textContent = "Check comment";
  }
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

updateCharCount();
loadHealth();
