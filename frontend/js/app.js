/* Vestígio: lógica da interface. JavaScript puro, sem dependências.
   Todo texto vindo do servidor é inserido com textContent (nunca innerHTML). */
(() => {
  "use strict";

  const FALLBACK_LIMITS = {
    max_upload_bytes: 10 * 1024 * 1024,
    mime_types: ["image/jpeg", "image/png", "image/webp"],
    extensions: [".jpg", ".jpeg", ".png", ".webp"],
  };
  const REQUEST_TIMEOUT_MS = 120000;
  const WEDGE_STEPS = 21;

  const FILE_ERROR_CODES = new Set([
    "NO_FILE", "EMPTY_FILE", "FILE_TOO_LARGE", "UNSUPPORTED_TYPE",
    "INVALID_IMAGE", "IMAGE_TOO_SMALL", "IMAGE_DIMENSIONS_TOO_LARGE",
  ]);
  const ERROR_HINTS = {
    ENGINE_LOADING: "Aguarde o indicador no topo ficar verde. Na primeira execução o modelo é baixado, o que pode levar alguns minutos.",
    ENGINE_UNAVAILABLE: "A causa aparece no terminal do VS Code. Os motivos mais comuns são falta de internet no primeiro download e dependências não instaladas.",
    BUSY: "Aguarde alguns segundos e clique em «Analisar imagem» de novo.",
    INFERENCE_FAILED: "O detalhe técnico está no terminal do VS Code. Se for pedir ajuda, informe a referência acima.",
    DETECTOR_BAD_RESPONSE: "O detalhe técnico está no terminal do VS Code. Se for pedir ajuda, informe a referência acima.",
    FILE_TOO_LARGE: "Reduza o tamanho do arquivo ou exporte a imagem em uma resolução menor.",
    UNSUPPORTED_TYPE: "Converta a imagem para JPEG, PNG ou WebP e tente de novo.",
    INVALID_IMAGE: "Tente abrir o arquivo em outro programa e salvá-lo de novo.",
    IMAGE_TOO_SMALL: "Use uma versão maior da imagem.",
    IMAGE_DIMENSIONS_TOO_LARGE: "Reduza a resolução da imagem e tente de novo.",
  };
  const SIGNAL_TAGS = {
    ai_declared: "Declara IA",
    ai_hint: "Indício de IA",
    camera: "Dados de câmera",
    info: "Informação",
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    engine: $("engine"), engineText: $("engine-text"), engineDetail: $("engine-detail"),
    printWrap: $("print-wrap"), print: $("print"), fileInput: $("file-input"),
    stage: $("stage"), preview: $("preview"), caption: $("caption"),
    capName: $("cap-name"), capDims: $("cap-dims"), capSize: $("cap-size"),
    fileError: $("file-error"), analyzeBtn: $("analyze-btn"), resetBtn: $("reset-btn"),
    actionHint: $("action-hint"), limitText: $("limit-text"),
    introNotice: $("intro-notice"), idleNotes: $("idle-notes"), busy: $("busy"), busyVisual: $("busy-visual"), busyText: $("busy-text"),
    errorBox: $("error-box"), errorTitle: $("error-title"), errorText: $("error-text"),
    errorHint: $("error-hint"), errorCode: $("error-code"),
    result: $("result"), resultTitle: $("result-title"), verdictSummary: $("verdict-summary"),
    models: $("models"), signals: $("signals"), reading: $("reading"), facts: $("facts"),
    resultDisclaimer: $("result-disclaimer"), againBtn: $("again-btn"), live: $("live"),
  };

  const state = {
    limits: FALLBACK_LIMITS,
    engine: null,          // última resposta de /api/health (null = servidor sem resposta)
    file: null,
    previewUrl: null,
    busy: false,
    healthTimer: null,
    busyTimer: null,
  };

  const nf1 = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 1 });
  const nfPct = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

  /* ---------- utilidades ---------- */
  function node(tag, className, text) {
    const n = document.createElement(tag);
    if (className) n.className = className;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function formatBytes(n) {
    if (n >= 1048576) return nf1.format(n / 1048576) + " MB";
    return Math.max(1, Math.round(n / 1024)) + " KB";
  }

  // Nunca mostra 0% nem 100%: o modelo estima, não afirma.
  function formatPercent(p) {
    const v = p * 100;
    if (v >= 99.95) return "mais de 99,9%";
    if (v < 0.05) return "menos de 0,1%";
    return nfPct.format(v) + "%";
  }

  function announce(message) {
    el.live.textContent = "";
    window.setTimeout(() => { el.live.textContent = message; }, 60);
  }

  function engineUsable(engine) {
    return Boolean(engine) && (engine.state === "ready" || engine.state === "degraded");
  }

  /* ---------- estado do motor ---------- */
  function normalizeLimits(raw) {
    if (!raw || typeof raw.max_upload_bytes !== "number" || !Array.isArray(raw.mime_types) || !Array.isArray(raw.extensions)) {
      return FALLBACK_LIMITS;
    }
    return raw;
  }

  async function checkHealth() {
    window.clearTimeout(state.healthTimer);
    let retryIn = 0;
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      if (!res.ok) throw new Error("health " + res.status);
      const data = await res.json();
      state.limits = normalizeLimits(data.limits);
      state.engine = data.engine;
      if (!engineUsable(data.engine) && data.engine.state !== "error") retryIn = 2000;
    } catch (_err) {
      state.engine = null;
      retryIn = 4000;
    }
    renderEngine();
    el.limitText.textContent = formatBytes(state.limits.max_upload_bytes);
    updateControls();
    if (retryIn) state.healthTimer = window.setTimeout(checkHealth, retryIn);
  }

  function renderEngine() {
    const engine = state.engine;
    let key, text, detail = "";
    if (!engine) {
      key = "unreachable"; text = "Servidor sem resposta";
    } else if (engine.state === "ready" || engine.state === "degraded") {
      key = engine.state === "ready" ? "ready" : "degraded";
      text = engine.state === "ready" ? "Motor pronto" : "Motor parcialmente pronto";
      const names = engine.models.filter((m) => m.loaded).map((m) => m.name);
      detail = names.join(", ") + " em " + String(engine.device || "cpu").toUpperCase();
    } else if (engine.state === "error") {
      key = "error"; text = "Motor indisponível";
    } else {
      key = "loading"; text = "Carregando o modelo…";
    }
    el.engine.dataset.state = key;
    el.engineText.textContent = text;
    el.engineDetail.textContent = detail;
    el.engine.title = detail;
  }

  function engineHint() {
    if (!state.engine) return "Sem conexão com o servidor. Confirme que o terminal ainda está executando o Vestígio.";
    if (state.engine.state === "error") return "O modelo não carregou. A causa aparece no terminal do VS Code.";
    return "Aguardando o modelo carregar. Na primeira execução ele é baixado, o que pode levar alguns minutos.";
  }

  function updateControls() {
    const usable = engineUsable(state.engine);
    const hasFile = Boolean(state.file);
    el.analyzeBtn.disabled = state.busy || !(hasFile && usable);
    el.resetBtn.disabled = state.busy || !hasFile;
    el.fileInput.disabled = state.busy;
    el.print.dataset.busy = String(state.busy);

    let hint;
    if (state.busy) hint = "Analisando. Isso costuma levar alguns segundos.";
    else if (!hasFile) hint = "Escolha uma imagem para começar.";
    else if (!usable) hint = engineHint();
    else if (state.engine.state === "degraded") hint = "Um dos modelos não carregou; a análise usará só os que estão prontos.";
    else hint = "Pronto para analisar.";
    el.actionHint.textContent = hint;
  }

  /* ---------- seleção e pré-visualização ---------- */
  function validateFile(file) {
    const limits = state.limits;
    const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
    if (file.type) {
      if (!limits.mime_types.includes(file.type)) return "Formato não suportado. Use JPEG, PNG ou WebP.";
    } else if (!limits.extensions.includes(ext)) {
      return "Formato não suportado. Use JPEG, PNG ou WebP.";
    }
    if (file.size === 0) return "O arquivo está vazio.";
    if (file.size > limits.max_upload_bytes) {
      return `O arquivo tem ${formatBytes(file.size)} e o limite é ${formatBytes(limits.max_upload_bytes)}.`;
    }
    return null;
  }

  function showFileError(message) {
    el.fileError.textContent = message;
    el.fileError.hidden = false;
  }

  function hideFileError() {
    el.fileError.hidden = true;
    el.fileError.textContent = "";
  }

  function handleFiles(files) {
    if (state.busy || !files.length) return;
    const file = files[0];
    const problem = validateFile(file);
    if (problem) {
      showFileError(problem);
      announce(problem);
      return;
    }
    hideFileError();
    if (files.length > 1) showFileError("Apenas uma imagem por vez. Usando a primeira: " + file.name + ".");

    const url = URL.createObjectURL(file);
    const probe = new Image();
    probe.onload = () => commitSelection(file, url, probe.naturalWidth, probe.naturalHeight);
    probe.onerror = () => {
      URL.revokeObjectURL(url);
      showFileError("O navegador não conseguiu abrir este arquivo como imagem. Ele pode estar corrompido.");
    };
    probe.src = url;
  }

  function commitSelection(file, url, width, height) {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = file;
    state.previewUrl = url;
    el.preview.src = url;
    el.capName.textContent = file.name;
    el.capName.title = file.name;
    el.capDims.textContent = `${width} × ${height} px`;
    el.capSize.textContent = formatBytes(file.size);
    el.stage.hidden = false;
    el.caption.hidden = false;
    el.print.dataset.hasImage = "true";
    resetPanels();
    updateControls();
    announce("Imagem selecionada: " + file.name + ".");
  }

  function clearSelection(focusInput) {
    if (state.busy) return;
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = null;
    state.previewUrl = null;
    el.preview.removeAttribute("src");
    el.stage.hidden = true;
    el.caption.hidden = true;
    el.print.dataset.hasImage = "false";
    hideFileError();
    resetPanels();
    updateControls();
    announce("Imagem removida. Escolha outra para analisar.");
    if (focusInput) el.fileInput.focus();
  }

  function resetPanels() {
    el.busy.hidden = true;
    el.errorBox.hidden = true;
    el.result.hidden = true;
    el.idleNotes.hidden = false;
    el.introNotice.hidden = false;
  }

  /* ---------- escala de cinzas ---------- */
  function buildWedge(probability, label) {
    const wedge = node("div", "wedge");
    const track = node("div", "wedge-track");
    const steps = node("div", "wedge-steps");
    for (let i = 0; i < WEDGE_STEPS; i++) {
      const cell = document.createElement("i");
      cell.style.setProperty("--i", String(i));
      steps.appendChild(cell);
    }

    if (probability === null) {            // indicador de andamento, sem valor
      wedge.classList.add("wedge-scanning");
      wedge.setAttribute("aria-hidden", "true");
      track.appendChild(steps);
      wedge.appendChild(track);
      return wedge;
    }

    wedge.setAttribute("role", "img");
    wedge.setAttribute("aria-label", label);
    const marker = node("span", "wedge-marker");
    marker.setAttribute("aria-hidden", "true");
    track.append(marker, steps);

    const zone = node("div", "wedge-zone");
    const labels = node("div", "wedge-labels");
    zone.setAttribute("aria-hidden", "true");
    labels.setAttribute("aria-hidden", "true");
    labels.append(node("span", "", "Humana"), node("span", "", "faixa inconclusiva"), node("span", "", "IA"));
    wedge.append(track, zone, labels);

    // O marcador parte da esquerda e desliza até a pontuação do modelo.
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
      marker.style.left = (probability * 100) + "%";
    }));
    return wedge;
  }

  /* ---------- análise ---------- */
  function setBusy(on) {
    state.busy = on;
    el.analyzeBtn.textContent = on ? "Analisando…" : "Analisar imagem";
    el.analyzeBtn.setAttribute("aria-busy", String(on));
    window.clearInterval(state.busyTimer);
    if (on) {
      const started = Date.now();
      el.busyText.textContent = "Analisando a imagem…";
      el.busyVisual.replaceChildren(buildWedge(null));
      state.busyTimer = window.setInterval(() => {
        const s = Math.floor((Date.now() - started) / 1000);
        el.busyText.textContent = s < 2 ? "Analisando a imagem…" : `Analisando a imagem há ${s} s…`;
      }, 500);
    } else {
      el.busyVisual.replaceChildren();
    }
    updateControls();
  }

  async function analyze() {
    if (state.busy || !state.file) return;           // evita envios duplicados
    if (!engineUsable(state.engine)) { updateControls(); return; }
    const problem = validateFile(state.file);
    if (problem) { showFileError(problem); return; }

    hideFileError();
    el.idleNotes.hidden = true;
    el.errorBox.hidden = true;
    el.result.hidden = true;
    el.busy.hidden = false;
    setBusy(true);
    announce("Análise iniciada.");

    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const body = new FormData();
    body.append("image", state.file, state.file.name);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body, signal: controller.signal });
      let payload = null;
      try { payload = await res.json(); } catch (_err) { payload = null; }

      if (!res.ok) {
        showServerError(res.status, payload && payload.error);
      } else if (!isValidPayload(payload)) {
        showError({
          title: "Resposta inesperada do servidor",
          text: "O servidor respondeu, mas os dados não têm o formato esperado.",
          hint: "Recarregue a página. Se o problema continuar, veja o terminal do VS Code.",
        });
      } else {
        renderResult(payload);
      }
    } catch (err) {
      if (err && err.name === "AbortError") {
        showError({
          title: "A análise demorou demais",
          text: "O servidor não respondeu em 2 minutos.",
          hint: "Em computadores mais lentos, tente uma imagem menor.",
        });
      } else {
        showError({
          title: "Sem conexão com o servidor",
          text: "Não foi possível falar com o Vestígio.",
          hint: `Confirme que o terminal do VS Code ainda está executando «python run.py» e que esta página foi aberta em ${location.origin}.`,
        });
        checkHealth();
      }
    } finally {
      window.clearTimeout(timeout);
      setBusy(false);
    }
  }

  function isValidPayload(p) {
    return Boolean(p)
      && p.verdict && typeof p.verdict.code === "string" && typeof p.verdict.headline === "string"
      && Array.isArray(p.models) && p.models.length > 0
      && p.models.every((m) => typeof m.name === "string" && typeof m.ai_probability === "number" && m.raw_labels)
      && p.metadata && Array.isArray(p.metadata.signals)
      && p.image && Array.isArray(p.notes);
  }

  function showServerError(status, error) {
    if (!error || typeof error.message !== "string") {
      showError({ title: "Resposta inesperada do servidor", text: `O servidor respondeu com o status ${status}.`, hint: "Veja o terminal do VS Code." });
      return;
    }
    const isFile = FILE_ERROR_CODES.has(error.code);
    showError({
      title: isFile ? "Não foi possível usar este arquivo" : "A análise não foi concluída",
      text: error.message,
      hint: ERROR_HINTS[error.code] || "",
      code: error.code,
    });
    if (error.code === "ENGINE_LOADING" || error.code === "ENGINE_UNAVAILABLE") checkHealth();
  }

  function showError({ title, text, hint, code }) {
    el.errorTitle.textContent = title;
    el.errorText.textContent = text;
    el.errorHint.textContent = hint || "";
    el.errorHint.hidden = !hint;
    el.errorCode.textContent = code ? "Código: " + code : "";
    el.errorCode.hidden = !code;
    el.idleNotes.hidden = true;
    el.busy.hidden = true;
    el.result.hidden = true;
    el.errorBox.hidden = false;
    announce(title + ". " + text);
  }

  /* ---------- laudo ---------- */
  function buildModelItem(m) {
    const li = node("li", "model");
    const head = node("div", "model-head");
    const score = node("span", "model-score");
    score.append(node("strong", "", formatPercent(m.ai_probability)), document.createTextNode(" de score para IA"));
    head.append(node("span", "model-name", m.name), score);

    const label = `${m.name}: score do modelo de ${formatPercent(m.ai_probability)} para IA, em uma escala que vai da classe humana à classe IA.`;
    const raw = Object.entries(m.raw_labels).map(([k, v]) => `${k} ${formatPercent(v)}`).join(", ");
    li.append(head, buildWedge(m.ai_probability, label), node("p", "model-raw", `Scores retornados pelo modelo: ${raw}.`));
    return li;
  }

  function buildSignalItem(s) {
    const li = node("li", "signal");
    const body = node("div");
    body.append(node("p", "signal-title", s.title), node("p", "signal-detail", s.detail));
    li.append(node("span", "tag tag-" + (SIGNAL_TAGS[s.kind] ? s.kind : "info"), SIGNAL_TAGS[s.kind] || SIGNAL_TAGS.info), body);
    return li;
  }

  function buildFact(name, value) {
    const group = node("div");
    group.append(node("dt", "", name), node("dd", "", value));
    return group;
  }

  function renderResult(data) {
    el.result.dataset.verdict = data.verdict.code;
    el.resultTitle.textContent = data.verdict.headline;
    el.verdictSummary.textContent = data.verdict.summary;
    el.models.replaceChildren(...data.models.map(buildModelItem));
    el.signals.replaceChildren(...data.metadata.signals.map(buildSignalItem));
    el.reading.replaceChildren(...data.notes.map((t) => node("li", "", t)));
    el.resultDisclaimer.textContent = data.disclaimer || "";

    const img = data.image;
    el.facts.replaceChildren(
      buildFact("Formato", String(img.format)),
      buildFact("Dimensões", `${img.width} × ${img.height} px`),
      buildFact("Tamanho", formatBytes(img.size_bytes)),
      buildFact("Tempo de análise", data.elapsed_ms < 1000 ? data.elapsed_ms + " ms" : nf1.format(data.elapsed_ms / 1000) + " s"),
    );

    el.idleNotes.hidden = true;
    el.errorBox.hidden = true;
    el.busy.hidden = true;
    el.introNotice.hidden = true;       // o laudo traz o mesmo aviso no final
    el.result.hidden = false;
    el.resultTitle.focus();
    announce("Análise concluída. " + data.verdict.headline + ". " + data.verdict.summary);
  }

  /* ---------- eventos ---------- */
  el.fileInput.addEventListener("change", () => {
    const files = Array.from(el.fileInput.files || []);
    el.fileInput.value = "";                 // permite escolher o mesmo arquivo de novo
    handleFiles(files);
  });

  el.analyzeBtn.addEventListener("click", analyze);
  el.resetBtn.addEventListener("click", () => clearSelection(true));
  el.againBtn.addEventListener("click", () => clearSelection(true));

  const hasFiles = (e) => Array.from((e.dataTransfer && e.dataTransfer.types) || []).includes("Files");

  ["dragenter", "dragover"].forEach((type) => {
    el.printWrap.addEventListener(type, (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = state.busy ? "none" : "copy";
      if (!state.busy) el.print.dataset.dragging = "true";
    });
  });
  el.printWrap.addEventListener("dragleave", (e) => {
    if (e.relatedTarget && el.printWrap.contains(e.relatedTarget)) return;
    el.print.dataset.dragging = "false";
  });
  el.printWrap.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    el.print.dataset.dragging = "false";
    handleFiles(Array.from(e.dataTransfer.files || []));
  });
  // Soltar um arquivo fora da área não deve fazer o navegador abrir a imagem e sair da página.
  ["dragover", "drop"].forEach((type) => {
    document.addEventListener(type, (e) => { if (hasFiles(e)) e.preventDefault(); });
  });

  checkHealth();
  updateControls();
})();
