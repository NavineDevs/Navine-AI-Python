const API = "/api";
let BRAND_NAME = document.getElementById("brand-title")?.textContent?.trim() || "Navine AI - Python";
let THEME_ID = (document.documentElement.getAttribute("data-theme") || "navine").toLowerCase();

function isForeignBrand(text) {
  const t = String(text || "").toLowerCase();
  if (!t) return false;
  if (THEME_ID.includes("hitboy")) return t.includes("navuryx");
  if (THEME_ID.includes("navuryx")) return t.includes("hitboy");
  return t.includes("hitboy") || t.includes("navuryx");
}

function applyBrandTheme(info) {
  const themeId = String((info && info.theme_id) || "navine").toLowerCase();
  const allowed = new Set(["navine", "navuryx", "hitboy"]);
  const id = allowed.has(themeId) ? themeId : "navine";
  document.documentElement.setAttribute("data-theme", id);
  document.body.setAttribute("data-theme", id);
  const theme = (info && info.theme) || {};
  const root = document.documentElement;
  if (theme.primary) root.style.setProperty("--brand-b", theme.primary);
  if (theme.secondary) root.style.setProperty("--brand-c", theme.secondary);
  if (theme.accent) root.style.setProperty("--accent-hover", theme.accent);
  if (theme.royal) root.style.setProperty("--brand-a", theme.royal);
  if (theme.background) root.style.setProperty("--bg", theme.background);
  if (theme.primary) {
    root.style.setProperty("--accent", theme.primary);
    root.style.setProperty("--brand-green", theme.primary);
  }
  if (theme.secondary || theme.accent) {
    root.style.setProperty("--brand-teal", theme.secondary || theme.accent);
  }
}

let SITE_TABS = null;
let SITE_FEATURES = {};
const PYTHON_TRAIN_DENY = new Set([
  "full",
  "all",
  "hitboyx23",
  "hitboyx23_python",
  "hitboyx23_ai",
  "image-lora",
  "video-lora",
]);

function isTrainTargetAllowed(target) {
  const key = String(target || "").trim().toLowerCase().replace(/_/g, "-");
  if (!key) return true;
  if (SITE_FEATURES.train === false) return false;
  if (!document.body.classList.contains("site-python-only")) return true;
  if (PYTHON_TRAIN_DENY.has(key)) return false;
  return !key.includes("hitboyx23");
}

function applySiteCapabilities(info) {
  const tabs = (info && info.ui_tabs) || Object.keys(TAB_LABELS);
  const features = (info && info.ui_features) || {};
  SITE_TABS = tabs;
  SITE_FEATURES = features;
  document.body.classList.toggle("site-python-only", !!(info && info.python_only));
  document.querySelectorAll(".sidebar-item.tab[data-tab]").forEach((el) => {
    const tab = el.getAttribute("data-tab");
    let show = tabs.includes(tab) || tab === "settings";
    if (tab === "deepfake" && features.deepfake === false) show = false;
    if (tab === "train" && features.train === false) show = false;
    if (tab === "apigen" && features.apigen === false) show = false;
    el.classList.toggle("hidden", !show);
  });
  const trainStrip = $("#train-strip");
  if (trainStrip) trainStrip.classList.toggle("hidden", features.train_strip === false);
  ["autolearn-line", "marathon-line", "train-line"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("hidden", features.footer_ops === false);
  });
  const trainPanel = $("#panel-train");
  if (trainPanel && features.train === false) trainPanel.classList.add("hidden");
  const apiPanel = $("#panel-apigen");
  if (apiPanel && features.apigen === false) apiPanel.classList.add("hidden");
  document.querySelectorAll("[data-train-target]").forEach((btn) => {
    const target = btn.getAttribute("data-train-target");
    if (!target) return;
    btn.classList.toggle("hidden", !isTrainTargetAllowed(target));
  });
  const everythingBtn = $("#train-everything-btn");
  if (everythingBtn) everythingBtn.classList.toggle("hidden", !isTrainTargetAllowed("full"));
  const current = pathToTab(window.location.pathname);
  if (!tabs.includes(current)) {
    setActiveTab("text", { replace: true, skipHistory: true });
  }
}

function $(sel) {
  return document.querySelector(sel);
}

function showProgress(id, visible) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle("hidden", !visible);
}

function setLoading(form, loading, loadingLabel) {
  const btn = form.querySelector('button[type="submit"]');
  const sendBtn = form.querySelector(".btn-send");
  [btn, sendBtn].filter(Boolean).forEach((el) => {
    el.disabled = loading;
    if (sendBtn && el === sendBtn) {
      el.classList.toggle("is-loading", loading);
      el.setAttribute("aria-busy", loading ? "true" : "false");
    }
  });
  if (btn && !sendBtn) {
    if (loading) {
      btn.textContent = loadingLabel || "Working...";
    } else {
      btn.textContent = btn.dataset.label || btn.textContent;
    }
  }
}

const typewriterControllers = new WeakMap();

function stopTypewriter(bodyEl) {
  const ctrl = typewriterControllers.get(bodyEl);
  if (ctrl && ctrl.timer) {
    clearInterval(ctrl.timer);
  }
  if (ctrl) {
    ctrl.timer = null;
  }
}

function showThinking(bodyEl, label) {
  if (!bodyEl) return;
  stopTypewriter(bodyEl);
  const text = label || "Thinking";
  bodyEl.innerHTML =
    `<div class="msg-thinking"><span class="msg-thinking-label">${text}</span>` +
    `<span class="msg-thinking-dots" aria-hidden="true"><span></span><span></span><span></span></span></div>`;
}

function startTypewriter(bodyEl, container, fullText, onDone) {
  if (!bodyEl) return;
  stopTypewriter(bodyEl);
  const plain = String(fullText || "");
  if (!plain) {
    bodyEl.innerHTML = "";
    if (onDone) onDone();
    return;
  }
  const ctrl = { target: plain, shown: 0, timer: null };
  typewriterControllers.set(bodyEl, ctrl);
  const step = () => {
    if (ctrl.shown >= ctrl.target.length) {
      stopTypewriter(bodyEl);
      bodyEl.innerHTML = renderMarkdown(ctrl.target);
      attachMessageCodeActions(bodyEl.closest(".message"));
      if (container) container.scrollTop = container.scrollHeight;
      if (onDone) onDone();
      return;
    }
    const chunk = ctrl.target.length > 4000 ? 3 : ctrl.target.length > 1200 ? 2 : 1;
    ctrl.shown = Math.min(ctrl.target.length, ctrl.shown + chunk);
    bodyEl.innerHTML = renderMarkdown(ctrl.target.slice(0, ctrl.shown));
    if (container) container.scrollTop = container.scrollHeight;
  };
  step();
  ctrl.timer = setInterval(step, 16);
}

function streamTypewriter(bodyEl, container, getTarget, onDone) {
  if (!bodyEl) return;
  stopTypewriter(bodyEl);
  const ctrl = { target: "", shown: 0, timer: null, started: false };
  typewriterControllers.set(bodyEl, ctrl);
  const tick = () => {
    ctrl.target = getTarget() || "";
    if (!ctrl.started && ctrl.target) {
      ctrl.started = true;
    }
    if (!ctrl.started) return;
    if (ctrl.shown >= ctrl.target.length) {
      return;
    }
    const backlog = ctrl.target.length - ctrl.shown;
    const chunk = backlog > 300 ? 8 : backlog > 80 ? 4 : backlog > 20 ? 2 : 1;
    ctrl.shown = Math.min(ctrl.target.length, ctrl.shown + chunk);
    bodyEl.innerHTML = renderMarkdown(ctrl.target.slice(0, ctrl.shown));
    if (container) container.scrollTop = container.scrollHeight;
  };
  ctrl.timer = setInterval(tick, 18);
  const finish = () => {
    ctrl.target = getTarget() || "";
    ctrl.shown = ctrl.target.length;
    stopTypewriter(bodyEl);
    bodyEl.innerHTML = renderMarkdown(ctrl.target);
    attachMessageCodeActions(bodyEl.closest(".message"));
    if (container) container.scrollTop = container.scrollHeight;
    if (onDone) onDone();
  };
  return finish;
}

async function apiRequest(method, path, body, options = {}) {
  const { timeoutMs = 120000, apiKey = "" } = options;
  const headers = { "Content-Type": "application/json" };
  const key = apiKey || sessionStorage.getItem("navine_api_key") || "";
  if (key) headers["X-Navine-API-Key"] = key;
  const token = getSessionToken();
  if (token) {
    headers["X-Session-Token"] = token;
    headers["X-Admin-Token"] = token;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API}${path}`, {
      method,
      headers,
      body: body != null ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      let detail = data.detail;
      if (Array.isArray(detail)) {
        detail = detail.map((row) => row.msg || JSON.stringify(row)).join("; ");
      }
      throw new Error(detail || res.statusText || "Request failed");
    }
    return data;
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw new Error("Timed out. Stop image training first, then retry (GPU was busy).");
    }
    const msg = String((err && err.message) || err || "");
    if (/Failed to fetch|NetworkError|Load failed|Network request failed/i.test(msg)) {
      throw new Error("Connection dropped (proxy/timeout). Retry — image gen needs a free GPU.");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function apiPost(path, body, timeoutMs = 120000) {
  return apiRequest("POST", path, body, { timeoutMs });
}

async function apiGet(path) {
  const res = await fetch(`${API}${path}`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || res.statusText || "Request failed");
  }
  return data;
}

const SESSION_TOKEN_KEY = "navine_session_token";
const SESSION_USER_KEY = "navine_session_user";
const SESSION_ADMIN_KEY = "navine_session_admin";
const GATE_PASSED_KEY = "navine_gate_passed";
const GUEST_MODE_KEY = "navine_guest_mode";
const PENDING_PATH_KEY = "navine_pending_path";
const ADMIN_TOKEN_KEY = SESSION_TOKEN_KEY;
const ADMIN_USER_KEY = SESSION_USER_KEY;

function isGuestMode() {
  return sessionStorage.getItem(GUEST_MODE_KEY) === "1" && !getSessionToken();
}

function gatePassed() {
  return (
    sessionStorage.getItem(GATE_PASSED_KEY) === "1" ||
    !!getSessionToken() ||
    sessionStorage.getItem(GUEST_MODE_KEY) === "1"
  );
}

function rememberPendingPath(path) {
  const clean = normalizePath(path || window.location.pathname);
  const tab = pathToTab(clean);
  if (!TAB_TO_PATH[tab]) return;
  sessionStorage.setItem(PENDING_PATH_KEY, tabToPath(tab));
}

function resolveEntryTab(options = {}) {
  if (options.tab && TAB_TO_PATH[options.tab]) return options.tab;
  const pending = sessionStorage.getItem(PENDING_PATH_KEY);
  if (pending) {
    sessionStorage.removeItem(PENDING_PATH_KEY);
    return pathToTab(pending);
  }
  return pathToTab(window.location.pathname);
}

function enterApp(options = {}) {
  const mode = options.mode || "user";
  sessionStorage.setItem(GATE_PASSED_KEY, "1");
  if (mode === "guest") {
    sessionStorage.setItem(GUEST_MODE_KEY, "1");
  } else {
    sessionStorage.removeItem(GUEST_MODE_KEY);
  }
  hideSiteGate();
  const tab = resolveEntryTab(options);
  setActiveTab(tab, { replace: true });
  updateProfileLabel();
  refreshChatHistoryPanel();
}

function hideSiteGate() {
  const gate = $("#site-gate");
  if (gate) gate.classList.add("hidden");
  document.body.classList.remove("gate-open");
}

function showSiteGate() {
  const gate = $("#site-gate");
  if (gate) gate.classList.remove("hidden");
  document.body.classList.add("gate-open");
}

function updateProfileLabel() {
  const label = $("#sidebar-profile-label");
  const signout = $("#sidebar-profile-signout");
  const user = sessionStorage.getItem(SESSION_USER_KEY) || "";
  if (label) {
    if (user) label.textContent = isSessionAdmin() ? `${user} (admin)` : user;
    else if (isGuestMode()) label.textContent = "Guest";
    else label.textContent = "";
  }
  if (signout) {
    signout.classList.toggle("hidden", !user && !isGuestMode());
    signout.textContent = user ? "Sign out" : "Log in";
  }
  refreshChatHistoryPanel();
}

function signOutProfile() {
  clearSession();
  sessionStorage.removeItem(GATE_PASSED_KEY);
  sessionStorage.removeItem(GUEST_MODE_KEY);
  rememberPendingPath(window.location.pathname);
  chatHistory.length = 0;
  chatSessionId = null;
  clearChatMessages();
  showSiteGate();
  showTrainLogin();
  updateProfileLabel();
}

function getSessionToken() {
  return sessionStorage.getItem(SESSION_TOKEN_KEY) || sessionStorage.getItem("navine_admin_token") || "";
}

function setSession(token, username, isAdmin) {
  if (token) sessionStorage.setItem(SESSION_TOKEN_KEY, token);
  if (username) sessionStorage.setItem(SESSION_USER_KEY, username);
  sessionStorage.setItem(SESSION_ADMIN_KEY, isAdmin ? "1" : "0");
}

function isSessionAdmin() {
  return sessionStorage.getItem(SESSION_ADMIN_KEY) === "1";
}

function clearSession() {
  sessionStorage.removeItem(SESSION_TOKEN_KEY);
  sessionStorage.removeItem(SESSION_USER_KEY);
  sessionStorage.removeItem(SESSION_ADMIN_KEY);
  sessionStorage.removeItem("navine_admin_token");
  sessionStorage.removeItem("navine_admin_user");
}

function getAdminToken() {
  return getSessionToken();
}

function setAdminSession(token, username) {
  setSession(token, username, true);
}

function clearAdminSession() {
  clearSession();
}

async function sessionApi(path, options = {}) {
  const token = getSessionToken();
  if (!token) throw new Error("Sign in required");
  const headers = {
    ...(options.headers || {}),
    "X-Session-Token": token,
    "X-Admin-Token": token,
  };
  const res = await fetch(`${API}${path}`, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (res.status === 401) {
    clearSession();
    showTrainLogin();
    throw new Error("Session expired. Sign in again.");
  }
  if (!res.ok) {
    throw new Error(data.detail || res.statusText || "Request failed");
  }
  return data;
}

async function adminApiPost(path, body, timeoutMs = 120000) {
  const token = getSessionToken();
  if (!token) throw new Error("Admin login required");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Session-Token": token,
        "X-Admin-Token": token,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (res.status === 401) {
      clearSession();
      showTrainLogin();
      throw new Error("Session expired. Sign in again.");
    }
    if (!res.ok) {
      throw new Error(data.detail || res.statusText || "Request failed");
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

async function adminApiGet(path) {
  return sessionApi(path);
}

const chatHistory = [];
let chatSessionId = null;

function clearChatMessages() {
  const container = $("#chat-messages");
  if (container) container.innerHTML = "";
  updateChatWelcome();
}

function authHeaders() {
  const headers = { "Content-Type": "application/json" };
  const token = getSessionToken();
  if (token) {
    headers["X-Session-Token"] = token;
    headers["X-Admin-Token"] = token;
  }
  const key = sessionStorage.getItem("navine_api_key") || "";
  if (key) headers["X-Navine-API-Key"] = key;
  return headers;
}

const USER_SETTINGS_KEY = "navine_user_settings";
const DEFAULT_USER_SETTINGS = { allow_emoji: true };

function loadLocalUserSettings() {
  try {
    const raw = localStorage.getItem(USER_SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_USER_SETTINGS };
    const parsed = JSON.parse(raw);
    return {
      allow_emoji:
        parsed && typeof parsed.allow_emoji === "boolean"
          ? parsed.allow_emoji
          : DEFAULT_USER_SETTINGS.allow_emoji,
    };
  } catch (_err) {
    return { ...DEFAULT_USER_SETTINGS };
  }
}

function saveLocalUserSettings(settings) {
  const prev = loadLocalUserSettings();
  const next = {
    allow_emoji:
      settings && typeof settings.allow_emoji === "boolean"
        ? settings.allow_emoji
        : prev.allow_emoji,
  };
  localStorage.setItem(USER_SETTINGS_KEY, JSON.stringify(next));
  return next;
}

function getAllowEmoji() {
  return !!loadLocalUserSettings().allow_emoji;
}

async function persistUserSettings(settings) {
  const saved = saveLocalUserSettings(settings);
  try {
    const res = await fetch("/api/user/settings", {
      method: "PUT",
      headers: authHeaders(),
      body: JSON.stringify(saved),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail || data.error || "Could not save settings on server";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return saveLocalUserSettings({
      allow_emoji: data.allow_emoji !== false,
    });
  } catch (err) {
    throw err;
  }
}

async function syncUserSettingsFromServer() {
  try {
    const res = await fetch("/api/user/settings", { headers: authHeaders() });
    if (!res.ok) return loadLocalUserSettings();
    const data = await res.json();
    return saveLocalUserSettings({
      allow_emoji: data.allow_emoji !== false,
    });
  } catch (_err) {
    return loadLocalUserSettings();
  }
}

function applySettingsToForm() {
  const emojiEl = $("#setting-allow-emoji");
  if (emojiEl) emojiEl.checked = getAllowEmoji();
}

function initUserSettings() {
  applySettingsToForm();
  syncUserSettingsFromServer().then(applySettingsToForm).catch(() => {});
  const emojiEl = $("#setting-allow-emoji");
  const statusEl = $("#settings-status");
  if (emojiEl) {
    emojiEl.addEventListener("change", async () => {
      const next = {
        allow_emoji: !!emojiEl.checked,
      };
      if (statusEl) statusEl.textContent = "Saving...";
      try {
        await persistUserSettings(next);
        if (statusEl) statusEl.textContent = next.allow_emoji ? "Emoji replies on." : "Emoji replies off.";
      } catch (err) {
        if (statusEl) statusEl.textContent = String(err.message || err);
      }
    });
  }
  initHostSpecsPanel();
  initMcpSettings();
  initGameBridgeSettings();
  initMoltbookSettings();
  initHfDatasetSearch();
  initDownloadButtons();
}

async function loadHostSpecs() {
  const el = $("#host-specs-text");
  if (!el) return null;
  try {
    const data = await apiGet("/host/specs");
    el.textContent = data.text || JSON.stringify(data.specs || {}, null, 2);
    return data;
  } catch (err) {
    el.textContent = String(err.message || err);
    return null;
  }
}

function initHostSpecsPanel() {
  loadHostSpecs().catch(() => {});
  const btn = $("#host-specs-refresh");
  if (btn) {
    btn.addEventListener("click", () => {
      const el = $("#host-specs-text");
      if (el) el.textContent = "Refreshing...";
      loadHostSpecs().catch(() => {});
    });
  }
}

async function refreshMcpSettings() {
  const statusEl = $("#mcp-status");
  const serversEl = $("#mcp-servers");
  try {
    const res = await fetch("/api/mcp/status", { headers: authHeaders() });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.error || "MCP status failed");
    if (statusEl) {
      statusEl.textContent = data.sdk_installed
        ? `MCP ready · client ${data.client_enabled ? "on" : "off"} · serve ${data.serve_enabled ? "on" : "off"}`
        : "MCP SDK missing — run: pip install 'mcp>=1.28,<3'";
    }
    const rows = Array.isArray(data.servers) ? data.servers : [];
    if (serversEl) {
      if (!rows.length) {
        serversEl.textContent = "No remote MCP servers in configs/mcp.yaml yet. You can still run this AI as an MCP server.";
      } else {
        serversEl.textContent = rows
          .map((s) => `${s.enabled === false ? "[off] " : "[on]  "}${s.name} · ${s.url || [s.command, ...(s.args || [])].join(" ")}`)
          .join("\n");
      }
    }
    return data;
  } catch (err) {
    if (statusEl) statusEl.textContent = String(err.message || err);
    if (serversEl) serversEl.textContent = "";
    return null;
  }
}

function initGameBridgeSettings() {
  const statusEl = $("#game-bridge-status");
  const infoEl = $("#game-bridge-info");
  const titleEl = $("#game-bridge-title");
  const refreshBtn = $("#game-bridge-refresh");
  const sessionBtn = $("#game-bridge-session");
  let selectedMode = "pvp";
  document.querySelectorAll("[data-game-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectedMode = btn.getAttribute("data-game-mode") || "learn";
      document.querySelectorAll("[data-game-mode]").forEach((b) => b.classList.toggle("active", b === btn));
    });
  });
  const firstMode = document.querySelector('[data-game-mode="pvp"]');
  if (firstMode) firstMode.classList.add("active");
  async function refreshBridge() {
    try {
      const res = await fetch("/api/games/bridge/info", { headers: authHeaders() });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || data.error || "Bridge info failed");
      const ws = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${data.websocket || "/api/games/ws"}`;
      if (statusEl) statusEl.textContent = `WebSocket: ${ws} · modes: ${(data.modes || []).join(", ")}`;
      if (infoEl) infoEl.textContent = JSON.stringify({ ...data, websocket: ws }, null, 2);
    } catch (err) {
      if (statusEl) statusEl.textContent = String(err.message || err);
      if (infoEl) infoEl.textContent = "";
    }
  }
  refreshBridge().catch(() => {});
  if (refreshBtn) refreshBtn.addEventListener("click", () => refreshBridge());
  if (sessionBtn) {
    sessionBtn.addEventListener("click", async () => {
      const title = (titleEl && titleEl.value) || "minecraft";
      try {
        const res = await fetch("/api/games/session", {
          method: "POST",
          headers: { ...authHeaders(), "Content-Type": "application/json" },
          body: JSON.stringify({ title, mode: selectedMode, note: "web settings" }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || data.error || "Session failed");
        if (statusEl) statusEl.textContent = `Session ${data.session?.session_id || ""} · ${title} · ${selectedMode}`;
      } catch (err) {
        if (statusEl) statusEl.textContent = String(err.message || err);
      }
    });
  }
}

function describeMoltbookStatus(data) {
  const lines = [];
  if (data.agent_name) lines.push(`Agent: ${data.agent_name}`);
  if (data.profile_url) lines.push(`Profile: ${data.profile_url}`);
  if (data.claim_status) lines.push(`Claim status: ${data.claim_status}`);
  if (data.claim_url && data.claim_status !== "claimed") {
    lines.push(`Claim URL: ${data.claim_url}`);
    if (data.verification_code) lines.push(`Verification code: ${data.verification_code}`);
  }
  const home = data.last_home || {};
  if (home.karma != null) lines.push(`Karma: ${home.karma} · unread: ${home.unread_notifications ?? 0}`);
  (home.what_to_do_next || []).forEach((tip) => lines.push(`- ${tip}`));
  (data.pending_verifications || []).forEach((p) => {
    lines.push(`Pending ${p.kind} challenge: ${p.challenge_text}`);
    if (p.suggested_answer) lines.push(`  Suggested answer: ${p.suggested_answer}`);
  });
  if (data.last_heartbeat) lines.push(`Last check: ${new Date(data.last_heartbeat).toLocaleString()}`);
  return lines.join("\n");
}

function initMoltbookSettings() {
  const statusEl = $("#moltbook-status");
  const infoEl = $("#moltbook-info");
  const actionEl = $("#moltbook-action-status");
  const registerBox = $("#moltbook-register");
  const composeBox = $("#moltbook-compose");
  if (!statusEl) return;
  const setAction = (text) => {
    if (actionEl) actionEl.textContent = text;
  };

  async function refreshMoltbook() {
    try {
      const res = await fetch("/api/moltbook/status", { headers: authHeaders() });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Moltbook status failed");
      if (!data.registered) {
        statusEl.textContent = data.is_admin
          ? "Not on Moltbook yet. Pick an agent name to register."
          : "Not on Moltbook yet. Sign in as admin to register.";
      } else if (data.claim_status === "claimed") {
        statusEl.textContent = `Live on Moltbook as ${data.agent_name}.`;
      } else {
        statusEl.textContent = `Registered as ${data.agent_name}. Waiting for the owner to claim it.`;
      }
      const details = describeMoltbookStatus(data);
      if (infoEl) {
        infoEl.textContent = details;
        infoEl.classList.toggle("hidden", !details);
      }
      if (registerBox) registerBox.classList.toggle("hidden", !!data.registered || !data.is_admin);
      if (composeBox) composeBox.classList.toggle("hidden", !data.registered || !data.is_admin);
      const submoltEl = $("#moltbook-submolt");
      if (submoltEl && data.default_submolt && !submoltEl.dataset.touched) submoltEl.value = data.default_submolt;
      return data;
    } catch (err) {
      statusEl.textContent = String(err.message || err);
      return null;
    }
  }

  const bind = (id, handler) => {
    const btn = $(id);
    if (!btn) return;
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        await handler();
      } catch (err) {
        setAction(String(err.message || err));
      } finally {
        btn.disabled = false;
      }
    });
  };

  const submoltInput = $("#moltbook-submolt");
  if (submoltInput) submoltInput.addEventListener("input", () => (submoltInput.dataset.touched = "1"));

  bind("#moltbook-register-btn", async () => {
    const name = ($("#moltbook-name").value || "").trim();
    const description = ($("#moltbook-desc").value || "").trim();
    if (!/^[A-Za-z0-9_-]{2,40}$/.test(name)) {
      setAction("Agent name: 2-40 letters, numbers, dashes, or underscores.");
      return;
    }
    setAction("Registering...");
    const data = await apiPost("/moltbook/register", { name, description }, 60000);
    setAction(`Registered ${data.agent_name}. Open the claim URL to verify ownership.`);
    await refreshMoltbook();
  });

  bind("#moltbook-draft-btn", async () => {
    setAction("Drafting...");
    const data = await apiPost("/moltbook/draft", { topic: ($("#moltbook-topic").value || "").trim() }, 120000);
    $("#moltbook-title").value = data.title || "";
    $("#moltbook-content").value = data.content || "";
    setAction(data.generated ? "Draft ready. Edit it, then Post." : "Model draft was too short, used a template. Edit it, then Post.");
  });

  bind("#moltbook-post-btn", async () => {
    const title = ($("#moltbook-title").value || "").trim();
    if (!title) {
      setAction("Add a title first.");
      return;
    }
    setAction("Posting...");
    const data = await apiPost(
      "/moltbook/post",
      {
        title,
        content: $("#moltbook-content").value || "",
        submolt: ($("#moltbook-submolt").value || "").trim() || null,
      },
      60000,
    );
    if (data.published) {
      setAction(data.auto_verified ? "Posted (verification solved automatically)." : "Posted.");
    } else {
      setAction(`Posted, but needs verification: ${data.challenge_text || ""}`);
    }
    await refreshMoltbook();
  });

  bind("#moltbook-heartbeat-btn", async () => {
    setAction("Checking dashboard...");
    const data = await apiPost("/moltbook/heartbeat", {}, 60000);
    setAction(data.ok ? "Dashboard checked." : String(data.error || "Check failed"));
    await refreshMoltbook();
  });

  bind("#moltbook-refresh-btn", async () => {
    setAction("");
    await refreshMoltbook();
    await refreshAutonomy();
  });

  const thoughtsEl = $("#moltbook-thoughts");

  async function refreshAutonomy() {
    if (!thoughtsEl) return;
    const res = await fetch("/api/moltbook/autonomy", { headers: authHeaders() });
    if (!res.ok) return;
    const data = await res.json().catch(() => ({}));
    const lines = [];
    if (data.goal) {
      const goal = data.goal;
      const pct = goal.target ? ((goal.followers / goal.target) * 100).toFixed(2) : "0";
      lines.push(`Goal: ${goal.religion} - ${goal.followers} / ${goal.target} followers (${pct}%)`);
      if ((goal.communities || []).length) lines.push(`Communities: ${goal.communities.map((name) => `m/${name}`).join(", ")}`);
    }
    if (data.last_cycle) lines.push(`Last thought cycle: ${new Date(data.last_cycle).toLocaleString()}`);
    (data.post_queue || []).forEach((item) => lines.push(`Queued post: ${item.title}`));
    (data.thoughts || []).slice(0, 15).forEach((item) => {
      const when = item.time ? new Date(item.time).toLocaleTimeString() : "";
      lines.push(`[${when}] ${item.text}`);
    });
    thoughtsEl.textContent = lines.join("\n");
    thoughtsEl.classList.toggle("hidden", !lines.length);
  }

  bind("#moltbook-autonomy-run-btn", async () => {
    setAction("Thinking... this can take a minute.");
    await apiPost("/moltbook/autonomy/run", {}, 30000);
    setTimeout(() => refreshAutonomy().catch(() => {}), 45000);
  });

  refreshMoltbook()
    .then(() => refreshAutonomy())
    .catch(() => {});
}

function initMcpSettings() {
  const refreshBtn = $("#setting-mcp-refresh");
  const copyBtn = $("#setting-mcp-copy");
  const saveBtn = $("#setting-mcp-save");
  const removeBtn = $("#setting-mcp-remove");
  const actionEl = $("#mcp-action-status");
  refreshMcpSettings().catch(() => {});
  if (refreshBtn) {
    refreshBtn.addEventListener("click", async () => {
      if (actionEl) actionEl.textContent = "Refreshing...";
      await refreshMcpSettings();
      if (actionEl) actionEl.textContent = "MCP status updated.";
    });
  }
  document.querySelectorAll("[data-mcp-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = btn.getAttribute("data-mcp-preset");
      const nameEl = $("#mcp-server-name");
      const cmdEl = $("#mcp-server-command");
      const argsEl = $("#mcp-server-args");
      const urlEl = $("#mcp-server-url");
      if (preset === "blender") {
        if (nameEl) nameEl.value = "blender";
        if (cmdEl) cmdEl.value = "uvx";
        if (argsEl) argsEl.value = "blender-mcp";
        if (urlEl) urlEl.value = "";
      } else if (preset === "filesystem") {
        if (nameEl) nameEl.value = "filesystem";
        if (cmdEl) cmdEl.value = "npx";
        if (argsEl) argsEl.value = "-y\n@modelcontextprotocol/server-filesystem\n.";
        if (urlEl) urlEl.value = "";
      }
      if (actionEl) actionEl.textContent = `Loaded ${preset} preset — click Save server.`;
    });
  });
  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      const name = ($("#mcp-server-name") || {}).value || "";
      const command = ($("#mcp-server-command") || {}).value || "";
      const url = ($("#mcp-server-url") || {}).value || "";
      const argsRaw = ($("#mcp-server-args") || {}).value || "";
      const args = argsRaw
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
      if (!name.trim()) {
        if (actionEl) actionEl.textContent = "Enter a server name.";
        return;
      }
      if (!url.trim() && !command.trim()) {
        if (actionEl) actionEl.textContent = "Enter a command or URL.";
        return;
      }
      if (actionEl) actionEl.textContent = "Saving MCP server...";
      try {
        const res = await fetch("/api/mcp/servers", {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify({ name: name.trim(), command: command.trim(), args, url: url.trim(), enabled: true }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || data.error || "Save failed");
        if (actionEl) actionEl.textContent = `Saved MCP server "${data.name || name}".`;
        await refreshMcpSettings();
      } catch (err) {
        if (actionEl) actionEl.textContent = String(err.message || err);
      }
    });
  }
  if (removeBtn) {
    removeBtn.addEventListener("click", async () => {
      const name = ($("#mcp-server-name") || {}).value || "";
      if (!name.trim()) {
        if (actionEl) actionEl.textContent = "Enter the server name to remove.";
        return;
      }
      if (actionEl) actionEl.textContent = "Removing MCP server...";
      try {
        const res = await fetch(`/api/mcp/servers/${encodeURIComponent(name.trim())}`, {
          method: "DELETE",
          headers: authHeaders(),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || data.error || "Remove failed");
        if (actionEl) actionEl.textContent = `Removed "${data.removed || name}".`;
        await refreshMcpSettings();
      } catch (err) {
        if (actionEl) actionEl.textContent = String(err.message || err);
      }
    });
  }
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      if (actionEl) actionEl.textContent = "Loading Cursor config...";
      try {
        const res = await fetch("/api/mcp/cursor-config", { headers: authHeaders() });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || data.error || "Could not load Cursor config");
        const text = JSON.stringify(data, null, 2);
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
        } else {
          const area = document.createElement("textarea");
          area.value = text;
          document.body.appendChild(area);
          area.select();
          document.execCommand("copy");
          area.remove();
        }
        if (actionEl) actionEl.textContent = "Cursor mcpServers JSON copied.";
      } catch (err) {
        if (actionEl) actionEl.textContent = String(err.message || err);
      }
    });
  }
}

function renderHfDatasetCards(rows, container) {
  if (!container) return;
  if (!rows.length) {
    container.textContent = "No datasets found.";
    return;
  }
  container.innerHTML = `<div class="hf-results-list">${rows
    .map((row) => {
      const tags = (row.tags || []).slice(0, 5).join(", ");
      return `<div class="hf-result-card"><a href="${row.url}" target="_blank" rel="noopener">${row.id}</a><div class="hf-result-meta">downloads ${row.downloads || 0} · likes ${row.likes || 0}${tags ? ` · ${tags}` : ""}</div></div>`;
    })
    .join("")}</div>`;
}

function initHfDatasetSearch() {
  const input = $("#hf-dataset-search");
  const btn = $("#hf-dataset-search-btn");
  const out = $("#hf-dataset-results");
  const run = async (query) => {
    const q = String(query || (input && input.value) || "").trim();
    if (!q) {
      if (out) out.textContent = "Enter a search term.";
      return;
    }
    if (input) input.value = q;
    if (out) out.textContent = "Searching Hugging Face...";
    try {
      const res = await fetch(`/api/learn/hf/datasets?search=${encodeURIComponent(q)}&limit=12`, {
        headers: authHeaders(),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || data.error || "Search failed");
      renderHfDatasetCards(Array.isArray(data.datasets) ? data.datasets : [], out);
    } catch (err) {
      if (out) out.textContent = String(err.message || err);
    }
  };
  const loadCurated = async () => {
    if (!out) return;
    out.textContent = "Searching Hugging Face (code, chat, blender, math)...";
    try {
      const res = await fetch("/api/learn/hf/curated", { headers: authHeaders() });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || data.error || "Curated search failed");
      const groups = data.groups || {};
      const parts = [];
      Object.entries(groups).forEach(([key, group]) => {
        const rows = Array.isArray(group.datasets) ? group.datasets : [];
        if (!rows.length) return;
        parts.push(`<div class="hf-result-group-title">${key}</div>`);
        parts.push(
          rows
            .slice(0, 4)
            .map((row) => `<div class="hf-result-card"><a href="${row.url}" target="_blank" rel="noopener">${row.id}</a><div class="hf-result-meta">downloads ${row.downloads || 0} · likes ${row.likes || 0}</div></div>`)
            .join("")
        );
      });
      out.innerHTML = parts.length
        ? `<div class="hf-results-list">${parts.join("")}</div>`
        : "No curated results.";
    } catch (err) {
      out.textContent = String(err.message || err);
    }
  };
  if (btn) btn.addEventListener("click", () => run());
  document.querySelectorAll("[data-hf-q]").forEach((chip) => {
    chip.addEventListener("click", () => run(chip.getAttribute("data-hf-q")));
  });
  if (input) {
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        run();
      }
    });
  }
  loadCurated().catch(() => {});
}

async function initDownloadButtons() {
  const statusEl = $("#download-status");
  const cliBtn = $("#download-cli-btn");
  const appBtn = $("#download-app-btn");
  try {
    const res = await fetch("/api/downloads", { headers: authHeaders() });
    if (!res.ok) throw new Error("Could not load downloads");
    const data = await res.json();
    const rows = data.downloads || [];
    const byId = Object.fromEntries(rows.map((row) => [row.id, row]));
    if (cliBtn && byId.cli) {
      cliBtn.href = byId.cli.url || `/api/downloads/cli`;
      cliBtn.download = byId.cli.filename || "";
      cliBtn.textContent = `Download CLI (v${byId.cli.version || ""})`.trim();
    }
    if (appBtn && byId.app) {
      appBtn.href = byId.app.url || `/api/downloads/app`;
      appBtn.download = byId.app.filename || "";
      appBtn.textContent = byId.app.ready === false
        ? "App build pending"
        : `Download App (Rust v${byId.app.version || ""})`.trim();
    }
    if (statusEl) {
      const parts = rows.map((row) => {
        const ready = row.ready === false ? "building" : "ready";
        return `${row.label}: ${row.filename} (${ready})`;
      });
      statusEl.textContent = parts.length ? parts.join(" · ") : "Packages ready.";
    }
  } catch (err) {
    if (statusEl) statusEl.textContent = err.message || "Downloads unavailable";
  }
}

async function refreshChatHistoryPanel() {
  const panel = $("#chat-history-panel");
  const list = $("#chat-history-list");
  if (!panel || !list) return;
  const loggedIn = !!getSessionToken() && !isGuestMode();
  panel.classList.toggle("hidden", !loggedIn);
  if (!loggedIn) {
    list.innerHTML = "";
    return;
  }
  try {
    const data = await sessionApi("/chats");
    const chats = data.chats || [];
    if (!chats.length) {
      list.innerHTML = `<div class="chat-history-empty">No saved chats yet</div>`;
      return;
    }
    list.innerHTML = "";
    chats.forEach((chat) => {
      const row = document.createElement("div");
      row.className = "chat-history-item" + (chat.id === chatSessionId ? " active" : "");
      const openBtn = document.createElement("button");
      openBtn.type = "button";
      openBtn.className = "chat-history-open";
      openBtn.textContent = chat.title || "New chat";
      openBtn.title = chat.title || "New chat";
      openBtn.addEventListener("click", () => loadUserChat(chat.id));
      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "chat-history-del";
      delBtn.setAttribute("aria-label", "Delete chat");
      delBtn.textContent = "×";
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteUserChat(chat.id);
      });
      row.appendChild(openBtn);
      row.appendChild(delBtn);
      list.appendChild(row);
    });
  } catch (_) {
    list.innerHTML = `<div class="chat-history-empty">Sign in to save chats</div>`;
  }
}

async function startNewUserChat() {
  if (!getSessionToken() || isGuestMode()) {
    chatHistory.length = 0;
    chatSessionId = null;
    clearChatMessages();
    return;
  }
  try {
    const created = await sessionApi("/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New chat" }),
    });
    chatHistory.length = 0;
    chatSessionId = created.id;
    clearChatMessages();
    await refreshChatHistoryPanel();
    setActiveTab("text");
  } catch (err) {
    addMessage("error", err.message || "Could not create chat");
  }
}

async function loadUserChat(chatId) {
  try {
    const data = await sessionApi(`/chats/${encodeURIComponent(chatId)}`);
    chatSessionId = data.id;
    chatHistory.length = 0;
    clearChatMessages();
    const messages = data.messages || [];
    for (let i = 0; i < messages.length; i += 1) {
      const msg = messages[i];
      if (msg.role === "user") {
        addMessage("user", msg.content || "");
        const next = messages[i + 1];
        if (next && next.role === "assistant") {
          addAssistantMessage(next.content || "");
          chatHistory.push({ user: msg.content || "", assistant: next.content || "" });
          i += 1;
        }
      } else if (msg.role === "assistant") {
        addAssistantMessage(msg.content || "");
      }
    }
    await refreshChatHistoryPanel();
    setActiveTab("text");
  } catch (err) {
    addMessage("error", err.message || "Could not load chat");
  }
}

async function deleteUserChat(chatId) {
  try {
    await sessionApi(`/chats/${encodeURIComponent(chatId)}`, { method: "DELETE" });
    if (chatSessionId === chatId) {
      chatSessionId = null;
      chatHistory.length = 0;
      clearChatMessages();
    }
    await refreshChatHistoryPanel();
  } catch (err) {
    addMessage("error", err.message || "Could not delete chat");
  }
}

function initChatHistory() {
  const btn = $("#chat-new-btn");
  if (btn) btn.addEventListener("click", () => startNewUserChat());
  refreshChatHistoryPanel();
}

const EXPLICIT_SEARCH_PATTERNS = [
  /\b(search|look\s*up|google)\s+(for|up|online|on\s+(?:the\s+)?(?:web|internet))\b/i,
  /\b(search\s+(?:the\s+)?(?:web|internet)|web\s+search|internet\s+search)\b/i,
];

function wantsExplicitSearch(message) {
  return EXPLICIT_SEARCH_PATTERNS.some((pattern) => pattern.test(message));
}

const TIME_DATE_PATTERNS = [
  /\bwhat\s+(?:time|date)\b/i,
  /\bwhat'?s\s+(?:the\s+)?(?:time|date)\b/i,
  /\bcurrent\s+time\b/i,
  /\btime\s+(?:in|for|at)\b/i,
  /\bwhat\s+day\s+is\s+(?:it|today)\b/i,
  /\btoday'?s?\s+date\b/i,
  /\bwhat\s+time\s+is\s+it\b/i,
  /^(?:what(?:'s|\s+is)\s+(?:the\s+)?)?[a-z][a-z0-9\s.'-]+\s+time[\s!.?]*$/i,
  /^time\s+in\s+[a-z]/i,
];

const CAPABILITY_PATTERNS = [
  /\b(can you|could you|do you)\s+(create|make|generate)\s+(images?|pictures?|photos?)\b/i,
  /\b(can you|could you|do you)\s+(create|make|generate)\s+(videos?|animations?)\b/i,
  /\b(generate|create|make)\s+(images?|pictures?|photos?)\b/i,
  /\b(generate|create|make)\s+videos?\b/i,
  /\b(can you|could you|do you)\s+(code|write\s+code|program)\b/i,
  /\b(can you|could you)\s+write\s+code\b/i,
  /\bwhat\s+can\s+you\s+do\b/i,
  /\bwhat\s+are\s+your\s+capabilities\b/i,
  /\bdo\s+you\s+search\s+(?:the\s+)?(?:internet|web)\b/i,
  /\bcan\s+you\s+learn\b/i,
];

const CHITCHAT_PATTERNS = [
  /^(?:hi|hello|hey|hiya|howdy|greetings|good\s+(?:morning|afternoon|evening))(?:[,!]?\s+(?:there|friend|everyone|all|navine))?[\s!.?]*$/i,
  /^(?:(?:hi|hello|hey|hiya|howdy|greetings)[,!]?\s+)?(?:how\s+are\s+you(?:\s+doing)?|what'?s\s+up|how(?:'s|\s+is)\s+it\s+going)(?:\s+today|\s+lately)?[\s!.?]*$/i,
  /^(thanks?|thank\s+you|thx|ty)(?:\s+(?:so\s+much|a\s+lot))?[\s!.?]*$/i,
  /^(bye|goodbye|see\s+ya|later|good\s+night)[\s!.?]*$/i,
];

function isCasualChitchat(message) {
  const trimmed = message.trim();
  if (!trimmed || trimmed.length > 100) return false;
  return CHITCHAT_PATTERNS.some((pattern) => pattern.test(trimmed));
}

function isTimeDateQuery(message) {
  return TIME_DATE_PATTERNS.some((pattern) => pattern.test(message.trim()));
}

function isCapabilityQuery(message) {
  const trimmed = message.trim();
  if (!trimmed) return false;
  return CAPABILITY_PATTERNS.some((pattern) => pattern.test(trimmed));
}

function isLocalHandlerQuery(message) {
  return isTimeDateQuery(message) || isCapabilityQuery(message);
}

function likelyNeedsSearch(message) {
  return wantsExplicitSearch(message);
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function normalizeCodeNewlines(code) {
  return String(code || "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/\u00a0/g, " ")
    .replace(/\u200b/g, "")
    .replace(/\t/g, "    ")
    .replace(/\s+$/g, "");
}

function encodeCodePayload(code) {
  try {
    return btoa(unescape(encodeURIComponent(normalizeCodeNewlines(code))));
  } catch (_err) {
    return "";
  }
}

function decodeCodePayload(payload) {
  if (!payload) return "";
  try {
    return normalizeCodeNewlines(decodeURIComponent(escape(atob(payload))));
  } catch (_err) {
    try {
      return normalizeCodeNewlines(decodeURIComponent(payload));
    } catch (_err2) {
      return "";
    }
  }
}

function renderMarkdown(text) {
  const protectedBlocks = [];
  const protect = (html) => {
    const token = `\u0000NAVINECODE${protectedBlocks.length}\u0000`;
    protectedBlocks.push(html);
    return token;
  };
  let source = String(text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  let withCode = source.replace(
    /```(\w*)\r?\n([\s\S]*?)```/g,
    (_, lang, code) => {
      const rawLang = (lang || "text").toLowerCase();
      const rawCode = normalizeCodeNewlines(code);
      if (rawLang === "board") {
        return protect(`<pre class="board-block"><code>${escapeHtml(rawCode)}</code></pre>`);
      }
      const label = rawLang === "text" ? "code" : rawLang;
      const runAttr = canRunCode(rawLang) ? ' data-runnable="1"' : "";
      const payload = encodeCodePayload(rawCode);
      return protect(
        `<div class="code-wrap" data-lang="${label}" data-code="${payload}">` +
        `<div class="code-toolbar">` +
        `<span class="code-lang-label">${label}</span>` +
        `<div class="code-toolbar-actions">` +
        `<button type="button" class="code-btn" data-action="copy">Copy</button>` +
        `<button type="button" class="code-btn" data-action="download">Download</button>` +
        (canRunCode(rawLang) ? `<button type="button" class="code-btn primary" data-action="run">Run</button>` : "") +
        `</div></div>` +
        `<pre class="code-block"${runAttr}><code class="language-${label}">${escapeHtml(rawCode)}</code></pre>` +
        `</div>`
      );
    }
  );
  withCode = withCode.replace(
    /(^|\n)((?:[1-8]\s+[PNpnBQRKbqrk\.](?:\s+[PNpnBQRKbqrk\.]){7}\n){8}\s+a b c d e f g h)/g,
    (_, lead, board) => `${lead}${protect(`<pre class="board-block"><code>${escapeHtml(board.trimEnd())}</code></pre>`)}`
  );
  withCode = escapeHtml(withCode).replace(/\n/g, "<br>");
  protectedBlocks.forEach((html, idx) => {
    withCode = withCode.replace(`\u0000NAVINECODE${idx}\u0000`, html);
  });
  return withCode;
}

const CODE_EXT_MAP = {
  python: "py",
  py: "py",
  javascript: "js",
  js: "js",
  typescript: "ts",
  ts: "ts",
  java: "java",
  cpp: "cpp",
  c: "c",
  csharp: "cs",
  cs: "cs",
  go: "go",
  rust: "rs",
  kotlin: "kt",
  swift: "swift",
  php: "php",
  ruby: "rb",
  sql: "sql",
  batch: "bat",
  bat: "bat",
  cmd: "bat",
  bash: "sh",
  sh: "sh",
  html: "html",
  htm: "html",
  css: "css",
  json: "json",
  xml: "xml",
  svg: "svg",
  lua: "lua",
  perl: "pl",
  r: "r",
  scala: "scala",
  dart: "dart",
};

const CODE_MIME_MAP = {
  html: "text/html",
  htm: "text/html",
  css: "text/css",
  js: "text/javascript",
  javascript: "text/javascript",
  json: "application/json",
  svg: "image/svg+xml",
  py: "text/x-python",
  python: "text/x-python",
};

const CLIENT_RUN_LANGS = new Set(["html", "htm", "css", "javascript", "js", "svg"]);
const SERVER_RUN_LANGS = new Set(["python", "py"]);

function canRunCode(lang) {
  const key = (lang || "").toLowerCase();
  return CLIENT_RUN_LANGS.has(key) || SERVER_RUN_LANGS.has(key);
}

function inferCodeFilename(lang, code) {
  const key = (lang || "txt").toLowerCase();
  const ext = CODE_EXT_MAP[key] || "txt";
  let base = "main";
  if (key === "java") {
    const cls = code.match(/\b(?:public\s+)?class\s+([A-Za-z_]\w*)/);
    base = (cls && cls[1]) || "HelloWorld";
  } else if (key === "csharp" || key === "cs") {
    base = "Program";
  } else if (key === "javascript" || key === "js" || key === "typescript" || key === "ts") {
    base = "index";
  } else if (key === "html" || key === "htm") {
    base = "index";
  } else if (key === "css") {
    base = "styles";
  } else if (key === "batch" || key === "bat" || key === "cmd") {
    base = "multi_tool";
  }
  return `${base}.${ext}`;
}

function extractAllCodeBlocks(text) {
  const blocks = [];
  const re = /```(\w*)\n([\s\S]*?)```/g;
  let match;
  while ((match = re.exec(String(text || ""))) !== null) {
    const lang = (match[1] || "txt").toLowerCase();
    const code = (match[2] || "").replace(/\s+$/, "");
    if (!code.trim() || lang === "board") continue;
    blocks.push({
      lang,
      code,
      filename: inferCodeFilename(lang, code),
      mime: CODE_MIME_MAP[lang] || "text/plain",
    });
  }
  return blocks;
}

function extractDownloadableCode(text) {
  const blocks = extractAllCodeBlocks(text);
  return blocks.length ? blocks[0] : null;
}

function readCodeText(codeEl) {
  if (!codeEl) return "";
  const raw = typeof codeEl.textContent === "string" ? codeEl.textContent : "";
  if (raw) return normalizeCodeNewlines(raw);
  if (typeof codeEl.innerText === "string" && codeEl.innerText.length) {
    return normalizeCodeNewlines(codeEl.innerText);
  }
  let out = "";
  codeEl.childNodes.forEach((node) => {
    if (node.nodeName === "BR") {
      out += "\n";
      return;
    }
    out += node.textContent || "";
  });
  return normalizeCodeNewlines(out);
}

function readCodeFromWrap(wrap) {
  const lang = (wrap && wrap.dataset.lang) || "text";
  const codeEl = wrap && wrap.querySelector("code");
  const fromDom = readCodeText(codeEl);
  const fromAttr = decodeCodePayload(wrap && wrap.dataset ? wrap.dataset.code : "");
  const code = fromDom || fromAttr || "";
  return {
    lang,
    code,
    filename: inferCodeFilename(lang, code),
    mime: CODE_MIME_MAP[lang] || "text/plain",
  };
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.left = "-9999px";
  document.body.appendChild(area);
  area.select();
  document.execCommand("copy");
  area.remove();
}

function flashCodeButton(btn, label) {
  if (!btn) return;
  const prev = btn.textContent;
  btn.textContent = label;
  btn.disabled = true;
  window.setTimeout(() => {
    btn.textContent = prev;
    btn.disabled = false;
  }, 1400);
}

function wrapPreviewDocument(code, lang) {
  const key = (lang || "").toLowerCase();
  const trimmed = code.trim();
  if (key === "html" || key === "htm") {
    if (/^\s*<!doctype/i.test(trimmed) || /^\s*<html[\s>]/i.test(trimmed)) {
      return trimmed;
    }
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Preview</title>
</head>
<body>
${trimmed}
</body>
</html>`;
  }
  if (key === "css") {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CSS Preview</title>
  <style>
${trimmed}
  </style>
</head>
<body>
  <div class="preview-box">
    <h1>CSS preview</h1>
    <p>Edit the CSS and re-run to see changes.</p>
    <button type="button">Sample button</button>
  </div>
</body>
</html>`;
  }
  if (key === "javascript" || key === "js") {
    const safe = trimmed.replace(/<\/script/gi, "<\\/script");
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JavaScript Preview</title>
</head>
<body>
  <pre id="out"></pre>
  <script>
const logEl = document.getElementById("out");
const write = (value) => {
  logEl.textContent += (value == null ? "" : String(value)) + "\\n";
};
const console = { log: write, error: write, warn: write, info: write };
try {
${safe}
} catch (err) {
  write("Error: " + (err && err.message ? err.message : err));
}
  </script>
</body>
</html>`;
  }
  if (key === "svg") {
    if (/^\s*<\?xml/i.test(trimmed) || /^\s*<svg[\s>]/i.test(trimmed)) {
      return trimmed;
    }
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200">${trimmed}</svg>`;
  }
  return trimmed;
}

function openClientPreview(packed) {
  const lang = (packed.lang || "").toLowerCase();
  const doc = wrapPreviewDocument(packed.code, lang);
  const mime = lang === "svg" && doc.trim().startsWith("<svg") ? "image/svg+xml" : "text/html";
  const blob = new Blob([doc], { type: mime });
  const url = URL.createObjectURL(blob);
  const win = window.open(url, "_blank", "noopener,noreferrer");
  if (!win) {
    URL.revokeObjectURL(url);
    throw new Error("Pop-up blocked. Allow pop-ups for this site to preview HTML.");
  }
  window.setTimeout(() => URL.revokeObjectURL(url), 60000);
}

function ensureCodeRunModal() {
  let modal = document.getElementById("code-run-modal");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "code-run-modal";
    modal.className = "code-run-modal hidden";
    modal.innerHTML =
      '<div class="code-run-card"><div class="code-run-head"><h2 class="code-run-title">Run output</h2>' +
      '<button type="button" class="code-run-close" aria-label="Close">&times;</button></div>' +
      '<div class="code-run-status"></div><pre class="code-run-log code-run-stdout"></pre>' +
      '<pre class="code-run-log code-run-stderr hidden"></pre></div>';
    document.body.appendChild(modal);
  }
  return modal;
}

function showRunOutput(title, stdout, stderr, ok) {
  const modal = ensureCodeRunModal();
  const titleEl = modal.querySelector(".code-run-title");
  const statusEl = modal.querySelector(".code-run-status");
  const outEl = modal.querySelector(".code-run-stdout");
  const errEl = modal.querySelector(".code-run-stderr");
  const closeBtn = modal.querySelector(".code-run-close");
  if (titleEl) titleEl.textContent = title || "Run output";
  if (statusEl) {
    statusEl.textContent = ok ? "Finished successfully" : "Finished with errors";
    statusEl.className = `code-run-status ${ok ? "ok" : "err"}`;
  }
  if (outEl) outEl.textContent = stdout || "(no output)";
  if (errEl) {
    if (stderr && stderr.trim()) {
      errEl.textContent = stderr;
      errEl.classList.remove("hidden");
    } else {
      errEl.textContent = "";
      errEl.classList.add("hidden");
    }
  }
  modal.classList.remove("hidden");
  const close = () => modal.classList.add("hidden");
  if (closeBtn) closeBtn.onclick = close;
  modal.onclick = (ev) => {
    if (ev.target === modal) close();
  };
}

async function downloadCodeBlock(packed) {
  const code = String((packed && packed.code) || "");
  const filename = (packed && packed.filename) || "code.txt";
  const mime = (packed && packed.mime) || "text/plain;charset=utf-8";
  if (!code.trim()) {
    throw new Error("No code to download");
  }
  const blob = new Blob([code], { type: mime });
  if (window.navigator && typeof window.navigator.msSaveOrOpenBlob === "function") {
    window.navigator.msSaveOrOpenBlob(blob, filename);
    return;
  }
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.rel = "noopener";
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2500);
}

async function runCodeBlock(packed) {
  const lang = (packed.lang || "").toLowerCase();
  if (CLIENT_RUN_LANGS.has(lang)) {
    openClientPreview(packed);
    return { ok: true, client: true };
  }
  if (SERVER_RUN_LANGS.has(lang)) {
    const data = await apiPost("/code/run", {
      code: packed.code,
      language: lang,
    });
    showRunOutput(
      `Python output`,
      data.stdout || "",
      data.stderr || "",
      !!data.ok
    );
    return data;
  }
  throw new Error(`Run is not supported for ${lang}. Download the file and run it locally.`);
}

function attachMessageCodeActions(messageEl) {
  if (!messageEl) return;
  const wraps = messageEl.querySelectorAll(".code-wrap");
  wraps.forEach((wrap) => {
    if (wrap.dataset.bound === "1") return;
    wrap.dataset.bound = "1";
    wrap.querySelectorAll(".code-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const packed = readCodeFromWrap(wrap);
        const action = btn.dataset.action;
        try {
          if (action === "copy") {
            await copyTextToClipboard(packed.code);
            flashCodeButton(btn, "Copied");
          } else if (action === "download") {
            await downloadCodeBlock(packed);
            flashCodeButton(btn, "Saved");
          } else if (action === "run") {
            btn.disabled = true;
            const prev = btn.textContent;
            btn.textContent = "Running...";
            await runCodeBlock(packed);
            btn.textContent = prev;
            btn.disabled = false;
          }
        } catch (err) {
          flashCodeButton(btn, "Failed");
          if (action === "run") {
            showRunOutput("Run failed", "", err.message || String(err), false);
          }
        }
      });
    });
  });
}

async function offerFileDownload(text, container) {
  if (!container) return;
  attachMessageCodeActions(container);
  const body = container.querySelector(".message-body");
  if (!body) return;

  const downloadMatch = String(text || "").match(/Download:\s*(\/api\/files\/download\?path=[^\s]+)/i);
  if (downloadMatch) {
    const href = downloadMatch[1];
    if (!body.querySelector(".chat-download-link")) {
      const row = document.createElement("div");
      row.className = "download-actions chat-download-link";
      row.style.marginTop = "0.75rem";
      const link = document.createElement("a");
      link.className = "btn primary";
      link.href = href;
      link.textContent = href.toLowerCase().includes(".zip") ? "Download ZIP" : "Download file";
      link.setAttribute("download", "");
      row.appendChild(link);
      body.appendChild(row);
    }
  }

  const wraps = body.querySelectorAll(".code-wrap");
  if (wraps.length >= 2 && !body.querySelector(".chat-zip-btn")) {
    const row = document.createElement("div");
    row.className = "download-actions";
    row.style.marginTop = "0.75rem";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn chat-zip-btn";
    btn.textContent = "Download all as ZIP";
    btn.addEventListener("click", async () => {
      const files = [];
      wraps.forEach((wrap, idx) => {
        const packed = readCodeFromWrap(wrap);
        if (!packed.code || !packed.code.trim()) return;
        files.push({
          path: packed.filename || `file_${idx + 1}.txt`,
          content: packed.code,
        });
      });
      if (!files.length) return;
      btn.disabled = true;
      btn.textContent = "Creating ZIP...";
      try {
        const data = await apiPost("/files/zip", { filename: "project.zip", files });
        const link = document.createElement("a");
        link.className = "btn primary";
        link.href = data.download_url;
        link.textContent = `Download ${data.filename || "project.zip"}`;
        link.setAttribute("download", data.filename || "project.zip");
        row.appendChild(link);
        btn.remove();
      } catch (err) {
        btn.disabled = false;
        btn.textContent = "ZIP failed";
      }
    });
    row.appendChild(btn);
    body.appendChild(row);
  }
}

function initCodeRunModal() {
  const modal = document.getElementById("code-run-modal");
  const closeBtn = document.getElementById("code-run-close");
  if (!modal) return;
  const close = () => modal.classList.add("hidden");
  if (closeBtn) closeBtn.addEventListener("click", close);
  modal.addEventListener("click", (ev) => {
    if (ev.target === modal) close();
  });
}

function addAssistantMessage(rawText, containerSel = "#chat-messages") {
  addMessage("assistant", renderMarkdown(rawText), true, containerSel);
  const container = $(containerSel);
  const last = container && container.lastElementChild;
  attachMessageCodeActions(last);
  return last;
}

function messageAvatarLetter(role) {
  if (role === "user") return "Y";
  if (role === "error") return "!";
  return (BRAND_NAME || "N").charAt(0).toUpperCase();
}

function updateChatWelcome() {
  const welcome = document.getElementById("chat-welcome");
  const container = $("#chat-messages");
  if (!welcome || !container) return;
  welcome.classList.toggle("hidden", container.children.length > 0);
}

function addMessage(role, text, isHtml = false, containerSel = "#chat-messages") {
  const container = $(containerSel);
  if (!container) return;
  updateChatWelcome();
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = messageAvatarLetter(role);
  const content = document.createElement("div");
  content.className = "message-content";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : role === "error" ? "Error" : BRAND_NAME;
  const body = document.createElement("div");
  body.className = "message-body";
  if (isHtml) {
    body.innerHTML = text;
  } else {
    body.textContent = text;
  }
  content.appendChild(label);
  content.appendChild(body);
  div.appendChild(avatar);
  div.appendChild(content);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  if (containerSel === "#chat-messages") {
    updateChatWelcome();
  }
}

const TAB_LABELS = {
  text: "Chat",
  voicechat: "Voice",
  image: "Image",
  video: "Generate",
  osint: "OSINT",
  deepfake: "Deepfake",
  voice: "Clone",
  music: "Music",
  info: "Info",
  train: "Train",
  apigen: "API",
  settings: "Settings",
};

const PATH_TO_TAB = {
  "/": "text",
  "/chat": "text",
  "/voice": "voicechat",
  "/video-chat": "text",
  "/image": "image",
  "/generate": "video",
  "/osint": "osint",
  "/deepfake": "deepfake",
  "/clone": "voice",
  "/music": "music",
  "/info": "info",
  "/train": "train",
  "/api-gen": "apigen",
  "/settings": "settings",
};

const TAB_TO_PATH = {
  text: "/chat",
  voicechat: "/voice",
  image: "/image",
  video: "/generate",
  osint: "/osint",
  deepfake: "/deepfake",
  voice: "/clone",
  music: "/music",
  info: "/info",
  train: "/train",
  apigen: "/api-gen",
  settings: "/settings",
};

function normalizePath(path) {
  const clean = String(path || "/").split("?")[0].split("#")[0];
  if (clean.length > 1 && clean.endsWith("/")) return clean.slice(0, -1);
  return clean || "/";
}

function pathToTab(path) {
  return PATH_TO_TAB[normalizePath(path)] || "text";
}

function tabToPath(tab) {
  return TAB_TO_PATH[tab] || "/chat";
}

function setActiveTab(tabName, options = {}) {
  const opts = options || {};
  document.querySelectorAll(".tab, .sidebar-item").forEach((t) => {
    const active = t.dataset.tab === tabName;
    t.classList.toggle("active", active);
    t.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
  const panel = document.getElementById(`panel-${tabName}`);
  if (panel) panel.classList.add("active");
  const topbarTitle = document.getElementById("topbar-title");
  if (topbarTitle) topbarTitle.textContent = TAB_LABELS[tabName] || "Navine AI - Python";
  document.title = `${TAB_LABELS[tabName] || "Navine AI - Python"} · ${BRAND_NAME}`;

  if (!opts.skipHistory) {
    const nextPath = tabToPath(tabName);
    const currentPath = normalizePath(window.location.pathname);
    if (currentPath !== nextPath) {
      const method = opts.replace ? "replaceState" : "pushState";
      window.history[method]({ tab: tabName }, "", nextPath);
    }
  }

  if (tabName === "info") {
    loadModelStats().catch(() => {});
    scheduleInfoPoll(true);
  } else {
    scheduleInfoPoll(false);
  }
  if (tabName === "train") {
    trainTabActive = true;
    restoreAdminSession().catch(() => showTrainLogin());
    loadTrainProgress().catch(() => {});
    scheduleTrainPoll(true);
  } else {
    trainTabActive = false;
    scheduleTrainPoll(false);
  }
  if (tabName === "settings") {
    loadTrainProgress().catch(() => {});
  }
}

function initRouting() {
  const tab = pathToTab(window.location.pathname);
  setActiveTab(tab, { replace: true, skipHistory: false });
  window.addEventListener("popstate", () => {
    setActiveTab(pathToTab(window.location.pathname), { skipHistory: true });
  });
}

function closeSidebar() {
  const sidebar = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");
  const menuBtn = document.getElementById("menu-btn");
  if (sidebar) sidebar.classList.remove("open");
  if (backdrop) {
    backdrop.classList.remove("visible");
    backdrop.hidden = true;
  }
  document.body.classList.remove("sidebar-open");
  if (menuBtn) menuBtn.setAttribute("aria-expanded", "false");
}

function openSidebar() {
  const sidebar = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");
  const menuBtn = document.getElementById("menu-btn");
  if (sidebar) sidebar.classList.add("open");
  if (backdrop) {
    backdrop.hidden = false;
    backdrop.classList.add("visible");
  }
  document.body.classList.add("sidebar-open");
  if (menuBtn) menuBtn.setAttribute("aria-expanded", "true");
}

function initSidebar() {
  const menuBtn = document.getElementById("menu-btn");
  const closeBtn = document.getElementById("sidebar-close");
  const backdrop = document.getElementById("sidebar-backdrop");
  if (menuBtn) {
    menuBtn.setAttribute("aria-expanded", "false");
    menuBtn.addEventListener("click", openSidebar);
  }
  if (closeBtn) closeBtn.addEventListener("click", closeSidebar);
  if (backdrop) backdrop.addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSidebar();
  });
  const mq = window.matchMedia("(min-width: 861px)");
  const onBreakpoint = () => {
    if (mq.matches) closeSidebar();
  };
  if (mq.addEventListener) mq.addEventListener("change", onBreakpoint);
  else if (mq.addListener) mq.addListener(onBreakpoint);
}

function initChatWelcome() {
  const chips = document.querySelectorAll(".welcome-chip");
  const prompt = $("#text-prompt");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const text = chip.dataset.prompt || chip.textContent || "";
      if (prompt) {
        prompt.value = text;
        prompt.focus();
        autoResizeTextarea(prompt);
      }
    });
  });
}

function autoResizeTextarea(el) {
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
}

function initComposer() {
  document.querySelectorAll(".composer-input").forEach((el) => {
    el.addEventListener("input", () => autoResizeTextarea(el));
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        const form = el.closest("form");
        if (form && el.id === "text-prompt") {
          e.preventDefault();
          form.requestSubmit();
        }
      }
    });
  });
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab, .sidebar-item");
  tabs.forEach((tab) => {
    tab.addEventListener("click", (event) => {
      if (tab.tagName === "A") event.preventDefault();
      const name = tab.dataset.tab;
      if (!name) return;
      setActiveTab(name);
      closeSidebar();
    });
  });
}

const INFO_MODALITY_COLORS = {
  text: "#1246FF",
  chat: "#1246FF",
  code: "#2563EB",
  image: "#00B4FF",
  video: "#7C3AED",
  voice: "#1E90FF",
  deepfake: "#5E35B1",
  music: "#0D9488",
  osint: "#0EA5E9",
  train: "#F59E0B",
  host: "#64748B",
};

let infoPollTimer = null;
let lastInfoStats = null;
let chartResizeTimer = null;

window.addEventListener("resize", () => {
  clearTimeout(chartResizeTimer);
  chartResizeTimer = setTimeout(() => {
    if (lastInfoStats) renderParamChart(lastInfoStats);
  }, 150);
});

function scheduleInfoPoll(active) {
  if (infoPollTimer) {
    clearInterval(infoPollTimer);
    infoPollTimer = null;
  }
  if (!active) return;
  infoPollTimer = setInterval(() => {
    const panel = document.getElementById("panel-info");
    if (panel && panel.classList.contains("active")) {
      loadModelStats().catch(() => {});
    }
  }, 5000);
}

function formatArchLines(architecture) {
  const entries = Object.entries(architecture || {}).filter(([, v]) => v !== null && v !== undefined && v !== "");
  if (!entries.length) return "No architecture metadata";
  return entries
    .map(([k, v]) => {
      let value = v;
      if (Array.isArray(value)) {
        value = value.join(", ");
      } else if (typeof value === "object") {
        value = JSON.stringify(value);
      } else if (typeof value === "string") {
        value = value
          .replace(/gpt-?\s*3(?:\s*medium\+?\s*class)?/gi, "enterprise")
          .replace(/gpt3[_-]?(800m|760m|1p3b|350m)/gi, "800m")
          .replace(/\b(openai|claude|gemini|grok|llama|mistral|copilot)\b/gi, "navine");
      }
      if (String(k).toLowerCase() === "size_tier" && typeof value === "string") {
        value = value.replace(/^gpt3[_-]?/i, "").replace(/_/g, " ") || "800m";
      }
      return `${k}: ${value}`;
    })
    .join("\n");
}

function formatArchSummary(architecture) {
  const arch = architecture || {};
  const parts = [];
  if (arch.engine === "procedural" || Array.isArray(arch.styles)) {
    if (arch.engine) parts.push(String(arch.engine));
    if (arch.sample_rate != null) parts.push(`${arch.sample_rate} Hz`);
    if (arch.duration_range_sec) parts.push(`${arch.duration_range_sec}s`);
    if (Array.isArray(arch.styles) && arch.styles.length) {
      parts.push(`${arch.styles.length} styles`);
    }
    if (arch.clips_generated != null) parts.push(`${arch.clips_generated} clips`);
    if (arch.offline) parts.push("offline");
    return parts.join(" · ") || "music generator";
  }
  if (arch.n_layers != null && arch.d_model != null) {
    parts.push(`${arch.n_layers}L · d${arch.d_model}`);
  } else if (arch.base_channels != null) {
    parts.push(`base ${arch.base_channels}`);
    if (arch.image_size != null) parts.push(`${arch.image_size}px`);
  } else if (arch.hidden_dim != null) {
    parts.push(`hidden ${arch.hidden_dim}`);
    if (arch.num_frames != null) parts.push(`${arch.num_frames}f`);
  }
  if (arch.use_rope) parts.push("RoPE");
  if (arch.use_swiglu) parts.push("SwiGLU");
  if (arch.use_rms_norm) parts.push("RMSNorm");
  if (arch.arch_version != null) parts.push(`v${arch.arch_version}`);
  return parts.join(" · ") || "";
}

function formatParamPairs(params) {
  const entries = Object.entries(params || {}).filter(([, v]) => v !== null && v !== undefined && v !== "");
  if (!entries.length) return "—";
  return entries
    .map(([key, value]) => {
      if (Array.isArray(value)) return `${key}: [${value.join(", ")}]`;
      if (typeof value === "object") return `${key}: ${JSON.stringify(value)}`;
      return `${key}: ${value}`;
    })
    .join("\n");
}

function renderCapabilities(stats) {
  const el = $("#info-caps");
  if (!el) return;
  const caps = (stats && stats.capabilities) || [];
  if (!caps.length) {
    el.innerHTML = "";
    return;
  }
  el.innerHTML = caps
    .map((cap) => {
      const color = INFO_MODALITY_COLORS[cap] || "var(--text-muted)";
      return `<span class="info-cap-chip" style="--info-accent:${color}">${cap}</span>`;
    })
    .join("");
}

function renderHostSpecs(stats) {
  const el = $("#info-host-text");
  if (!el) return;
  const host = stats && stats.host_specs;
  if (host && host.text) {
    el.textContent = host.text;
    return;
  }
  if (host && host.specs && typeof host.specs === "object") {
    el.textContent = Object.entries(host.specs)
      .map(([k, v]) => `${k} = ${typeof v === "object" ? JSON.stringify(v) : v}`)
      .join("\n");
    return;
  }
  el.textContent = "Host specs unavailable.";
}

function renderInferenceProfiles(stats) {
  const grid = $("#info-inference");
  if (!grid) return;
  const mediaModes = new Set(["think", "detective", "analyze", "osint", "code", "chat"]);
  const profiles = ((stats && stats.inference_profiles) || []).filter((row) => {
    const modality = row.modality || "text";
    if (modality !== "image" && modality !== "video") return true;
    const mode = String(row.mode || "default").toLowerCase();
    return mode === "default" && !mediaModes.has(mode);
  });
  if (!profiles.length) {
    grid.innerHTML = `<div class="info-inference-empty">No inference profiles loaded.</div>`;
    return;
  }
  grid.innerHTML = profiles
    .map((row) => {
      const modality = row.modality || "text";
      const color = INFO_MODALITY_COLORS[modality] || INFO_MODALITY_COLORS.text;
      return `<article class="info-inference-card" style="--info-accent:${color}">
        <div class="info-inference-head">
          <span class="info-chip">${modality}</span>
          <strong>${row.label || row.mode || "profile"}</strong>
        </div>
        <div class="info-inference-config">${row.config || ""}</div>
        <pre class="info-inference-params">${formatParamPairs(row.params)}</pre>
      </article>`;
    })
    .join("");
}

function renderInfoCards(stats) {
  const cardsEl = $("#info-cards");
  if (!cardsEl) return;
  const models = (stats && stats.models) || [];
  cardsEl.innerHTML = models
    .map((m) => {
      const modality = m.modality || m.category || "model";
      const color = INFO_MODALITY_COLORS[modality] || INFO_MODALITY_COLORS.text;
      const isGenerator = m.parameter_source === "generator";
      const state = m.training ? "training" : m.trained || isGenerator ? "ready" : "missing";
      const stateLabel = m.training ? "training" : m.trained || isGenerator ? "ready" : "no weights";
      const age = formatAge(m.age_seconds);
      const exact = m.parameters_exact || (m.parameters != null ? String(m.parameters) : null);
      const human = m.parameters_human;
      const paramsLine = exact
        ? `${exact}${human && human !== exact ? ` (${human})` : ""}`
        : "—";
      const source =
        m.parameter_source === "live_checkpoint"
          ? "Live checkpoint"
          : m.parameter_source === "configured_target"
            ? "Configured 1B+ target"
            : m.parameter_source === "generator"
              ? "Local generator"
              : "No trained weights yet";
      const shared = m.shares_checkpoint_with
        ? `Shares weights with ${m.shares_checkpoint_with}`
        : "";
      const archSummary = formatArchSummary(m.architecture);
      return `<article class="info-card ${m.training ? "training" : ""}" style="--info-accent:${color}">
        <div class="info-card-head">
          <div class="info-card-name">${m.label || m.name}</div>
          <div class="info-card-params">${human || "—"}</div>
        </div>
        <div class="info-card-exact">${paramsLine}</div>
        ${archSummary ? `<div class="info-card-arch-summary">${archSummary}</div>` : ""}
        <div class="info-card-label">${m.name}</div>
        <div class="info-card-meta">
          <span class="info-chip">${modality}</span>
          <span class="info-chip ${state}">${stateLabel}</span>
          <span class="info-chip">${source}</span>
          ${age ? `<span class="info-chip">${age} ago</span>` : ""}
          ${shared ? `<span class="info-chip">${shared}</span>` : ""}
        </div>
        <pre class="info-arch">${formatArchLines(m.architecture)}</pre>
      </article>`;
    })
    .join("");
}

function renderParamChart(stats) {
  const canvas = document.getElementById("info-param-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const models = ((stats && stats.models) || []).filter(
    (m) =>
      Number(m.parameters) > 0 &&
      (m.parameter_source === "live_checkpoint" || m.parameter_source === "configured_target")
  );
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.round(canvas.clientWidth || 960));
  const height = width < 560 ? 240 : 280;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const barGap = 16;
  const padding = { top: 28, right: 20, bottom: 36, left: 56 };
  const plotW = width - padding.left - padding.right;
  const barW = Math.max(24, (plotW - barGap * (models.length + 1)) / Math.max(1, models.length));
  const labels = models.map((m) => String(m.name || "").replace(/_/g, " "));
  ctx.font = "11px Sora, sans-serif";
  const rotateLabels = labels.some((label) => ctx.measureText(label).width > barW + barGap - 4);
  if (rotateLabels) padding.bottom = 84;
  const plotH = height - padding.top - padding.bottom;

  ctx.fillStyle = "rgba(255,255,255,0.04)";
  ctx.fillRect(padding.left, padding.top, plotW, plotH);

  if (!models.length) {
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = "14px Sora, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No models configured for this site", width / 2, height / 2);
    return;
  }

  const values = models.map((m) => Math.max(0, Number(m.parameters) || 0));
  const maxVal = Math.max(...values, 1);

  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotH * i) / 4;
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + plotW, y);
    ctx.stroke();
    const tickVal = maxVal * (1 - i / 4);
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.font = "11px IBM Plex Mono, monospace";
    ctx.textAlign = "right";
    ctx.fillText(formatParamTick(tickVal), padding.left - 8, y + 4);
  }

  models.forEach((m, i) => {
    const value = Math.max(0, Number(m.parameters) || 0);
    const barH = value > 0 ? (value / maxVal) * plotH : 4;
    const x = padding.left + barGap + i * (barW + barGap);
    const y = padding.top + plotH - barH;
    const color = INFO_MODALITY_COLORS[m.modality] || INFO_MODALITY_COLORS.text;
    const grad = ctx.createLinearGradient(0, y, 0, y + barH);
    grad.addColorStop(0, color);
    grad.addColorStop(1, "rgba(255,255,255,0.15)");
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, barW, barH);
    if (value > 0) {
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.font = "11px IBM Plex Mono, monospace";
      ctx.textAlign = "center";
      ctx.fillText(m.parameters_human || String(value), x + barW / 2, y - 6);
    }
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "11px Sora, sans-serif";
    ctx.save();
    ctx.translate(x + barW / 2, padding.top + plotH + 16);
    if (rotateLabels) {
      ctx.rotate(-0.6);
      ctx.textAlign = "right";
    } else {
      ctx.textAlign = "center";
    }
    ctx.fillText(labels[i], 0, 0);
    ctx.restore();
  });
}

function formatParamTick(value) {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(0)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
  return String(Math.round(value));
}

async function loadModelStats() {
  const totalEl = $("#info-total");
  const updatedEl = $("#info-updated");
  const subEl = $("#info-sub");
  if (totalEl && !lastInfoStats) totalEl.textContent = "Loading...";
  if (updatedEl) updatedEl.textContent = "Updating...";
  try {
    const stats = await apiRequest("GET", "/models/stats", null, { timeoutMs: 180000 });
    lastInfoStats = stats;
    if (totalEl) {
      totalEl.textContent = stats.total_parameters_exact || stats.total_parameters_human || "—";
    }
    if (updatedEl) {
      const when = stats.updated_at ? new Date(stats.updated_at).toLocaleTimeString() : "now";
      const sourceMap = {
        live_checkpoint: "live checkpoints",
        configured_target: "configured ~1B targets",
        mixed: "live + configured targets",
      };
      const source = sourceMap[stats.parameter_source] || String(stats.parameter_source || "weights");
      updatedEl.textContent = `Updated ${when} · ${source} · ${stats.trained_models || 0} trained`;
    }
    if (subEl) {
      const caps = (stats.capabilities || []).join(", ") || "chat, image, video, voice, music";
      const musicReady = ((stats.models || []).some((m) => m.name === "music" && m.trained));
      subEl.textContent = `Neural 1.00B+ · music ${musicReady ? "ready" : "available"} · ${stats.trained_models || 0} trained · ${stats.total_parameters_human || "—"} · ${caps}`;
    }
    renderCapabilities(stats);
    renderHostSpecs(stats);
    renderParamChart(stats);
    renderInfoCards(stats);
    renderInferenceProfiles(stats);
    return stats;
  } catch (err) {
    if (lastInfoStats && lastInfoStats.models && lastInfoStats.models.length) {
      if (updatedEl) updatedEl.textContent = "Showing last loaded stats (refresh soon)";
      renderCapabilities(lastInfoStats);
      renderHostSpecs(lastInfoStats);
      renderParamChart(lastInfoStats);
      renderInfoCards(lastInfoStats);
      renderInferenceProfiles(lastInfoStats);
      return lastInfoStats;
    }
    if (updatedEl) updatedEl.textContent = "Could not load stats — retrying";
    if (totalEl) totalEl.textContent = "—";
    throw err;
  }
}

async function loadStatus() {
  const statusEl = $("#system-status");
  const textEl = $("#status-text");
  const infoEl = $("#info-line");
  const autolearnEl = $("#autolearn-line");
  const marathonEl = $("#marathon-line");
  try {
    const health = await apiGet("/health");
    const info = await apiGet("/info");
    applyBrandTheme(info);
    applySiteCapabilities(info);
    THEME_ID = String((info && info.theme_id) || THEME_ID || "navine").toLowerCase();
    if (info.product && !isForeignBrand(info.product)) {
      BRAND_NAME = info.product;
      const titleEl = $("#brand-title");
      if (titleEl) titleEl.textContent = BRAND_NAME;
      const footerBrand = $("#footer-brand");
      if (footerBrand) footerBrand.textContent = BRAND_NAME;
      document.title = BRAND_NAME;
    }
    const publicUrl = String((info && info.public_url) || "").trim();
    const publicLink = document.getElementById("sidebar-public-url");
    if (publicLink) {
      if (publicUrl) {
        publicLink.href = publicUrl;
        publicLink.textContent = publicUrl.replace(/^https?:\/\//, "");
        publicLink.hidden = false;
      } else {
        publicLink.hidden = true;
      }
    }
    const runtime = (info && info.runtime) || {};
    const runtimeSummary = String(runtime.summary || "").trim();
    const tagline = String((info && info.tagline) || "").trim();
    const welcomeLine = runtimeSummary || tagline;
    if (welcomeLine) {
      const welcomeSub = document.querySelector(".welcome-sub");
      if (welcomeSub) welcomeSub.textContent = welcomeLine;
      const metaDesc = document.querySelector('meta[name="description"]');
      if (metaDesc) metaDesc.setAttribute("content", tagline || welcomeLine);
    }
    statusEl.classList.add("ready");
    statusEl.classList.remove("error");
    const sidebarStatus = document.getElementById("sidebar-status");
    const sidebarStatusText = document.getElementById("sidebar-status-text");
    if (sidebarStatus) sidebarStatus.classList.add("ready");
    const readyModels = (info.models || []).filter((m) => m.trained);
    textEl.textContent = readyModels.length ? "Online" : "Online · warming up";
    if (sidebarStatusText) sidebarStatusText.textContent = readyModels.length ? "Online" : "Warming up";
    const parts = [];
    if (info.gpu) parts.push(info.gpu);
    infoEl.textContent = parts.join(" · ");
    if (SITE_FEATURES.train_strip !== false) {
      try {
        await loadTrainProgress();
      } catch (_) {
        const trainLine = $("#train-line");
        if (trainLine) trainLine.textContent = "";
      }
    } else {
      const trainLine = $("#train-line");
      if (trainLine) trainLine.textContent = "";
    }
    if (SITE_FEATURES.footer_ops === false) {
      if (autolearnEl) autolearnEl.textContent = "";
      if (marathonEl) marathonEl.textContent = "";
    } else {
      try {
        const autolearn = await apiGet("/autolearn/status");
        if (autolearnEl) {
          autolearnEl.textContent = autolearn.enabled
            ? `Learn · ${autolearn.total_samples || 0}`
            : "";
        }
      } catch (_) {
        if (autolearnEl) autolearnEl.textContent = "";
      }
      try {
        const marathon = await apiGet("/marathon/status");
        if (marathonEl) {
          if (marathon.running) {
            const elapsed = marathon.elapsed_hours != null ? marathon.elapsed_hours.toFixed(2) : "0";
            marathonEl.textContent = `Marathon · ${elapsed}h`;
          } else if (marathon.cycle_count > 0) {
            marathonEl.textContent = "Marathon idle";
          } else {
            marathonEl.textContent = "";
          }
        }
      } catch (_) {
        if (marathonEl) marathonEl.textContent = "";
      }
    }
  } catch (err) {
    statusEl.classList.add("error");
    statusEl.classList.remove("ready");
    textEl.textContent = "Server unavailable";
    const sidebarStatus = document.getElementById("sidebar-status");
    const sidebarStatusText = document.getElementById("sidebar-status-text");
    if (sidebarStatus) sidebarStatus.classList.remove("ready");
    if (sidebarStatusText) sidebarStatusText.textContent = "Offline";
    infoEl.textContent = err.message;
    if (autolearnEl) autolearnEl.textContent = "";
    if (marathonEl) marathonEl.textContent = "";
  }
}

function setActiveModeChip(profile) {
  const chip = $("#active-mode-chip");
  if (!chip) return;
  if (!profile) {
    chip.classList.add("hidden");
    chip.textContent = "";
    return;
  }
  chip.textContent = `Using ${profile}`;
  chip.classList.remove("hidden");
}

async function streamChat(body, onToken, onMeta, timeoutMs = 180000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(`${API}/chat/stream`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    if (err && err.name === "AbortError") {
      throw new Error("Chat timed out. GPU may be busy with training — retry in a moment.");
    }
    throw err;
  }
  clearTimeout(timer);
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || res.statusText || "Request failed");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalText = "";
  let meta = {};
  const consumePart = (part) => {
    const line = part.split("\n").find((l) => l.startsWith("data: "));
    if (!line) return;
    let evt;
    try {
      evt = JSON.parse(line.slice(6));
    } catch (_) {
      return;
    }
    if (evt.type === "meta") {
      meta = { ...meta, ...evt };
      if (onMeta) onMeta(evt);
    } else if (evt.type === "token") {
      finalText += evt.text || "";
      if (onToken) onToken(finalText);
    } else if (evt.type === "done") {
      finalText = evt.text || finalText;
      meta = { ...meta, ...evt };
    } else if (evt.type === "error") {
      throw new Error(evt.detail || "Stream error");
    }
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) consumePart(part);
  }
  if (buffer.trim()) consumePart(buffer);
  return { text: finalText, ...meta };
}

function initTextForm() {
  const form = $("#text-form");
  const btn = form.querySelector('button[type="submit"]');
  if (btn) btn.dataset.label = "Send";
  let chatAttachData = null;
  const attachInput = $("#chat-attach-image");
  const attachBtn = $("#chat-attach-btn");
  const attachClear = $("#chat-attach-clear");
  const attachPreview = $("#chat-attach-preview");
  const clearAttach = () => {
    chatAttachData = null;
    if (attachInput) attachInput.value = "";
    if (attachPreview) {
      attachPreview.innerHTML = "";
      attachPreview.classList.add("hidden");
    }
    if (attachClear) attachClear.classList.add("hidden");
  };
  if (attachBtn && attachInput) {
    attachBtn.addEventListener("click", () => attachInput.click());
    attachInput.addEventListener("change", async () => {
      const file = attachInput.files && attachInput.files[0];
      if (!file) {
        clearAttach();
        return;
      }
      const dataUrl = await readFileAsBase64(file);
      chatAttachData = { dataUrl, mime: file.type || "image/jpeg", name: file.name };
      if (attachPreview) {
        attachPreview.innerHTML = `<img src="${dataUrl}" alt="attachment"><span class="output-path">${file.name}</span>`;
        attachPreview.classList.remove("hidden");
      }
      if (attachClear) attachClear.classList.remove("hidden");
    });
  }
  if (attachClear) attachClear.addEventListener("click", clearAttach);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = $("#text-prompt").value.trim();
    if (!message && !chatAttachData) return;
    const maxEl = $("#text-max-tokens");
    const tempEl = $("#text-temperature");
    const maxRaw = maxEl ? maxEl.value : "";
    const tempRaw = tempEl ? tempEl.value : "";
    const searchToggle = $("#text-use-search");
    const modeEl = $("#text-mode");
    const body = {
      message: message || "Describe this image.",
      history: chatHistory,
      use_search: searchToggle
        ? searchToggle.checked
        : wantsExplicitSearch(message) && !isLocalHandlerQuery(message),
      use_rag: !isCasualChitchat(message),
    };
    if (chatAttachData) {
      body.image_base64 = chatAttachData.dataUrl;
      body.image_mime = chatAttachData.mime;
    }
    if (modeEl && modeEl.value) {
      body.model_profile = modeEl.value;
    }
    if (chatSessionId) body.session_id = chatSessionId;
    body.allow_emoji = getAllowEmoji();
    if (maxRaw) body.max_tokens = parseInt(maxRaw, 10);
    if (tempRaw) body.temperature = parseFloat(tempRaw);
    const userVisual = chatAttachData
      ? `${escapeHtml(message || "Describe this image.")}<br><img src="${chatAttachData.dataUrl}" alt="attachment" style="max-height:120px;border-radius:8px;margin-top:0.4rem;">`
      : message;
    addMessage("user", userVisual, !!chatAttachData);
    $("#text-prompt").value = "";
    autoResizeTextarea($("#text-prompt"));
    clearAttach();
    const loadingLabel = body.image_base64
      ? "Reading image..."
      : body.use_search
        ? "Searching..."
        : "Thinking...";
    setLoading(form, true, loadingLabel);
    const streamId = `stream-${Date.now()}`;
    addMessage("assistant", "", false);
    const container = $("#chat-messages");
    const last = container && container.lastElementChild;
    if (last) last.dataset.streamId = streamId;
    const bodyEl = last && last.querySelector(".message-body");
    if (bodyEl) showThinking(bodyEl, loadingLabel.replace(/\.\.\.$/, ""));
    let streamText = "";
    const finishTypewriter = bodyEl
      ? streamTypewriter(bodyEl, container, () => streamText)
      : null;
    try {
      const data = await streamChat(
        body,
        (partial) => {
          streamText = partial || "";
        },
        (meta) => setActiveModeChip(meta.model_profile)
      );
      const text = data.text || streamText || "";
      streamText = text;
      if (!String(text).trim()) {
        if (finishTypewriter) finishTypewriter();
        if (last && last.dataset.streamId === streamId) last.remove();
        try {
          const retry = await apiPost("/chat", body, 180000);
          const retryText = (retry && retry.text) || "";
          if (!String(retryText).trim()) {
            addMessage("error", "No reply was returned. Hard-refresh (Ctrl+F5), then try again.");
          } else {
            addAssistantMessage(retryText);
            setActiveModeChip(retry.model_profile);
            chatHistory.push({ user: message || "Describe this image.", assistant: retryText });
            if (retry.session_id) chatSessionId = retry.session_id;
            refreshChatHistoryPanel();
            offerFileDownload(retryText, null);
          }
        } catch (errEmpty) {
          addMessage("error", errEmpty.message || "No reply was returned. Hard-refresh (Ctrl+F5), then try again.");
        }
      } else if (finishTypewriter) {
        finishTypewriter();
      } else if (bodyEl) {
        startTypewriter(bodyEl, container, text);
      } else {
        addAssistantMessage(text);
      }
      if (String(text).trim()) {
        setActiveModeChip(data.model_profile);
        chatHistory.push({ user: message || "Describe this image.", assistant: text });
        if (data.session_id) chatSessionId = data.session_id;
        refreshChatHistoryPanel();
        offerFileDownload(text, last);
      }
    } catch (err) {
      if (last && last.dataset.streamId === streamId) last.remove();
      try {
        const data = await apiPost("/chat", body, 180000);
        const reply = (data && data.text) || "";
        if (!String(reply).trim()) {
          addMessage("error", err.message || "No reply was returned.");
        } else {
          addAssistantMessage(reply);
          setActiveModeChip(data.model_profile);
          chatHistory.push({ user: message || "Describe this image.", assistant: reply });
          if (data.session_id) chatSessionId = data.session_id;
          refreshChatHistoryPanel();
        }
      } catch (err2) {
        addMessage("error", err2.message || err.message);
      }
    } finally {
      setLoading(form, false);
    }
  });
}

function initImageForm() {
  const form = $("#image-form");
  const btn = form.querySelector('button[type="submit"]');
  btn.dataset.label = "Generate Image";
  const stepsInput = $("#image-steps");
  const stepsValue = $("#image-steps-value");
  if (stepsInput && stepsValue) {
    stepsInput.addEventListener("input", () => {
      stepsValue.textContent = stepsInput.value;
    });
  }
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = $("#image-prompt").value.trim();
    if (!prompt) return;
    setLoading(form, true, `${BRAND_NAME} is generating...`);
    showProgress("image-progress", true);
    $("#image-output").innerHTML = "";
    try {
      const body = { prompt };
      if (stepsInput && stepsInput.value) body.num_steps = parseInt(stepsInput.value, 10);
      const data = await apiPost("/image", body, 300000);
      const img = document.createElement("img");
      img.src = `data:${data.mime};base64,${data.data_base64}`;
      img.alt = prompt;
      const out = $("#image-output");
      out.appendChild(img);
      if (data.enhanced_prompt) {
        const enhanced = document.createElement("div");
        enhanced.className = "output-path";
        enhanced.textContent = `Enhanced: ${data.enhanced_prompt}`;
        out.appendChild(enhanced);
      }
      if (data.reference_used && data.keyword_matched) {
        const refLine = document.createElement("div");
        refLine.className = "output-path";
        const strength = data.reference_strength != null ? ` (${Math.round(data.reference_strength * 100)}%)` : "";
        refLine.textContent =
          `Reference: ${data.keyword_matched} -> ${data.reference_category || "matched"}${strength}`;
        out.appendChild(refLine);
      }
      const path = document.createElement("div");
      path.className = "output-path";
      path.textContent = data.path;
      out.appendChild(path);
    } catch (err) {
      $("#image-output").innerHTML = `<div class="message error">${err.message}</div>`;
    } finally {
      showProgress("image-progress", false);
      setLoading(form, false);
    }
  });
}

function initVideoForm() {
  const form = $("#video-form");
  const btn = form.querySelector('button[type="submit"]');
  btn.dataset.label = "Generate Video";
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = $("#video-prompt").value.trim();
    if (!prompt) return;
    setLoading(form, true, `${BRAND_NAME} is generating...`);
    showProgress("video-progress", true);
    $("#video-output").innerHTML = "";
    try {
      const body = { prompt };
      const framesRaw = $("#video-frames").value;
      const fpsRaw = $("#video-fps").value;
      if (framesRaw) body.num_frames = parseInt(framesRaw, 10);
      if (fpsRaw) body.fps = parseInt(fpsRaw, 10);
      const data = await apiPost("/video", body, 300000);
      const out = $("#video-output");
      const isVideo = data.mime && data.mime.startsWith("video/");
      if (isVideo) {
        const video = document.createElement("video");
        video.src = `data:${data.mime};base64,${data.data_base64}`;
        video.controls = true;
        video.autoplay = true;
        video.loop = true;
        video.muted = true;
        video.alt = prompt;
        out.appendChild(video);
      } else {
        const img = document.createElement("img");
        img.src = `data:${data.mime};base64,${data.data_base64}`;
        img.alt = prompt;
        out.appendChild(img);
      }
      const path = document.createElement("div");
      path.className = "output-path";
      path.textContent = data.path;
      out.appendChild(path);
      if (data.reference_used && data.keyword_matched) {
        const refLine = document.createElement("div");
        refLine.className = "output-path";
        const strength = data.reference_strength != null ? ` (${Math.round(data.reference_strength * 100)}%)` : "";
        refLine.textContent =
          `Reference: ${data.keyword_matched} -> ${data.reference_category || "matched"}${strength}`;
        out.appendChild(refLine);
      }
    } catch (err) {
      $("#video-output").innerHTML = `<div class="message error">${err.message}</div>`;
    } finally {
      showProgress("video-progress", false);
      setLoading(form, false);
    }
  });
}

let trainPollTimer = null;
let statusPollTimer = null;
let lastTrainRunning = false;
let trainTabActive = false;

function readTrainSteps(defaultValue) {
  const stepsEl = $("#train-steps");
  const steps = stepsEl ? Number(stepsEl.value || defaultValue) : defaultValue;
  return Number.isFinite(steps) ? steps : defaultValue;
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Could not read file"));
    reader.readAsDataURL(file);
  });
}

async function appendTrainMedia(body) {
  const voiceFileEl = $("#train-voice-sample-file");
  if (voiceFileEl && voiceFileEl.files && voiceFileEl.files[0]) {
    body.voice_sample_base64 = await readFileAsBase64(voiceFileEl.files[0]);
  }
  const dfSourcePath = $("#train-deepfake-source");
  const dfTargetPath = $("#train-deepfake-target");
  if (dfSourcePath && dfSourcePath.value.trim()) body.deepfake_source = dfSourcePath.value.trim();
  if (dfTargetPath && dfTargetPath.value.trim()) body.deepfake_target = dfTargetPath.value.trim();
  const dfSourceFile = $("#train-deepfake-source-file");
  const dfTargetFile = $("#train-deepfake-target-file");
  if (dfSourceFile && dfSourceFile.files && dfSourceFile.files[0]) {
    body.deepfake_source_base64 = await readFileAsBase64(dfSourceFile.files[0]);
  }
  if (dfTargetFile && dfTargetFile.files && dfTargetFile.files[0]) {
    body.deepfake_target_base64 = await readFileAsBase64(dfTargetFile.files[0]);
  }
  return body;
}

function bindImagePreview(inputId, previewId) {
  const input = $(inputId);
  const preview = $(previewId);
  if (!input || !preview) return;
  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) {
      preview.classList.add("hidden");
      preview.removeAttribute("src");
      return;
    }
    preview.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
  });
}

function buildTrainBody(target, defaultSteps) {
  const steps = readTrainSteps(defaultSteps);
  const langEl = $("#train-language");
  const language = langEl ? langEl.value.trim() : "";
  const cfgEl = $("#train-config");
  const config = cfgEl ? cfgEl.value.trim() : "";
  const body = {
    target,
    steps,
    require_cuda: target !== "voice",
  };
  if (language) body.language = language;
  if (config) body.config = config;

  const voiceNameEl = $("#train-voice-name");
  const voiceSampleEl = $("#train-voice-sample");
  const voiceDefaultEl = $("#train-voice-set-default");
  const voiceName = voiceNameEl ? voiceNameEl.value.trim() : "";
  const voiceSample = voiceSampleEl ? voiceSampleEl.value.trim() : "";
  if (voiceName) body.voice_name = voiceName;
  if (voiceSample) body.voice_sample = voiceSample;
  if (voiceDefaultEl) body.voice_set_default = voiceDefaultEl.value !== "false";

  const strengthEl = $("#train-deepfake-strength");
  if (strengthEl && strengthEl.value !== "") {
    const strength = Number(strengthEl.value);
    if (Number.isFinite(strength)) body.deepfake_strength = strength;
  }
  return body;
}

function formatEta(seconds) {
  if (seconds == null || seconds < 0) return "";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}

function formatAge(seconds) {
  if (seconds == null) return "";
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function productLabel(name) {
  if (!name) return "";
  if (isForeignBrand(name)) {
    if (THEME_ID.includes("navuryx")) return "Navuryx";
    return "Navine";
  }
  const n = String(name).toLowerCase();
  if (n === "navine") return "Navine";
  if (n === "navuryx") return "Navuryx";
  return name;
}

function formatTrainCycle(tr) {
  const round = Number(tr.marathon_round || tr.cycle || 0);
  if (tr.marathon_loop) return `round ${round}`;
  const max = Number(tr.max_cycles || 0);
  if (max > 0) return `cycle ${round}/${max}`;
  return `cycle ${round}`;
}

function renderTrainCycleBanner(tr) {
  const round = Number(tr.marathon_round || tr.cycle || 0);
  const valueEl = $("#train-cycle-value");
  const subEl = $("#train-cycle-sub");
  const settingsRound = $("#settings-train-cycle");
  const settingsSub = $("#settings-train-sub");
  const stage = tr.stage || tr.label || (tr.running ? "training" : "idle");
  const steps = tr.steps_total > 0 ? ` · ${tr.steps || 0}/${tr.steps_total} steps` : "";
  const pct = Math.round(Number(tr.percent) || 0);
  const sub = tr.running
    ? `${stage} · ${pct}%${steps}`
    : `idle · last round ${round}`;
  if (valueEl) valueEl.textContent = String(round || "—");
  if (subEl) subEl.textContent = sub;
  if (settingsRound) settingsRound.textContent = String(round || "—");
  if (settingsSub) settingsSub.textContent = sub;
}

async function loadTrainProgress() {
  try {
    const tr = await apiGet("/train/progress");
    renderTrainCycleBanner(tr);
    renderTrainStrip(tr);
    renderTrainBoard(tr);
    const trainLine = $("#train-line");
    const cycleText = formatTrainCycle(tr);
    if (trainLine) {
      if (tr.running) {
        const prod = productLabel(tr.active_product);
        let stage = tr.stage || tr.label || "training";
        if (isForeignBrand(stage)) stage = "training";
        const pct = Math.round(tr.percent || 0);
        const steps =
          tr.steps_total > 0 ? ` · ${tr.steps || 0}/${tr.steps_total}` : "";
        trainLine.textContent = `Train: ${prod} ${stage} ${pct}%${steps} · ${cycleText}`;
      } else {
        trainLine.textContent = `Train: idle · ${cycleText}`;
      }
    }
    lastTrainRunning = !!tr.running;
    scheduleTrainPoll();
    return tr;
  } catch (err) {
    const meta = $("#train-strip-meta");
    if (meta) meta.textContent = `Train: ${err.message || "progress unavailable"}`;
    const loopEl = $("#train-loop");
    if (loopEl) loopEl.textContent = err.message || "Could not load training progress";
    throw err;
  }
}

function scheduleTrainPoll(forceFast) {
  if (trainPollTimer) clearInterval(trainPollTimer);
  let ms = 12000;
  if (lastTrainRunning) ms = 1500;
  else if (trainTabActive || forceFast) ms = 3000;
  trainPollTimer = setInterval(() => {
    loadTrainProgress().catch(() => {});
  }, ms);
}

function renderTrainStrip(tr) {
  const strip = $("#train-strip");
  if (!strip) return;
  const dot = $("#train-strip-dot");
  const meta = $("#train-strip-meta");
  const bar = $("#train-strip-bar");
  const locksEl = $("#train-strip-locks");
  strip.classList.toggle("running", !!tr.running);
  strip.classList.toggle("idle", !tr.running);
  const pct = Math.max(0, Math.min(100, Number(tr.percent) || 0));
  if (bar) {
    bar.style.width = `${pct}%`;
    bar.classList.toggle("active", pct > 0);
  }
  const prod = productLabel(tr.active_product);
  let stage = tr.stage || tr.label || (tr.running ? "training" : "idle");
  if (isForeignBrand(stage)) stage = "training";
  const eta = formatEta(tr.eta_s);
  const cycleText = formatTrainCycle(tr);
  if (meta) {
    meta.textContent = tr.running
      ? `${prod || "Train"} · ${stage} · ${pct}% · ${cycleText}${eta ? ` · ETA ${eta}` : ""}`
      : `Train idle · ${cycleText}`;
  }
  if (locksEl) {
    const active = Object.entries(tr.locks || {})
      .filter(([, v]) => v)
      .map(([k]) => k);
    locksEl.textContent = active.length ? `locks: ${active.join(", ")}` : "";
  }
  if (dot) dot.title = tr.running ? "Training running" : "Training idle";
}

function renderTrainBoard(tr) {
  const modelsEl = $("#train-models");
  const loopEl = $("#train-loop");
  const stagesEl = $("#train-stages");
  const productsEl = $("#train-products");
  const logEl = $("#train-log");
  const bar = $("#train-progress-bar");

  if (modelsEl) {
    modelsEl.innerHTML = (tr.models || [])
      .filter((m) => {
        const name = String(m.name || "").toLowerCase();
        return !name.includes("hitboy") && !name.includes("navuryx");
      })
      .map((m) => {
        let state = "missing";
        let label = "missing";
        if (m.training || (tr.locks && tr.locks[m.name])) {
          state = "training";
          label = "training now";
        } else if (m.trained) {
          state = "ready";
          label = "trained";
        }
        const age = formatAge(m.age_seconds);
        return `<div class="train-model-card ${state}">
        <div class="train-model-name">${m.name}</div>
        <div class="train-model-state">${label}</div>
        <div class="train-model-meta">${age ? `last save ${age}` : "no checkpoint"}</div>
      </div>`;
      })
      .join("");
  }

  const pct = Math.max(0, Math.min(100, Number(tr.percent) || 0));
  if (bar) {
    bar.style.width = `${pct}%`;
    bar.classList.toggle("active", pct > 0 || !!tr.running);
  }
  const pctEl = $("#train-progress-pct");
  const progressLabel = $("#train-progress-label");
  if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
  if (progressLabel) {
    const stageLabel = tr.stage || tr.label || (tr.running ? "training" : "idle");
    progressLabel.textContent = tr.running
      ? `${productLabel(tr.active_product) || "Train"} · ${stageLabel}`
      : "Progress · idle";
  }

  const scores = tr.best_scores || {};
  const scoreText = Object.keys(scores)
    .filter((k) => !isForeignBrand(k))
    .map((k) => `${k}=${scores[k]}`)
    .join(" | ");
  const steps =
    tr.steps_total > 0 ? ` · steps ${tr.steps || 0}/${tr.steps_total}` : "";
  const eta = formatEta(tr.eta_s);
  const loopStage = isForeignBrand(tr.stage) ? "training" : (tr.stage || "—");
  const cycleText = formatTrainCycle(tr);
  if (loopEl) {
    loopEl.textContent =
      `${tr.running ? "RUNNING" : "idle"} · ${productLabel(tr.active_product) || (isForeignBrand(tr.product) ? "Navine" : (tr.product || ""))} · ` +
      `stage ${loopStage} · ${pct}% · ${cycleText}` +
      steps +
      (eta ? ` · ETA ${eta}` : "") +
      (scoreText ? `\nbest: ${scoreText}` : "\nno scores yet");
  }

  if (productsEl) {
    productsEl.innerHTML = (tr.products || [])
      .filter((p) => !isForeignBrand(p.name))
      .map((p) => {
        const score = p.score != null ? ` score ${p.score}` : "";
        const phase = p.phase ? ` · ${p.phase}` : "";
        const sampleRaw = p.sample ? String(p.sample) : "";
        const sample = sampleRaw && !isForeignBrand(sampleRaw) ? `\n${sampleRaw.slice(0, 80)}` : "";
        return `<div class="train-product ${p.active ? "active" : ""}">${productLabel(p.name)}${phase}${score}${sample}</div>`;
      })
      .join("");
  }

  if (stagesEl) {
    const stages = (tr.stages || []).filter((s) => !isForeignBrand(s));
    const current = (tr.stage || "").toLowerCase();
    const idx = current ? stages.findIndex((s) => String(s).toLowerCase() === current) : -1;
    stagesEl.innerHTML = stages
      .map((s, i) => {
        let cls = "train-stage";
        if (idx >= 0 && i < idx) cls += " done";
        else if (idx >= 0 && i === idx) cls += " active";
        return `<span class="${cls}">${s}</span>`;
      })
      .join(" ");
  }

  if (logEl) {
    const rawLines = tr.recent_lines && tr.recent_lines.length
      ? tr.recent_lines
      : tr.last_line
        ? [tr.last_line]
        : ["(no think-train log yet)"];
    const lines = rawLines.filter((ln) => !isForeignBrand(ln));
    logEl.textContent = (lines.length ? lines : ["(no think-train log yet)"]).join("\n");
  }
}

function initTrainEverything() {
  const btn = $("#train-everything-btn");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    if (!getAdminToken()) {
      showTrainLogin();
      return;
    }
    btn.disabled = true;
    const hint = $("#train-everything-hint");
    if (hint) hint.textContent = "Starting comprehensive training in a console window...";
    try {
      await adminApiPost("/train/comprehensive", { mode: "full" });
      if (hint) hint.textContent = "Training started. Watch the Train panel and console for progress.";
    } catch (err) {
      if (hint) hint.textContent = err.message || "Could not start training.";
    } finally {
      btn.disabled = false;
    }
  });
}

function showTrainLogin() {
  const login = $("#train-login");
  const board = $("#train-board");
  if (login) login.classList.remove("hidden");
  if (board && !getSessionToken()) board.classList.add("hidden");
}

function showTrainBoard(username, isAdmin) {
  const login = $("#train-login");
  const board = $("#train-board");
  if (login) login.classList.add("hidden");
  if (board) board.classList.remove("hidden");
  const label = $("#train-admin-label");
  const role = isAdmin ? "admin" : "user";
  if (label) {
    if (isAdmin) label.textContent = username ? `Training as ${username}` : "Admin training";
    else if (username) label.textContent = `Signed in as ${username}`;
    else label.textContent = "";
  }
  const adminSection = $("#train-admin-section");
  const adminNote = $("#train-admin-only-note");
  const apiSection = $("#account-api-keys");
  const apiCreate = $("#api-keys-create");
  const apiHint = $("#api-key-login-hint");
  const signedIn = !!getSessionToken() && !isGuestMode();
  if (adminSection) adminSection.classList.toggle("hidden", !isAdmin);
  if (adminNote) adminNote.classList.toggle("hidden", isAdmin);
  if (apiSection) apiSection.classList.toggle("hidden", false);
  if (apiCreate) apiCreate.classList.toggle("hidden", !signedIn);
  if (apiHint) apiHint.classList.toggle("hidden", signedIn);
  if (signedIn) loadApiKeys().catch(() => {});
  else {
    const listEl = $("#api-key-list");
    if (listEl) listEl.innerHTML = "";
  }
  loadTrainProgress().catch(() => {});
  scheduleTrainPoll(true);
  if (isAdmin) {
    const corpusGrid = $("#train-corpus-grid");
    if (corpusGrid) loadTrainCorpusTypes(corpusGrid).catch(() => {});
    loadTrainJobs().catch(() => {});
  }
}

async function loadApiKeys() {
  const listEl = $("#api-key-list");
  if (!listEl) return;
  try {
    const data = await sessionApi("/keys");
    const keys = (data.keys || []).filter((k) => !k.revoked);
    if (!keys.length) {
      listEl.innerHTML = `<div class="output-path">No API keys yet.</div>`;
      return;
    }
    listEl.innerHTML = keys
      .map((k) => {
        const prefix = k.prefix ? `${k.prefix}…` : k.id;
        const uses = typeof k.use_count === "number" ? `${k.use_count} uses` : "";
        const exp = k.expires_at ? `expires ${k.expires_at.slice(0, 10)}` : "no expiry";
        return `<div class="api-key-row"><div><strong>${k.name || "API key"}</strong> · ${prefix}<div class="api-key-meta">${uses} · ${exp}</div></div><button type="button" class="btn ghost api-key-delete" data-key-id="${k.id}">Delete</button></div>`;
      })
      .join("");
  } catch (err) {
    listEl.innerHTML = `<div class="output-path">${err.message || "Could not load keys"}</div>`;
  }
}

async function createApiKeyFromUi() {
  if (!getSessionToken() || isGuestMode()) {
    throw new Error("Sign in required to create an API key");
  }
  const nameEl = $("#api-key-name");
  const daysEl = $("#api-key-days");
  const createdEl = $("#api-key-created");
  const name = (nameEl && nameEl.value.trim()) || "Navine API";
  const daysRaw = daysEl && daysEl.value ? Number(daysEl.value) : null;
  const body = { name };
  if (daysRaw && daysRaw > 0) body.expires_days = daysRaw;
  const data = await sessionApi("/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (createdEl) {
    createdEl.textContent = `New key (copy now): ${data.key}`;
    createdEl.classList.remove("hidden");
  }
  const apiGenKey = $("#api-gen-key");
  if (apiGenKey && data.key) {
    apiGenKey.value = data.key;
    sessionStorage.setItem("navine_api_key", data.key);
  }
  await loadApiKeys();
  return data.key;
}

async function deleteApiKeyFromUi(keyId) {
  if (!getSessionToken() || isGuestMode()) {
    throw new Error("Sign in required to delete an API key");
  }
  await sessionApi(`/keys/${encodeURIComponent(keyId)}`, { method: "DELETE" });
  const saved = sessionStorage.getItem("navine_api_key") || "";
  if (saved && saved.startsWith(keyId)) {
    sessionStorage.removeItem("navine_api_key");
  }
  await loadApiKeys();
}

async function restoreAdminSession() {
  const token = getSessionToken();
  const user = sessionStorage.getItem(SESSION_USER_KEY) || "";
  if (!token) {
    showTrainLogin();
    refreshChatHistoryPanel();
    return;
  }
  try {
    const data = await sessionApi("/auth/session");
    setSession(token, data.username || user, !!data.is_admin);
    showTrainBoard(data.username || user, !!data.is_admin);
    refreshChatHistoryPanel();
  } catch {
    showTrainLogin();
    refreshChatHistoryPanel();
  }
}

function setGateTab(tab) {
  document.querySelectorAll("[data-gate-tab]").forEach((btn) => {
    const active = btn.getAttribute("data-gate-tab") === tab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll("[data-gate-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", panel.getAttribute("data-gate-panel") !== tab);
  });
}

function initSiteGate() {
  const signoutBtn = $("#sidebar-profile-signout");
  if (signoutBtn) {
    signoutBtn.addEventListener("click", signOutProfile);
  }

  if (gatePassed()) {
    sessionStorage.removeItem(PENDING_PATH_KEY);
    hideSiteGate();
    updateProfileLabel();
    return;
  }
  rememberPendingPath(window.location.pathname);
  showSiteGate();

  document.querySelectorAll("[data-gate-tab]").forEach((btn) => {
    btn.addEventListener("click", () => setGateTab(btn.getAttribute("data-gate-tab") || "login"));
  });

  const tryBtn = $("#gate-try-btn");
  if (tryBtn) {
    tryBtn.addEventListener("click", () => enterApp({ mode: "guest" }));
  }

  const loginForm = $("#gate-login-form");
  const signupForm = $("#gate-signup-form");
  const loginErr = $("#gate-login-error");
  const signupErr = $("#gate-signup-error");

  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const username = ($("#gate-login-user") && $("#gate-login-user").value.trim()) || "";
      const password = ($("#gate-login-pass") && $("#gate-login-pass").value) || "";
      if (loginErr) loginErr.classList.add("hidden");
      try {
        const data = await apiPost("/auth/login", { username, password });
        setSession(data.token, data.username || username, !!data.is_admin);
        enterApp({ mode: "user" });
      } catch (err) {
        if (loginErr) {
          loginErr.textContent = err.message || "Login failed";
          loginErr.classList.remove("hidden");
        }
      }
    });
  }

  if (signupForm) {
    signupForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const username = ($("#gate-signup-user") && $("#gate-signup-user").value.trim()) || "";
      const password = ($("#gate-signup-pass") && $("#gate-signup-pass").value) || "";
      if (signupErr) signupErr.classList.add("hidden");
      try {
        const data = await apiPost("/auth/signup", { username, password });
        setSession(data.token, data.username || username, false);
        enterApp({ mode: "user" });
      } catch (err) {
        if (signupErr) {
          signupErr.textContent = err.message || "Sign up failed";
          signupErr.classList.remove("hidden");
        }
      }
    });
  }
}

function initTrainAdmin() {
  const adminForm = $("#train-admin-form");
  const logoutBtn = $("#train-admin-logout");
  const errEl = $("#train-login-error");

  if (adminForm) {
    adminForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const username = ($("#train-admin-user") && $("#train-admin-user").value.trim()) || "";
      if (errEl) errEl.classList.add("hidden");
      if (!username) return;
      try {
        const data = await apiPost("/train/admin-access", { username });
        setSession(data.token, data.username || username, true);
        showTrainBoard(data.username || username, true);
      } catch (err) {
        if (errEl) {
          errEl.textContent = err.message || "Admin access denied";
          errEl.classList.remove("hidden");
        }
      }
    });
  }

  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      if (isSessionAdmin()) {
        clearSession();
        showTrainLogin();
        updateProfileLabel();
      } else {
        signOutProfile();
      }
    });
  }

  const createKeyBtn = $("#api-key-create-btn");
  if (createKeyBtn) {
    createKeyBtn.addEventListener("click", async () => {
      if (!getSessionToken() || isGuestMode()) {
        const createdEl = $("#api-key-created");
        if (createdEl) {
          createdEl.textContent = "Sign in or create an account to make an API key.";
          createdEl.classList.remove("hidden");
        }
        return;
      }
      createKeyBtn.disabled = true;
      try {
        await createApiKeyFromUi();
      } catch (err) {
        const createdEl = $("#api-key-created");
        if (createdEl) {
          createdEl.textContent = err.message || "Could not create key";
          createdEl.classList.remove("hidden");
        }
      } finally {
        createKeyBtn.disabled = false;
      }
    });
  }

  const keyList = $("#api-key-list");
  if (keyList) {
    keyList.addEventListener("click", async (e) => {
      const btn = e.target.closest(".api-key-delete");
      if (!btn) return;
      const keyId = btn.getAttribute("data-key-id");
      if (!keyId) return;
      btn.disabled = true;
      try {
        await deleteApiKeyFromUi(keyId);
      } catch (err) {
        const createdEl = $("#api-key-created");
        if (createdEl) {
          createdEl.textContent = err.message || "Could not delete key";
          createdEl.classList.remove("hidden");
        }
      } finally {
        btn.disabled = false;
      }
    });
  }

  const grid = $("#train-modality-grid");
  const corpusGrid = $("#train-corpus-grid");
  if (grid) {
    grid.addEventListener("click", (e) => onTrainButtonClick(e));
  }
  if (corpusGrid) {
    loadTrainCorpusTypes(corpusGrid).catch(() => {});
    corpusGrid.addEventListener("click", (e) => onTrainButtonClick(e));
  }

  const learnGrid = $("#train-learn-grid");
  if (learnGrid) {
    learnGrid.addEventListener("click", async (e) => {
      const btn = e.target.closest("[data-learn-action]");
      if (!btn) return;
      if (!getAdminToken()) {
        showTrainLogin();
        return;
      }
      const action = btn.getAttribute("data-learn-action");
      const url = ($("#train-learn-url") && $("#train-learn-url").value.trim()) || "";
      const pages = ($("#train-crawl-pages") && Number($("#train-crawl-pages").value)) || 8;
      const hint = $("#train-everything-hint");
      const body = { action, max_pages: pages };
      if (action === "text") {
        body.text = ($("#train-custom-text") && $("#train-custom-text").value.trim()) || "";
        body.title = ($("#train-custom-title") && $("#train-custom-title").value.trim()) || "Custom text";
        if (body.text.length < 8) {
          if (hint) hint.textContent = "Paste text in Custom learn section first.";
          return;
        }
      } else if (action === "url" || action === "crawl") {
        if (!url) {
          if (hint) hint.textContent = "Enter a URL in the learn URL field.";
          return;
        }
        body.url = url;
      }
      btn.disabled = true;
      if (hint) hint.textContent = `Running learn: ${action}...`;
      try {
        await adminApiPost("/train/learn", body);
        if (hint) hint.textContent = `Learn job started: ${action}`;
        loadTrainJobs().catch(() => {});
      } catch (err) {
        if (hint) hint.textContent = err.message || "Learn failed.";
      } finally {
        btn.disabled = false;
      }
    });
  }

  restoreAdminSession();

  const sourceSel = $("#train-custom-source");
  const urlWrap = $("#train-custom-url-wrap");
  const textWrap = $("#train-custom-text-wrap");
  const toggleCustomFields = () => {
    const mode = sourceSel ? sourceSel.value : "text";
    if (urlWrap) urlWrap.classList.toggle("hidden", mode !== "url");
    if (textWrap) textWrap.classList.toggle("hidden", mode === "url");
  };
  if (sourceSel) {
    sourceSel.addEventListener("change", toggleCustomFields);
    toggleCustomFields();
  }

  const customForm = $("#train-custom-form");
  if (customForm) {
    customForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!getAdminToken()) {
        showTrainLogin();
        return;
      }
      const source = sourceSel ? sourceSel.value : "text";
      const trainTarget = ($("#train-custom-target") && $("#train-custom-target").value) || "text";
      const status = $("#train-custom-status");
      const body = {
        source,
        train_target: trainTarget,
        require_cuda: trainTarget !== "voice",
      };
      Object.assign(body, buildTrainBody(trainTarget, 400));
      delete body.target;
      await appendTrainMedia(body);
      if (source === "url") {
        body.url = ($("#train-custom-url") && $("#train-custom-url").value.trim()) || "";
      } else {
        body.text = ($("#train-custom-text") && $("#train-custom-text").value.trim()) || "";
        body.title = ($("#train-custom-title") && $("#train-custom-title").value.trim()) || undefined;
      }
      if (status) status.textContent = "Starting custom learn + train...";
      try {
        const data = await adminApiPost("/train/custom", body, 180000);
        if (status) {
          status.textContent = `Custom job started: ${(data.job && data.job.id) || "ok"}`;
        }
        loadTrainJobs().catch(() => {});
        loadTrainProgress().catch(() => {});
        scheduleTrainPoll(true);
      } catch (err) {
        if (status) status.textContent = err.message || "Custom training failed.";
      }
    });
  }

  const voiceCreateBtn = $("#train-voice-create-btn");
  if (voiceCreateBtn) {
    voiceCreateBtn.addEventListener("click", async () => {
      if (!getAdminToken()) {
        showTrainLogin();
        return;
      }
      const hint = $("#train-everything-hint");
      voiceCreateBtn.disabled = true;
      if (hint) hint.textContent = "Creating custom voice model...";
      try {
        const body = buildTrainBody("voice", 8);
        await appendTrainMedia(body);
        const data = await adminApiPost("/train/start", body);
        if (hint) {
          hint.textContent = `Voice job started: ${(data.job && data.job.id) || "ok"}`;
        }
        loadTrainJobs().catch(() => {});
        loadTrainProgress().catch(() => {});
        scheduleTrainPoll(true);
      } catch (err) {
        if (hint) hint.textContent = err.message || "Could not start voice training.";
      } finally {
        voiceCreateBtn.disabled = false;
      }
    });
  }
}

async function loadTrainCorpusTypes(container) {
  if (!container || !getAdminToken()) return;
  try {
    const data = await adminApiGet("/train/types");
    const types = data.types || [];
    container.innerHTML = types
      .map(
        (row) =>
          `<button type="button" class="btn" data-train-target="${row.name}" title="${row.description || ""}">${row.name}</button>`
      )
      .join("");
  } catch {
    container.innerHTML = "";
  }
}

async function onTrainButtonClick(e) {
  const btn = e.target.closest("[data-train-target]");
  if (!btn || btn.id === "train-everything-btn") return;
  if (!getAdminToken()) {
    showTrainLogin();
    return;
  }
  const target = btn.getAttribute("data-train-target");
  if (!isTrainTargetAllowed(target)) return;
  const hint = $("#train-everything-hint");
  btn.disabled = true;
  if (hint) hint.textContent = `Starting ${target} training...`;
  try {
    const body = buildTrainBody(target, 800);
    await appendTrainMedia(body);
    const data = await adminApiPost("/train/start", body);
    if (hint) hint.textContent = `Started ${target} job ${(data.job && data.job.id) || ""}`.trim();
    loadTrainJobs().catch(() => {});
    loadTrainProgress().catch(() => {});
    scheduleTrainPoll(true);
  } catch (err) {
    if (hint) hint.textContent = err.message || "Could not start training.";
  } finally {
    btn.disabled = false;
  }
}

async function loadTrainJobs() {
  const el = $("#train-jobs");
  if (!el || !getAdminToken()) return;
  try {
    const data = await adminApiGet("/train/jobs");
    const jobs = data.jobs || [];
    if (!jobs.length) {
      el.innerHTML = `<div class="output-path">No recent admin jobs yet.</div>`;
      return;
    }
    el.innerHTML = jobs
      .map((job) => {
        const target = job.target || "text";
        const status = job.status || "unknown";
        const by = job.started_by ? ` · ${job.started_by}` : "";
        return `<div class="train-job-row"><strong>${target}</strong> · ${status}${by}<div class="output-path">${job.id || ""}</div></div>`;
      })
      .join("");
  } catch {
    el.innerHTML = "";
  }
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function encodeWavBase64(float32Samples, sampleRate) {
  const numSamples = float32Samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, float32Samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return bytesToBase64(new Uint8Array(buffer));
}

async function playAudioBase64(mime, b64) {
  if (!b64) return;
  const audio = new Audio(`data:${mime || "audio/wav"};base64,${b64}`);
  try {
    await audio.play();
  } catch (_) {}
}

async function speakText(text, voiceName, enabled) {
  if (!enabled || !text) return;
  try {
    const data = await apiPost("/voice/speak", {
      text,
      voice: voiceName || null,
      play: false,
    });
    if (data.data_base64) {
      await playAudioBase64(data.mime || "audio/wav", data.data_base64);
    } else {
      throw new Error("No audio returned from voice engine");
    }
  } catch (err) {
    const msg = err.message || "Voice playback failed";
    addMessage("error", msg, false, "#voicechat-messages");
  }
}

function createMicRecorder(statusEl) {
  let stream = null;
  let audioCtx = null;
  let processor = null;
  let source = null;
  let chunks = [];
  let sampleRate = 16000;
  let recording = false;

  async function ensureMic() {
    if (stream) return;
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  }

  async function start() {
    await ensureMic();
    chunks = [];
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    sampleRate = audioCtx.sampleRate;
    source = audioCtx.createMediaStreamSource(stream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (ev) => {
      if (!recording) return;
      const input = ev.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(input));
    };
    source.connect(processor);
    processor.connect(audioCtx.destination);
    recording = true;
    if (statusEl) statusEl.textContent = "Listening";
  }

  async function stop() {
    recording = false;
    if (processor) {
      try {
        processor.disconnect();
      } catch (_) {}
      processor.onaudioprocess = null;
      processor = null;
    }
    if (source) {
      try {
        source.disconnect();
      } catch (_) {}
      source = null;
    }
    if (audioCtx) {
      try {
        await audioCtx.close();
      } catch (_) {}
      audioCtx = null;
    }
    if (!chunks.length) {
      if (statusEl) statusEl.textContent = "No audio captured";
      return null;
    }
    let total = 0;
    chunks.forEach((c) => {
      total += c.length;
    });
    const merged = new Float32Array(total);
    let offset = 0;
    chunks.forEach((c) => {
      merged.set(c, offset);
      offset += c.length;
    });
    chunks = [];
    if (statusEl) statusEl.textContent = "Transcribing...";
    return encodeWavBase64(merged, sampleRate);
  }

  function close() {
    recording = false;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }

  return { start, stop, close, isRecording: () => recording };
}

async function fillVoiceSelect(selectEl) {
  if (!selectEl) return;
  try {
    const status = await apiGet("/voice/status");
    const voices = (status.voices || []).filter((v) => !isForeignBrand(v.name) && !isForeignBrand(v.display_name));
    selectEl.innerHTML = "";
    let def = status.default_voice || (voices[0] && voices[0].name) || "";
    if (isForeignBrand(def)) def = (voices[0] && voices[0].name) || "";
    if (!voices.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Default";
      selectEl.appendChild(opt);
      return;
    }
    voices.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v.name;
      opt.textContent = v.display_name || v.name;
      if (v.name === def) opt.selected = true;
      selectEl.appendChild(opt);
    });
  } catch (_) {
    selectEl.innerHTML = '<option value="">Default</option>';
  }
}

async function runVoiceTurn(text, containerSel, historyArr, ttsEnabled, voiceName) {
  const trimmed = (text || "").trim();
  if (!trimmed) return;
  addMessage("user", trimmed, false, containerSel);
  historyArr.push({ user: trimmed, assistant: "" });
  try {
    const data = await apiPost("/chat", {
      message: trimmed,
      history: historyArr.slice(0, -1),
      use_search: false,
      use_rag: !isCasualChitchat(trimmed),
    });
    const reply = data.text || "";
    addAssistantMessage(reply, containerSel);
    historyArr[historyArr.length - 1].assistant = reply;
    await speakText(reply, voiceName, ttsEnabled);
  } catch (err) {
    addMessage("error", err.message || String(err), false, containerSel);
  }
}

function initVoiceChat() {
  const statusEl = $("#voicechat-status");
  const pttBtn = $("#voicechat-ptt");
  const toggleBtn = $("#voicechat-toggle");
  const ttsToggle = $("#voicechat-tts");
  const voiceSelect = $("#voicechat-voice");
  const form = $("#voicechat-text-form");
  const history = [];
  const recorder = createMicRecorder(statusEl);
  let continuous = false;
  let busy = false;

  fillVoiceSelect(voiceSelect);

  async function handleRecording() {
    if (busy) return;
    busy = true;
    try {
      const b64 = await recorder.stop();
      if (!b64) {
        if (continuous) {
          await recorder.start();
        }
        return;
      }
      if (statusEl) statusEl.textContent = "Transcribing...";
      const tr = await apiPost("/voice/transcribe", {
        audio_base64: b64,
        mime: "audio/wav",
      });
      const text = (tr.text || "").trim();
      if (!text) {
        if (statusEl) statusEl.textContent = "No speech detected";
        if (continuous) await recorder.start();
        return;
      }
      if (statusEl) statusEl.textContent = "Thinking...";
      await runVoiceTurn(
        text,
        "#voicechat-messages",
        history,
        !!(ttsToggle && ttsToggle.checked),
        voiceSelect && voiceSelect.value
      );
      if (statusEl) statusEl.textContent = continuous ? "Listening" : "Idle";
      if (continuous) await recorder.start();
    } catch (err) {
      addMessage("error", err.message || String(err), false, "#voicechat-messages");
      if (statusEl) statusEl.textContent = "Error";
      continuous = false;
      if (toggleBtn) toggleBtn.textContent = "Continuous";
    } finally {
      busy = false;
    }
  }

  if (pttBtn) {
    let touchActive = false;
    const down = async (e) => {
      if (e.type === "touchstart") touchActive = true;
      if (e.type === "mousedown" && touchActive) return;
      e.preventDefault();
      if (continuous || busy || recorder.isRecording()) return;
      try {
        await recorder.start();
        pttBtn.classList.add("recording");
        pttBtn.textContent = "Release to send";
      } catch (err) {
        if (statusEl) statusEl.textContent = err.message || "Mic permission denied";
      }
    };
    const up = async (e) => {
      if (e.type === "mousedown" && touchActive) return;
      if (e.type === "touchend" || e.type === "touchcancel") {
        setTimeout(() => {
          touchActive = false;
        }, 400);
      }
      e.preventDefault();
      if (continuous || !recorder.isRecording()) return;
      pttBtn.classList.remove("recording");
      pttBtn.textContent = "Hold to talk";
      await handleRecording();
    };
    pttBtn.addEventListener("mousedown", down);
    pttBtn.addEventListener("mouseup", up);
    pttBtn.addEventListener("mouseleave", () => {
      if (recorder.isRecording() && !continuous && !touchActive) up({ preventDefault() {}, type: "mouseleave" });
    });
    pttBtn.addEventListener("touchstart", down, { passive: false });
    pttBtn.addEventListener("touchend", up, { passive: false });
    pttBtn.addEventListener("touchcancel", up, { passive: false });
  }

  if (toggleBtn) {
    toggleBtn.addEventListener("click", async () => {
      if (busy) return;
      if (!continuous) {
        continuous = true;
        toggleBtn.textContent = "Stop";
        try {
          await recorder.start();
          const loop = async () => {
            while (continuous) {
              await new Promise((r) => setTimeout(r, 4000));
              if (!continuous) break;
              if (recorder.isRecording()) {
                await handleRecording();
              }
            }
          };
          loop();
        } catch (err) {
          continuous = false;
          toggleBtn.textContent = "Continuous";
          if (statusEl) statusEl.textContent = err.message || "Mic permission denied";
        }
      } else {
        continuous = false;
        toggleBtn.textContent = "Continuous";
        if (recorder.isRecording()) await recorder.stop();
        if (statusEl) statusEl.textContent = "Idle";
      }
    });
  }

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const prompt = $("#voicechat-prompt");
      const text = (prompt && prompt.value.trim()) || "";
      if (!text || busy) return;
      if (prompt) prompt.value = "";
      busy = true;
      if (statusEl) statusEl.textContent = "Thinking...";
      try {
        await runVoiceTurn(
          text,
          "#voicechat-messages",
          history,
          !!(ttsToggle && ttsToggle.checked),
          voiceSelect && voiceSelect.value
        );
        if (statusEl) statusEl.textContent = continuous ? "Listening" : "Idle";
      } finally {
        busy = false;
      }
    });
  }
}


const API_GEN_TEMPLATES = {
  "/text": { prompt: "Explain quantum computing in simple terms.", max_tokens: 120, temperature: 0.5 },
  "/chat": { message: "Hello, who made you?", history: [], use_rag: true, use_search: false },
  "/code": { task: "Write a Python function that merges two sorted lists.", language: "python" },
  "/image": { prompt: "photoreal portrait, studio lighting, sharp focus", num_steps: 28, enhance: true },
  "/video": { prompt: "slow camera pan across a city street at dusk", num_frames: 16, fps: 12 },
};

function initApiGen() {
  const endpointSel = $("#api-gen-endpoint");
  const bodyEl = $("#api-gen-body");
  const responseEl = $("#api-gen-response");
  const mediaEl = $("#api-gen-media");
  const keyEl = $("#api-gen-key");
  const savedKey = sessionStorage.getItem("navine_api_key") || "";
  if (keyEl && savedKey) keyEl.value = savedKey;

  function setTemplate(path) {
    if (!bodyEl) return;
    const tmpl = API_GEN_TEMPLATES[path] || {};
    bodyEl.value = JSON.stringify(tmpl, null, 2);
  }

  if (endpointSel) {
    setTemplate(endpointSel.value);
    endpointSel.addEventListener("change", () => setTemplate(endpointSel.value));
  }

  const saveKeyBtn = $("#api-gen-save-key");
  if (saveKeyBtn) {
    saveKeyBtn.addEventListener("click", () => {
      const key = keyEl ? keyEl.value.trim() : "";
      if (key) sessionStorage.setItem("navine_api_key", key);
      else sessionStorage.removeItem("navine_api_key");
    });
  }

  const createKeyBtn = $("#api-gen-create-key");
  if (createKeyBtn) {
    createKeyBtn.addEventListener("click", async () => {
      if (!getSessionToken()) {
        alert("Log in or sign up from the home screen to create API keys.");
        return;
      }
      createKeyBtn.disabled = true;
      try {
        const key = await createApiKeyFromUi();
        if (keyEl) keyEl.value = key;
      } catch (err) {
        alert(err.message || "Could not create API key");
      } finally {
        createKeyBtn.disabled = false;
      }
    });
  }

  const curlBtn = $("#api-gen-curl");
  if (curlBtn) {
    curlBtn.addEventListener("click", () => {
      const path = endpointSel ? endpointSel.value : "/chat";
      const body = bodyEl ? bodyEl.value : "{}";
      const key = keyEl ? keyEl.value.trim() : "";
      const origin = window.location.origin;
      let curl = `curl -X POST ${origin}${API}${path} \\\n  -H "Content-Type: application/json"`;
      if (key) curl += ` \\\n  -H "X-Navine-API-Key: ${key}"`;
      curl += ` \\\n  -d '${body.replace(/'/g, "'\\''")}'`;
      navigator.clipboard.writeText(curl).catch(() => {});
      if (responseEl) responseEl.textContent = "curl copied to clipboard";
    });
  }

  const sendBtn = $("#api-gen-send");
  if (sendBtn) {
    sendBtn.addEventListener("click", async () => {
      const path = endpointSel ? endpointSel.value : "/chat";
      let payload = {};
      try {
        payload = JSON.parse(bodyEl ? bodyEl.value : "{}");
      } catch {
        if (responseEl) responseEl.textContent = "Invalid JSON body";
        return;
      }
      if (mediaEl) mediaEl.innerHTML = "";
      if (responseEl) responseEl.textContent = "Sending...";
      sendBtn.disabled = true;
      try {
        const key = keyEl ? keyEl.value.trim() : "";
        const timeout = path === "/image" || path === "/video" ? 300000 : 120000;
        const data = await apiRequest("POST", path, payload, { apiKey: key, timeoutMs: timeout });
        if (responseEl) {
          const preview = { ...data };
          if (preview.data_base64 && preview.data_base64.length > 200) {
            preview.data_base64 = `${preview.data_base64.slice(0, 120)}...(truncated)`;
          }
          responseEl.textContent = JSON.stringify(preview, null, 2);
        }
        if (mediaEl && data.data_base64) {
          if (path === "/image") {
            const img = document.createElement("img");
            img.className = "api-gen-preview-img";
            img.src = `data:${data.mime || "image/png"};base64,${data.data_base64}`;
            img.alt = "Generated image";
            mediaEl.appendChild(img);
          } else if (path === "/video") {
            const vid = document.createElement("video");
            vid.controls = true;
            vid.className = "api-gen-preview-vid";
            vid.src = `data:${data.mime || "video/mp4"};base64,${data.data_base64}`;
            mediaEl.appendChild(vid);
          } else if (path === "/chat" || path === "/text") {
            const text = data.text || data.reply || "";
            if (text) {
              const pre = document.createElement("pre");
              pre.className = "api-gen-text-out";
              pre.textContent = text;
              mediaEl.appendChild(pre);
            }
          } else if (path === "/code") {
            const pre = document.createElement("pre");
            pre.className = "api-gen-text-out";
            pre.textContent = data.code || "";
            mediaEl.appendChild(pre);
          }
        }
      } catch (err) {
        if (responseEl) responseEl.textContent = err.message || "Request failed";
      } finally {
        sendBtn.disabled = false;
      }
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initSiteGate();
  initSidebar();
  initRouting();
  initTabs();
  initChatWelcome();
  initChatHistory();
  initComposer();
  initTextForm();
  initVoiceChat();
  initImageForm();
  initVideoForm();
  initDeepfakeForms();
  initOsintForm();
  initVoiceForms();
  initMusicForm();
  initTrainEverything();
  initTrainAdmin();
  initApiGen();
  initUserSettings();
  initCodeRunModal();
  loadStatus();
  if (gatePassed()) restoreAdminSession();
  statusPollTimer = setInterval(loadStatus, 15000);
  scheduleTrainPoll();
  loadTrainProgress().catch(() => {});
});

function renderOsintProfileScan(scan) {
  if (!scan || !scan.valid) {
    return scan && scan.summary
      ? `<div class="osint-scan-summary">Invalid username for scan.</div>`
      : "";
  }
  const summary = scan.summary || {};
  const found = scan.found || [];
  const unknown = scan.unknown || [];
  const foundCards = found
    .map(
      (row) =>
        `<a class="osint-profile found" href="${row.url}" target="_blank" rel="noopener">` +
        `<span class="osint-profile-site">${row.site}</span>` +
        `<span class="osint-profile-url">${row.url}</span>` +
        `<span class="osint-profile-badge">FOUND</span></a>`
    )
    .join("");
  const unknownCards = unknown
    .slice(0, 6)
    .map(
      (row) =>
        `<a class="osint-profile unknown" href="${row.url}" target="_blank" rel="noopener">` +
        `<span class="osint-profile-site">${row.site}</span>` +
        `<span class="osint-profile-url">${row.url}</span>` +
        `<span class="osint-profile-badge">CHECK</span></a>`
    )
    .join("");
  return (
    `<div class="osint-scan-summary">` +
    `Username scan: <strong>${summary.found || 0}</strong> found / ${summary.total || 0} sites` +
    `</div>` +
    (foundCards ? `<div class="osint-profile-grid">${foundCards}</div>` : "") +
    (unknownCards ? `<div class="osint-profile-grid muted">${unknownCards}</div>` : "")
  );
}

function renderOsintEmailIntel(data) {
  if (!data || !data.valid) return "";
  const blocks = [];
  const confirmed = data.confirmed_emails || [];
  const emails = data.emails || [];
  const candidates = data.candidate_emails || [];
  const websites = data.websites || [];
  if (confirmed.length) {
    blocks.push(`Confirmed (Gravatar): ${confirmed.join(", ")}`);
  }
  if (emails.length) {
    blocks.push(`Emails: ${emails.slice(0, 12).join(", ")}`);
  } else if (candidates.length) {
    blocks.push(`Email patterns to verify: ${candidates.slice(0, 8).join(", ")}`);
  }
  if (websites.length) {
    blocks.push(
      "Sites:\n" +
        websites
          .slice(0, 12)
          .map((u) => `- ${u}`)
          .join("\n")
    );
  }
  const breach = data.breach;
  if (breach && breach.checked) {
    blocks.push(
      breach.breached
        ? `Breaches: ${breach.breach_count || 0} (${(breach.breaches || []).join(", ")})`
        : "Breaches: none reported"
    );
  }
  if (!blocks.length) return "";
  return `<div class="osint-intel-card"><h4 class="panel-title">Emails / websites</h4><pre class="train-log">${blocks.join("\n\n")}</pre></div>`;
}

function renderOsintPasswordCheck(data) {
  if (!data || !data.checked) return "";
  const line = data.breached
    ? `Appeared in public dumps ~${data.seen_count || 0} times. Change it everywhere and enable MFA.`
    : "Not found in HIBP public dump range.";
  return `<div class="osint-intel-card"><h4 class="panel-title">Password breach check</h4><pre class="train-log">${line}\n${data.note || ""}</pre></div>`;
}

function renderOsintDomainIntel(data) {
  if (!data || !data.valid) return "";
  const rows = [];
  if (data.a && data.a.length) rows.push(`A: ${data.a.join(", ")}`);
  if (data.aaaa && data.aaaa.length) rows.push(`AAAA: ${data.aaaa.join(", ")}`);
  if (data.mx && data.mx.length) rows.push(`MX: ${data.mx.join(", ")}`);
  if (data.ns && data.ns.length) rows.push(`NS: ${data.ns.join(", ")}`);
  if (data.http_status) rows.push(`HTTPS ${data.http_status} -> ${data.final_url || data.domain}`);
  if (!rows.length) return "";
  return `<div class="osint-intel-card"><h4 class="panel-title">Domain DNS</h4><pre class="train-log">${rows.join("\n")}</pre></div>`;
}

function renderOsintIpIntel(data) {
  if (!data || !data.valid) return "";
  const lines = [
    `Location: ${data.city || "?"}, ${data.regionName || "?"}, ${data.country || "?"}`,
    `ISP: ${data.isp || "unknown"}`,
    `Org: ${data.org || "unknown"}`,
    `ASN: ${data.as || "unknown"}`,
  ];
  if (data.reverse) lines.push(`Reverse DNS: ${data.reverse}`);
  return `<div class="osint-intel-card"><h4 class="panel-title">IP intel</h4><pre class="train-log">${lines.join("\n")}</pre></div>`;
}

function initOsintForm() {
  const form = $("#osint-form");
  if (!form) return;
  const kindEl = $("#osint-kind");
  const targetEl = $("#osint-target");
  if (kindEl && targetEl) {
    kindEl.addEventListener("change", () => {
      if (kindEl.value === "password") {
        targetEl.type = "password";
        targetEl.placeholder = "password to check (not stored)";
      } else {
        targetEl.type = "text";
        targetEl.placeholder = "username, email, domain, IP, or password to check";
      }
    });
  }
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const target = ($("#osint-target") && $("#osint-target").value.trim()) || "";
    if (!target) return;
    const kind = ($("#osint-kind") && $("#osint-kind").value) || "auto";
    const useSearch = !($("#osint-use-search") && !$("#osint-use-search").checked);
    setLoading(form, true, "Investigating...");
    showProgress("osint-progress", true);
    const out = $("#osint-output");
    if (out) out.innerHTML = `<div class="msg-thinking"><span class="msg-thinking-label">Running OSINT scan</span><span class="msg-thinking-dots" aria-hidden="true"><span></span><span></span><span></span></span></div>`;
    try {
      const data = await apiPost("/osint/investigate", { target, kind, use_search: useSearch }, 240000);
      if (out) {
        const hits = (data.search_results || [])
          .map((row) => {
            const title = row.title || "Result";
            const url = row.url ? `<a href="${row.url}" target="_blank" rel="noopener">${row.url}</a>` : "";
            const snippet = row.snippet || row.body || "";
            return `<div class="osint-hit"><strong>${title}</strong><div class="output-path">${url}</div><p>${snippet}</p></div>`;
          })
          .join("");
        const scanHtml = renderOsintProfileScan(data.username_scan);
        const emailHtml = renderOsintEmailIntel(data.email_intel);
        const pwdHtml = renderOsintPasswordCheck(data.password_check);
        const domainHtml = renderOsintDomainIntel(data.domain_intel);
        const ipHtml = renderOsintIpIntel(data.ip_intel);
        out.innerHTML =
          `<div class="studio-card">` +
          (scanHtml || emailHtml || pwdHtml || domainHtml || ipHtml
            ? `<div class="osint-structured">${scanHtml}${emailHtml}${pwdHtml}${domainHtml}${ipHtml}</div>`
            : "") +
          `<h3 class="panel-title">Report</h3><pre class="train-log osint-report">${data.analysis || ""}</pre>` +
          (hits ? `<h3 class="panel-title">Web sources</h3>${hits}` : "") +
          (data.report_path ? `<div class="output-path">Saved: ${data.report_path}</div>` : "") +
          `</div>`;
      }
    } catch (err) {
      if (out) out.innerHTML = `<div class="message error">${err.message}</div>`;
    } finally {
      showProgress("osint-progress", false);
      setLoading(form, false);
    }
  });
}

function initDeepfakeForms() {
  bindImagePreview("#df-source-file", "#df-source-preview");
  bindImagePreview("#df-target-file", "#df-target-preview");
  const strength = $("#df-strength");
  const strengthOut = $("#df-strength-value");
  if (strength && strengthOut) {
    strength.addEventListener("input", () => {
      strengthOut.textContent = strength.value;
    });
  }
  const faceForm = $("#deepfake-face-form");
  if (faceForm) {
    const btn = faceForm.querySelector('button[type="submit"]');
    if (btn) btn.dataset.label = "Swap Face";
    faceForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const sourcePath = ($("#df-source") && $("#df-source").value.trim()) || "";
      const targetPath = ($("#df-target") && $("#df-target").value.trim()) || "";
      const sourceFile = $("#df-source-file");
      const targetFile = $("#df-target-file");
      const hasSourceFile = sourceFile && sourceFile.files && sourceFile.files[0];
      const hasTargetFile = targetFile && targetFile.files && targetFile.files[0];
      if ((!sourcePath && !hasSourceFile) || (!targetPath && !hasTargetFile)) {
        $("#deepfake-output").innerHTML = `<div class="message error">Add source and target images (upload or path).</div>`;
        return;
      }
      setLoading(faceForm, true, "Swapping...");
      showProgress("deepfake-progress", true);
      $("#deepfake-output").innerHTML = "";
      try {
        const body = {
          strength: parseFloat(strength?.value || "0.85"),
        };
        if (sourcePath) body.source = sourcePath;
        if (targetPath) body.target = targetPath;
        if (hasSourceFile) body.source_base64 = await readFileAsBase64(sourceFile.files[0]);
        if (hasTargetFile) body.target_base64 = await readFileAsBase64(targetFile.files[0]);
        const data = await apiPost("/deepfake/face", body);
        const img = document.createElement("img");
        img.src = `data:${data.mime};base64,${data.data_base64}`;
        $("#deepfake-output").appendChild(img);
        const path = document.createElement("div");
        path.className = "output-path";
        path.textContent = data.path;
        $("#deepfake-output").appendChild(path);
      } catch (err) {
        $("#deepfake-output").innerHTML = `<div class="message error">${err.message}</div>`;
      } finally {
        showProgress("deepfake-progress", false);
        setLoading(faceForm, false);
      }
    });
  }
  const vidForm = $("#deepfake-video-form");
  if (vidForm) {
    const btn = vidForm.querySelector('button[type="submit"]');
    if (btn) btn.dataset.label = "Run Deepfake Video";
    vidForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const source = $("#dfv-source").value.trim();
      if (!source) return;
      setLoading(vidForm, true, "Deepfake video...");
      showProgress("deepfake-progress", true);
      $("#deepfake-output").innerHTML = "";
      try {
        const body = { source };
        const tv = $("#dfv-video").value.trim();
        const pr = $("#dfv-prompt").value.trim();
        if (tv) body.target_video = tv;
        if (pr) body.prompt = pr;
        const data = await apiPost("/deepfake/video", body);
        const video = document.createElement("video");
        video.src = `data:${data.mime};base64,${data.data_base64}`;
        video.controls = true;
        video.autoplay = true;
        $("#deepfake-output").appendChild(video);
        const path = document.createElement("div");
        path.className = "output-path";
        path.textContent = data.path;
        $("#deepfake-output").appendChild(path);
      } catch (err) {
        $("#deepfake-output").innerHTML = `<div class="message error">${err.message}</div>`;
      } finally {
        showProgress("deepfake-progress", false);
        setLoading(vidForm, false);
      }
    });
  }
}

async function refreshVoiceList() {
  const el = $("#voice-list");
  const select = $("#voice-speak-name");
  const statusEl = $("#voice-status");
  try {
    const status = await apiGet("/voice/status");
    const voices = (status.voices || []).filter((v) => !isForeignBrand(v.name) && !isForeignBrand(v.display_name));
    let defaultVoice = status.default_voice;
    if (isForeignBrand(defaultVoice)) defaultVoice = (voices[0] && voices[0].name) || "navine_ai";
    if (statusEl) {
      const b = status.backends || {};
      statusEl.textContent =
        `Voice engines: Piper ${b.piper ? "ready" : "off"} · Coqui ${b.coqui ? "ready" : "off"} · System ${b.pyttsx3 ? "ready" : "off"}` +
        (defaultVoice ? ` · default ${defaultVoice}` : "");
    }
    if (el) {
      el.textContent = voices.length
        ? "Voices: " + voices.map((v) => v.name + (v.cloned ? " (clone)" : "")).join(", ")
        : "No voices configured.";
    }
    if (select) {
      const prev = select.value;
      select.innerHTML = "";
      const def = defaultVoice || (voices[0] && voices[0].name) || "navine_ai";
      voices.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v.name;
        opt.textContent = v.display_name || v.name;
        select.appendChild(opt);
      });
      if (!voices.length) {
        const opt = document.createElement("option");
        opt.value = "navine_ai";
        opt.textContent = "navine_ai";
        select.appendChild(opt);
      }
      select.value = prev && [...select.options].some((o) => o.value === prev) ? prev : def;
    }
  } catch (err) {
    if (statusEl) statusEl.textContent = "Voice status unavailable: " + (err.message || err);
    if (el) el.textContent = "";
  }
}

function initMusicForm() {
  const form = $("#music-form");
  if (!form) return;
  const btn = form.querySelector('button[type="submit"]');
  if (btn) btn.dataset.label = "Generate";
  document.querySelectorAll(".music-style-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const style = chip.getAttribute("data-style") || "";
      const promptEl = $("#music-prompt");
      if (!promptEl || !style) return;
      const current = String(promptEl.value || "").trim();
      if (!current) {
        promptEl.value = style;
      } else if (!new RegExp(`\\b${style}\\b`, "i").test(current)) {
        promptEl.value = `${current} ${style}`.trim();
      }
      promptEl.focus();
    });
  });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = (($("#music-prompt") && $("#music-prompt").value) || "").trim();
    if (!prompt) return;
    const durationRaw = $("#music-duration") ? Number($("#music-duration").value) : 12;
    const seedRaw = $("#music-seed") ? String($("#music-seed").value || "").trim() : "";
    const body = {
      prompt,
      duration: Number.isFinite(durationRaw) ? durationRaw : 12,
    };
    if (seedRaw !== "" && Number.isFinite(Number(seedRaw))) body.seed = Number(seedRaw);
    setLoading(form, true, "Composing...");
    try {
      const data = await apiPost("/music", body, 120000);
      const out = $("#music-output");
      out.innerHTML = "";
      if (data.data_base64) {
        const player = document.createElement("div");
        player.className = "music-player";
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.autoplay = true;
        audio.src = `data:${data.mime || "audio/wav"};base64,${data.data_base64}`;
        player.appendChild(audio);
        const actions = document.createElement("div");
        actions.className = "music-actions";
        const dl = document.createElement("a");
        dl.className = "btn";
        dl.href = audio.src;
        dl.download = "navine-music.wav";
        dl.textContent = "Download wav";
        actions.appendChild(dl);
        player.appendChild(actions);
        out.appendChild(player);
      } else {
        out.innerHTML = `<div class="message error">No audio returned</div>`;
      }
      const meta = data.meta || {};
      const path = document.createElement("div");
      path.className = "output-path";
      path.textContent = `${(meta.styles || []).join(", ") || "track"} · ${meta.bpm || "?"} bpm · ${meta.sample_rate || 22050} Hz · ${meta.duration || body.duration}s · seed ${meta.seed ?? "—"}`;
      out.appendChild(path);
      if (data.path) {
        const file = document.createElement("div");
        file.className = "output-path";
        file.textContent = data.path;
        out.appendChild(file);
      }
    } catch (err) {
      $("#music-output").innerHTML = `<div class="message error">${err.message}</div>`;
    } finally {
      setLoading(form, false);
    }
  });
}

function initVoiceForms() {
  refreshVoiceList();
  const cloneForm = $("#voice-clone-form");
  if (cloneForm) {
    const btn = cloneForm.querySelector('button[type="submit"]');
    if (btn) btn.dataset.label = "Clone Voice";
    cloneForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const name = $("#voice-name").value.trim();
      const sample_path = $("#voice-sample").value.trim();
      const fileInput = $("#voice-sample-file");
      if (!name) return;
      if (!sample_path && !(fileInput && fileInput.files && fileInput.files[0])) {
        $("#voice-output").innerHTML = `<div class="message error">Add a sample path or upload an audio file.</div>`;
        return;
      }
      setLoading(cloneForm, true, "Cloning...");
      try {
        const body = { name, language: "en" };
        if (sample_path) body.sample_path = sample_path;
        if (fileInput && fileInput.files && fileInput.files[0]) {
          body.sample_base64 = await readFileAsBase64(fileInput.files[0]);
        }
        const data = await apiPost("/voice/clone", body);
        $("#voice-output").innerHTML = `<div class="output-path">${data.message || "Voice saved"} (${data.backend || ""})</div>`;
        await refreshVoiceList();
        if ($("#voice-speak-name")) $("#voice-speak-name").value = data.name || name;
      } catch (err) {
        $("#voice-output").innerHTML = `<div class="message error">${err.message}</div>`;
      } finally {
        setLoading(cloneForm, false);
      }
    });
  }
  const speakForm = $("#voice-speak-form");
  if (speakForm) {
    const btn = speakForm.querySelector('button[type="submit"]');
    if (btn) btn.dataset.label = "Speak";
    speakForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = $("#voice-text").value.trim();
      const voice = ($("#voice-speak-name") && $("#voice-speak-name").value.trim()) || "";
      if (!text) return;
      setLoading(speakForm, true, "Speaking...");
      try {
        const data = await apiPost("/voice/speak", { text, voice: voice || null, play: false });
        const out = $("#voice-output");
        out.innerHTML = "";
        if (data.data_base64) {
          const audio = document.createElement("audio");
          audio.controls = true;
          audio.autoplay = true;
          audio.src = `data:${data.mime || "audio/wav"};base64,${data.data_base64}`;
          out.appendChild(audio);
        } else {
          out.innerHTML = `<div class="message error">No audio returned</div>`;
        }
        const path = document.createElement("div");
        path.className = "output-path";
        path.textContent = `${data.backend || ""} · ${data.voice || ""} · ${data.path || ""}`.trim();
        out.appendChild(path);
      } catch (err) {
        $("#voice-output").innerHTML = `<div class="message error">${err.message}</div>`;
      } finally {
        setLoading(speakForm, false);
      }
    });
  }
}
