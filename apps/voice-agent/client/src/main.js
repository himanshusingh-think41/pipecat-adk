import { App } from "./App.js";

const root = document.getElementById("app");

if (root) {
  root.innerHTML = App();
}

const healthStatus = document.getElementById("health-status");
const sessionIdNode = document.getElementById("session-id");
const errorBanner = document.getElementById("error-banner");
const messagesNode = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");

let sessionId = "";

function setBanner(message = "") {
  if (!errorBanner) return;
  errorBanner.textContent = message;
  errorBanner.classList.toggle("hidden", !message);
}

function setStatus(node, text, variant) {
  if (!node) return;
  node.textContent = text;
  node.className = `status-pill status-${variant}`;
}

function setSendEnabled(enabled) {
  if (!sendButton) return;
  sendButton.disabled = !enabled;
}

function appendMessage(role, text) {
  if (!messagesNode) return;
  const item = document.createElement("div");
  item.className = `message message-${role}`;
  const roleNode = document.createElement("span");
  roleNode.className = "message-role";
  roleNode.textContent = role === "user" ? "You" : "Agent";
  const textNode = document.createElement("p");
  textNode.textContent = text;
  item.append(roleNode, textNode);
  messagesNode.appendChild(item);
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    setStatus(healthStatus, "Ready", "ready");
  } catch (error) {
    setStatus(healthStatus, `Unavailable: ${error.message}`, "error");
    setBanner("Backend health check failed. Confirm the FastAPI server is still running.");
  }
}

async function createSession() {
    try {
    const response = await fetch("/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enableDefaultIceServers: true }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    sessionId = payload.sessionId || "";
    if (!sessionId) throw new Error("Session id missing from response");
    setStatus(sessionIdNode, sessionId, "ready");
    setSendEnabled(true);
  } catch (error) {
    setStatus(sessionIdNode, `Failed: ${error.message}`, "error");
    setBanner("Session creation failed. Reload the page after checking server logs.");
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const message = messageInput?.value.trim() || "";
  if (!message || !sessionId) return;

  setBanner("");
  appendMessage("user", message);
  messageInput.value = "";
  if (sendButton) {
    sendButton.disabled = true;
    sendButton.textContent = "Thinking...";
  }

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }

    appendMessage("assistant", payload.assistant_text);
  } catch (error) {
    setBanner(`Agent request failed: ${error.message}`);
    appendMessage("assistant", `Error: ${error.message}`);
  } finally {
    if (sendButton) {
      sendButton.disabled = false;
      sendButton.textContent = "Send";
    }
  }
}

void checkHealth();
void createSession();
chatForm?.addEventListener("submit", sendMessage);
