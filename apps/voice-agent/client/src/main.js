import "./styles/globals.css";
import { App } from "./App.js";

const root = document.getElementById("app");

if (root) {
  root.innerHTML = App();
}

const healthStatus = document.getElementById("health-status");
const sessionIdNode = document.getElementById("session-id");
const messagesNode = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");

let sessionId = "";

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
    if (healthStatus) healthStatus.textContent = "Ready";
  } catch (error) {
    if (healthStatus) healthStatus.textContent = `Unavailable: ${error.message}`;
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
    if (sessionIdNode) {
      sessionIdNode.textContent = sessionId || "Session not created";
    }
  } catch (error) {
    if (sessionIdNode) sessionIdNode.textContent = `Failed: ${error.message}`;
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const message = messageInput?.value.trim() || "";
  if (!message || !sessionId) return;

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
