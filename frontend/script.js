/**
 * script.js
 * ---------
 * GenAI RAG Assistant – Frontend interaction logic.
 * Manages session, sends messages to the FastAPI backend, renders responses.
 */

// ── Configuration ──────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000"; // Change to your deployed backend URL

// ── Session management ─────────────────────────────────────────────────────
function getSessionId() {
  let sid = localStorage.getItem("rag_session_id");
  if (!sid) {
    sid = "sess_" + Math.random().toString(36).slice(2, 11) + "_" + Date.now();
    localStorage.setItem("rag_session_id", sid);
  }
  return sid;
}

let sessionId = getSessionId();

// ── DOM references ─────────────────────────────────────────────────────────
const chatMessages    = document.getElementById("chatMessages");
const messageInput    = document.getElementById("messageInput");
const btnSend         = document.getElementById("btnSend");
const btnNewChat      = document.getElementById("btnNewChat");
const typingIndicator = document.getElementById("typingIndicator");

// ── Auto-resize textarea ────────────────────────────────────────────────────
messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
});

// ── Send on Enter (Shift+Enter = newline) ───────────────────────────────────
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

btnSend.addEventListener("click", handleSend);
btnNewChat.addEventListener("click", startNewChat);

// ── Suggestion chips ────────────────────────────────────────────────────────
function sendSuggestion(text) {
  messageInput.value = text;
  handleSend();
}

// ── New chat ────────────────────────────────────────────────────────────────
function startNewChat() {
  // Generate fresh session ID
  sessionId = "sess_" + Math.random().toString(36).slice(2, 11) + "_" + Date.now();
  localStorage.setItem("rag_session_id", sessionId);

  // Reset chat area to welcome card
  chatMessages.innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">👋</div>
      <h2>Hello! I'm your support assistant.</h2>
      <p>I can help you with questions about passwords, VPN, leave policy, expense claims, IT support, and more.</p>
      <div class="suggestion-chips">
        <button class="chip" onclick="sendSuggestion('How do I reset my password?')">🔑 Password reset</button>
        <button class="chip" onclick="sendSuggestion('How do I set up VPN access?')">🌐 VPN setup</button>
        <button class="chip" onclick="sendSuggestion('What is the leave policy?')">📅 Leave policy</button>
        <button class="chip" onclick="sendSuggestion('How do I claim expenses?')">💰 Expense claims</button>
      </div>
    </div>`;
}

// ── Main send handler ───────────────────────────────────────────────────────
async function handleSend() {
  const message = messageInput.value.trim();
  if (!message || btnSend.disabled) return;

  // Clear welcome card on first message
  const welcomeCard = chatMessages.querySelector(".welcome-card");
  if (welcomeCard) welcomeCard.remove();

  // Render user bubble
  appendMessage("user", message);

  // Reset input
  messageInput.value = "";
  messageInput.style.height = "auto";

  // Show typing
  setLoading(true);

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, message }),
    });

    const data = await response.json();

    if (!response.ok) {
      const errMsg = data.error || data.detail || "An error occurred.";
      appendMessage("bot", errMsg, null, null, true);
      return;
    }

    appendMessage("bot", data.reply, data.tokensUsed, data.retrievedChunks);
  } catch (err) {
    console.error("Network error:", err);
    appendMessage(
      "bot",
      "⚠️ Could not reach the server. Please ensure the backend is running.",
      null,
      null,
      true
    );
  } finally {
    setLoading(false);
  }
}

// ── Render a message bubble ─────────────────────────────────────────────────
function appendMessage(role, text, tokensUsed = null, retrievedChunks = null, isError = false) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatarEmoji = role === "user" ? "👤" : "🤖";
  const bubbleClass = isError ? "bubble error" : "bubble";

  // Build meta line for bot messages
  let metaHtml = "";
  if (role === "bot" && !isError) {
    const parts = [];
    if (retrievedChunks !== null && retrievedChunks !== undefined) {
      parts.push(`<span class="chunk-badge">📄 ${retrievedChunks} chunk${retrievedChunks !== 1 ? "s" : ""} retrieved</span>`);
    }
    if (tokensUsed) {
      parts.push(`<span>${tokensUsed} tokens</span>`);
    }
    if (parts.length) {
      metaHtml = `<div class="meta">${parts.join("")}</div>`;
    }
  }

  row.innerHTML = `
    <div class="avatar ${role}">${avatarEmoji}</div>
    <div>
      <div class="${bubbleClass}">${escapeHtml(text)}</div>
      ${metaHtml}
    </div>`;

  chatMessages.appendChild(row);
  scrollToBottom();
}

// ── Loading state ───────────────────────────────────────────────────────────
function setLoading(loading) {
  btnSend.disabled = loading;
  messageInput.disabled = loading;
  typingIndicator.classList.toggle("hidden", !loading);
  if (!loading) scrollToBottom();
}

// ── Helpers ─────────────────────────────────────────────────────────────────
function scrollToBottom() {
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/\n/g, "<br>");
}
