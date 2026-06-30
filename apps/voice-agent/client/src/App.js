export function App() {
  return `
    <main class="app-shell">
      <section class="hero">
        <div>
          <p class="eyebrow">Single Agent Dashboard</p>
          <h1>Voice Agent</h1>
          <p class="lede">
            Ask questions in chat right now, or open the live voice console for
            microphone-based interaction.
          </p>
        </div>
        <a class="voice-link" href="/voice-console/" target="_blank" rel="noreferrer">
          Open Voice Console
        </a>
      </section>

      <section class="status-card">
        <div>
          <h2>Backend Status</h2>
          <p id="health-status" class="status-pill status-pending">Checking backend health...</p>
        </div>
        <div>
          <h2>Session</h2>
          <p id="session-id" class="status-pill status-pending">Creating session...</p>
        </div>
      </section>

      <div id="error-banner" class="error-banner hidden" role="alert"></div>

      <section class="chat-layout">
        <div class="chat-panel">
          <div id="messages" class="messages"></div>
          <form id="chat-form" class="chat-form">
            <textarea
              id="message-input"
              rows="3"
              placeholder="Ask the voice agent a question..."
            ></textarea>
            <div class="chat-actions">
              <span class="muted">Text chat is already connected to the same agent runtime.</span>
              <button id="send-button" type="submit" disabled>Send</button>
            </div>
          </form>
        </div>

        <aside class="info-panel">
          <h3>How To Use</h3>
          <ol>
            <li>Wait until backend health shows as ready.</li>
            <li>Type a question to test the agent through Gemini.</li>
            <li>Use "Open Voice Console" for live microphone interaction.</li>
          </ol>
          <p class="note">
            The voice console reuses the working Pipecat WebRTC UI so we can
            ship a reliable single-agent experience quickly.
          </p>
        </aside>
      </section>
    </main>
  `;
}
