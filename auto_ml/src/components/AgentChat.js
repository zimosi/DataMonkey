import React, { useState, useEffect, useRef } from 'react';
import './AgentChat.css';

function AgentChat({ jobId, agentId, agentName, isOpen, onClose }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const messagesEndRef = useRef(null);

  // Load conversation history when chat opens
  useEffect(() => {
    if (isOpen && jobId && agentId) {
      loadConversation();
    }
  }, [isOpen, jobId, agentId]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadConversation = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/agent/conversation/${jobId}/${agentId}`
      );
      const data = await response.json();
      setMessages(data.conversation || []);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || loading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setLoading(true);

    // Add user message to UI immediately
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);

    try {
      const response = await fetch('http://localhost:8000/api/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobId,
          agentId,
          message: userMessage,
        }),
      });

      const data = await response.json();

      if (data.success) {
        // Update with full conversation history from backend
        setMessages(data.conversation_history);
        setSuggestions(data.suggestions || []);
      } else {
        // Add error message
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: data.response || 'Sorry, I encountered an error. Please try again.',
          },
        ]);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I could not process your message. Please try again.',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setInputMessage(suggestion);
  };

  if (!isOpen) return null;

  return (
    <div className="agent-chat-overlay" onClick={onClose}>
      <div className="agent-chat-container" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="agent-chat-header">
          <div className="agent-chat-title">
            <div className="agent-avatar">{agentName.charAt(0)}</div>
            <div>
              <h3>{agentName}</h3>
              <p>Ask me about my decisions and results</p>
            </div>
          </div>
          <button className="chat-close-btn" onClick={onClose}>
            ×
          </button>
        </div>

        {/* Messages */}
        <div className="agent-chat-messages">
          {messages.length === 0 && (
            <div className="chat-empty-state">
              <div className="empty-icon">💬</div>
              <p>Start a conversation!</p>
              <p className="empty-subtitle">
                Ask me about preprocessing steps, decisions, or recommendations.
              </p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`chat-message ${msg.role === 'user' ? 'user' : 'assistant'}`}
            >
              <div className="message-content">
                {msg.role === 'assistant' && (
                  <div className="message-avatar">{agentName.charAt(0)}</div>
                )}
                <div className="message-bubble">{msg.content}</div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="chat-message assistant">
              <div className="message-content">
                <div className="message-avatar">{agentName.charAt(0)}</div>
                <div className="message-bubble typing">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Suggestions */}
        {suggestions.length > 0 && messages.length === 0 && (
          <div className="chat-suggestions">
            <p className="suggestions-label">Suggested questions:</p>
            {suggestions.map((suggestion, idx) => (
              <button
                key={idx}
                className="suggestion-chip"
                onClick={() => handleSuggestionClick(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}

        {/* Input */}
        <div className="agent-chat-input">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            rows={2}
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={!inputMessage.trim() || loading}>
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default AgentChat;
