// Chatbot JavaScript
(function() {
    'use strict';

    // State
    let chatInitialized = false;
    let currentProvider = 'auto';
    let isWaitingForResponse = false;

    // DOM Elements
    const configCard = document.getElementById('config-card');
    const chatContainer = document.getElementById('chat-container');
    const samplesCard = document.getElementById('samples-card');
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const initBtn = document.getElementById('init-btn');
    const clearBtn = document.getElementById('clear-btn');
    const exportBtn = document.getElementById('export-btn');
    const providerSelect = document.getElementById('provider-select');
    const statusIndicator = document.getElementById('status-indicator');
    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');
    const providerInfo = document.getElementById('provider-info');
    const sampleQuestions = document.getElementById('sample-questions');

    // Utility Functions
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function formatTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function updateStatus(icon, text, alertClass) {
        statusIcon.textContent = icon;
        statusText.textContent = text;
        statusIndicator.className = `alert ${alertClass} mb-0`;
    }

    function setLoadingState(loading) {
        isWaitingForResponse = loading;
        sendBtn.disabled = loading;
        userInput.disabled = loading;
        
        if (loading) {
            sendBtn.querySelector('#send-icon').textContent = '⏳';
            sendBtn.querySelector('#send-text').textContent = window.TRANSLATIONS.thinking || 'Thinking...';
        } else {
            sendBtn.querySelector('#send-icon').textContent = '📤';
            sendBtn.querySelector('#send-text').textContent = window.TRANSLATIONS.send || 'Send';
        }
    }

    // Message Functions
    function addMessage(content, isUser, time = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
        
        const headerDiv = document.createElement('div');
        headerDiv.className = 'message-header';
        headerDiv.textContent = isUser ? (window.TRANSLATIONS.you || 'You') : (window.TRANSLATIONS.bot || 'AI Assistant');
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Format content (convert line breaks, etc.)
        const formattedContent = escapeHtml(content)
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        contentDiv.innerHTML = formattedContent;
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = time || formatTime();
        
        messageDiv.appendChild(headerDiv);
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
        
        return messageDiv;
    }

    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            typingDiv.appendChild(dot);
        }
        
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const typing = document.getElementById('typing-indicator');
        if (typing) {
            typing.remove();
        }
    }

    function addWelcomeMessage(message) {
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <h3>🌾 ${window.TRANSLATIONS.welcome || 'Welcome!'}</h3>
            <p>${escapeHtml(message)}</p>
        `;
        chatMessages.appendChild(welcomeDiv);
    }

    function addErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = `❌ ${message}`;
        chatMessages.appendChild(errorDiv);
        scrollToBottom();
    }

    // Sample Questions
    function loadSampleQuestions() {
        const lang = window.TRANSLATIONS.currentLang || 'en';
        const questions = window.SAMPLE_QUESTIONS[lang] || window.SAMPLE_QUESTIONS['en'];
        
        sampleQuestions.innerHTML = '';
        questions.forEach(question => {
            const btn = document.createElement('button');
            btn.className = 'sample-question';
            btn.textContent = question;
            btn.onclick = () => {
                userInput.value = question;
                userInput.focus();
            };
            sampleQuestions.appendChild(btn);
        });
    }

    // API Functions
    async function initializeChatbot() {
        try {
            updateStatus('⏳', window.TRANSLATIONS.initializing || 'Initializing...', 'alert-warning');
            
            const response = await fetch('/api/chatbot/init', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    provider: providerSelect.value
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                chatInitialized = true;
                currentProvider = data.provider;
                
                // Update UI
                configCard.style.display = 'none';
                chatContainer.style.display = 'block';
                samplesCard.style.display = 'block';
                
                // Update provider info
                providerInfo.textContent = `${window.TRANSLATIONS.connected || 'Connected'}: ${currentProvider}`;
                
                // Add welcome message
                chatMessages.innerHTML = '';
                addWelcomeMessage(data.welcome_message);
                
                // Load sample questions
                loadSampleQuestions();
                
                // Focus input
                userInput.focus();
                
                console.log('Chatbot initialized successfully');
            } else {
                throw new Error(data.message || 'Failed to initialize chatbot');
            }
        } catch (error) {
            console.error('Initialization error:', error);
            updateStatus('❌', window.TRANSLATIONS.error || 'Error', 'alert-danger');
            addErrorMessage(error.message);
        }
    }

    async function sendMessage(message) {
        if (!message.trim() || isWaitingForResponse) return;

        try {
            // Add user message to UI
            addMessage(message, true);
            
            // Clear input
            userInput.value = '';
            
            // Show typing indicator
            addTypingIndicator();
            setLoadingState(true);

            // Send to API
            const response = await fetch('/api/chatbot/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            // Remove typing indicator
            removeTypingIndicator();
            setLoadingState(false);

            if (data.status === 'success') {
                // Add bot response
                addMessage(data.response, false);
            } else {
                addErrorMessage(data.message || data.response || 'Failed to get response');
            }
        } catch (error) {
            console.error('Chat error:', error);
            removeTypingIndicator();
            setLoadingState(false);
            addErrorMessage(error.message);
        }
    }

    async function clearChat() {
        if (!confirm(window.TRANSLATIONS.confirm_clear || 'Are you sure you want to clear the chat history?')) {
            return;
        }

        try {
            const response = await fetch('/api/chatbot/clear', {
                method: 'POST'
            });

            const data = await response.json();

            if (data.status === 'success') {
                chatMessages.innerHTML = '';
                addWelcomeMessage(window.TRANSLATIONS.cleared || 'Chat history cleared. Start a new conversation!');
                console.log('Chat cleared');
            }
        } catch (error) {
            console.error('Clear error:', error);
            addErrorMessage(error.message);
        }
    }

    async function exportChat() {
        try {
            const response = await fetch('/api/chatbot/export');
            const data = await response.json();

            if (data.status === 'success') {
                // Create download link
                const blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `agrovision-chat-${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                console.log('Chat exported');
            }
        } catch (error) {
            console.error('Export error:', error);
            addErrorMessage(error.message);
        }
    }

    // Event Listeners
    initBtn.addEventListener('click', initializeChatbot);

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (message) {
            sendMessage(message);
        }
    });

    clearBtn.addEventListener('click', clearChat);
    exportBtn.addEventListener('click', exportChat);

    // Allow Enter to send, Shift+Enter for new line
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Initialize on load
    document.addEventListener('DOMContentLoaded', () => {
        console.log('Chatbot page loaded');
        updateStatus('⏳', window.TRANSLATIONS.ready || 'Ready to start', 'alert-info');
    });

    // Add to window for debugging
    window.chatbot = {
        initialized: () => chatInitialized,
        provider: () => currentProvider,
        send: sendMessage,
        clear: clearChat,
        export: exportChat
    };
})();
