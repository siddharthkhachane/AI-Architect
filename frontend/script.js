class RAGAssistant {
    constructor() {
        this.apiBase = 'http://localhost:8000';
        this.conversationId = null;
        this.documents = [];
        
        this.initializeElements();
        this.setupEventListeners();
        this.checkStatus();
        this.loadDocuments();
        this.loadStats();
    }

    initializeElements() {
        this.elements = {
            status: document.getElementById('status'),
            uploadArea: document.getElementById('uploadArea'),
            fileInput: document.getElementById('fileInput'),
            documentsList: document.getElementById('documentsList'),
            chatMessages: document.getElementById('chatMessages'),
            messageInput: document.getElementById('messageInput'),
            sendButton: document.getElementById('sendButton'),
            clearChat: document.getElementById('clearChat'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            docCount: document.getElementById('docCount'),
            chunkCount: document.getElementById('chunkCount'),
            sourceCount: document.getElementById('sourceCount')
        };
    }

    setupEventListeners() {
        this.elements.uploadArea.addEventListener('click', () => {
            this.elements.fileInput.click();
        });

        this.elements.fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        this.elements.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.add('dragover');
        });

        this.elements.uploadArea.addEventListener('dragleave', () => {
            this.elements.uploadArea.classList.remove('dragover');
        });

        this.elements.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.remove('dragover');
            this.handleFileUpload(e.dataTransfer.files);
        });

        this.elements.sendButton.addEventListener('click', () => {
            this.sendMessage();
        });

        this.elements.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.elements.clearChat.addEventListener('click', () => {
            this.clearChat();
        });

        setInterval(() => {
            this.loadDocuments();
            this.loadStats();
        }, 5000);
    }

    async checkStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            this.updateStatus(data.status === 'healthy' && data.services.ollama === 'connected');
        } catch (error) {
            this.updateStatus(false);
        }
    }

    updateStatus(isConnected) {
        const status = this.elements.status;
        if (isConnected) {
            status.className = 'status connected';
            status.innerHTML = '<i class="fas fa-circle"></i> Connected';
        } else {
            status.className = 'status disconnected';
            status.innerHTML = '<i class="fas fa-circle"></i> Disconnected';
        }
    }

    async handleFileUpload(files) {
        for (const file of files) {
            await this.uploadFile(file);
        }
        this.loadDocuments();
        this.loadStats();
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            this.showLoading();
            const response = await fetch(`${this.apiBase}/api/v1/documents/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                this.showNotification(`✅ ${file.name} uploaded successfully!`, 'success');
            } else {
                this.showNotification(`❌ Failed to upload ${file.name}: ${data.detail}`, 'error');
            }
        } catch (error) {
            this.showNotification(`❌ Error uploading ${file.name}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async loadDocuments() {
        try {
            const response = await fetch(`${this.apiBase}/api/v1/documents/`);
            const documents = await response.json();
            this.documents = documents;
            this.renderDocuments(documents);
        } catch (error) {
            console.error('Error loading documents:', error);
        }
    }

    renderDocuments(documents) {
        const container = this.elements.documentsList;
        
        if (documents.length === 0) {
            container.innerHTML = '<p style="color: #999; text-align: center; padding: 1rem;">No documents uploaded yet</p>';
            return;
        }

        container.innerHTML = documents.map(doc => `
            <div class="document-item">
                <div class="document-info">
                    <div class="document-name">${doc.title}</div>
                    <div class="document-status ${doc.status}">${doc.status}</div>
                </div>
                <i class="fas fa-file-alt" style="color: #667eea;"></i>
            </div>
        `).join('');
    }

    async loadStats() {
        try {
            const response = await fetch(`${this.apiBase}/api/v1/documents/stats`);
            const stats = await response.json();
            
            this.elements.docCount.textContent = stats.total_documents;
            this.elements.chunkCount.textContent = stats.total_chunks;
            this.elements.sourceCount.textContent = stats.total_chunks;
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }

    async sendMessage() {
        const message = this.elements.messageInput.value.trim();
        if (!message) return;

        this.elements.messageInput.value = '';
        this.elements.sendButton.disabled = true;

        this.addMessage('user', message);

        try {
            const response = await fetch(`${this.apiBase}/api/v1/chat/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.conversationId,
                    include_sources: true,
                    max_sources: 3
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.conversationId = data.conversation_id;
                this.addMessage('bot', data.response, data.sources);
            } else {
                this.addMessage('bot', `Sorry, I encountered an error: ${data.detail}`);
            }
        } catch (error) {
            this.addMessage('bot', 'Sorry, I\'m having trouble connecting. Please check if the backend is running.');
        } finally {
            this.elements.sendButton.disabled = false;
        }
    }

    addMessage(role, content, sources = []) {
        const messagesContainer = this.elements.chatMessages;
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message';

        const avatar = role === 'user' ? '👤' : '🤖';
        const messageClass = role === 'user' ? 'user-message' : 'bot-message';

        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="sources">
                    <div class="sources-title">📚 Sources (${sources.length}):</div>
                    ${sources.map(source => `
                        <div class="source-item">
                            <div class="source-title">${source.document_title}</div>
                            <div class="source-text">${source.chunk_text.substring(0, 150)}...</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="${messageClass}">
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">
                    <p>${content}</p>
                    ${sourcesHtml}
                </div>
            </div>
        `;

        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage && role === 'user') {
            welcomeMessage.remove();
        }

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    clearChat() {
        this.conversationId = null;
        this.elements.chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="bot-message">
                    <div class="message-avatar">🤖</div>
                    <div class="message-content">
                        <p>👋 Chat cleared! Ask me anything about your documents.</p>
                    </div>
                </div>
            </div>
        `;
    }

    showLoading() {
        this.elements.loadingOverlay.classList.add('show');
    }

    hideLoading() {
        this.elements.loadingOverlay.classList.remove('show');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#10b981' : '#ef4444'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            font-weight: 500;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

function sendSuggestion(message) {
    const assistant = window.ragAssistant;
    assistant.elements.messageInput.value = message;
    assistant.sendMessage();
}

document.addEventListener('DOMContentLoaded', () => {
    window.ragAssistant = new RAGAssistant();
});