const API_BASE = 'http://localhost:8000';

let isConnected = false;
let conversationHistory = [];

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkConnection();
    loadStats();
});

function initializeApp() {
    updateStatus('⚡ Connecting...');
}

function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearChat');
    const uploadRepoBtn = document.getElementById('uploadRepo');
    const suggestions = document.querySelectorAll('.suggestion');

    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    fileInput.addEventListener('change', handleFileSelect);
    uploadRepoBtn.addEventListener('click', handleRepoUpload);

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);
    clearBtn.addEventListener('click', clearChat);

    suggestions.forEach(suggestion => {
        suggestion.addEventListener('click', () => {
            chatInput.value = suggestion.textContent;
            sendMessage();
        });
    });
}

async function checkConnection() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            isConnected = true;
            updateStatus('🟢 Connected');
        } else {
            throw new Error('Connection failed');
        }
    } catch (error) {
        isConnected = false;
        updateStatus('🔴 Disconnected');
        console.error('Connection error:', error);
    }
}

function updateStatus(status) {
    document.getElementById('status').textContent = status;
}

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/v1/documents/collections`);
        const data = await response.json();
        
        if (data.collections && data.collections.length > 0) {
            const collection = data.collections[0];
            document.getElementById('docCount').textContent = collection.count || 0;
            document.getElementById('chunkCount').textContent = collection.count || 0;
            document.getElementById('sourceCount').textContent = `${collection.count || 0} sources available`;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    uploadFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    uploadFiles(files);
}

async function uploadFiles(files) {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    try {
        updateStatus('📤 Uploading...');
        const response = await fetch(`${API_BASE}/api/v1/documents/upload`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            updateStatus('✅ Upload Complete');
            loadStats();
            addBotMessage(`Successfully uploaded ${files.length} file(s)!`);
        } else {
            throw new Error('Upload failed');
        }
    } catch (error) {
        updateStatus('❌ Upload Failed');
        addBotMessage('Sorry, upload failed. Please try again.');
        console.error('Upload error:', error);
    }
}

async function handleRepoUpload() {
    const repoUrl = document.getElementById('repoUrl').value.trim();
    const branch = document.getElementById('repoBranch').value.trim() || 'main';

    if (!repoUrl) {
        addBotMessage('Please enter a repository URL.');
        return;
    }

    try {
        updateStatus('📤 Cloning repository...');
        document.getElementById('uploadRepo').disabled = true;
        document.getElementById('uploadRepo').textContent = 'Uploading...';

        const response = await fetch(`${API_BASE}/api/v1/documents/upload`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                repo_url: repoUrl,
                branch: branch
            })
        });

        const result = await response.json();

        if (response.ok) {
            updateStatus('✅ Repository Uploaded');
            loadStats();
            document.getElementById('repoUrl').value = '';
            document.getElementById('repoBranch').value = 'main';
            addBotMessage(`Successfully processed repository! Added ${result.documents_added} documents.`);
        } else {
            throw new Error(result.error || 'Repository upload failed');
        }
    } catch (error) {
        updateStatus('❌ Upload Failed');
        addBotMessage(`Repository upload failed: ${error.message}`);
        console.error('Repository upload error:', error);
    } finally {
        document.getElementById('uploadRepo').disabled = false;
        document.getElementById('uploadRepo').textContent = 'Upload Repo';
    }
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message || !isConnected) return;

    input.value = '';
    document.getElementById('sendBtn').disabled = true;

    addUserMessage(message);

    try {
        const response = await fetch(`${API_BASE}/api/v1/chat/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                include_sources: true
            })
        });

        const data = await response.json();

        if (response.ok) {
            addBotMessage(data.response, data.sources, data.processing_time, data.model_used);
        } else {
            throw new Error('Chat request failed');
        }
    } catch (error) {
        addBotMessage('Sorry, I encountered an error. Please try again.');
        console.error('Chat error:', error);
    } finally {
        document.getElementById('sendBtn').disabled = false;
    }
}

function addUserMessage(message) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.innerHTML = `
        <div class="message-avatar">👤</div>
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addBotMessage(message, sources = [], processingTime = 0, model = '') {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="sources">
                <strong>📚 Sources (${sources.length}):</strong>
                <div style="margin-top: 5px; font-size: 0.8em; max-height: 100px; overflow-y: auto;">
                    ${sources.map(source => `<div style="margin: 2px 0; padding: 2px; background: rgba(255,255,255,0.5); border-radius: 3px;">${escapeHtml(source.substring(0, 150))}${source.length > 150 ? '...' : ''}</div>`).join('')}
                </div>
            </div>
        `;
    }

    let metaHtml = '';
    if (processingTime > 0) {
        metaHtml = `
            <div class="message-meta">
                <span>⚡ ${processingTime.toFixed(2)}s</span>
                ${model ? `<span>🤖 ${model}</span>` : ''}
                ${sources.length > 0 ? `<span>📚 ${sources.length} sources</span>` : ''}
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
            ${sourcesHtml}
            ${metaHtml}
        </div>
    `;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function clearChat() {
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = `
        <div class="message bot-message">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <p>👋 Hi! I'm your RAG Assistant. Upload some documents and ask me questions about them!</p>
                <div class="suggestions">
                    <span class="suggestion">What is machine learning?</span>
                    <span class="suggestion">Summarize the key concepts</span>
                    <span class="suggestion">What are the main topics?</span>
                </div>
            </div>
        </div>
    `;
    
    const suggestions = document.querySelectorAll('.suggestion');
    suggestions.forEach(suggestion => {
        suggestion.addEventListener('click', () => {
            document.getElementById('chatInput').value = suggestion.textContent;
            sendMessage();
        });
    });
    
    conversationHistory = [];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

setInterval(checkConnection, 30000);
setInterval(loadStats, 10000);
