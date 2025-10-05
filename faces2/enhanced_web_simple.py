#!/usr/bin/env python3
"""
Enhanced Web-based Face Processing GUI (No numpy required)
Complete system with batch download, vector database simulation, and search functionality
"""

import os
import sys
import json
import time
import hashlib
import threading
import random
import math
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socket

# Check dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class SimpleVectorDB:
    """Simple vector database simulation (no ChromaDB required)"""
    def __init__(self, config):
        self.config = config
        self.db_file = os.path.join(config.db_path, "simple_vectors.json")
        self.embeddings = []
        self.initialized = False

    def initialize(self):
        """Initialize the simple vector database"""
        try:
            os.makedirs(self.config.db_path, exist_ok=True)
            self.load_embeddings()
            self.initialized = True
            return True
        except Exception as e:
            print(f"Vector database initialization failed: {e}")
            return False

    def load_embeddings(self):
        """Load embeddings from file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.embeddings = json.load(f)
            except:
                self.embeddings = []
        else:
            self.embeddings = []

    def save_embeddings(self):
        """Save embeddings to file"""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.embeddings, f, indent=2)
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def export_metadata(self, export_path=None):
        """Export metadata to JSON file"""
        if export_path is None:
            export_path = os.path.join(self.config.db_path, "face_metadata.json")

        try:
            metadata = []
            for entry in self.embeddings:
                metadata.append({
                    'id': entry['id'],
                    'file_path': entry['file_path'],
                    'timestamp': entry['timestamp'],
                    'file_size': entry['file_size'],
                    'file_mtime': entry['file_mtime']
                })

            with open(export_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True, export_path
        except Exception as e:
            return False, f"Error exporting metadata: {e}"

    def create_embedding(self, image_path):
        """Create a simple embedding for an image"""
        try:
            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    # Resize for consistency
                    img = img.resize((32, 32))

                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Create embedding from pixel statistics
                    pixels = list(img.getdata())

                    # Statistical features
                    features = []

                    # Color channel statistics
                    for channel in range(3):  # RGB
                        channel_values = [p[channel] for p in pixels]
                        features.extend([
                            sum(channel_values) / len(channel_values),  # mean
                            max(channel_values) - min(channel_values),  # range
                            len([v for v in channel_values if v > 128]) / len(channel_values)  # bright ratio
                        ])

                    # Histogram features (simplified)
                    for channel in range(3):
                        channel_values = [p[channel] for p in pixels]
                        # Create simple histogram
                        hist = [0] * 8
                        for val in channel_values:
                            hist[val // 32] += 1
                        # Normalize
                        total = sum(hist)
                        if total > 0:
                            hist = [h / total for h in hist]
                        features.extend(hist)

                    # Ensure 64 dimensions
                    while len(features) < 64:
                        features.append(0.0)
                    features = features[:64]

                    # Normalize vector
                    magnitude = math.sqrt(sum(f * f for f in features))
                    if magnitude > 0:
                        features = [f / magnitude for f in features]

                    return features
            else:
                # Fallback: random embedding based on file properties
                file_stat = os.stat(image_path)
                random.seed(file_stat.st_size + int(file_stat.st_mtime))
                return [random.random() for _ in range(64)]

        except Exception as e:
            print(f"Error creating embedding: {e}")
            return [random.random() for _ in range(64)]

    def add_face(self, file_path, face_id):
        """Add face to vector database"""
        if not self.initialized:
            return False

        try:
            embedding = self.create_embedding(file_path)
            file_stat = os.stat(file_path)

            entry = {
                'id': face_id,
                'file_path': file_path,
                'timestamp': datetime.now().isoformat(),
                'file_size': file_stat.st_size,
                'file_mtime': file_stat.st_mtime,
                'embedding': embedding
            }

            self.embeddings.append(entry)
            self.save_embeddings()
            return True

        except Exception as e:
            print(f"Error adding face to database: {e}")
            return False

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)

    def search_similar(self, query_path, n_results=5):
        """Search for similar faces"""
        if not self.initialized or not self.embeddings:
            return []

        try:
            query_embedding = self.create_embedding(query_path)
            similarities = []

            for entry in self.embeddings:
                similarity = self.cosine_similarity(query_embedding, entry['embedding'])
                distance = 1 - similarity  # Convert to distance

                similarities.append({
                    'id': entry['id'],
                    'distance': distance,
                    'metadata': {
                        'file_path': entry['file_path'],
                        'file_size': entry['file_size'],
                        'timestamp': entry['timestamp']
                    }
                })

            # Sort by distance (lower = more similar)
            similarities.sort(key=lambda x: x['distance'])
            return similarities[:n_results]

        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def get_stats(self):
        """Get database statistics"""
        return {
            'initialized': self.initialized,
            'available': True,  # Our simple version is always available
            'embeddings_count': len(self.embeddings),
            'collection_name': self.config.collection_name,
            'db_path': self.config.db_path,
            'embedding_dimensions': 64
        }

class EnhancedConfig:
    """Enhanced configuration"""
    def __init__(self):
        self.faces_dir = "./faces"
        self.db_path = "./simple_db"
        self.collection_name = "faces"
        self.download_delay = 1.0
        self.port = 8080
        self.config_file = "enhanced_simple_config.json"

    def save(self):
        data = {
            'faces_dir': self.faces_dir,
            'db_path': self.db_path,
            'collection_name': self.collection_name,
            'download_delay': self.download_delay,
            'port': self.port
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

class EnhancedDownloader:
    """Enhanced downloader with batch support"""
    def __init__(self, config, vector_db):
        self.config = config
        self.vector_db = vector_db
        self.running = False
        self.downloaded_count = 0
        self.error_count = 0
        self.status = "Ready"
        self.last_download = None
        self.batch_progress = {"current": 0, "total": 0, "running": False}
        self.download_history = []

        os.makedirs(self.config.faces_dir, exist_ok=True)

    def download_single_face(self):
        """Download a single face"""
        if not REQUESTS_AVAILABLE:
            return False, "requests module not available"

        try:
            self.status = "Downloading..."
            response = requests.get(
                "https://thispersondoesnotexist.com/",
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30
            )
            response.raise_for_status()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_hash = hashlib.md5(response.content).hexdigest()[:8]
            filename = f"face_{timestamp}_{image_hash}.jpg"
            file_path = os.path.join(self.config.faces_dir, filename)

            with open(file_path, 'wb') as f:
                f.write(response.content)

            # Add to vector database
            face_id = f"face_{timestamp}_{image_hash}"
            embedding_success = self.vector_db.add_face(file_path, face_id)

            self.downloaded_count += 1
            self.last_download = filename
            self.status = "Ready"

            # Add to history
            download_info = {
                'filename': filename,
                'size': len(response.content),
                'timestamp': time.time(),
                'embedding_added': embedding_success
            }
            self.download_history.append(download_info)
            if len(self.download_history) > 20:  # Keep only last 20
                self.download_history = self.download_history[-20:]

            return True, f"Downloaded: {filename} (Embedding: {'✅' if embedding_success else '❌'})"

        except Exception as e:
            self.error_count += 1
            self.status = "Error"
            return False, f"Error: {str(e)}"

    def download_batch(self, count):
        """Download multiple faces"""
        self.batch_progress = {"current": 0, "total": count, "running": True}
        self.status = f"Batch downloading {count} faces..."

        successful = 0
        failed = 0

        for i in range(count):
            if not self.batch_progress["running"]:
                break

            self.batch_progress["current"] = i + 1
            success, message = self.download_single_face()

            if success:
                successful += 1
            else:
                failed += 1

            if i < count - 1:  # Don't delay after last download
                time.sleep(self.config.download_delay)

        self.batch_progress["running"] = False
        self.status = "Ready"

        return {
            "successful": successful,
            "failed": failed,
            "total": count
        }

    def stop_batch(self):
        """Stop batch download"""
        self.batch_progress["running"] = False
        self.status = "Stopped"

    def get_stats(self):
        """Get download statistics"""
        face_files = list(Path(self.config.faces_dir).glob("*.jpg"))
        total_size = sum(f.stat().st_size for f in face_files if f.exists())

        return {
            'downloaded_count': self.downloaded_count,
            'error_count': self.error_count,
            'total_files': len(face_files),
            'total_size_mb': total_size / (1024 * 1024),
            'status': self.status,
            'last_download': self.last_download,
            'batch_progress': self.batch_progress,
            'download_history': self.download_history[-10:]  # Last 10 downloads
        }

# Import the web handler from the previous version but modify for simple vector DB
class EnhancedWebHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP request handler"""

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query = parse_qs(parsed_path.query)

        if path == '/':
            self.serve_main_page()
        elif path == '/api/status':
            self.serve_status()
        elif path == '/api/download':
            self.serve_download(query)
        elif path == '/api/batch_download':
            self.serve_batch_download(query)
        elif path == '/api/stop_batch':
            self.serve_stop_batch()
        elif path == '/api/files':
            self.serve_files()
        elif path == '/api/vector_stats':
            self.serve_vector_stats()
        elif path == '/api/search':
            self.serve_search(query)
        elif path == '/api/process_all':
            self.serve_process_all()
        elif path == '/api/export_metadata':
            self.serve_export_metadata()
        elif path.startswith('/api/image/'):
            self.serve_image(path)
        elif path == '/style.css':
            self.serve_css()
        else:
            self.send_error(404)

    def serve_main_page(self):
        """Serve the enhanced main HTML page"""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Face Processing System</title>
    <link rel="stylesheet" href="/style.css">
    <script>
        let autoRefresh = false;
        let refreshInterval;

        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update basic stats
                    document.getElementById('downloaded-count').textContent = data.downloaded_count;
                    document.getElementById('error-count').textContent = data.error_count;
                    document.getElementById('total-files').textContent = data.total_files;
                    document.getElementById('total-size').textContent = data.total_size_mb.toFixed(2);
                    document.getElementById('status').textContent = data.status;
                    document.getElementById('last-download').textContent = data.last_download || 'None';

                    // Update batch progress
                    const progress = data.batch_progress;
                    if (progress.running) {
                        document.getElementById('batch-progress').style.display = 'block';
                        document.getElementById('batch-current').textContent = progress.current;
                        document.getElementById('batch-total').textContent = progress.total;
                        const percentage = (progress.current / progress.total * 100).toFixed(1);
                        document.getElementById('batch-percentage').textContent = percentage;

                        const progressBar = document.getElementById('progress-bar');
                        progressBar.style.width = percentage + '%';
                    } else {
                        document.getElementById('batch-progress').style.display = 'none';
                    }

                    // Update download history
                    updateDownloadHistory(data.download_history);
                })
                .catch(err => console.error('Error updating status:', err));
        }

        function updateVectorStats() {
            fetch('/api/vector_stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('vector-available').textContent = data.available ? '✅ Yes' : '❌ No';
                    document.getElementById('vector-initialized').textContent = data.initialized ? '✅ Yes' : '❌ No';
                    document.getElementById('embeddings-count').textContent = data.embeddings_count;
                    document.getElementById('collection-name').textContent = data.collection_name;
                    document.getElementById('db-path').textContent = data.db_path;
                    document.getElementById('embedding-dims').textContent = data.embedding_dimensions || 'N/A';
                })
                .catch(err => console.error('Error updating vector stats:', err));
        }

        function updateDownloadHistory(history) {
            const container = document.getElementById('download-history');
            container.innerHTML = '';

            history.forEach(item => {
                const div = document.createElement('div');
                div.className = 'history-item';
                div.innerHTML = `
                    <strong>${item.filename}</strong><br>
                    Size: ${(item.size / 1024).toFixed(1)} KB |
                    Embedding: ${item.embedding_added ? '✅' : '❌'} |
                    ${new Date(item.timestamp * 1000).toLocaleTimeString()}
                `;
                container.appendChild(div);
            });
        }

        function downloadFace() {
            const btn = document.getElementById('download-btn');
            btn.disabled = true;
            btn.textContent = 'Downloading...';

            fetch('/api/download')
                .then(response => response.json())
                .then(data => {
                    logMessage(data.message, data.success ? 'success' : 'error');
                    updateStatus();
                    updateFilesList();
                    updateVectorStats();

                    // Show downloaded image preview
                    if (data.success) {
                        showLatestDownload();
                    }
                })
                .catch(err => {
                    logMessage(`Error: ${err.message}`, 'error');
                })
                .finally(() => {
                    btn.disabled = false;
                    btn.textContent = 'Download Single Face';
                });
        }

        function showLatestDownload() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.last_download) {
                        const preview = document.getElementById('download-preview');
                        preview.innerHTML = `
                            <h4>📥 Latest Download</h4>
                            <img src="/api/image/${data.last_download}" class="preview-image" alt="Latest Download">
                            <p class="image-label">${data.last_download}</p>
                        `;
                        preview.style.display = 'block';
                    }
                })
                .catch(err => console.error('Error showing preview:', err));
        }

        function downloadBatch() {
            const count = parseInt(document.getElementById('batch-count').value);
            if (!count || count < 1 || count > 50) {
                alert('Please enter a valid number between 1 and 50');
                return;
            }

            const btn = document.getElementById('batch-download-btn');
            btn.disabled = true;
            btn.textContent = 'Downloading...';

            fetch(`/api/batch_download?count=${count}`)
                .then(response => response.json())
                .then(data => {
                    logMessage(`Batch download completed: ${data.successful} successful, ${data.failed} failed`, 'success');
                    updateStatus();
                    updateFilesList();
                    updateVectorStats();
                })
                .catch(err => {
                    logMessage(`Batch download error: ${err.message}`, 'error');
                })
                .finally(() => {
                    btn.disabled = false;
                    btn.textContent = 'Start Batch Download';
                });
        }

        function stopBatch() {
            fetch('/api/stop_batch')
                .then(response => response.json())
                .then(data => {
                    logMessage('Batch download stopped', 'info');
                    updateStatus();
                })
                .catch(err => console.error('Error stopping batch:', err));
        }

        function processAllFaces() {
            const btn = document.getElementById('process-btn');
            btn.disabled = true;
            btn.textContent = 'Processing...';

            const preview = document.getElementById('embedding-preview');
            preview.innerHTML = '<h4>⚙️ Processing Embeddings...</h4>';
            preview.style.display = 'block';

            fetch('/api/process_all')
                .then(response => response.json())
                .then(data => {
                    logMessage(`Processing completed: ${data.processed} faces processed`, 'success');
                    updateVectorStats();

                    preview.innerHTML = `
                        <h4>✅ Embedding Complete</h4>
                        <p>${data.processed} of ${data.total} faces embedded successfully</p>
                    `;
                })
                .catch(err => {
                    logMessage(`Processing error: ${err.message}`, 'error');
                    preview.style.display = 'none';
                })
                .finally(() => {
                    btn.disabled = false;
                    btn.textContent = 'Process All Faces';
                });
        }

        function exportMetadata() {
            const btn = document.getElementById('export-btn');
            btn.disabled = true;
            btn.textContent = 'Exporting...';

            fetch('/api/export_metadata')
                .then(response => response.json())
                .then(data => {
                    logMessage(data.message, data.success ? 'success' : 'error');
                })
                .catch(err => {
                    logMessage(`Export error: ${err.message}`, 'error');
                })
                .finally(() => {
                    btn.disabled = false;
                    btn.textContent = 'Export Metadata to JSON';
                });
        }

        function searchSimilar() {
            const numResults = document.getElementById('search-count').value;

            const btn = document.getElementById('search-btn');
            btn.disabled = true;
            btn.textContent = 'Searching...';

            fetch(`/api/search?count=${numResults}`)
                .then(response => response.json())
                .then(data => {
                    displaySearchResults(data.results, data.query_image);
                })
                .catch(err => {
                    logMessage(`Search error: ${err.message}`, 'error');
                })
                .finally(() => {
                    btn.disabled = false;
                    btn.textContent = 'Search Similar Faces';
                });
        }

        function displaySearchResults(results, queryImage) {
            const container = document.getElementById('search-results');
            container.innerHTML = '<h3>🔍 Search Results</h3>';

            if (results.length === 0) {
                container.innerHTML += '<p>No results found. Add faces to the database first.</p>';
                return;
            }

            // Show query image
            if (queryImage) {
                const queryDiv = document.createElement('div');
                queryDiv.className = 'query-image-container';
                queryDiv.innerHTML = `
                    <h4>Query Image (Random Selection)</h4>
                    <img src="/api/image/${queryImage}" class="result-image" alt="Query Image">
                    <p class="image-label">${queryImage}</p>
                `;
                container.appendChild(queryDiv);
            }

            // Show results
            const resultsTitle = document.createElement('h4');
            resultsTitle.textContent = 'Similar Faces';
            resultsTitle.style.marginTop = '20px';
            container.appendChild(resultsTitle);

            results.forEach((result, index) => {
                const filename = result.metadata.file_path.split('/').pop();
                const div = document.createElement('div');
                div.className = 'search-result';
                div.innerHTML = `
                    <img src="/api/image/${filename}" class="result-image" alt="Result ${index + 1}">
                    <div class="result-info">
                        <strong>Result ${index + 1}</strong><br>
                        File: ${filename}<br>
                        Similarity: ${(1 - result.distance).toFixed(3)} (Distance: ${result.distance.toFixed(3)})<br>
                        Size: ${(result.metadata.file_size / 1024).toFixed(1)} KB
                    </div>
                `;
                container.appendChild(div);
            });
        }

        function updateFilesList() {
            fetch('/api/files')
                .then(response => response.json())
                .then(data => {
                    const list = document.getElementById('files-list');
                    list.innerHTML = '';
                    data.files.forEach(file => {
                        const item = document.createElement('div');
                        item.className = 'file-item';
                        item.innerHTML = `
                            <strong>${file.name}</strong><br>
                            Size: ${(file.size / 1024).toFixed(1)} KB<br>
                            Modified: ${new Date(file.mtime * 1000).toLocaleString()}
                        `;
                        list.appendChild(item);
                    });
                })
                .catch(err => console.error('Error updating files:', err));
        }

        function logMessage(message, type = 'info') {
            const log = document.getElementById('download-log');
            const timestamp = new Date().toLocaleTimeString();
            const div = document.createElement('div');
            div.className = type;
            div.textContent = `[${timestamp}] ${message}`;
            log.appendChild(div);
            log.scrollTop = log.scrollHeight;

            // Keep only last 50 messages
            while (log.children.length > 50) {
                log.removeChild(log.firstChild);
            }
        }

        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const btn = document.getElementById('auto-refresh-btn');
            btn.textContent = autoRefresh ? 'Stop Auto Refresh' : 'Start Auto Refresh';
            btn.className = autoRefresh ? 'btn btn-danger' : 'btn btn-secondary';

            if (autoRefresh) {
                refreshInterval = setInterval(() => {
                    updateStatus();
                    updateFilesList();
                    updateVectorStats();
                }, 2000);
            } else {
                clearInterval(refreshInterval);
            }
        }

        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            updateStatus();
            updateFilesList();
            updateVectorStats();
            setInterval(updateStatus, 5000);
        });
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎭 Enhanced Face Processing System</h1>
            <p>Complete Web Interface with Batch Download & Vector Search</p>
        </header>

        <div class="grid">
            <!-- System Status -->
            <div class="card">
                <h2>📊 System Status</h2>
                <div class="stats">
                    <div class="stat">
                        <label>Downloaded this session:</label>
                        <span id="downloaded-count">0</span>
                    </div>
                    <div class="stat">
                        <label>Errors:</label>
                        <span id="error-count">0</span>
                    </div>
                    <div class="stat">
                        <label>Total files:</label>
                        <span id="total-files">0</span>
                    </div>
                    <div class="stat">
                        <label>Total size (MB):</label>
                        <span id="total-size">0.00</span>
                    </div>
                    <div class="stat">
                        <label>Status:</label>
                        <span id="status">Ready</span>
                    </div>
                    <div class="stat">
                        <label>Last download:</label>
                        <span id="last-download">None</span>
                    </div>
                </div>
            </div>

            <!-- Download Control -->
            <div class="card">
                <h2>📥 Download Control</h2>
                <div class="control-section">
                    <h3>Single Download</h3>
                    <button id="download-btn" class="btn btn-primary" onclick="downloadFace()">
                        Download Single Face
                    </button>
                    <div id="download-preview" class="image-preview" style="display: none;"></div>
                </div>

                <div class="control-section">
                    <h3>Batch Download</h3>
                    <div class="input-group">
                        <label for="batch-count">Number of faces (1-50):</label>
                        <input type="number" id="batch-count" min="1" max="50" value="5">
                    </div>
                    <button id="batch-download-btn" class="btn btn-success" onclick="downloadBatch()">
                        Start Batch Download
                    </button>
                    <button class="btn btn-warning" onclick="stopBatch()">
                        Stop Batch
                    </button>
                </div>

                <div id="batch-progress" class="progress-container" style="display: none;">
                    <h4>Batch Progress</h4>
                    <div class="progress-bar-container">
                        <div id="progress-bar" class="progress-bar"></div>
                    </div>
                    <p><span id="batch-current">0</span> / <span id="batch-total">0</span> (<span id="batch-percentage">0</span>%)</p>
                </div>

                <button id="auto-refresh-btn" class="btn btn-secondary" onclick="toggleAutoRefresh()">
                    Start Auto Refresh
                </button>
            </div>

            <!-- Vector Database -->
            <div class="card">
                <h2>🧠 Vector Database</h2>
                <div class="stats">
                    <div class="stat">
                        <label>Database Available:</label>
                        <span id="vector-available">Checking...</span>
                    </div>
                    <div class="stat">
                        <label>Database Initialized:</label>
                        <span id="vector-initialized">Checking...</span>
                    </div>
                    <div class="stat">
                        <label>Embeddings Count:</label>
                        <span id="embeddings-count">0</span>
                    </div>
                    <div class="stat">
                        <label>Collection Name:</label>
                        <span id="collection-name">faces</span>
                    </div>
                    <div class="stat">
                        <label>Database Path:</label>
                        <span id="db-path">./simple_db</span>
                    </div>
                    <div class="stat">
                        <label>Embedding Dimensions:</label>
                        <span id="embedding-dims">64</span>
                    </div>
                </div>

                <button id="process-btn" class="btn btn-info" onclick="processAllFaces()">
                    Process All Faces for Embeddings
                </button>
                <button id="export-btn" class="btn btn-success" onclick="exportMetadata()">
                    Export Metadata to JSON
                </button>
                <div id="embedding-preview" class="image-preview" style="display: none;"></div>
            </div>

            <!-- Search -->
            <div class="card">
                <h2>🔍 Face Search</h2>
                <div class="search-controls">
                    <p><strong>Note:</strong> Search uses the first available image as query</p>
                    <div class="input-group">
                        <label for="search-count">Number of results:</label>
                        <input type="number" id="search-count" min="1" max="20" value="5">
                    </div>
                    <button id="search-btn" class="btn btn-primary" onclick="searchSimilar()">
                        Search Similar Faces
                    </button>
                </div>
                <div id="search-results" class="search-results"></div>
            </div>

            <!-- Download Log & History -->
            <div class="card">
                <h2>📋 Download Log & History</h2>
                <div id="download-log" class="log"></div>

                <h3>Recent Downloads</h3>
                <div id="download-history" class="download-history"></div>
            </div>

            <!-- Files List -->
            <div class="card">
                <h2>📁 Downloaded Files</h2>
                <div id="files-list" class="files-list"></div>
            </div>

            <!-- System Info -->
            <div class="card">
                <h2>ℹ️ System Information</h2>
                <div class="info">
                    <p><strong>Requests available:</strong> """ + ("✅ Yes" if REQUESTS_AVAILABLE else "❌ No") + """</p>
                    <p><strong>PIL available:</strong> """ + ("✅ Yes" if PIL_AVAILABLE else "❌ No") + """</p>
                    <p><strong>Vector Database:</strong> ✅ Simple Implementation</p>
                    <p><strong>Faces directory:</strong> ./faces</p>
                    <p><strong>Source:</strong> ThisPersonDoesNotExist.com</p>
                    <p><strong>Embedding type:</strong> Statistical + Histogram</p>
                    <p><strong>Similarity metric:</strong> Cosine similarity</p>
                </div>
            </div>
        </div>

        <footer>
            <p>Enhanced Face Processing System - No Dependencies Required!</p>
        </footer>
    </div>
</body>
</html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_css(self):
        """Serve enhanced CSS styles (same as before)"""
        css = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { text-align: center; color: white; margin-bottom: 30px; }
        header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .card h2 { color: #4a5568; margin-bottom: 15px; font-size: 1.3rem; }
        .card h3 { color: #2d3748; margin: 15px 0 10px 0; font-size: 1.1rem; }
        .stats { display: grid; gap: 8px; }
        .stat { display: flex; justify-content: space-between; padding: 8px; background: #f7fafc; border-radius: 5px; }
        .stat label { font-weight: 500; }
        .stat span { color: #2d3748; font-weight: bold; }
        .control-section { margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }
        .input-group { display: flex; align-items: center; margin: 10px 0; }
        .input-group label { margin-right: 10px; min-width: 120px; font-weight: 500; }
        .input-group input { padding: 8px; border: 1px solid #e2e8f0; border-radius: 4px; flex: 1; }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; font-weight: 500; margin: 5px; transition: all 0.2s; }
        .btn-primary { background: #4299e1; color: white; }
        .btn-primary:hover { background: #3182ce; }
        .btn-success { background: #48bb78; color: white; }
        .btn-success:hover { background: #38a169; }
        .btn-warning { background: #ed8936; color: white; }
        .btn-warning:hover { background: #dd6b20; }
        .btn-info { background: #4fd1c7; color: white; }
        .btn-info:hover { background: #38b2ac; }
        .btn-secondary { background: #718096; color: white; }
        .btn-secondary:hover { background: #4a5568; }
        .btn-danger { background: #f56565; color: white; }
        .btn-danger:hover { background: #e53e3e; }
        .btn:disabled { background: #cbd5e0; cursor: not-allowed; }
        .progress-container { margin: 15px 0; padding: 15px; background: #edf2f7; border-radius: 8px; }
        .progress-bar-container { width: 100%; height: 20px; background: #e2e8f0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #48bb78, #38a169); transition: width 0.3s ease; width: 0%; }
        .log { background: #1a202c; color: #e2e8f0; padding: 15px; border-radius: 5px; height: 200px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 12px; }
        .log .success { color: #68d391; }
        .log .error { color: #feb2b2; }
        .log .info { color: #90cdf4; }
        .download-history { max-height: 200px; overflow-y: auto; }
        .history-item { padding: 8px; border: 1px solid #e2e8f0; border-radius: 4px; margin-bottom: 5px; background: #f7fafc; font-size: 12px; }
        .files-list { max-height: 300px; overflow-y: auto; }
        .file-item { padding: 10px; border: 1px solid #e2e8f0; border-radius: 5px; margin-bottom: 10px; background: #f7fafc; }
        .search-controls { margin-bottom: 20px; }
        .search-results { max-height: 600px; overflow-y: auto; }
        .search-result { padding: 15px; border: 1px solid #e2e8f0; border-radius: 5px; margin-bottom: 15px; background: #f7fafc; display: flex; gap: 15px; align-items: center; }
        .result-info { font-size: 14px; flex: 1; }
        .result-image { max-width: 150px; max-height: 150px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); object-fit: cover; }
        .preview-image { max-width: 200px; max-height: 200px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); object-fit: cover; margin: 10px auto; display: block; }
        .image-preview { margin-top: 15px; padding: 15px; background: #edf2f7; border-radius: 8px; text-align: center; }
        .image-preview h4 { margin-bottom: 10px; color: #2d3748; }
        .image-label { font-size: 12px; color: #4a5568; margin-top: 5px; word-break: break-all; }
        .query-image-container { padding: 15px; background: #fff5e6; border: 2px solid #ffa500; border-radius: 8px; margin-bottom: 15px; text-align: center; }
        .query-image-container h4 { color: #d97706; margin-bottom: 10px; }
        .info p { margin-bottom: 10px; padding: 5px 0; }
        footer { text-align: center; color: white; margin-top: 30px; opacity: 0.8; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } header h1 { font-size: 2rem; } .input-group { flex-direction: column; align-items: flex-start; } .input-group label { margin-bottom: 5px; } .search-result { flex-direction: column; } .result-image { max-width: 100%; } }
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/css')
        self.end_headers()
        self.wfile.write(css.encode())

    def serve_status(self):
        """Serve status API endpoint"""
        stats = self.server.downloader.get_stats()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode())

    def serve_download(self, query):
        """Serve download API endpoint"""
        success, message = self.server.downloader.download_single_face()
        response = {'success': success, 'message': message}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def serve_batch_download(self, query):
        """Serve batch download API endpoint"""
        count = int(query.get('count', [5])[0])
        count = min(count, 50)  # Limit to 50

        result = self.server.downloader.download_batch(count)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def serve_stop_batch(self):
        """Serve stop batch API endpoint"""
        self.server.downloader.stop_batch()
        response = {'success': True, 'message': 'Batch download stopped'}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def serve_vector_stats(self):
        """Serve vector database stats"""
        stats = self.server.vector_db.get_stats()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode())

    def serve_search(self, query):
        """Serve search API endpoint"""
        count = int(query.get('count', [5])[0])

        # Get all face files
        face_files = list(Path(self.server.config.faces_dir).glob("*.jpg"))

        if face_files:
            # Randomly select a query image
            import random
            query_file = random.choice(face_files)
            query_filename = query_file.name

            # Perform search
            results = self.server.vector_db.search_similar(str(query_file), count)

            # Add query image to response
            response = {
                'results': results,
                'query_image': query_filename
            }
        else:
            response = {
                'results': [],
                'query_image': None
            }

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def serve_process_all(self):
        """Process all faces for embeddings"""
        face_files = list(Path(self.server.config.faces_dir).glob("*.jpg"))
        processed = 0

        for file_path in face_files:
            face_id = f"processed_{int(time.time())}_{file_path.stem}"
            if self.server.vector_db.add_face(str(file_path), face_id):
                processed += 1

        response = {'processed': processed, 'total': len(face_files)}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def serve_files(self):
        """Serve files list API endpoint"""
        face_files = list(Path(self.server.config.faces_dir).glob("*.jpg"))
        files_data = []

        for file_path in sorted(face_files, key=lambda x: x.stat().st_mtime, reverse=True):
            stat = file_path.stat()
            files_data.append({
                'name': file_path.name,
                'size': stat.st_size,
                'mtime': stat.st_mtime
            })

        response = {'files': files_data}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def serve_export_metadata(self):
        """Export metadata to JSON file"""
        success, result = self.server.vector_db.export_metadata()
        response = {
            'success': success,
            'message': f'Metadata exported to {result}' if success else result
        }
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def serve_image(self, path):
        """Serve image file"""
        try:
            # Extract filename from path (format: /api/image/filename.jpg)
            filename = path.split('/api/image/')[-1]
            file_path = os.path.join(self.server.config.faces_dir, filename)

            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    image_data = f.read()

                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-Length', str(len(image_data)))
                self.end_headers()
                self.wfile.write(image_data)
            else:
                self.send_error(404)
        except Exception as e:
            print(f"Error serving image: {e}")
            self.send_error(500)

    def log_message(self, format, *args):
        """Suppress default logging"""
        return

class EnhancedWebServer:
    """Enhanced web server with simple vector database"""

    def __init__(self):
        self.config = EnhancedConfig()
        self.config.load()
        self.vector_db = SimpleVectorDB(self.config)
        self.downloader = EnhancedDownloader(self.config, self.vector_db)

    def find_available_port(self, start_port=8080):
        """Find an available port"""
        for port in range(start_port, start_port + 10):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()
                return port
            except OSError:
                continue
        return start_port

    def start(self):
        """Start the enhanced web server"""
        # Initialize vector database
        self.vector_db.initialize()

        port = self.find_available_port(self.config.port)

        httpd = HTTPServer(('localhost', port), EnhancedWebHandler)
        httpd.config = self.config
        httpd.downloader = self.downloader
        httpd.vector_db = self.vector_db

        print("=" * 70)
        print("🌐 ENHANCED FACE PROCESSING WEB INTERFACE (SIMPLE VERSION)")
        print("=" * 70)
        print(f"🚀 Server starting on http://localhost:{port}")
        print(f"📁 Faces directory: {self.config.faces_dir}")
        print(f"🗄️  Database path: {self.config.db_path}")
        print(f"🌐 Requests available: {'✅' if REQUESTS_AVAILABLE else '❌'}")
        print(f"🖼️  PIL available: {'✅' if PIL_AVAILABLE else '❌'}")
        print(f"🧠 Vector database: ✅ Simple Implementation (No ChromaDB required)")
        print(f"📐 Embedding dimensions: 64")
        print()
        print("📖 Features:")
        print("   • Single and batch face downloading (1-50 faces)")
        print("   • Simple vector database with statistical embeddings")
        print("   • Cosine similarity search functionality")
        print("   • Real-time progress tracking with progress bars")
        print("   • Detailed download history and statistics")
        print("   • Auto-refresh capability")
        print("   • Works without numpy or ChromaDB dependencies")
        print()
        print(f"📱 Open your browser to: http://localhost:{port}")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 70)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n⏹️  Server stopped")
            httpd.server_close()

def main():
    """Main entry point"""
    server = EnhancedWebServer()
    server.start()

if __name__ == "__main__":
    main()