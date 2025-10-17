#!/usr/bin/env python3
"""
Web-based Face Processing GUI
A simple web interface for the integrated face processing system
"""

import os
import sys
import json
import time
import hashlib
import threading
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

class FaceWebConfig:
    """Configuration for web interface"""
    def __init__(self):
        self.faces_dir = "./faces"
        self.download_delay = 1.0
        self.config_file = "web_config.json"
        self.port = 8080

    def save(self):
        data = {
            'faces_dir': self.faces_dir,
            'download_delay': self.download_delay,
            'port': self.port
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.faces_dir = data.get('faces_dir', './faces')
                self.download_delay = data.get('download_delay', 1.0)
                self.port = data.get('port', 8080)

class FaceWebDownloader:
    """Web-compatible face downloader"""
    def __init__(self, config):
        self.config = config
        self.running = False
        self.downloaded_count = 0
        self.error_count = 0
        self.status = "Ready"
        self.last_download = None

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

            self.downloaded_count += 1
            self.last_download = filename
            self.status = "Ready"
            return True, f"Downloaded: {filename}"

        except Exception as e:
            self.error_count += 1
            self.status = "Error"
            return False, f"Error: {str(e)}"

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
            'last_download': self.last_download
        }

class FaceWebHandler(BaseHTTPRequestHandler):
    """HTTP request handler for face processing web interface"""

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
            self.serve_download()
        elif path == '/api/files':
            self.serve_files()
        elif path == '/style.css':
            self.serve_css()
        else:
            self.send_error(404)

    def serve_main_page(self):
        """Serve the main HTML page"""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integrated Face Processing System</title>
    <link rel="stylesheet" href="/style.css">
    <script>
        let autoRefresh = false;

        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('downloaded-count').textContent = data.downloaded_count;
                    document.getElementById('error-count').textContent = data.error_count;
                    document.getElementById('total-files').textContent = data.total_files;
                    document.getElementById('total-size').textContent = data.total_size_mb.toFixed(2);
                    document.getElementById('status').textContent = data.status;
                    document.getElementById('last-download').textContent = data.last_download || 'None';
                })
                .catch(err => console.error('Error updating status:', err));
        }

        function downloadFace() {
            document.getElementById('download-btn').disabled = true;
            document.getElementById('download-btn').textContent = 'Downloading...';

            fetch('/api/download')
                .then(response => response.json())
                .then(data => {
                    const log = document.getElementById('download-log');
                    const timestamp = new Date().toLocaleTimeString();
                    log.innerHTML += `<div>[${timestamp}] ${data.message}</div>`;
                    log.scrollTop = log.scrollHeight;
                    updateStatus();
                    updateFilesList();
                })
                .catch(err => {
                    console.error('Download error:', err);
                    const log = document.getElementById('download-log');
                    log.innerHTML += `<div class="error">Error: ${err.message}</div>`;
                })
                .finally(() => {
                    document.getElementById('download-btn').disabled = false;
                    document.getElementById('download-btn').textContent = 'Download Single Face';
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

        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const btn = document.getElementById('auto-refresh-btn');
            btn.textContent = autoRefresh ? 'Stop Auto Refresh' : 'Start Auto Refresh';
            btn.className = autoRefresh ? 'btn btn-danger' : 'btn btn-secondary';

            if (autoRefresh) {
                refreshInterval = setInterval(() => {
                    updateStatus();
                    updateFilesList();
                }, 2000);
            } else {
                clearInterval(refreshInterval);
            }
        }

        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            updateStatus();
            updateFilesList();
            setInterval(updateStatus, 5000); // Update every 5 seconds
        });
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎭 Integrated Face Processing System</h1>
            <p>Web Interface for Face Download and Management</p>
        </header>

        <div class="grid">
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

            <div class="card">
                <h2>📥 Download Control</h2>
                <button id="download-btn" class="btn btn-primary" onclick="downloadFace()">
                    Download Single Face
                </button>
                <button id="auto-refresh-btn" class="btn btn-secondary" onclick="toggleAutoRefresh()">
                    Start Auto Refresh
                </button>
                <div class="log-container">
                    <h3>Download Log</h3>
                    <div id="download-log" class="log"></div>
                </div>
            </div>

            <div class="card">
                <h2>📁 Downloaded Files</h2>
                <div id="files-list" class="files-list"></div>
            </div>

            <div class="card">
                <h2>ℹ️ System Information</h2>
                <div class="info">
                    <p><strong>Requests available:</strong> """ + ("✅ Yes" if REQUESTS_AVAILABLE else "❌ No") + """</p>
                    <p><strong>PIL available:</strong> """ + ("✅ Yes" if PIL_AVAILABLE else "❌ No") + """</p>
                    <p><strong>Faces directory:</strong> ./faces</p>
                    <p><strong>Source:</strong> ThisPersonDoesNotExist.com</p>
                </div>
            </div>
        </div>

        <footer>
            <p>Integrated Face Processing System - Web Interface</p>
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
        """Serve CSS styles"""
        css = """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .card h2 {
            color: #4a5568;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }

        .stats {
            display: grid;
            gap: 10px;
        }

        .stat {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            background: #f7fafc;
            border-radius: 5px;
        }

        .stat label {
            font-weight: 500;
        }

        .stat span {
            color: #2d3748;
            font-weight: bold;
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            margin: 5px;
            transition: all 0.2s;
        }

        .btn-primary {
            background: #4299e1;
            color: white;
        }

        .btn-primary:hover {
            background: #3182ce;
        }

        .btn-secondary {
            background: #718096;
            color: white;
        }

        .btn-secondary:hover {
            background: #4a5568;
        }

        .btn-danger {
            background: #f56565;
            color: white;
        }

        .btn-danger:hover {
            background: #e53e3e;
        }

        .btn:disabled {
            background: #cbd5e0;
            cursor: not-allowed;
        }

        .log-container {
            margin-top: 20px;
        }

        .log {
            background: #1a202c;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 5px;
            height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }

        .log .error {
            color: #feb2b2;
        }

        .files-list {
            max-height: 300px;
            overflow-y: auto;
        }

        .file-item {
            padding: 10px;
            border: 1px solid #e2e8f0;
            border-radius: 5px;
            margin-bottom: 10px;
            background: #f7fafc;
        }

        .info p {
            margin-bottom: 10px;
            padding: 5px 0;
        }

        footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.8;
        }

        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }

            header h1 {
                font-size: 2rem;
            }
        }
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

    def serve_download(self):
        """Serve download API endpoint"""
        success, message = self.server.downloader.download_single_face()
        response = {'success': success, 'message': message}
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

    def log_message(self, format, *args):
        """Suppress default logging"""
        return

class FaceWebServer:
    """Web server for face processing interface"""

    def __init__(self):
        self.config = FaceWebConfig()
        self.config.load()
        self.downloader = FaceWebDownloader(self.config)

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
        return start_port  # Fallback

    def start(self):
        """Start the web server"""
        port = self.find_available_port(self.config.port)

        httpd = HTTPServer(('localhost', port), FaceWebHandler)
        httpd.config = self.config
        httpd.downloader = self.downloader

        print("=" * 60)
        print("🌐 FACE PROCESSING WEB INTERFACE")
        print("=" * 60)
        print(f"🚀 Server starting on http://localhost:{port}")
        print(f"📁 Faces directory: {self.config.faces_dir}")
        print(f"🌐 Requests available: {'✅' if REQUESTS_AVAILABLE else '❌'}")
        print(f"🖼️  PIL available: {'✅' if PIL_AVAILABLE else '❌'}")
        print()
        print("📖 Usage:")
        print(f"   Open your browser to: http://localhost:{port}")
        print("   Click 'Download Single Face' to download faces")
        print("   Use 'Start Auto Refresh' for live updates")
        print()
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n⏹️  Server stopped")
            httpd.server_close()

def main():
    """Main entry point"""
    server = FaceWebServer()
    server.start()

if __name__ == "__main__":
    main()