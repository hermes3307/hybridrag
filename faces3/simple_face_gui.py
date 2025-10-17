#!/usr/bin/env python3
"""
Simple Face Processing GUI - Basic Version
Works with minimal dependencies for demonstration
"""

import os
import sys
import json
import time
import hashlib
import threading
from datetime import datetime
from pathlib import Path

# Check if we can import tkinter
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("❌ tkinter not available. Please install: sudo apt-get install python3-tk")

# Check other basic dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("❌ requests not available. Install with: pip install requests")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("❌ PIL not available. Install with: pip install Pillow")

class SimpleConfig:
    """Simple configuration class"""
    def __init__(self):
        self.faces_dir = "./faces"
        self.download_delay = 1.0
        self.config_file = "simple_config.json"

    def save(self):
        data = {
            'faces_dir': self.faces_dir,
            'download_delay': self.download_delay
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.faces_dir = data.get('faces_dir', './faces')
                self.download_delay = data.get('download_delay', 1.0)

class SimpleDownloader:
    """Simple face downloader"""
    def __init__(self, config):
        self.config = config
        self.running = False
        self.downloaded_count = 0
        self.error_count = 0

        # Create faces directory
        os.makedirs(self.config.faces_dir, exist_ok=True)

    def download_single_face(self):
        """Download a single face"""
        if not REQUESTS_AVAILABLE:
            return False, "requests module not available"

        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/",
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30
            )
            response.raise_for_status()

            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_hash = hashlib.md5(response.content).hexdigest()[:8]
            filename = f"face_{timestamp}_{image_hash}.jpg"
            file_path = os.path.join(self.config.faces_dir, filename)

            # Save image
            with open(file_path, 'wb') as f:
                f.write(response.content)

            self.downloaded_count += 1
            return True, f"Downloaded: {filename}"

        except Exception as e:
            self.error_count += 1
            return False, f"Error: {str(e)}"

    def start_download_loop(self, callback=None):
        """Start continuous download"""
        self.running = True

        def download_worker():
            while self.running:
                success, message = self.download_single_face()
                if callback:
                    callback(success, message)
                time.sleep(self.config.download_delay)

        thread = threading.Thread(target=download_worker, daemon=True)
        thread.start()
        return thread

    def stop_download_loop(self):
        """Stop download"""
        self.running = False

class SimpleFaceGUI:
    """Simple GUI for face downloading"""

    def __init__(self):
        if not TKINTER_AVAILABLE:
            print("Cannot start GUI - tkinter not available")
            return

        self.root = tk.Tk()
        self.root.title("Simple Face Downloader")
        self.root.geometry("800x600")

        self.config = SimpleConfig()
        self.config.load()

        self.downloader = SimpleDownloader(self.config)
        self.is_downloading = False

        self.create_widgets()
        self.setup_layout()

    def create_widgets(self):
        """Create GUI widgets"""

        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding=10)
        config_frame.pack(fill="x", pady=(0, 10))

        # Faces directory
        ttk.Label(config_frame, text="Faces Directory:").grid(row=0, column=0, sticky="w")
        self.faces_dir_var = tk.StringVar(value=self.config.faces_dir)
        ttk.Entry(config_frame, textvariable=self.faces_dir_var, width=50).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        ttk.Button(config_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=5)

        # Download delay
        ttk.Label(config_frame, text="Download Delay (seconds):").grid(row=1, column=0, sticky="w")
        self.delay_var = tk.DoubleVar(value=self.config.download_delay)
        ttk.Spinbox(config_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.delay_var, width=10).grid(row=1, column=1, sticky="w", padx=(5, 0))

        config_frame.columnconfigure(1, weight=1)

        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Download Control", padding=10)
        control_frame.pack(fill="x", pady=(0, 10))

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x")

        self.download_button = ttk.Button(button_frame, text="Start Download", command=self.toggle_download)
        self.download_button.pack(side="left", padx=(0, 5))

        ttk.Button(button_frame, text="Download Single", command=self.download_single).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Open Faces Folder", command=self.open_faces_folder).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(side="left", padx=5)

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
        status_frame.pack(fill="both", expand=True)

        # Statistics
        stats_frame = ttk.Frame(status_frame)
        stats_frame.pack(fill="x", pady=(0, 10))

        self.downloaded_label = ttk.Label(stats_frame, text="Downloaded: 0")
        self.downloaded_label.pack(side="left")

        self.errors_label = ttk.Label(stats_frame, text="Errors: 0")
        self.errors_label.pack(side="left", padx=(20, 0))

        self.status_label = ttk.Label(stats_frame, text="Status: Ready")
        self.status_label.pack(side="left", padx=(20, 0))

        # Log
        ttk.Label(status_frame, text="Download Log:").pack(anchor="w")
        self.log_text = scrolledtext.ScrolledText(status_frame, height=15)
        self.log_text.pack(fill="both", expand=True)

        # Update display
        self.update_display()

    def setup_layout(self):
        """Setup responsive layout"""
        pass

    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, formatted)
        self.log_text.see(tk.END)

    def browse_directory(self):
        """Browse for faces directory"""
        directory = filedialog.askdirectory(initialdir=self.faces_dir_var.get())
        if directory:
            self.faces_dir_var.set(directory)

    def toggle_download(self):
        """Toggle download process"""
        if not self.is_downloading:
            self.start_download()
        else:
            self.stop_download()

    def start_download(self):
        """Start downloading"""
        # Update config
        self.config.faces_dir = self.faces_dir_var.get()
        self.config.download_delay = self.delay_var.get()

        # Create directory
        os.makedirs(self.config.faces_dir, exist_ok=True)

        # Start download
        self.downloader.config = self.config
        self.downloader.start_download_loop(callback=self.on_download_result)

        self.is_downloading = True
        self.download_button.config(text="Stop Download")
        self.log_message("Download started")

    def stop_download(self):
        """Stop downloading"""
        self.downloader.stop_download_loop()
        self.is_downloading = False
        self.download_button.config(text="Start Download")
        self.log_message("Download stopped")

    def download_single(self):
        """Download single face"""
        # Update config
        self.config.faces_dir = self.faces_dir_var.get()
        self.downloader.config = self.config

        success, message = self.downloader.download_single_face()
        self.log_message(message)

    def on_download_result(self, success, message):
        """Handle download result"""
        self.log_message(message)

    def open_faces_folder(self):
        """Open faces folder in file manager"""
        try:
            if os.path.exists(self.config.faces_dir):
                os.system(f'xdg-open "{self.config.faces_dir}"')
            else:
                messagebox.showwarning("Warning", "Faces directory does not exist")
        except Exception as e:
            self.log_message(f"Error opening folder: {e}")

    def save_config(self):
        """Save configuration"""
        self.config.faces_dir = self.faces_dir_var.get()
        self.config.download_delay = self.delay_var.get()
        self.config.save()
        self.log_message("Configuration saved")

    def update_display(self):
        """Update display elements"""
        if hasattr(self, 'downloaded_label'):
            self.downloaded_label.config(text=f"Downloaded: {self.downloader.downloaded_count}")
            self.errors_label.config(text=f"Errors: {self.downloader.error_count}")

            status = "Downloading..." if self.is_downloading else "Ready"
            self.status_label.config(text=f"Status: {status}")

        # Schedule next update
        self.root.after(1000, self.update_display)

    def run(self):
        """Run the application"""
        if TKINTER_AVAILABLE:
            try:
                self.root.mainloop()
            except KeyboardInterrupt:
                self.log_message("Application interrupted")
            finally:
                if self.downloader:
                    self.downloader.stop_download_loop()
        else:
            print("GUI not available")

def check_dependencies():
    """Check and report dependencies"""
    print("="*50)
    print("DEPENDENCY CHECK")
    print("="*50)

    deps = [
        ("tkinter", TKINTER_AVAILABLE, "sudo apt-get install python3-tk"),
        ("requests", REQUESTS_AVAILABLE, "pip install requests"),
        ("PIL/Pillow", PIL_AVAILABLE, "pip install Pillow")
    ]

    all_available = True
    for name, available, install_cmd in deps:
        status = "✅" if available else "❌"
        print(f"{status} {name}: {'Available' if available else 'Missing'}")
        if not available:
            print(f"   Install with: {install_cmd}")
            all_available = False

    print()

    if all_available:
        print("🎉 All dependencies available!")
    else:
        print("⚠️  Some dependencies missing. GUI may not work properly.")

    return all_available

def main():
    """Main entry point"""
    print("Simple Face Processing GUI")
    print("=" * 30)

    # Check dependencies
    deps_ok = check_dependencies()

    if not deps_ok:
        print("\nTo install missing dependencies:")
        print("1. For tkinter: sudo apt-get install python3-tk")
        print("2. For requests: pip3 install requests --user")
        print("3. For PIL: pip3 install Pillow --user")
        print("\nOr create a virtual environment:")
        print("python3 -m venv venv")
        print("source venv/bin/activate")
        print("pip install requests Pillow")
        print()

    if TKINTER_AVAILABLE:
        app = SimpleFaceGUI()
        app.run()
    else:
        print("❌ Cannot start GUI - tkinter not available")
        print("Install tkinter with: sudo apt-get install python3-tk")

if __name__ == "__main__":
    main()