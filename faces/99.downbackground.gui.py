#!/usr/bin/env python3
"""
Background Face Download System - GUI Version
Graphical interface for downloading faces with real-time statistics and controls
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import json
from datetime import datetime
from typing import Optional, List, Tuple
import sys
import shutil
from PIL import Image
import hashlib

# Import the download system components
from importlib import import_module
import importlib.util

# Import the BackgroundDownloader from the main script
spec = importlib.util.spec_from_file_location("downloader", "99.downbackground.py")
downloader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader_module)

BackgroundDownloader = downloader_module.BackgroundDownloader
DownloadConfig = downloader_module.DownloadConfig
DownloadStats = downloader_module.DownloadStats

class DownloadGUI:
    """GUI for the background face download system"""

    def __init__(self, root):
        self.root = root
        self.root.title("🎭 Background Face Downloader")
        self.root.geometry("800x700")
        self.root.resizable(True, True)

        # Download system
        self.config = DownloadConfig()
        self.downloader: Optional[BackgroundDownloader] = None
        self.download_thread: Optional[threading.Thread] = None
        self.is_downloading = False
        self.update_thread_running = False

        # Image validation
        self.corrupt_files: List[str] = []
        self.validation_thread: Optional[threading.Thread] = None
        self.is_validating = False

        # Duplicate detection
        self.duplicate_groups: List[List[str]] = []
        self.duplicate_thread: Optional[threading.Thread] = None
        self.is_detecting_duplicates = False

        # Load configuration
        self.load_config()

        # Create GUI
        self.create_widgets()
        self.update_display()

        # Start update loop
        self.start_update_loop()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="🎭 Background Face Downloader",
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)

        # Faces directory
        ttk.Label(config_frame, text="Faces Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.faces_dir_var = tk.StringVar(value=self.config.faces_dir)
        faces_dir_frame = ttk.Frame(config_frame)
        faces_dir_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        faces_dir_frame.columnconfigure(0, weight=1)

        self.faces_dir_entry = ttk.Entry(faces_dir_frame, textvariable=self.faces_dir_var)
        self.faces_dir_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(faces_dir_frame, text="Browse", command=self.browse_directory).grid(row=0, column=1)

        # Delay
        ttk.Label(config_frame, text="Delay (seconds):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.delay_var = tk.DoubleVar(value=self.config.delay)
        delay_spin = ttk.Spinbox(config_frame, from_=0.1, to=10.0, increment=0.1,
                                textvariable=self.delay_var, width=10)
        delay_spin.grid(row=1, column=1, sticky=tk.W, pady=2)

        # Workers
        ttk.Label(config_frame, text="Worker Threads:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.workers_var = tk.IntVar(value=self.config.max_workers)
        workers_spin = ttk.Spinbox(config_frame, from_=1, to=10, textvariable=self.workers_var, width=10)
        workers_spin.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Download limit
        limit_frame = ttk.Frame(config_frame)
        limit_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        self.unlimited_var = tk.BooleanVar(value=self.config.unlimited_download)
        unlimited_check = ttk.Checkbutton(limit_frame, text="Unlimited Download",
                                        variable=self.unlimited_var, command=self.toggle_limit)
        unlimited_check.grid(row=0, column=0, sticky=tk.W)

        ttk.Label(limit_frame, text="Limit:").grid(row=0, column=1, sticky=tk.W, padx=(20, 5))
        self.limit_var = tk.IntVar(value=self.config.download_limit or 100)
        self.limit_spin = ttk.Spinbox(limit_frame, from_=1, to=10000, textvariable=self.limit_var, width=10)
        self.limit_spin.grid(row=0, column=2, sticky=tk.W)

        # Duplicate checking
        self.check_duplicates_var = tk.BooleanVar(value=self.config.check_duplicates)
        dup_check = ttk.Checkbutton(config_frame, text="Check for Duplicates",
                                   variable=self.check_duplicates_var)
        dup_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.start_button = ttk.Button(control_frame, text="🚀 Start Download",
                                      command=self.start_download, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = ttk.Button(control_frame, text="⏹️ Stop Download",
                                     command=self.stop_download, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))

        self.force_stop_button = ttk.Button(control_frame, text="🛑 Force Stop",
                                           command=self.force_stop_download, state=tk.DISABLED)
        self.force_stop_button.pack(side=tk.LEFT, padx=(0, 10))

        self.restart_button = ttk.Button(control_frame, text="🔄 Restart Download",
                                        command=self.restart_download, state=tk.DISABLED)
        self.restart_button.pack(side=tk.LEFT, padx=(0, 10))

        self.save_config_button = ttk.Button(control_frame, text="💾 Save Config",
                                           command=self.save_config)
        self.save_config_button.pack(side=tk.LEFT, padx=(0, 10))

        self.open_folder_button = ttk.Button(control_frame, text="📁 Open Folder",
                                           command=self.open_faces_folder)
        self.open_folder_button.pack(side=tk.LEFT, padx=(0, 10))

        self.validate_button = ttk.Button(control_frame, text="🔍 Validate Images",
                                        command=self.validate_images)
        self.validate_button.pack(side=tk.LEFT, padx=(0, 10))

        self.clean_button = ttk.Button(control_frame, text="🗑️ Remove Corrupt",
                                     command=self.remove_corrupt_files, state=tk.DISABLED)
        self.clean_button.pack(side=tk.LEFT, padx=(0, 10))

        self.detect_duplicates_button = ttk.Button(control_frame, text="🔍 Find Duplicates",
                                                 command=self.detect_duplicates)
        self.detect_duplicates_button.pack(side=tk.LEFT, padx=(0, 10))

        self.remove_duplicates_button = ttk.Button(control_frame, text="🗑️ Remove Duplicates",
                                                 command=self.remove_duplicate_files, state=tk.DISABLED)
        self.remove_duplicates_button.pack(side=tk.LEFT)

        # Statistics section
        stats_frame = ttk.LabelFrame(main_frame, text="Download Statistics", padding="10")
        stats_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        stats_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Status
        ttk.Label(stats_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.status_label = ttk.Label(stats_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        # Statistics labels
        self.stats_labels = {}
        stats_names = [
            ("Total Files in Directory", "total_files"),
            ("Directory Size", "directory_size"),
            ("Available Disk Space", "available_space"),
            ("Total Attempts", "total_attempts"),
            ("Successful Downloads", "successful_downloads"),
            ("Duplicates Skipped", "duplicates_skipped"),
            ("Errors", "errors"),
            ("Elapsed Time", "elapsed_time"),
            ("Download Rate", "download_rate")
        ]

        for i, (name, key) in enumerate(stats_names, start=1):
            ttk.Label(stats_frame, text=f"{name}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            label = ttk.Label(stats_frame, text="0")
            label.grid(row=i, column=1, sticky=tk.W, pady=2)
            self.stats_labels[key] = label

        # Progress bar
        ttk.Label(stats_frame, text="Progress:").grid(row=len(stats_names)+1, column=0, sticky=tk.W, pady=2)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(stats_frame, variable=self.progress_var,
                                          mode='indeterminate')
        self.progress_bar.grid(row=len(stats_names)+1, column=1, sticky=(tk.W, tk.E), pady=2)

        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Download Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(row=1, column=0, pady=(5, 0))

        # Initialize limit spinner state
        self.toggle_limit()

    def browse_directory(self):
        """Browse for faces directory"""
        directory = filedialog.askdirectory(initialdir=self.faces_dir_var.get())
        if directory:
            self.faces_dir_var.set(directory)

    def toggle_limit(self):
        """Toggle limit spinner state"""
        if self.unlimited_var.get():
            self.limit_spin.config(state=tk.DISABLED)
        else:
            self.limit_spin.config(state=tk.NORMAL)

    def load_config(self):
        """Load configuration from file"""
        try:
            self.config = DownloadConfig.from_file()
            self.log("Configuration loaded successfully")
        except Exception as e:
            self.log(f"Error loading configuration: {e}")

    def save_config(self):
        """Save current configuration"""
        try:
            # Update config from GUI
            self.config.faces_dir = self.faces_dir_var.get()
            self.config.delay = self.delay_var.get()
            self.config.max_workers = self.workers_var.get()
            self.config.unlimited_download = self.unlimited_var.get()
            self.config.download_limit = None if self.unlimited_var.get() else self.limit_var.get()
            self.config.check_duplicates = self.check_duplicates_var.get()

            # Save to file
            self.config.save_to_file()
            self.log("Configuration saved successfully")
        except Exception as e:
            self.log(f"Error saving configuration: {e}")
            messagebox.showerror("Error", f"Error saving configuration: {e}")

    def start_download(self):
        """Start the download process"""
        if self.is_downloading:
            return

        try:
            # Update config from GUI
            self.save_config()

            # Create downloader
            self.downloader = BackgroundDownloader(self.config)

            # Start download thread
            self.download_thread = threading.Thread(target=self._download_worker, daemon=True)
            self.download_thread.start()

            # Update UI
            self.is_downloading = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.force_stop_button.config(state=tk.NORMAL)
            self.restart_button.config(state=tk.DISABLED)
            self.status_label.config(text="Downloading...", foreground="green")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()

            self.log("Download started")

        except Exception as e:
            self.log(f"Error starting download: {e}")
            messagebox.showerror("Error", f"Error starting download: {e}")

    def stop_download(self):
        """Stop the download process immediately"""
        if not self.is_downloading:
            return

        try:
            self.log("Stopping download immediately...")

            # Set running flag to False immediately
            if self.downloader:
                self.downloader.running = False

            # Force immediate UI update
            self.is_downloading = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.force_stop_button.config(state=tk.DISABLED)
            self.restart_button.config(state=tk.NORMAL)
            self.status_label.config(text="Stopped", foreground="red")
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate')

            # Cancel any pending download threads more aggressively
            if self.download_thread and self.download_thread.is_alive():
                # We can't force kill threads in Python, but we set the flag
                # The thread should check this flag frequently
                pass

            self.log("Download stopped immediately")

        except Exception as e:
            self.log(f"Error stopping download: {e}")

    def force_stop_download(self):
        """Force stop the download process with system termination"""
        if not self.is_downloading:
            return

        try:
            self.log("FORCE STOPPING all downloads...")

            # Immediately set all flags
            if self.downloader:
                self.downloader.running = False

            # Force immediate UI update
            self.is_downloading = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.force_stop_button.config(state=tk.DISABLED)
            self.restart_button.config(state=tk.NORMAL)
            self.status_label.config(text="FORCE STOPPED", foreground="red")
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate')

            # Kill any related Python processes (aggressive approach)
            import subprocess
            import os

            try:
                # Try using psutil if available
                try:
                    import psutil
                    current_pid = os.getpid()
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            if (proc.info['name'] == 'python3' and
                                any('99.downbackground' in cmd for cmd in proc.info['cmdline'] or []) and
                                proc.info['pid'] != current_pid):
                                proc.terminate()
                                self.log(f"Terminated background process PID: {proc.info['pid']}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except ImportError:
                    # Fallback: use system commands
                    try:
                        # On macOS/Linux, use pkill
                        subprocess.run(['pkill', '-f', '99.downbackground.py'],
                                     capture_output=True, timeout=5)
                        self.log("Terminated background processes using pkill")
                    except Exception:
                        self.log("Could not terminate background processes (psutil not available)")
            except Exception as e:
                self.log(f"Process termination failed: {e}")

            self.log("FORCE STOP completed")

        except Exception as e:
            self.log(f"Error during force stop: {e}")

    def restart_download(self):
        """Restart the download process"""
        if self.is_downloading:
            self.log("Stopping current download before restart...")
            self.stop_download()
            # Give it a moment to stop
            self.root.after(1000, self._restart_after_stop)
        else:
            self.start_download()

    def _restart_after_stop(self):
        """Helper method to restart after stopping"""
        if not self.is_downloading:
            self.log("Restarting download...")
            self.start_download()
        else:
            # Still stopping, wait a bit more
            self.root.after(500, self._restart_after_stop)

    def _download_worker(self):
        """Worker thread for downloading"""
        try:
            self.downloader.start_background_download()
        except Exception as e:
            self.log(f"Download error: {e}")
        finally:
            # Reset UI state
            self.root.after(0, self._reset_ui_after_download)

    def _reset_ui_after_download(self):
        """Reset UI state after download completes"""
        self.is_downloading = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.force_stop_button.config(state=tk.DISABLED)
        self.restart_button.config(state=tk.NORMAL)
        self.status_label.config(text="Stopped", foreground="red")
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        self.log("Download stopped")

    def open_faces_folder(self):
        """Open the faces folder in file explorer"""
        try:
            faces_dir = self.faces_dir_var.get()
            if not os.path.exists(faces_dir):
                os.makedirs(faces_dir, exist_ok=True)

            # Open folder based on OS
            import subprocess
            import platform

            if platform.system() == "Darwin":  # macOS
                subprocess.call(["open", faces_dir])
            elif platform.system() == "Windows":
                os.startfile(faces_dir)
            else:  # Linux
                subprocess.call(["xdg-open", faces_dir])

        except Exception as e:
            self.log(f"Error opening folder: {e}")
            messagebox.showerror("Error", f"Error opening folder: {e}")

    def start_update_loop(self):
        """Start the statistics update loop"""
        self.update_thread_running = True
        self.update_stats()

    def get_faces_count(self):
        """Get the total number of face files in the directory"""
        try:
            faces_dir = self.faces_dir_var.get()
            if os.path.exists(faces_dir):
                return len([f for f in os.listdir(faces_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            return 0
        except Exception:
            return 0

    def get_directory_size(self):
        """Get the total size of the faces directory in bytes"""
        try:
            faces_dir = self.faces_dir_var.get()
            if not os.path.exists(faces_dir):
                return 0

            total_size = 0
            for dirpath, dirnames, filenames in os.walk(faces_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except (OSError, IOError):
                            continue
            return total_size
        except Exception:
            return 0

    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                if unit == 'B':
                    return f"{size_bytes:.0f} {unit}"
                else:
                    return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def get_available_disk_space(self):
        """Get available disk space for the download location"""
        try:
            faces_dir = self.faces_dir_var.get()
            if not faces_dir:
                return 0

            # Create directory if it doesn't exist to get disk space
            if not os.path.exists(faces_dir):
                os.makedirs(faces_dir, exist_ok=True)

            # Get disk usage statistics
            total, used, free = shutil.disk_usage(faces_dir)
            return free
        except Exception:
            return 0

    def update_stats(self):
        """Update statistics display"""
        if not self.update_thread_running:
            return

        try:
            # Always update total files count, directory size, and available space
            total_files = self.get_faces_count()
            directory_size = self.get_directory_size()
            available_space = self.get_available_disk_space()

            self.stats_labels["total_files"].config(text=str(total_files))
            self.stats_labels["directory_size"].config(text=self.format_file_size(directory_size))
            self.stats_labels["available_space"].config(text=self.format_file_size(available_space))

            if self.downloader and self.is_downloading:
                stats = self.downloader.stats.get_stats()

                # Update statistics labels
                self.stats_labels["total_attempts"].config(text=str(stats["total_attempts"]))
                self.stats_labels["successful_downloads"].config(text=str(stats["successful_downloads"]))
                self.stats_labels["duplicates_skipped"].config(text=str(stats["duplicates_skipped"]))
                self.stats_labels["errors"].config(text=str(stats["errors"]))
                self.stats_labels["elapsed_time"].config(text=f"{stats['elapsed_time']:.1f}s")
                self.stats_labels["download_rate"].config(text=f"{stats['download_rate']:.2f}/s")

                # Update progress if limited download
                if not self.config.unlimited_download and self.config.download_limit:
                    progress = (stats["successful_downloads"] / self.config.download_limit) * 100
                    self.progress_var.set(min(progress, 100))
                    self.progress_bar.config(mode='determinate')
            else:
                # Reset download-specific stats when not downloading
                if not self.is_downloading:
                    self.stats_labels["total_attempts"].config(text="0")
                    self.stats_labels["successful_downloads"].config(text="0")
                    self.stats_labels["duplicates_skipped"].config(text="0")
                    self.stats_labels["errors"].config(text="0")
                    self.stats_labels["elapsed_time"].config(text="0.0s")
                    self.stats_labels["download_rate"].config(text="0.00/s")

        except Exception as e:
            self.log(f"Error updating stats: {e}")

        # Schedule next update
        self.root.after(1000, self.update_stats)  # Update every second

    def update_display(self):
        """Update the GUI display"""
        try:
            # Update faces directory info if it exists
            face_count = self.get_faces_count()
            directory_size = self.get_directory_size()
            available_space = self.get_available_disk_space()

            if face_count > 0:
                self.log(f"Found {face_count} existing faces in directory ({self.format_file_size(directory_size)})")

            # Update the stats display immediately
            self.stats_labels["total_files"].config(text=str(face_count))
            self.stats_labels["directory_size"].config(text=self.format_file_size(directory_size))
            self.stats_labels["available_space"].config(text=self.format_file_size(available_space))
        except Exception as e:
            self.log(f"Error updating display: {e}")

    def log(self, message):
        """Add message to log"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}\n"

            self.log_text.insert(tk.END, log_message)
            self.log_text.see(tk.END)
        except Exception:
            pass  # Ignore log errors

    def clear_log(self):
        """Clear the log"""
        self.log_text.delete(1.0, tk.END)

    def detect_duplicates(self):
        """Start duplicate detection in a separate thread"""
        if self.is_detecting_duplicates or self.is_downloading or self.is_validating:
            messagebox.showwarning("Warning", "Please wait for current operation to complete.")
            return

        faces_dir = self.faces_dir_var.get()
        if not os.path.exists(faces_dir):
            messagebox.showerror("Error", f"Faces directory does not exist: {faces_dir}")
            return

        # Reset duplicate groups
        self.duplicate_groups = []
        self.remove_duplicates_button.config(state=tk.DISABLED)

        # Start duplicate detection
        self.is_detecting_duplicates = True
        self.detect_duplicates_button.config(state=tk.DISABLED)
        self.status_label.config(text="Detecting duplicates...", foreground="orange")
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()

        # Start detection thread
        self.duplicate_thread = threading.Thread(target=self._detect_duplicates_worker, daemon=True)
        self.duplicate_thread.start()

        self.log("Starting duplicate detection...")

    def _detect_duplicates_worker(self):
        """Worker thread for duplicate detection"""
        try:
            faces_dir = self.faces_dir_var.get()
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

            # Get all image files
            image_files = []
            for root, dirs, files in os.walk(faces_dir):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_files.append(os.path.join(root, file))

            total_files = len(image_files)
            self.log(f"Found {total_files} image files to analyze for duplicates")

            if total_files < 2:
                self.root.after(0, lambda: self._duplicate_detection_complete([]))
                return

            # Multiple detection methods
            duplicate_groups = []

            # Method 1: File hash duplicates (exact matches)
            self.log("🔍 Phase 1: Detecting exact file duplicates...")
            hash_duplicates = self._find_hash_duplicates(image_files)
            duplicate_groups.extend(hash_duplicates)

            # Method 2: Image content hash duplicates (visually identical)
            if self.is_detecting_duplicates:
                self.log("🔍 Phase 2: Detecting visually identical images...")
                content_duplicates = self._find_content_duplicates(image_files)
                duplicate_groups.extend(content_duplicates)

            # Method 3: Similar images (perceptual hash)
            if self.is_detecting_duplicates:
                self.log("🔍 Phase 3: Detecting visually similar images...")
                similar_duplicates = self._find_similar_images(image_files)
                duplicate_groups.extend(similar_duplicates)

            # Remove overlapping groups and filter small groups
            duplicate_groups = self._merge_duplicate_groups(duplicate_groups)

            # Store results
            self.duplicate_groups = duplicate_groups

            # Update UI on completion
            self.root.after(0, lambda: self._duplicate_detection_complete(duplicate_groups))

        except Exception as e:
            self.log(f"Duplicate detection error: {e}")
            self.root.after(0, lambda: self._duplicate_detection_complete([]))

    def _find_hash_duplicates(self, image_files: List[str]) -> List[List[str]]:
        """Find exact file duplicates using file hash"""
        hash_map = {}

        for file_path in image_files:
            if not self.is_detecting_duplicates:
                break

            try:
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in hash_map:
                    hash_map[file_hash].append(file_path)
                else:
                    hash_map[file_hash] = [file_path]

            except Exception as e:
                self.log(f"Error hashing {os.path.basename(file_path)}: {e}")

        # Return groups with 2+ files
        return [files for files in hash_map.values() if len(files) > 1]

    def _find_content_duplicates(self, image_files: List[str]) -> List[List[str]]:
        """Find duplicates using image content hash (pixel-based)"""
        content_hash_map = {}
        processed = 0

        for file_path in image_files:
            if not self.is_detecting_duplicates:
                break

            try:
                # Calculate perceptual hash using PIL
                with Image.open(file_path) as img:
                    # Resize to small size for comparison
                    img_resized = img.resize((8, 8), Image.Resampling.LANCZOS)
                    img_gray = img_resized.convert('L')

                    # Convert to binary based on average
                    pixels = list(img_gray.getdata())
                    avg = sum(pixels) / len(pixels)
                    bits = ''.join('1' if pixel > avg else '0' for pixel in pixels)

                    # Convert binary to hex for storage
                    content_hash = format(int(bits, 2), '016x')

                if content_hash in content_hash_map:
                    content_hash_map[content_hash].append(file_path)
                else:
                    content_hash_map[content_hash] = [file_path]

                processed += 1
                if processed % 100 == 0:
                    self.root.after(0, lambda: self.log(f"Processed {processed} images for content analysis..."))

            except Exception as e:
                self.log(f"Error processing {os.path.basename(file_path)}: {e}")

        # Return groups with 2+ files
        return [files for files in content_hash_map.values() if len(files) > 1]

    def _find_similar_images(self, image_files: List[str]) -> List[List[str]]:
        """Find similar images using advanced perceptual hashing"""
        similar_groups = []
        processed = 0

        # Calculate hashes for all images
        image_hashes = {}

        for file_path in image_files:
            if not self.is_detecting_duplicates:
                break

            try:
                with Image.open(file_path) as img:
                    # Use difference hash (dHash) for better similarity detection
                    img_resized = img.resize((9, 8), Image.Resampling.LANCZOS)
                    img_gray = img_resized.convert('L')
                    pixels = list(img_gray.getdata())

                    # Calculate differences between adjacent pixels
                    diff_hash = []
                    for row in range(8):
                        for col in range(8):
                            pixel_left = pixels[row * 9 + col]
                            pixel_right = pixels[row * 9 + col + 1]
                            diff_hash.append(pixel_left > pixel_right)

                    # Convert to integer for Hamming distance calculation
                    hash_int = sum(2**i for i, bit in enumerate(diff_hash) if bit)
                    image_hashes[file_path] = hash_int

                processed += 1
                if processed % 100 == 0:
                    self.root.after(0, lambda: self.log(f"Processed {processed} images for similarity analysis..."))

            except Exception as e:
                self.log(f"Error hashing {os.path.basename(file_path)}: {e}")

        # Find similar images using Hamming distance
        threshold = 5  # Max different bits for similarity
        processed_files = set()

        for file1, hash1 in image_hashes.items():
            if file1 in processed_files or not self.is_detecting_duplicates:
                continue

            similar_group = [file1]
            processed_files.add(file1)

            for file2, hash2 in image_hashes.items():
                if file2 == file1 or file2 in processed_files:
                    continue

                # Calculate Hamming distance (number of different bits)
                hamming_distance = bin(hash1 ^ hash2).count('1')

                if hamming_distance <= threshold:
                    similar_group.append(file2)
                    processed_files.add(file2)

            if len(similar_group) > 1:
                similar_groups.append(similar_group)

        return similar_groups

    def _merge_duplicate_groups(self, duplicate_groups: List[List[str]]) -> List[List[str]]:
        """Merge overlapping duplicate groups and remove small groups"""
        if not duplicate_groups:
            return []

        # Merge overlapping groups
        merged_groups = []

        for group in duplicate_groups:
            merged = False
            for merged_group in merged_groups:
                # Check if any file in the group is already in a merged group
                if any(file in merged_group for file in group):
                    # Merge the groups
                    merged_group.extend([f for f in group if f not in merged_group])
                    merged = True
                    break

            if not merged:
                merged_groups.append(group[:])  # Copy the list

        # Remove duplicates within groups and filter small groups
        final_groups = []
        for group in merged_groups:
            unique_group = list(set(group))  # Remove duplicates
            if len(unique_group) > 1:  # Keep only groups with 2+ files
                final_groups.append(unique_group)

        return final_groups

    def _duplicate_detection_complete(self, duplicate_groups: List[List[str]]):
        """Called when duplicate detection is complete"""
        self.is_detecting_duplicates = False
        self.detect_duplicates_button.config(state=tk.NORMAL)
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')

        total_duplicates = sum(len(group) - 1 for group in duplicate_groups)  # -1 because we keep one from each group
        total_groups = len(duplicate_groups)

        if total_duplicates == 0:
            self.status_label.config(text="No duplicates found ✓", foreground="green")
            self.log("✅ Duplicate detection complete - No duplicates found!")
            messagebox.showinfo("Detection Complete", "No duplicate files found!")
        else:
            self.status_label.config(text=f"Found {total_duplicates} duplicate files in {total_groups} groups", foreground="red")
            self.remove_duplicates_button.config(state=tk.NORMAL)

            # Calculate total size of duplicates
            total_duplicate_size = 0
            for group in duplicate_groups:
                for file_path in group[1:]:  # Skip first file (kept)
                    try:
                        total_duplicate_size += os.path.getsize(file_path)
                    except:
                        pass

            size_str = self.format_file_size(total_duplicate_size)
            self.log(f"❌ Found {total_duplicates} duplicate files in {total_groups} groups ({size_str})")

            # Show detailed results
            msg = f"Found {total_duplicates} duplicate files in {total_groups} groups:\n\n"

            for i, group in enumerate(duplicate_groups[:5], 1):  # Show first 5 groups
                msg += f"Group {i}: {len(group)} files\n"
                for j, file_path in enumerate(group[:3]):  # Show first 3 files in each group
                    msg += f"  • {os.path.basename(file_path)}\n"
                if len(group) > 3:
                    msg += f"  ... and {len(group) - 3} more\n"
                msg += "\n"

            if total_groups > 5:
                msg += f"... and {total_groups - 5} more groups\n\n"

            msg += f"Total space to be freed: {size_str}\n\n"
            msg += "Note: One file from each group will be kept (usually the first one)."

            result = messagebox.askyesno("Duplicates Found",
                                       f"{msg}\n\nWould you like to remove all duplicate files now?")

            if result:
                self.remove_duplicate_files()

    def remove_duplicate_files(self):
        """Remove duplicate files, keeping one from each group"""
        if not self.duplicate_groups:
            messagebox.showinfo("Info", "No duplicate files to remove.")
            return

        total_duplicates = sum(len(group) - 1 for group in self.duplicate_groups)
        total_groups = len(self.duplicate_groups)

        # Calculate total size
        total_size = 0
        for group in self.duplicate_groups:
            for file_path in group[1:]:  # Skip first file (kept)
                try:
                    total_size += os.path.getsize(file_path)
                except:
                    pass

        size_str = self.format_file_size(total_size)

        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Duplicate Removal",
            f"Are you sure you want to delete {total_duplicates} duplicate files?\n\n"
            f"Groups: {total_groups}\n"
            f"Total size: {size_str}\n\n"
            f"One file from each group will be kept.\n"
            f"This action cannot be undone!"
        )

        if not result:
            return

        # Remove duplicate files
        removed_count = 0
        failed_count = 0
        kept_count = 0

        self.log(f"🗑️ Removing {total_duplicates} duplicate files from {total_groups} groups...")
        self.remove_duplicates_button.config(state=tk.DISABLED)

        for group_idx, group in enumerate(self.duplicate_groups, 1):
            if len(group) < 2:
                continue

            # Keep the first file (usually the oldest or first found)
            kept_file = group[0]
            kept_count += 1
            self.log(f"📌 Group {group_idx}: Keeping {os.path.basename(kept_file)}")

            # Remove the rest
            for file_path in group[1:]:
                filename = os.path.basename(file_path)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        removed_count += 1
                        self.log(f"✅ Removed duplicate: {filename}")
                    else:
                        self.log(f"⚠️ File not found: {filename}")

                except Exception as e:
                    failed_count += 1
                    self.log(f"❌ Failed to remove {filename}: {e}")

        # Clear the duplicate groups list
        self.duplicate_groups = []

        # Update display
        self.update_display()

        # Show results
        if failed_count == 0:
            self.status_label.config(text=f"Removed {removed_count} duplicates ✓", foreground="green")
            self.log(f"✅ Successfully removed {removed_count} duplicates, kept {kept_count} unique files")
            messagebox.showinfo("Cleanup Complete",
                              f"Successfully removed {removed_count} duplicate files!\n"
                              f"Kept {kept_count} unique files\n"
                              f"Freed up {size_str} of disk space.")
        else:
            self.status_label.config(text=f"Removed {removed_count}, failed {failed_count}", foreground="orange")
            self.log(f"⚠️ Removed {removed_count} files, {failed_count} failures, kept {kept_count} unique files")
            messagebox.showwarning("Cleanup Partial",
                                 f"Removed {removed_count} files successfully.\n"
                                 f"{failed_count} files could not be removed.\n"
                                 f"Kept {kept_count} unique files.")

    def validate_images(self):
        """Start image validation in a separate thread"""
        if self.is_validating or self.is_downloading:
            messagebox.showwarning("Warning", "Please wait for current operation to complete.")
            return

        faces_dir = self.faces_dir_var.get()
        if not os.path.exists(faces_dir):
            messagebox.showerror("Error", f"Faces directory does not exist: {faces_dir}")
            return

        # Reset corrupt files list
        self.corrupt_files = []
        self.clean_button.config(state=tk.DISABLED)

        # Start validation
        self.is_validating = True
        self.validate_button.config(state=tk.DISABLED)
        self.status_label.config(text="Validating images...", foreground="orange")
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()

        # Start validation thread
        self.validation_thread = threading.Thread(target=self._validate_images_worker, daemon=True)
        self.validation_thread.start()

        self.log("Starting image validation...")

    def _validate_images_worker(self):
        """Worker thread for image validation"""
        try:
            faces_dir = self.faces_dir_var.get()
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

            # Get all image files
            image_files = []
            for root, dirs, files in os.walk(faces_dir):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_files.append(os.path.join(root, file))

            total_files = len(image_files)
            corrupt_files = []
            processed = 0

            self.log(f"Found {total_files} image files to validate")

            for file_path in image_files:
                if not self.is_validating:  # Check if validation was cancelled
                    break

                try:
                    validation_result = self._validate_single_image(file_path)
                    if not validation_result['valid']:
                        corrupt_files.append({
                            'path': file_path,
                            'filename': os.path.basename(file_path),
                            'error': validation_result['error'],
                            'size': validation_result['size']
                        })
                        self.log(f"❌ Corrupt: {os.path.basename(file_path)} - {validation_result['error']}")

                    processed += 1

                    # Update progress occasionally
                    if processed % 50 == 0:
                        progress_msg = f"Validated {processed}/{total_files} files, found {len(corrupt_files)} corrupt"
                        self.root.after(0, lambda: self.log(progress_msg))

                except Exception as e:
                    corrupt_files.append({
                        'path': file_path,
                        'filename': os.path.basename(file_path),
                        'error': f"Validation failed: {str(e)}",
                        'size': 0
                    })
                    self.log(f"❌ Error validating {os.path.basename(file_path)}: {e}")

            # Store results
            self.corrupt_files = corrupt_files

            # Update UI on completion
            self.root.after(0, self._validation_complete)

        except Exception as e:
            self.log(f"Validation error: {e}")
            self.root.after(0, self._validation_complete)

    def _validate_single_image(self, file_path: str) -> dict:
        """Validate a single image file"""
        try:
            file_size = os.path.getsize(file_path)

            # Check if file is empty
            if file_size == 0:
                return {'valid': False, 'error': 'Empty file (0 bytes)', 'size': file_size}

            # Check if file is suspiciously small
            if file_size < 100:  # Less than 100 bytes is suspicious for an image
                return {'valid': False, 'error': f'File too small ({file_size} bytes)', 'size': file_size}

            # Try to open and verify image with PIL
            with Image.open(file_path) as img:
                # Verify the image integrity
                img.verify()

                # Check image dimensions
                if hasattr(img, 'size'):
                    width, height = img.size
                    if width == 0 or height == 0:
                        return {'valid': False, 'error': 'Invalid dimensions (0x0)', 'size': file_size}

                    # Check if dimensions are reasonable (not too small)
                    if width < 10 or height < 10:
                        return {'valid': False, 'error': f'Dimensions too small ({width}x{height})', 'size': file_size}

                # Try to load the image data to catch truncated files
                with Image.open(file_path) as img2:
                    img2.load()  # This will raise an exception if the image is truncated

                return {'valid': True, 'error': None, 'size': file_size}

        except Image.UnidentifiedImageError:
            return {'valid': False, 'error': 'Not a valid image format', 'size': file_size}
        except Image.DecompressionBombError:
            return {'valid': False, 'error': 'Decompression bomb detected', 'size': file_size}
        except OSError as e:
            if "truncated" in str(e).lower():
                return {'valid': False, 'error': 'Truncated or incomplete image', 'size': file_size}
            elif "cannot identify" in str(e).lower():
                return {'valid': False, 'error': 'Cannot identify image format', 'size': file_size}
            else:
                return {'valid': False, 'error': f'OS error: {str(e)}', 'size': file_size}
        except Exception as e:
            return {'valid': False, 'error': f'Unknown error: {str(e)}', 'size': file_size}

    def _validation_complete(self):
        """Called when validation is complete"""
        self.is_validating = False
        self.validate_button.config(state=tk.NORMAL)
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')

        corrupt_count = len(self.corrupt_files)

        if corrupt_count == 0:
            self.status_label.config(text="All images valid ✓", foreground="green")
            self.log("✅ Image validation complete - All images are valid!")
            messagebox.showinfo("Validation Complete", "All images are valid! No corrupt files found.")
        else:
            self.status_label.config(text=f"Found {corrupt_count} corrupt files", foreground="red")
            self.clean_button.config(state=tk.NORMAL)

            # Calculate total size of corrupt files
            total_corrupt_size = sum(f.get('size', 0) for f in self.corrupt_files)
            size_str = self.format_file_size(total_corrupt_size)

            self.log(f"❌ Found {corrupt_count} corrupt files ({size_str})")

            # Show detailed results
            msg = f"Found {corrupt_count} corrupt files:\n\n"
            for i, f in enumerate(self.corrupt_files[:10]):  # Show first 10
                msg += f"• {f['filename']}: {f['error']}\n"

            if corrupt_count > 10:
                msg += f"\n... and {corrupt_count - 10} more files"

            msg += f"\nTotal size: {size_str}\n\nClick 'Remove Corrupt' to delete these files."

            # Use a scrollable message box for long lists
            result = messagebox.askyesno("Corrupt Files Found",
                                       f"{msg}\n\nWould you like to remove all corrupt files now?")

            if result:
                self.remove_corrupt_files()

    def remove_corrupt_files(self):
        """Remove all corrupt files identified during validation"""
        if not self.corrupt_files:
            messagebox.showinfo("Info", "No corrupt files to remove.")
            return

        corrupt_count = len(self.corrupt_files)
        total_size = sum(f.get('size', 0) for f in self.corrupt_files)
        size_str = self.format_file_size(total_size)

        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete {corrupt_count} corrupt files?\n\n"
            f"Total size: {size_str}\n\n"
            f"This action cannot be undone!"
        )

        if not result:
            return

        # Remove files
        removed_count = 0
        failed_count = 0

        self.log(f"🗑️ Removing {corrupt_count} corrupt files...")
        self.clean_button.config(state=tk.DISABLED)

        for file_info in self.corrupt_files:
            file_path = file_info['path']
            filename = file_info['filename']

            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    removed_count += 1
                    self.log(f"✅ Removed: {filename}")
                else:
                    self.log(f"⚠️ File not found: {filename}")

            except Exception as e:
                failed_count += 1
                self.log(f"❌ Failed to remove {filename}: {e}")

        # Clear the corrupt files list
        self.corrupt_files = []

        # Update display
        self.update_display()

        # Show results
        if failed_count == 0:
            self.status_label.config(text=f"Removed {removed_count} corrupt files ✓", foreground="green")
            self.log(f"✅ Successfully removed {removed_count} corrupt files")
            messagebox.showinfo("Cleanup Complete",
                              f"Successfully removed {removed_count} corrupt files!\n"
                              f"Freed up {size_str} of disk space.")
        else:
            self.status_label.config(text=f"Removed {removed_count}, failed {failed_count}", foreground="orange")
            self.log(f"⚠️ Removed {removed_count} files, {failed_count} failures")
            messagebox.showwarning("Cleanup Partial",
                                 f"Removed {removed_count} files successfully.\n"
                                 f"{failed_count} files could not be removed.")

    def on_closing(self):
        """Handle window closing"""
        if self.is_downloading or self.is_validating or self.is_detecting_duplicates:
            if messagebox.askokcancel("Quit", "Operation is in progress. Do you want to stop and quit?"):
                if self.is_downloading:
                    self.stop_download()
                if self.is_validating:
                    self.is_validating = False
                if self.is_detecting_duplicates:
                    self.is_detecting_duplicates = False
                self.update_thread_running = False
                self.root.after(500, self.root.destroy)  # Give time to stop
        else:
            self.update_thread_running = False
            self.root.destroy()

def main():
    """Main function"""
    # Check if 99.downbackground.py exists
    if not os.path.exists("99.downbackground.py"):
        print("Error: 99.downbackground.py not found in current directory")
        print("Please ensure both files are in the same directory")
        sys.exit(1)

    # Create and run GUI
    root = tk.Tk()

    # Configure style
    style = ttk.Style()
    style.theme_use('clam')  # Use a modern theme

    app = DownloadGUI(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_closing()

if __name__ == "__main__":
    main()