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
from typing import Optional
import sys
import shutil

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
        self.open_folder_button.pack(side=tk.LEFT)

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

    def on_closing(self):
        """Handle window closing"""
        if self.is_downloading:
            if messagebox.askokcancel("Quit", "Download is in progress. Do you want to stop and quit?"):
                self.stop_download()
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