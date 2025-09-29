#!/usr/bin/env python3
"""
Vector Embedding GUI - GUI Version
Graphical interface for embedding faces into vector database with real-time progress tracking
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

# Import the embedding system components
from importlib import import_module
import importlib.util

# Import the VectorEmbeddingProcessor
spec = importlib.util.spec_from_file_location("embedding", "100.embedintoVector.py")
embedding_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embedding_module)

VectorEmbeddingProcessor = embedding_module.VectorEmbeddingProcessor

class EmbeddingGUI:
    """GUI for the vector embedding system"""

    def __init__(self, root):
        self.root = root
        self.root.title("🔮 Vector Embedding Processor")
        self.root.geometry("900x800")
        self.root.resizable(True, True)

        # Embedding system
        self.processor: Optional[VectorEmbeddingProcessor] = None
        self.embedding_thread: Optional[threading.Thread] = None
        self.is_processing = False
        self.update_thread_running = False

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
        title_label = ttk.Label(main_frame, text="🔮 Vector Embedding Processor",
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)

        # Faces directory
        ttk.Label(config_frame, text="Faces Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.faces_dir_var = tk.StringVar(value="./faces")
        faces_dir_frame = ttk.Frame(config_frame)
        faces_dir_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        faces_dir_frame.columnconfigure(0, weight=1)

        self.faces_dir_entry = ttk.Entry(faces_dir_frame, textvariable=self.faces_dir_var)
        self.faces_dir_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(faces_dir_frame, text="Browse", command=self.browse_directory).grid(row=0, column=1)

        # Batch size
        ttk.Label(config_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.IntVar(value=50)
        batch_spin = ttk.Spinbox(config_frame, from_=10, to=200, increment=10,
                                textvariable=self.batch_size_var, width=10)
        batch_spin.grid(row=1, column=1, sticky=tk.W, pady=2)

        # Workers
        ttk.Label(config_frame, text="Worker Threads:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.workers_var = tk.IntVar(value=4)
        workers_spin = ttk.Spinbox(config_frame, from_=1, to=8, textvariable=self.workers_var, width=10)
        workers_spin.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Clear existing option
        self.clear_existing_var = tk.BooleanVar(value=False)
        clear_check = ttk.Checkbutton(config_frame, text="Clear Existing Embeddings",
                                     variable=self.clear_existing_var)
        clear_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.start_button = ttk.Button(control_frame, text="🚀 Start Embedding",
                                      command=self.start_embedding, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = ttk.Button(control_frame, text="⏹️ Stop Embedding",
                                     command=self.stop_embedding, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))

        self.open_folder_button = ttk.Button(control_frame, text="📁 Open Folder",
                                           command=self.open_faces_folder)
        self.open_folder_button.pack(side=tk.LEFT, padx=(0, 10))

        self.db_info_button = ttk.Button(control_frame, text="📊 Database Info",
                                        command=self.show_db_info)
        self.db_info_button.pack(side=tk.LEFT)

        # Directory Information section
        dir_info_frame = ttk.LabelFrame(main_frame, text="Directory Information", padding="10")
        dir_info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        dir_info_frame.columnconfigure(1, weight=1)

        # Directory info labels
        self.dir_info_labels = {}
        dir_info_names = [
            ("Total Files", "total_files"),
            ("Directory Size", "directory_size"),
            ("Available Disk Space", "available_space")
        ]

        for i, (name, key) in enumerate(dir_info_names):
            ttk.Label(dir_info_frame, text=f"{name}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            label = ttk.Label(dir_info_frame, text="0")
            label.grid(row=i, column=1, sticky=tk.W, pady=2)
            self.dir_info_labels[key] = label

        # Statistics section
        stats_frame = ttk.LabelFrame(main_frame, text="Embedding Statistics", padding="10")
        stats_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        stats_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # Status
        ttk.Label(stats_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.status_label = ttk.Label(stats_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        # Statistics labels
        self.stats_labels = {}
        stats_names = [
            ("Files to Process", "total_files"),
            ("Files Processed", "processed_files"),
            ("Successful Embeddings", "successful_embeddings"),
            ("Duplicates Skipped", "duplicates_skipped"),
            ("Errors", "errors"),
            ("Elapsed Time", "elapsed_time"),
            ("Processing Rate", "processing_rate"),
            ("Progress", "progress_percentage"),
            ("Remaining Files", "remaining_files")
        ]

        for i, (name, key) in enumerate(stats_names, start=1):
            ttk.Label(stats_frame, text=f"{name}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            label = ttk.Label(stats_frame, text="0")
            label.grid(row=i, column=1, sticky=tk.W, pady=2)
            self.stats_labels[key] = label

        # Progress bar
        ttk.Label(stats_frame, text="Progress Bar:").grid(row=len(stats_names)+1, column=0, sticky=tk.W, pady=2)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(stats_frame, variable=self.progress_var,
                                          mode='determinate', maximum=100)
        self.progress_bar.grid(row=len(stats_names)+1, column=1, sticky=(tk.W, tk.E), pady=2)

        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(row=1, column=0, pady=(5, 0))

    def browse_directory(self):
        """Browse for faces directory"""
        directory = filedialog.askdirectory(initialdir=self.faces_dir_var.get())
        if directory:
            self.faces_dir_var.set(directory)
            self.update_directory_info()

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

    def get_directory_info(self):
        """Get directory information"""
        try:
            faces_dir = self.faces_dir_var.get()
            if not os.path.exists(faces_dir):
                return {'total_files': 0, 'total_size_bytes': 0, 'free_space_bytes': 0}

            total_size = 0
            file_count = 0

            for root, dirs, files in os.walk(faces_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            file_path = os.path.join(root, file)
                            total_size += os.path.getsize(file_path)
                            file_count += 1
                        except OSError:
                            continue

            # Get available disk space
            free_space = shutil.disk_usage(faces_dir)[2]

            return {
                'total_files': file_count,
                'total_size_bytes': total_size,
                'free_space_bytes': free_space
            }
        except Exception:
            return {'total_files': 0, 'total_size_bytes': 0, 'free_space_bytes': 0}

    def update_directory_info(self):
        """Update directory information display"""
        try:
            dir_info = self.get_directory_info()

            self.dir_info_labels["total_files"].config(text=f"{dir_info['total_files']:,}")
            self.dir_info_labels["directory_size"].config(text=self.format_file_size(dir_info['total_size_bytes']))
            self.dir_info_labels["available_space"].config(text=self.format_file_size(dir_info['free_space_bytes']))

            if dir_info['total_files'] > 0:
                self.log(f"Directory contains {dir_info['total_files']:,} face files "
                        f"({self.format_file_size(dir_info['total_size_bytes'])})")

        except Exception as e:
            self.log(f"Error updating directory info: {e}")

    def start_embedding(self):
        """Start the embedding process"""
        if self.is_processing:
            return

        try:
            # Validate directory
            faces_dir = self.faces_dir_var.get()
            if not os.path.exists(faces_dir):
                messagebox.showerror("Error", f"Faces directory not found: {faces_dir}")
                return

            # Create processor
            self.processor = VectorEmbeddingProcessor(
                faces_dir=faces_dir,
                batch_size=self.batch_size_var.get(),
                max_workers=self.workers_var.get()
            )

            # Start embedding thread
            self.embedding_thread = threading.Thread(target=self._embedding_worker, daemon=True)
            self.embedding_thread.start()

            # Update UI
            self.is_processing = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Embedding...", foreground="green")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()

            self.log("Embedding process started")

        except Exception as e:
            self.log(f"Error starting embedding: {e}")
            messagebox.showerror("Error", f"Error starting embedding: {e}")

    def stop_embedding(self):
        """Stop the embedding process"""
        if not self.is_processing:
            return

        try:
            self.log("Stopping embedding process...")

            # Stop processor
            if self.processor:
                self.processor.stop_process()

            # Update UI
            self.is_processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Stopped", foreground="red")
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate')

            self.log("Embedding process stopped")

        except Exception as e:
            self.log(f"Error stopping embedding: {e}")

    def _embedding_worker(self):
        """Worker thread for embedding"""
        try:
            clear_existing = self.clear_existing_var.get()
            success = self.processor.start_embedding_process(clear_existing=clear_existing)

            if success:
                self.log("Embedding completed successfully!")
            else:
                self.log("Embedding process failed!")

        except Exception as e:
            self.log(f"Embedding error: {e}")
        finally:
            # Reset UI state
            self.root.after(0, self._reset_ui_after_embedding)

    def _reset_ui_after_embedding(self):
        """Reset UI state after embedding completes"""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Completed", foreground="blue")
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')

    def open_faces_folder(self):
        """Open the faces folder in file explorer"""
        try:
            faces_dir = self.faces_dir_var.get()
            if not os.path.exists(faces_dir):
                messagebox.showwarning("Warning", f"Directory not found: {faces_dir}")
                return

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

    def show_db_info(self):
        """Show database information"""
        try:
            # Run database info command
            import subprocess
            result = subprocess.run(['python3', 'run_chroma_info.py'],
                                  capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                # Show in a new window
                info_window = tk.Toplevel(self.root)
                info_window.title("Database Information")
                info_window.geometry("600x400")

                info_text = scrolledtext.ScrolledText(info_window, wrap=tk.WORD)
                info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                info_text.insert(tk.END, result.stdout)
                info_text.config(state=tk.DISABLED)
            else:
                self.log("Error getting database info")
                messagebox.showerror("Error", "Could not retrieve database information")

        except Exception as e:
            self.log(f"Error showing database info: {e}")
            messagebox.showerror("Error", f"Error showing database info: {e}")

    def start_update_loop(self):
        """Start the statistics update loop"""
        self.update_thread_running = True
        self.update_stats()

    def update_stats(self):
        """Update statistics display"""
        if not self.update_thread_running:
            return

        try:
            # Always update directory info
            self.update_directory_info()

            if self.processor and self.is_processing:
                stats = self.processor.get_stats()

                # Update statistics labels
                self.stats_labels["total_files"].config(text=str(stats.get("total_files", 0)))
                self.stats_labels["processed_files"].config(text=str(stats.get("processed_files", 0)))
                self.stats_labels["successful_embeddings"].config(text=str(stats.get("successful_embeddings", 0)))
                self.stats_labels["duplicates_skipped"].config(text=str(stats.get("duplicates_skipped", 0)))
                self.stats_labels["errors"].config(text=str(stats.get("errors", 0)))
                self.stats_labels["elapsed_time"].config(text=f"{stats.get('elapsed_time', 0):.1f}s")
                self.stats_labels["processing_rate"].config(text=f"{stats.get('processing_rate', 0):.2f}/s")
                self.stats_labels["progress_percentage"].config(text=f"{stats.get('progress_percentage', 0):.1f}%")
                self.stats_labels["remaining_files"].config(text=str(stats.get("remaining_files", 0)))

                # Update progress bar
                progress = stats.get("progress_percentage", 0)
                self.progress_var.set(progress)
                self.progress_bar.config(mode='determinate')
            else:
                # Reset stats when not processing
                if not self.is_processing:
                    for key in ["total_files", "processed_files", "successful_embeddings",
                               "duplicates_skipped", "errors", "remaining_files"]:
                        self.stats_labels[key].config(text="0")
                    self.stats_labels["elapsed_time"].config(text="0.0s")
                    self.stats_labels["processing_rate"].config(text="0.00/s")
                    self.stats_labels["progress_percentage"].config(text="0.0%")

        except Exception as e:
            self.log(f"Error updating stats: {e}")

        # Schedule next update
        self.root.after(1000, self.update_stats)  # Update every second

    def update_display(self):
        """Update the GUI display"""
        try:
            self.update_directory_info()
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
        if self.is_processing:
            if messagebox.askokcancel("Quit", "Embedding is in progress. Do you want to stop and quit?"):
                self.stop_embedding()
                self.update_thread_running = False
                self.root.after(500, self.root.destroy)  # Give time to stop
        else:
            self.update_thread_running = False
            self.root.destroy()

def main():
    """Main function"""
    # Check if required files exist
    required_files = ["100.embedintoVector.py", "face_collector.py", "face_database.py"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found in current directory")
            print("Please ensure all required files are in the same directory")
            sys.exit(1)

    # Create and run GUI
    root = tk.Tk()

    # Configure style
    style = ttk.Style()
    style.theme_use('clam')  # Use a modern theme

    app = EmbeddingGUI(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_closing()

if __name__ == "__main__":
    main()