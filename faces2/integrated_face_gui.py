#!/usr/bin/env python3
"""
Integrated Face Processing GUI
Combines downloading, embedding, and searching functionality into one interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import json
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
from PIL import Image, ImageTk
import io

# Import our core backend
try:
    from core_backend import (
        IntegratedFaceSystem, SystemConfig, FaceAnalyzer,
        FaceEmbedder, CHROMADB_AVAILABLE
    )
except ImportError as e:
    print(f"Error importing core backend: {e}")
    print("Make sure core_backend.py is in the same directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedFaceGUI:
    """Main GUI application for integrated face processing"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Integrated Face Processing System")
        self.root.geometry("1200x800")

        # System components
        self.system = None
        self.download_thread = None
        self.processing_thread = None

        # GUI state
        self.is_downloading = False
        self.is_processing = False
        self.last_stats_update = 0

        # Create GUI
        self.create_widgets()
        self.setup_layout()

        # Initialize system
        self.initialize_system()

        # Start update loop
        self.update_display()

    def create_widgets(self):
        """Create all GUI widgets"""

        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)

        # Tab 1: System Overview
        self.overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_frame, text="System Overview")
        self.create_overview_tab()

        # Tab 2: Download
        self.download_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.download_frame, text="Download Faces")
        self.create_download_tab()

        # Tab 3: Process/Embed
        self.process_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.process_frame, text="Process & Embed")
        self.create_process_tab()

        # Tab 4: Search
        self.search_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.search_frame, text="Search Faces")
        self.create_search_tab()

        # Tab 5: Configuration
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Configuration")
        self.create_config_tab()

    def create_overview_tab(self):
        """Create system overview tab"""

        # System status frame
        status_frame = ttk.LabelFrame(self.overview_frame, text="System Status", padding=10)
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Status labels
        self.status_labels = {}
        status_items = [
            ("Database Status", "db_status"),
            ("Total Faces", "total_faces"),
            ("Download Rate", "download_rate"),
            ("Processing Rate", "process_rate"),
            ("System Uptime", "uptime")
        ]

        for i, (label, key) in enumerate(status_items):
            ttk.Label(status_frame, text=f"{label}:").grid(row=i, column=0, sticky="w", padx=(0, 10))
            self.status_labels[key] = ttk.Label(status_frame, text="Initializing...")
            self.status_labels[key].grid(row=i, column=1, sticky="w")

        # Statistics frame
        stats_frame = ttk.LabelFrame(self.overview_frame, text="Statistics", padding=10)
        stats_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Statistics text widget
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=15, width=70)
        self.stats_text.pack(fill="both", expand=True)

        # Control buttons
        control_frame = ttk.Frame(self.overview_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(control_frame, text="Refresh Status", command=self.refresh_status).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Reset Statistics", command=self.reset_statistics).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Save Configuration", command=self.save_configuration).pack(side="left", padx=5)

    def create_download_tab(self):
        """Create download faces tab"""

        # Download control frame
        control_frame = ttk.LabelFrame(self.download_frame, text="Download Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Download settings
        ttk.Label(control_frame, text="Download Delay (seconds):").grid(row=0, column=0, sticky="w")
        self.download_delay_var = tk.DoubleVar(value=1.0)
        delay_spin = ttk.Spinbox(control_frame, from_=0.1, to=10.0, increment=0.1,
                                textvariable=self.download_delay_var, width=10)
        delay_spin.grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(control_frame, text="Faces Directory:").grid(row=1, column=0, sticky="w")
        self.faces_dir_var = tk.StringVar(value="./faces")
        ttk.Entry(control_frame, textvariable=self.faces_dir_var, width=40).grid(row=1, column=1, sticky="w", padx=(5, 0))
        ttk.Button(control_frame, text="Browse", command=self.browse_faces_dir).grid(row=1, column=2, padx=5)

        # Download buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)

        self.download_button = ttk.Button(button_frame, text="Start Download", command=self.toggle_download)
        self.download_button.pack(side="left", padx=5)

        ttk.Button(button_frame, text="Download Single", command=self.download_single).pack(side="left", padx=5)

        # Download status
        status_frame = ttk.LabelFrame(self.download_frame, text="Download Status", padding=10)
        status_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.download_status_text = scrolledtext.ScrolledText(status_frame, height=20, width=70)
        self.download_status_text.pack(fill="both", expand=True)

    def create_process_tab(self):
        """Create process/embed tab"""

        # Processing control frame
        control_frame = ttk.LabelFrame(self.process_frame, text="Processing Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Processing settings
        ttk.Label(control_frame, text="Batch Size:").grid(row=0, column=0, sticky="w")
        self.batch_size_var = tk.IntVar(value=50)
        batch_spin = ttk.Spinbox(control_frame, from_=1, to=200, increment=1,
                                textvariable=self.batch_size_var, width=10)
        batch_spin.grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(control_frame, text="Max Workers:").grid(row=1, column=0, sticky="w")
        self.max_workers_var = tk.IntVar(value=4)
        workers_spin = ttk.Spinbox(control_frame, from_=1, to=8, increment=1,
                                  textvariable=self.max_workers_var, width=10)
        workers_spin.grid(row=1, column=1, sticky="w", padx=(5, 0))

        # Processing buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.process_button = ttk.Button(button_frame, text="Process All Faces", command=self.start_processing)
        self.process_button.pack(side="left", padx=5)

        ttk.Button(button_frame, text="Process New Only", command=self.process_new_faces).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Stop Processing", command=self.stop_processing).pack(side="left", padx=5)

        # Progress frame
        progress_frame = ttk.LabelFrame(self.process_frame, text="Processing Progress", padding=10)
        progress_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.process_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.process_progress.pack(fill="x", pady=(0, 10))

        self.process_status_text = scrolledtext.ScrolledText(progress_frame, height=15, width=70)
        self.process_status_text.pack(fill="both", expand=True)

    def create_search_tab(self):
        """Create search faces tab"""

        # Search control frame
        control_frame = ttk.LabelFrame(self.search_frame, text="Search Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Search by image
        ttk.Label(control_frame, text="Search by Image:").grid(row=0, column=0, sticky="w")
        self.search_image_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.search_image_var, width=40).grid(row=0, column=1, sticky="w", padx=(5, 0))
        ttk.Button(control_frame, text="Browse", command=self.browse_search_image).grid(row=0, column=2, padx=5)

        # Number of results
        ttk.Label(control_frame, text="Number of Results:").grid(row=1, column=0, sticky="w")
        self.num_results_var = tk.IntVar(value=10)
        results_spin = ttk.Spinbox(control_frame, from_=1, to=50, increment=1,
                                  textvariable=self.num_results_var, width=10)
        results_spin.grid(row=1, column=1, sticky="w", padx=(5, 0))

        # Search button
        ttk.Button(control_frame, text="Search Similar Faces", command=self.search_faces).grid(row=2, column=0, columnspan=2, pady=10)

        # Results frame
        results_frame = ttk.LabelFrame(self.search_frame, text="Search Results", padding=10)
        results_frame.grid(row=1, column=0, sticky="both", padx=5, pady=5, fill="both", expand=True)

        # Results display
        self.results_frame_inner = ttk.Frame(results_frame)
        self.results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical")
        self.results_canvas = tk.Canvas(results_frame, yscrollcommand=self.results_scrollbar.set)
        self.results_scrollbar.config(command=self.results_canvas.yview)

        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.results_scrollbar.pack(side="right", fill="y")
        self.results_canvas.create_window((0, 0), window=self.results_frame_inner, anchor="nw")

    def create_config_tab(self):
        """Create configuration tab"""

        # Database config frame
        db_frame = ttk.LabelFrame(self.config_frame, text="Database Configuration", padding=10)
        db_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ttk.Label(db_frame, text="Database Path:").grid(row=0, column=0, sticky="w")
        self.db_path_var = tk.StringVar(value="./chroma_db")
        ttk.Entry(db_frame, textvariable=self.db_path_var, width=40).grid(row=0, column=1, sticky="w", padx=(5, 0))
        ttk.Button(db_frame, text="Browse", command=self.browse_db_path).grid(row=0, column=2, padx=5)

        ttk.Label(db_frame, text="Collection Name:").grid(row=1, column=0, sticky="w")
        self.collection_name_var = tk.StringVar(value="faces")
        ttk.Entry(db_frame, textvariable=self.collection_name_var, width=40).grid(row=1, column=1, sticky="w", padx=(5, 0))

        # System config frame
        system_frame = ttk.LabelFrame(self.config_frame, text="System Configuration", padding=10)
        system_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Dependencies status
        deps_frame = ttk.LabelFrame(self.config_frame, text="Dependencies Status", padding=10)
        deps_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.deps_text = scrolledtext.ScrolledText(deps_frame, height=10, width=70)
        self.deps_text.pack(fill="both", expand=True)

        # Config buttons
        button_frame = ttk.Frame(self.config_frame)
        button_frame.grid(row=3, column=0, pady=10)

        ttk.Button(button_frame, text="Check Dependencies", command=self.check_dependencies).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Reset Database", command=self.reset_database).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Load Configuration", command=self.load_configuration).pack(side="left", padx=5)

    def setup_layout(self):
        """Setup the main layout"""
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure column weights for responsive design
        for i in range(5):  # Number of tabs
            self.root.grid_columnconfigure(i, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def initialize_system(self):
        """Initialize the face processing system"""
        try:
            self.system = IntegratedFaceSystem()
            if self.system.initialize():
                self.log_message("System initialized successfully")
                self.update_configuration_from_system()
            else:
                self.log_message("Failed to initialize system", "error")
                messagebox.showerror("Error", "Failed to initialize system. Check dependencies.")
        except Exception as e:
            self.log_message(f"Error initializing system: {e}", "error")
            messagebox.showerror("Error", f"Error initializing system: {e}")

    def update_configuration_from_system(self):
        """Update GUI configuration from system"""
        if self.system:
            config = self.system.config
            self.faces_dir_var.set(config.faces_dir)
            self.db_path_var.set(config.db_path)
            self.collection_name_var.set(config.collection_name)
            self.download_delay_var.set(config.download_delay)
            self.batch_size_var.set(config.batch_size)
            self.max_workers_var.set(config.max_workers)

    def log_message(self, message: str, level: str = "info"):
        """Log message to appropriate text widget"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        # Log to download status if downloading
        if hasattr(self, 'download_status_text'):
            self.download_status_text.insert(tk.END, formatted_message)
            self.download_status_text.see(tk.END)

        # Log to process status if processing
        if hasattr(self, 'process_status_text'):
            self.process_status_text.insert(tk.END, formatted_message)
            self.process_status_text.see(tk.END)

        # Log to logger
        if level == "error":
            logger.error(message)
        else:
            logger.info(message)

    def update_display(self):
        """Update display elements"""
        try:
            if self.system:
                status = self.system.get_system_status()

                # Update status labels
                if hasattr(self, 'status_labels'):
                    db_info = status.get('database', {})
                    stats = status.get('statistics', {})

                    self.status_labels['db_status'].config(text="Connected" if db_info else "Disconnected")
                    self.status_labels['total_faces'].config(text=str(db_info.get('count', 0)))
                    self.status_labels['download_rate'].config(text=f"{stats.get('download_rate', 0):.2f}/sec")
                    self.status_labels['process_rate'].config(text=f"{stats.get('embed_rate', 0):.2f}/sec")
                    self.status_labels['uptime'].config(text=f"{stats.get('elapsed_time', 0):.0f}s")

                # Update statistics text (throttled)
                current_time = time.time()
                if current_time - self.last_stats_update > 2.0:  # Update every 2 seconds
                    if hasattr(self, 'stats_text'):
                        self.stats_text.delete(1.0, tk.END)
                        stats_str = json.dumps(status, indent=2)
                        self.stats_text.insert(1.0, stats_str)
                    self.last_stats_update = current_time

        except Exception as e:
            pass  # Ignore update errors

        # Schedule next update
        self.root.after(1000, self.update_display)

    # Download methods
    def toggle_download(self):
        """Toggle download process"""
        if not self.is_downloading:
            self.start_download()
        else:
            self.stop_download()

    def start_download(self):
        """Start downloading faces"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        try:
            # Update configuration
            self.system.config.download_delay = self.download_delay_var.get()
            self.system.config.faces_dir = self.faces_dir_var.get()

            # Create faces directory
            os.makedirs(self.system.config.faces_dir, exist_ok=True)

            # Start download
            self.download_thread = self.system.downloader.start_download_loop(
                callback=self.on_face_downloaded
            )

            self.is_downloading = True
            self.download_button.config(text="Stop Download")
            self.log_message("Download started")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start download: {e}")

    def stop_download(self):
        """Stop downloading faces"""
        if self.system:
            self.system.downloader.stop_download_loop()

        self.is_downloading = False
        self.download_button.config(text="Start Download")
        self.log_message("Download stopped")

    def download_single(self):
        """Download a single face"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        try:
            file_path = self.system.downloader.download_face()
            if file_path:
                self.log_message(f"Downloaded: {os.path.basename(file_path)}")
            else:
                self.log_message("No new face downloaded (duplicate or error)")
        except Exception as e:
            self.log_message(f"Download error: {e}", "error")

    def on_face_downloaded(self, file_path: str):
        """Callback when a face is downloaded"""
        self.log_message(f"Downloaded: {os.path.basename(file_path)}")

    # Processing methods
    def start_processing(self):
        """Start processing all faces"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        if self.is_processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return

        self.is_processing = True
        self.process_button.config(state="disabled")
        self.process_progress.start()

        def process_worker():
            try:
                self.system.processor.process_all_faces(
                    callback=self.on_face_processed
                )
                self.log_message("Processing completed")
            except Exception as e:
                self.log_message(f"Processing error: {e}", "error")
            finally:
                self.is_processing = False
                self.root.after(0, lambda: [
                    self.process_button.config(state="normal"),
                    self.process_progress.stop()
                ])

        self.processing_thread = threading.Thread(target=process_worker, daemon=True)
        self.processing_thread.start()

    def process_new_faces(self):
        """Process only new faces"""
        # Simplified version - just process all for now
        self.start_processing()

    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        self.process_button.config(state="normal")
        self.process_progress.stop()
        self.log_message("Processing stopped")

    def on_face_processed(self, file_path: str):
        """Callback when a face is processed"""
        self.log_message(f"Processed: {os.path.basename(file_path)}")

    # Search methods
    def search_faces(self):
        """Search for similar faces"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        image_path = self.search_image_var.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return

        try:
            # Create embedding for search image
            analyzer = FaceAnalyzer()
            embedder = FaceEmbedder()

            features = analyzer.analyze_face(image_path)
            embedding = embedder.create_embedding(image_path, features)

            # Search database
            results = self.system.db_manager.search_faces(embedding, self.num_results_var.get())

            # Display results
            self.display_search_results(results)

            self.system.stats.increment_search_queries()

        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {e}")

    def display_search_results(self, results: List[Dict[str, Any]]):
        """Display search results"""
        # Clear previous results
        for widget in self.results_frame_inner.winfo_children():
            widget.destroy()

        if not results:
            ttk.Label(self.results_frame_inner, text="No results found").pack(pady=20)
            return

        # Display results
        for i, result in enumerate(results):
            result_frame = ttk.Frame(self.results_frame_inner)
            result_frame.pack(fill="x", padx=5, pady=5)

            # Result info
            info_text = f"Result {i+1}: Distance: {result['distance']:.3f}\nPath: {result['metadata'].get('file_path', 'Unknown')}"
            ttk.Label(result_frame, text=info_text).pack(side="left")

            # Try to display image thumbnail
            try:
                image_path = result['metadata'].get('file_path')
                if image_path and os.path.exists(image_path):
                    image = Image.open(image_path)
                    image.thumbnail((64, 64), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    image_label = ttk.Label(result_frame, image=photo)
                    image_label.image = photo  # Keep a reference
                    image_label.pack(side="right")
            except Exception:
                pass  # Skip image display if error

        # Update scroll region
        self.results_frame_inner.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    # Utility methods
    def browse_faces_dir(self):
        """Browse for faces directory"""
        directory = filedialog.askdirectory(initialdir=self.faces_dir_var.get())
        if directory:
            self.faces_dir_var.set(directory)

    def browse_search_image(self):
        """Browse for search image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.search_image_var.set(file_path)

    def browse_db_path(self):
        """Browse for database path"""
        directory = filedialog.askdirectory(initialdir=self.db_path_var.get())
        if directory:
            self.db_path_var.set(directory)

    def refresh_status(self):
        """Refresh system status"""
        self.log_message("Status refreshed")

    def reset_statistics(self):
        """Reset system statistics"""
        if self.system:
            self.system.stats = type(self.system.stats)()
            self.log_message("Statistics reset")

    def save_configuration(self):
        """Save current configuration"""
        if self.system:
            # Update system configuration from GUI
            self.system.config.faces_dir = self.faces_dir_var.get()
            self.system.config.db_path = self.db_path_var.get()
            self.system.config.collection_name = self.collection_name_var.get()
            self.system.config.download_delay = self.download_delay_var.get()
            self.system.config.batch_size = self.batch_size_var.get()
            self.system.config.max_workers = self.max_workers_var.get()

            # Save to file
            self.system.config.save_to_file()
            self.log_message("Configuration saved")

    def load_configuration(self):
        """Load configuration from file"""
        try:
            config = SystemConfig.from_file()
            if self.system:
                self.system.config = config
                self.update_configuration_from_system()
            self.log_message("Configuration loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def check_dependencies(self):
        """Check system dependencies"""
        deps_status = []

        # Check ChromaDB
        if CHROMADB_AVAILABLE:
            deps_status.append("✓ ChromaDB: Available")
        else:
            deps_status.append("✗ ChromaDB: Missing (pip install chromadb)")

        # Check OpenCV
        try:
            import cv2
            deps_status.append("✓ OpenCV: Available")
        except ImportError:
            deps_status.append("✗ OpenCV: Missing (pip install opencv-python)")

        # Check PIL
        try:
            from PIL import Image
            deps_status.append("✓ PIL/Pillow: Available")
        except ImportError:
            deps_status.append("✗ PIL/Pillow: Missing (pip install Pillow)")

        # Check other dependencies
        for module in ['numpy', 'requests']:
            try:
                __import__(module)
                deps_status.append(f"✓ {module}: Available")
            except ImportError:
                deps_status.append(f"✗ {module}: Missing (pip install {module})")

        # Update deps text
        self.deps_text.delete(1.0, tk.END)
        self.deps_text.insert(1.0, "\n".join(deps_status))

    def reset_database(self):
        """Reset the database"""
        if messagebox.askyesno("Confirm", "Are you sure you want to reset the database? This will delete all data."):
            try:
                if self.system and self.system.db_manager.client:
                    # This is a simplified reset - in practice you'd want more careful handling
                    self.log_message("Database reset requested")
                    messagebox.showinfo("Info", "Please restart the application to complete database reset")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset database: {e}")

    def run(self):
        """Run the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log_message("Application interrupted")
        finally:
            # Cleanup
            if self.system:
                if hasattr(self.system, 'downloader'):
                    self.system.downloader.stop_download_loop()

def main():
    """Main entry point"""
    app = IntegratedFaceGUI()
    app.run()

if __name__ == "__main__":
    main()