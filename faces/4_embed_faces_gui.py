#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import sys
import io
from datetime import datetime
from typing import Optional
import shutil
import numpy as np

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass  # If already wrapped or not available, skip

# Import the embedding system components
from importlib import import_module
import importlib.util

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the VectorEmbeddingProcessor
try:
    spec = importlib.util.spec_from_file_location("embedding",
                                                   os.path.join(os.path.dirname(__file__), "4_embed_faces.py"))
    embedding_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(embedding_module)

    VectorEmbeddingProcessor = embedding_module.VectorEmbeddingProcessor
except Exception as e:
    print(f"Error importing embedding module: {e}")
    print("Make sure 4_embed_faces.py is in the same directory")
    sys.exit(1)

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

        # Refresh collections on startup
        self.refresh_collections()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Create GUI widgets"""
        # Create canvas and scrollbar for entire window
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)

        # Main frame inside canvas
        main_frame = ttk.Frame(canvas, padding="10")

        # Configure canvas
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrollbar and canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create window in canvas
        canvas_frame = canvas.create_window((0, 0), window=main_frame, anchor="nw")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Update scrollregion when frame changes size
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        main_frame.bind("<Configure>", on_frame_configure)

        # Update canvas window width when canvas is resized
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_frame, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

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

        # Collection name selection
        ttk.Label(config_frame, text="Collection Name:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.collection_name_var = tk.StringVar(value="faces")
        collection_frame = ttk.Frame(config_frame)
        collection_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        collection_frame.columnconfigure(0, weight=1)

        self.collection_entry = ttk.Entry(collection_frame, textvariable=self.collection_name_var)
        self.collection_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(collection_frame, text="Refresh", command=self.refresh_collections).grid(row=0, column=1)

        # Available collections dropdown
        ttk.Label(config_frame, text="Existing Collections:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.available_collections_var = tk.StringVar()
        self.collections_combo = ttk.Combobox(config_frame, textvariable=self.available_collections_var,
                                             state="readonly", width=30)
        self.collections_combo.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=2)
        self.collections_combo.bind('<<ComboboxSelected>>', self.on_collection_selected)

        # Clear existing option
        self.clear_existing_var = tk.BooleanVar(value=False)
        clear_check = ttk.Checkbutton(config_frame, text="Clear Existing Embeddings",
                                     variable=self.clear_existing_var)
        clear_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)

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
        self.db_info_button.pack(side=tk.LEFT, padx=(0, 10))

        self.delete_collection_button = ttk.Button(control_frame, text="🗑️ Delete Collection",
                                                 command=self.delete_collection)
        self.delete_collection_button.pack(side=tk.LEFT, padx=(0, 10))

        self.show_embeddings_button = ttk.Button(control_frame, text="🔍 View Embeddings",
                                               command=self.show_embedding_viewer)
        self.show_embeddings_button.pack(side=tk.LEFT)

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
            ("Metadata Loaded", "metadata_loaded"),
            ("Metadata Missing", "metadata_missing"),
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

        # Refresh collections after embedding
        self.refresh_collections()

        # Show completion statistics
        self.show_completion_statistics()

    def show_completion_statistics(self):
        """Show detailed statistics after embedding completion"""
        try:
            # Create completion statistics window
            stats_window = tk.Toplevel(self.root)
            stats_window.title("🎉 Embedding Completed - Statistics")
            stats_window.geometry("700x500")
            stats_window.resizable(True, True)

            # Main frame
            main_frame = ttk.Frame(stats_window, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Title
            title_label = ttk.Label(main_frame, text="🎉 Embedding Process Completed!",
                                   font=("Arial", 16, "bold"))
            title_label.pack(pady=(0, 20))

            # Statistics text area
            stats_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=("Consolas", 10))
            stats_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

            # Generate completion statistics
            self._generate_completion_stats(stats_text)

            # Action buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(10, 0))

            ttk.Button(button_frame, text="📊 View Full Database Info",
                      command=self.show_db_info).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="📁 Open Faces Folder",
                      command=self.open_faces_folder).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="✅ Close",
                      command=stats_window.destroy).pack(side=tk.RIGHT)

        except Exception as e:
            self.log(f"Error showing completion statistics: {e}")

    def _generate_completion_stats(self, text_widget):
        """Generate detailed completion statistics"""
        try:
            import chromadb
            import os
            from datetime import datetime

            text_widget.insert(tk.END, f"📊 EMBEDDING COMPLETION REPORT\n")
            text_widget.insert(tk.END, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            text_widget.insert(tk.END, "=" * 60 + "\n\n")

            # Processor statistics
            if self.processor:
                stats = self.processor.get_stats()
                text_widget.insert(tk.END, "🎯 PROCESSING SUMMARY\n")
                text_widget.insert(tk.END, "-" * 30 + "\n")
                text_widget.insert(tk.END, f"📁 Files Processed: {stats.get('processed_files', 0):,}\n")
                text_widget.insert(tk.END, f"✅ Successful Embeddings: {stats.get('successful_embeddings', 0):,}\n")
                text_widget.insert(tk.END, f"📋 Metadata Loaded: {stats.get('metadata_loaded', 0):,}\n")
                text_widget.insert(tk.END, f"⚠️  Metadata Missing: {stats.get('metadata_missing', 0):,}\n")
                text_widget.insert(tk.END, f"⏭️  Duplicates Skipped: {stats.get('duplicates_skipped', 0):,}\n")
                text_widget.insert(tk.END, f"❌ Errors: {stats.get('errors', 0):,}\n")
                text_widget.insert(tk.END, f"⏱️  Total Time: {stats.get('elapsed_time', 0):.1f} seconds\n")
                text_widget.insert(tk.END, f"⚡ Processing Rate: {stats.get('processing_rate', 0):.2f} files/sec\n\n")

            # Database statistics
            try:
                client = chromadb.PersistentClient(path="./chroma_db")
                collections = client.list_collections()

                text_widget.insert(tk.END, "🗄️  DATABASE STATISTICS\n")
                text_widget.insert(tk.END, "-" * 30 + "\n")
                text_widget.insert(tk.END, f"📚 Total Collections: {len(collections)}\n")

                total_vectors = 0
                for collection in collections:
                    try:
                        count = collection.count()
                        total_vectors += count
                        text_widget.insert(tk.END, f"   • {collection.name}: {count:,} vectors\n")
                    except Exception as e:
                        text_widget.insert(tk.END, f"   • {collection.name}: Error ({e})\n")

                text_widget.insert(tk.END, f"\n🔢 Total Vectors: {total_vectors:,}\n")

                # Collection-specific analysis for faces
                if any(col.name == "faces" for col in collections):
                    faces_collection = client.get_collection("faces")
                    faces_count = faces_collection.count()

                    text_widget.insert(tk.END, f"\n🎭 FACE COLLECTION ANALYSIS\n")
                    text_widget.insert(tk.END, "-" * 30 + "\n")
                    text_widget.insert(tk.END, f"👥 Total Face Vectors: {faces_count:,}\n")

                    if faces_count > 0:
                        # Sample analysis
                        sample_size = min(100, faces_count)
                        results = faces_collection.get(limit=sample_size, include=['metadatas'])

                        if results['metadatas']:
                            age_groups = {}
                            skin_tones = {}
                            qualities = {}

                            for metadata in results['metadatas']:
                                age_group = metadata.get('estimated_age_group', 'unknown')
                                age_groups[age_group] = age_groups.get(age_group, 0) + 1

                                skin_tone = metadata.get('estimated_skin_tone', 'unknown')
                                skin_tones[skin_tone] = skin_tones.get(skin_tone, 0) + 1

                                quality = metadata.get('image_quality', 'unknown')
                                qualities[quality] = qualities.get(quality, 0) + 1

                            text_widget.insert(tk.END, f"🎂 Age Distribution: {dict(sorted(age_groups.items()))}\n")
                            text_widget.insert(tk.END, f"🎨 Skin Tone Distribution: {dict(sorted(skin_tones.items()))}\n")
                            text_widget.insert(tk.END, f"📸 Quality Distribution: {dict(sorted(qualities.items()))}\n")

            except Exception as e:
                text_widget.insert(tk.END, f"⚠️ Error getting database statistics: {e}\n")

            # Storage information
            try:
                db_path = "./chroma_db"
                if os.path.exists(db_path):
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(db_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            total_size += os.path.getsize(filepath)

                    size_mb = total_size / (1024 * 1024)
                    size_gb = size_mb / 1024

                    text_widget.insert(tk.END, f"\n💾 STORAGE INFORMATION\n")
                    text_widget.insert(tk.END, "-" * 30 + "\n")
                    text_widget.insert(tk.END, f"📁 Database Size: {size_mb:.2f} MB ({size_gb:.3f} GB)\n")
                    text_widget.insert(tk.END, f"📂 Database Path: {os.path.abspath(db_path)}\n")

                    if total_vectors > 0:
                        avg_size_per_vector = (total_size / total_vectors) / 1024
                        text_widget.insert(tk.END, f"📏 Average Size per Vector: {avg_size_per_vector:.2f} KB\n")

            except Exception as e:
                text_widget.insert(tk.END, f"⚠️ Error getting storage information: {e}\n")

            # Recommendations
            text_widget.insert(tk.END, f"\n💡 RECOMMENDATIONS\n")
            text_widget.insert(tk.END, "-" * 30 + "\n")
            text_widget.insert(tk.END, "• Use the 'Database Info' button for detailed analysis\n")
            text_widget.insert(tk.END, "• Check the Collections tab to explore individual collections\n")
            text_widget.insert(tk.END, "• Consider backing up your database after successful embedding\n")
            text_widget.insert(tk.END, "• Monitor storage space as your database grows\n")

            text_widget.config(state=tk.DISABLED)

        except Exception as e:
            text_widget.insert(tk.END, f"❌ Error generating completion statistics: {e}")
            text_widget.config(state=tk.DISABLED)

    def delete_collection(self):
        """Delete a selected collection"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            collections = client.list_collections()

            if not collections:
                messagebox.showinfo("No Collections", "No collections available to delete.")
                return

            # Create selection dialog
            delete_window = tk.Toplevel(self.root)
            delete_window.title("🗑️ Delete Collection")
            delete_window.geometry("550x450")
            delete_window.resizable(True, True)

            # Center the window
            delete_window.transient(self.root)
            delete_window.grab_set()

            main_frame = ttk.Frame(delete_window, padding="20")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Warning label
            warning_label = ttk.Label(main_frame, text="⚠️ WARNING: This action cannot be undone!",
                                    font=("Arial", 12, "bold"), foreground="red")
            warning_label.pack(pady=(0, 10))

            # Instructions
            ttk.Label(main_frame, text="Select a collection to delete:").pack(pady=(0, 10))

            # Collection listbox
            collection_frame = ttk.Frame(main_frame)
            collection_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

            scrollbar = ttk.Scrollbar(collection_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            collection_listbox = tk.Listbox(collection_frame, yscrollcommand=scrollbar.set)
            collection_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=collection_listbox.yview)

            # Populate collections
            collection_info = []
            for collection in collections:
                try:
                    count = collection.count()
                    info = f"{collection.name} ({count:,} vectors)"
                    collection_info.append((collection.name, info))
                    collection_listbox.insert(tk.END, info)
                except:
                    info = f"{collection.name} (unknown count)"
                    collection_info.append((collection.name, info))
                    collection_listbox.insert(tk.END, info)

            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X)

            def confirm_delete():
                selection = collection_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a collection to delete.")
                    return

                selected_idx = selection[0]
                collection_name = collection_info[selected_idx][0]

                # Final confirmation
                confirm = messagebox.askyesno(
                    "Confirm Deletion",
                    f"Are you sure you want to delete the collection '{collection_name}'?\n\n"
                    f"This will permanently remove all vectors and data in this collection.",
                    icon="warning"
                )

                if confirm:
                    try:
                        client.delete_collection(collection_name)
                        self.log(f"✅ Collection '{collection_name}' deleted successfully")
                        messagebox.showinfo("Success", f"Collection '{collection_name}' has been deleted.")

                        # Refresh collections in main GUI
                        self.refresh_collections()
                        delete_window.destroy()

                    except Exception as e:
                        error_msg = f"Failed to delete collection '{collection_name}': {e}"
                        self.log(f"❌ {error_msg}")
                        messagebox.showerror("Error", error_msg)

            ttk.Button(button_frame, text="🗑️ Delete Selected",
                      command=confirm_delete, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="Cancel",
                      command=delete_window.destroy).pack(side=tk.RIGHT)

        except Exception as e:
            error_msg = f"Error accessing collections: {e}"
            self.log(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)

    def show_embedding_viewer(self):
        """Show real-time embedding viewer"""
        try:
            # Create embedding viewer window
            embedding_window = tk.Toplevel(self.root)
            embedding_window.title("🔍 Real-time Embedding Viewer")
            embedding_window.geometry("900x600")
            embedding_window.resizable(True, True)

            main_frame = ttk.Frame(embedding_window, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Title
            title_label = ttk.Label(main_frame, text="🔍 Live Embedding Vector Display",
                                   font=("Arial", 14, "bold"))
            title_label.pack(pady=(0, 10))

            # Control frame
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(fill=tk.X, pady=(0, 10))

            # Collection selector
            ttk.Label(control_frame, text="Collection:").pack(side=tk.LEFT, padx=(0, 5))
            collection_var = tk.StringVar()
            collection_combo = ttk.Combobox(control_frame, textvariable=collection_var, state="readonly", width=20)
            collection_combo.pack(side=tk.LEFT, padx=(0, 10))

            # Populate collections
            try:
                import chromadb
                client = chromadb.PersistentClient(path="./chroma_db")
                collections = client.list_collections()
                collection_names = [col.name for col in collections]
                collection_combo['values'] = collection_names
                if collection_names:
                    collection_combo.set(collection_names[0])
            except:
                pass

            # Embedding display frame
            display_frame = ttk.LabelFrame(main_frame, text="Recent Embeddings", padding="10")
            display_frame.pack(fill=tk.BOTH, expand=True)

            # Create notebook for different views
            notebook = ttk.Notebook(display_frame)
            notebook.pack(fill=tk.BOTH, expand=True)

            # Vector Display Tab
            vector_frame = ttk.Frame(notebook)
            notebook.add(vector_frame, text="📊 Vector Values")

            vector_text = scrolledtext.ScrolledText(vector_frame, wrap=tk.WORD, font=("Consolas", 9))
            vector_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Visualization Tab
            viz_frame = ttk.Frame(notebook)
            notebook.add(viz_frame, text="📈 Visualization")

            viz_text = scrolledtext.ScrolledText(viz_frame, wrap=tk.WORD, font=("Consolas", 9))
            viz_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Statistics Tab
            stats_frame = ttk.Frame(notebook)
            notebook.add(stats_frame, text="📊 Statistics")

            stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, font=("Consolas", 9))
            stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            def update_embedding_display():
                """Update the embedding display with latest data"""
                try:
                    selected_collection = collection_var.get()
                    if not selected_collection:
                        return

                    import chromadb
                    client = chromadb.PersistentClient(path="./chroma_db")
                    collection = client.get_collection(selected_collection)

                    # Get recent embeddings
                    results = collection.get(limit=10, include=['embeddings', 'metadatas', 'documents'])

                    # Update vector display
                    vector_text.delete(1.0, tk.END)
                    if results.get('embeddings') is not None and len(results['embeddings']) > 0:
                        vector_text.insert(tk.END, f"🔍 LATEST EMBEDDINGS FROM '{selected_collection}'\n")
                        vector_text.insert(tk.END, "=" * 80 + "\n\n")

                        for i, embedding in enumerate(results['embeddings']):
                            if embedding is not None and len(embedding) > 0:
                                metadata = results['metadatas'][i] if results.get('metadatas') is not None and i < len(results['metadatas']) else {}

                                vector_text.insert(tk.END, f"📄 Vector #{i+1}\n")
                                vector_text.insert(tk.END, "-" * 80 + "\n")

                                # Show metadata information
                                if metadata:
                                    file_path = metadata.get('file_path', 'Unknown')
                                    vector_text.insert(tk.END, f"   File: {os.path.basename(file_path) if file_path != 'Unknown' else 'Unknown'}\n")

                                    # Show additional metadata if available
                                    if 'estimated_age_group' in metadata:
                                        vector_text.insert(tk.END, f"   Age Group: {metadata.get('estimated_age_group', 'N/A')}\n")
                                    if 'estimated_skin_tone' in metadata:
                                        vector_text.insert(tk.END, f"   Skin Tone: {metadata.get('estimated_skin_tone', 'N/A')}\n")
                                    if 'image_quality' in metadata:
                                        vector_text.insert(tk.END, f"   Quality: {metadata.get('image_quality', 'N/A')}\n")

                                # Vector statistics
                                vector_array = np.array(embedding)
                                vector_text.insert(tk.END, f"\n   📊 Vector Properties:\n")
                                vector_text.insert(tk.END, f"      • Dimensions: {len(embedding)}\n")
                                vector_text.insert(tk.END, f"      • L2 Norm: {np.linalg.norm(vector_array):.6f}\n")
                                vector_text.insert(tk.END, f"      • Mean: {np.mean(vector_array):.6f}\n")
                                vector_text.insert(tk.END, f"      • Std Dev: {np.std(vector_array):.6f}\n")
                                vector_text.insert(tk.END, f"      • Min: {np.min(vector_array):.6f}\n")
                                vector_text.insert(tk.END, f"      • Max: {np.max(vector_array):.6f}\n")

                                # Show vector preview with better formatting
                                vector_text.insert(tk.END, f"\n   🔢 Vector Values (first 10 and last 10 dimensions):\n")
                                if len(embedding) >= 20:
                                    vector_text.insert(tk.END, f"      First 10:  [{', '.join(f'{x:7.4f}' for x in embedding[:10])}]\n")
                                    vector_text.insert(tk.END, f"      Last 10:   [{', '.join(f'{x:7.4f}' for x in embedding[-10:])}]\n")
                                else:
                                    vector_text.insert(tk.END, f"      All values: [{', '.join(f'{x:7.4f}' for x in embedding)}]\n")

                                vector_text.insert(tk.END, "\n")
                    else:
                        vector_text.insert(tk.END, f"❌ No embeddings found in collection '{selected_collection}'\n")
                        vector_text.insert(tk.END, "\nThis could mean:\n")
                        vector_text.insert(tk.END, "  • The collection is empty\n")
                        vector_text.insert(tk.END, "  • Embeddings were not stored during the embedding process\n")
                        vector_text.insert(tk.END, "  • The collection name is incorrect\n")

                    # Update visualization
                    viz_text.delete(1.0, tk.END)
                    try:
                        if results.get('embeddings') is not None and len(results['embeddings']) > 0:
                            viz_text.insert(tk.END, f"📈 EMBEDDING VISUALIZATION\n")
                            viz_text.insert(tk.END, "=" * 80 + "\n\n")

                            for i, embedding in enumerate(results['embeddings'][:5]):  # Show first 5
                                if embedding is not None and len(embedding) > 0:
                                    vector_array = np.array(embedding)

                                    # Metadata info
                                    metadata = results['metadatas'][i] if results.get('metadatas') is not None and i < len(results['metadatas']) else {}
                                    file_name = "Unknown"
                                    if metadata:
                                        file_path = metadata.get('file_path', 'Unknown')
                                        file_name = os.path.basename(file_path) if file_path != 'Unknown' else 'Unknown'

                                    viz_text.insert(tk.END, f"📊 Vector #{i+1}: {file_name}\n")
                                    viz_text.insert(tk.END, "-" * 80 + "\n")

                                    # Show overall distribution first
                                    viz_text.insert(tk.END, f"Overall Statistics:\n")
                                    viz_text.insert(tk.END, f"  Range: [{np.min(vector_array):.4f}, {np.max(vector_array):.4f}]\n")
                                    viz_text.insert(tk.END, f"  Mean: {np.mean(vector_array):.4f}, Std: {np.std(vector_array):.4f}\n\n")

                                    # Create scaled ASCII bar chart (first 30 dimensions)
                                    viz_text.insert(tk.END, f"First 30 dimensions (scaled to max={np.max(np.abs(vector_array[:30])):.3f}):\n")

                                    # Find max absolute value for scaling
                                    max_abs_val = np.max(np.abs(vector_array[:30])) if len(vector_array) >= 30 else np.max(np.abs(vector_array))
                                    if max_abs_val == 0:
                                        max_abs_val = 1.0  # Avoid division by zero

                                    for j, val in enumerate(embedding[:30]):
                                        # Scale to 40 character width
                                        normalized_val = val / max_abs_val
                                        bar_length = int(abs(normalized_val) * 40)
                                        bar_length = max(1, bar_length) if abs(val) > 0.001 else 0

                                        bar = "█" * bar_length
                                        sign = "+" if val >= 0 else "-"

                                        # Color coding with text
                                        magnitude = abs(val)
                                        if magnitude > max_abs_val * 0.7:
                                            intensity = "HIGH"
                                        elif magnitude > max_abs_val * 0.3:
                                            intensity = "MED "
                                        else:
                                            intensity = "LOW "

                                        viz_text.insert(tk.END, f"  [{j:3d}] {sign} {bar:<40} {val:8.4f} ({intensity})\n")

                                    # Show distribution histogram
                                    viz_text.insert(tk.END, f"\nValue Distribution (all {len(embedding)} dimensions):\n")
                                    bins = [-float('inf'), -0.5, -0.2, -0.05, 0, 0.05, 0.2, 0.5, float('inf')]
                                    bin_labels = ["< -0.5", "-0.5 to -0.2", "-0.2 to -0.05", "-0.05 to 0", "0 to 0.05", "0.05 to 0.2", "0.2 to 0.5", "> 0.5"]
                                    hist, _ = np.histogram(vector_array, bins=bins)

                                    max_count = max(hist) if max(hist) > 0 else 1
                                    for label, count in zip(bin_labels, hist):
                                        bar_width = int((count / max_count) * 30)
                                        bar = "▓" * bar_width
                                        percentage = (count / len(vector_array)) * 100
                                        viz_text.insert(tk.END, f"  {label:>15s}: {bar:<30} {count:4d} ({percentage:5.1f}%)\n")

                                    viz_text.insert(tk.END, "\n")
                        else:
                            viz_text.insert(tk.END, f"❌ No embeddings available for visualization\n")
                    except Exception as viz_error:
                        viz_text.insert(tk.END, f"❌ Error in visualization: {viz_error}\n")
                        import traceback
                        viz_text.insert(tk.END, f"\n{traceback.format_exc()}\n")

                    # Update statistics
                    stats_text.delete(1.0, tk.END)
                    try:
                        if results.get('embeddings') is not None and len(results['embeddings']) > 0:
                            embeddings_array = np.array([emb for emb in results['embeddings'] if emb is not None and len(emb) > 0])
                            if len(embeddings_array) > 0:
                                stats_text.insert(tk.END, f"📊 COMPREHENSIVE EMBEDDING STATISTICS\n")
                                stats_text.insert(tk.END, "=" * 80 + "\n\n")

                                # Basic information
                                stats_text.insert(tk.END, f"🔢 Dataset Overview:\n")
                                stats_text.insert(tk.END, f"   Total Vectors Analyzed: {len(embeddings_array)}\n")
                                stats_text.insert(tk.END, f"   Dimensions per Vector: {embeddings_array.shape[1]}\n")
                                stats_text.insert(tk.END, f"   Total Data Points: {embeddings_array.shape[0] * embeddings_array.shape[1]:,}\n")
                                stats_text.insert(tk.END, f"   Memory Size: ~{(embeddings_array.nbytes / 1024):.2f} KB\n\n")

                                # Aggregate statistics across all vectors
                                stats_text.insert(tk.END, f"📈 Global Statistics (across all vectors and dimensions):\n")
                                all_values = embeddings_array.flatten()
                                stats_text.insert(tk.END, f"   Mean (overall): {np.mean(all_values):.6f}\n")
                                stats_text.insert(tk.END, f"   Std Dev (overall): {np.std(all_values):.6f}\n")
                                stats_text.insert(tk.END, f"   Min (overall): {np.min(all_values):.6f}\n")
                                stats_text.insert(tk.END, f"   Max (overall): {np.max(all_values):.6f}\n")
                                stats_text.insert(tk.END, f"   Median (overall): {np.median(all_values):.6f}\n")
                                stats_text.insert(tk.END, f"   25th Percentile: {np.percentile(all_values, 25):.6f}\n")
                                stats_text.insert(tk.END, f"   75th Percentile: {np.percentile(all_values, 75):.6f}\n\n")

                                # Per-vector statistics
                                stats_text.insert(tk.END, f"📊 Per-Vector Statistics:\n")
                                norms = np.linalg.norm(embeddings_array, axis=1)
                                stats_text.insert(tk.END, f"   L2 Norms - Mean: {np.mean(norms):.6f}, Std: {np.std(norms):.6f}\n")
                                stats_text.insert(tk.END, f"   L2 Norms - Min: {np.min(norms):.6f}, Max: {np.max(norms):.6f}\n\n")

                                # Per-dimension statistics summary
                                stats_text.insert(tk.END, f"📐 Per-Dimension Statistics (aggregated across {embeddings_array.shape[1]} dimensions):\n")
                                dim_means = np.mean(embeddings_array, axis=0)
                                dim_stds = np.std(embeddings_array, axis=0)
                                dim_mins = np.min(embeddings_array, axis=0)
                                dim_maxs = np.max(embeddings_array, axis=0)

                                stats_text.insert(tk.END, f"   Dimension Means - Avg: {np.mean(dim_means):.6f}, Std: {np.std(dim_means):.6f}\n")
                                stats_text.insert(tk.END, f"   Dimension Means - Range: [{np.min(dim_means):.6f}, {np.max(dim_means):.6f}]\n")
                                stats_text.insert(tk.END, f"   Dimension StdDevs - Avg: {np.mean(dim_stds):.6f}, Std: {np.std(dim_stds):.6f}\n")
                                stats_text.insert(tk.END, f"   Dimension StdDevs - Range: [{np.min(dim_stds):.6f}, {np.max(dim_stds):.6f}]\n\n")

                                # Show detailed statistics for first 10 dimensions
                                stats_text.insert(tk.END, f"🔍 Detailed Statistics for First 10 Dimensions:\n")
                                stats_text.insert(tk.END, f"{'Dim':>4} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}\n")
                                stats_text.insert(tk.END, "-" * 70 + "\n")
                                for i in range(min(10, embeddings_array.shape[1])):
                                    stats_text.insert(tk.END,
                                        f"{i:4d} | {dim_means[i]:10.6f} | {dim_stds[i]:10.6f} | "
                                        f"{dim_mins[i]:10.6f} | {dim_maxs[i]:10.6f}\n")

                                # Distribution analysis
                                stats_text.insert(tk.END, f"\n📊 Value Distribution Analysis:\n")
                                positive_count = np.sum(all_values > 0)
                                negative_count = np.sum(all_values < 0)
                                zero_count = np.sum(all_values == 0)
                                total_count = len(all_values)

                                stats_text.insert(tk.END, f"   Positive values: {positive_count:,} ({100*positive_count/total_count:.2f}%)\n")
                                stats_text.insert(tk.END, f"   Negative values: {negative_count:,} ({100*negative_count/total_count:.2f}%)\n")
                                stats_text.insert(tk.END, f"   Zero values: {zero_count:,} ({100*zero_count/total_count:.2f}%)\n\n")

                                # Sparsity analysis
                                near_zero_count = np.sum(np.abs(all_values) < 0.01)
                                stats_text.insert(tk.END, f"   Near-zero (|x| < 0.01): {near_zero_count:,} ({100*near_zero_count/total_count:.2f}%)\n")
                                stats_text.insert(tk.END, f"   Sparsity: {100*near_zero_count/total_count:.2f}%\n\n")

                                # Similarity analysis
                                if len(embeddings_array) > 1:
                                    stats_text.insert(tk.END, f"🔗 Vector Similarity Analysis:\n")
                                    # Calculate cosine similarity between pairs
                                    from sklearn.metrics.pairwise import cosine_similarity
                                    similarity_matrix = cosine_similarity(embeddings_array)
                                    # Get upper triangle (excluding diagonal)
                                    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                                    similarities = similarity_matrix[mask]

                                    stats_text.insert(tk.END, f"   Avg Cosine Similarity: {np.mean(similarities):.6f}\n")
                                    stats_text.insert(tk.END, f"   Std Cosine Similarity: {np.std(similarities):.6f}\n")
                                    stats_text.insert(tk.END, f"   Min Similarity: {np.min(similarities):.6f}\n")
                                    stats_text.insert(tk.END, f"   Max Similarity: {np.max(similarities):.6f}\n")
                            else:
                                stats_text.insert(tk.END, "❌ No valid embeddings to analyze\n")
                        else:
                            stats_text.insert(tk.END, "❌ No embeddings available for statistics\n")
                    except Exception as stats_error:
                        stats_text.insert(tk.END, f"❌ Error in statistics: {stats_error}\n")
                        import traceback
                        stats_text.insert(tk.END, f"\n{traceback.format_exc()}\n")

                except Exception as e:
                    vector_text.delete(1.0, tk.END)
                    vector_text.insert(tk.END, f"Error loading embeddings: {e}")

            # Control buttons
            ttk.Button(control_frame, text="🔄 Refresh",
                      command=update_embedding_display).pack(side=tk.LEFT, padx=(10, 0))

            # Auto-refresh checkbox
            auto_refresh_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(control_frame, text="Auto-refresh",
                           variable=auto_refresh_var).pack(side=tk.LEFT, padx=(10, 0))

            # Initial load
            update_embedding_display()

            # Auto-refresh timer
            def auto_refresh():
                if auto_refresh_var.get() and embedding_window.winfo_exists():
                    update_embedding_display()
                    embedding_window.after(5000, auto_refresh)  # Refresh every 5 seconds

            embedding_window.after(5000, auto_refresh)

        except Exception as e:
            self.log(f"Error showing embedding viewer: {e}")
            messagebox.showerror("Error", f"Error opening embedding viewer: {e}")

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

    def refresh_collections(self):
        """Refresh the list of available collections"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            collections = client.list_collections()

            collection_names = [col.name for col in collections]
            collection_info = []

            for col in collections:
                try:
                    count = col.count()
                    collection_info.append(f"{col.name} ({count:,} vectors)")
                except:
                    collection_info.append(f"{col.name} (unknown count)")

            self.collections_combo['values'] = collection_info
            self.log(f"Found {len(collections)} collections: {', '.join(collection_names)}")

        except Exception as e:
            self.log(f"Error refreshing collections: {e}")
            self.collections_combo['values'] = []

    def on_collection_selected(self, event):
        """Handle collection selection from dropdown"""
        selected = self.available_collections_var.get()
        if selected:
            # Extract collection name (before the parentheses)
            collection_name = selected.split(' (')[0]
            self.collection_name_var.set(collection_name)
            self.log(f"Selected collection: {collection_name}")

    def show_db_info(self):
        """Show enhanced database information"""
        try:
            # Run database info command
            import subprocess
            import sys

            # Use the same python interpreter that's running this script
            python_exe = sys.executable

            result = subprocess.run([python_exe, '2_database_info.py'],
                                  capture_output=True, text=True, timeout=30,
                                  encoding='utf-8', errors='replace')

            if result.returncode == 0:
                # Show in a new window with enhanced layout
                info_window = tk.Toplevel(self.root)
                info_window.title("📊 Enhanced Database Information")
                info_window.geometry("800x600")
                info_window.resizable(True, True)

                # Create notebook for tabbed interface
                notebook = ttk.Notebook(info_window)
                notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # Database Overview Tab
                overview_frame = ttk.Frame(notebook)
                notebook.add(overview_frame, text="📊 Overview")

                overview_text = scrolledtext.ScrolledText(overview_frame, wrap=tk.WORD, font=("Consolas", 10))
                overview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                overview_text.insert(tk.END, result.stdout)
                overview_text.config(state=tk.DISABLED)

                # Collection Details Tab
                collections_frame = ttk.Frame(notebook)
                notebook.add(collections_frame, text="🗂️ Collections")

                # Collection selection and details
                col_select_frame = ttk.Frame(collections_frame)
                col_select_frame.pack(fill=tk.X, padx=5, pady=5)

                ttk.Label(col_select_frame, text="Select Collection:").pack(side=tk.LEFT, padx=(0, 5))

                collection_var = tk.StringVar()
                collection_dropdown = ttk.Combobox(col_select_frame, textvariable=collection_var, state="readonly")
                collection_dropdown.pack(side=tk.LEFT, padx=(0, 5))

                def update_collection_details():
                    self._show_collection_details(collection_var.get(), collections_detail_text)

                ttk.Button(col_select_frame, text="Show Details", command=update_collection_details).pack(side=tk.LEFT)

                collections_detail_text = scrolledtext.ScrolledText(collections_frame, wrap=tk.WORD, font=("Consolas", 10))
                collections_detail_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                # Populate collection dropdown
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path="./chroma_db")
                    collections = client.list_collections()
                    collection_names = [col.name for col in collections]
                    collection_dropdown['values'] = collection_names
                    if collection_names:
                        collection_dropdown.set(collection_names[0])
                        self._show_collection_details(collection_names[0], collections_detail_text)
                except Exception as e:
                    collections_detail_text.insert(tk.END, f"Error loading collections: {e}")

                # Statistics Tab
                stats_frame = ttk.Frame(notebook)
                notebook.add(stats_frame, text="📈 Statistics")

                stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, font=("Consolas", 10))
                stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                # Generate statistics
                self._generate_statistics(stats_text)

            else:
                # Show error details
                error_msg = f"Failed to retrieve database information.\n\nReturn code: {result.returncode}\n\nError output:\n{result.stderr}"
                self.log(f"Error getting database info: {result.stderr}")
                messagebox.showerror("Error", error_msg)

        except subprocess.TimeoutExpired:
            self.log("Database info command timed out")
            messagebox.showerror("Error", "Database info command timed out after 30 seconds")
        except Exception as e:
            self.log(f"Error showing database info: {e}")
            messagebox.showerror("Error", f"Error showing database info: {e}")

    def _show_collection_details(self, collection_name, text_widget):
        """Show detailed information about a specific collection"""
        if not collection_name:
            return

        try:
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection(collection_name)

            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)

            # Basic collection info
            count = collection.count()
            metadata = collection.metadata or {}

            text_widget.insert(tk.END, f"🗂️ Collection: {collection_name}\n")
            text_widget.insert(tk.END, "=" * 50 + "\n\n")
            text_widget.insert(tk.END, f"📄 Total Documents: {count:,}\n")
            text_widget.insert(tk.END, f"🔢 Total Vectors: {count:,}\n\n")

            if metadata:
                text_widget.insert(tk.END, "🏷️ Collection Metadata:\n")
                for key, value in metadata.items():
                    text_widget.insert(tk.END, f"   • {key}: {value}\n")
                text_widget.insert(tk.END, "\n")

            # Sample data analysis
            if count > 0:
                try:
                    sample_size = min(10, count)
                    results = collection.peek(limit=sample_size)

                    if results['embeddings']:
                        dimensions = len(results['embeddings'][0])
                        text_widget.insert(tk.END, f"📐 Vector Dimensions: {dimensions}\n")

                        vector_size_mb = (count * dimensions * 4) / (1024 * 1024)
                        text_widget.insert(tk.END, f"💾 Estimated Vector Storage: {vector_size_mb:.2f} MB\n\n")

                    # Show sample IDs
                    if results['ids']:
                        text_widget.insert(tk.END, "🔍 Sample IDs:\n")
                        for i, id_ in enumerate(results['ids'][:5]):
                            text_widget.insert(tk.END, f"   {i+1}. {id_}\n")
                        text_widget.insert(tk.END, "\n")

                    # Analyze metadata for face collections
                    if collection_name == "faces" and results['metadatas']:
                        text_widget.insert(tk.END, "🎭 Face Feature Analysis:\n")

                        # Analyze larger sample for better statistics
                        larger_sample = min(100, count)
                        full_results = collection.get(limit=larger_sample, include=['metadatas'])

                        if full_results['metadatas']:
                            age_groups = {}
                            skin_tones = {}
                            qualities = {}

                            for metadata in full_results['metadatas']:
                                age_group = metadata.get('estimated_age_group', 'unknown')
                                age_groups[age_group] = age_groups.get(age_group, 0) + 1

                                skin_tone = metadata.get('estimated_skin_tone', 'unknown')
                                skin_tones[skin_tone] = skin_tones.get(skin_tone, 0) + 1

                                quality = metadata.get('image_quality', 'unknown')
                                qualities[quality] = qualities.get(quality, 0) + 1

                            text_widget.insert(tk.END, f"   🎂 Age Groups: {dict(sorted(age_groups.items()))}\n")
                            text_widget.insert(tk.END, f"   🎨 Skin Tones: {dict(sorted(skin_tones.items()))}\n")
                            text_widget.insert(tk.END, f"   📸 Qualities: {dict(sorted(qualities.items()))}\n")

                except Exception as e:
                    text_widget.insert(tk.END, f"⚠️ Could not analyze collection data: {e}\n")

            text_widget.config(state=tk.DISABLED)

        except Exception as e:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, f"❌ Error loading collection details: {e}")
            text_widget.config(state=tk.DISABLED)

    def _generate_statistics(self, text_widget):
        """Generate comprehensive database statistics"""
        try:
            import chromadb
            import os

            client = chromadb.PersistentClient(path="./chroma_db")
            collections = client.list_collections()

            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)

            text_widget.insert(tk.END, "📈 COMPREHENSIVE DATABASE STATISTICS\n")
            text_widget.insert(tk.END, "=" * 60 + "\n\n")

            total_vectors = 0
            total_size = 0

            # Collection statistics
            text_widget.insert(tk.END, "📊 Collection Summary:\n")
            for collection in collections:
                try:
                    count = collection.count()
                    total_vectors += count
                    text_widget.insert(tk.END, f"   • {collection.name}: {count:,} vectors\n")
                except Exception as e:
                    text_widget.insert(tk.END, f"   • {collection.name}: Error ({e})\n")

            text_widget.insert(tk.END, f"\n🔢 Total Vectors: {total_vectors:,}\n")
            text_widget.insert(tk.END, f"🗂️ Total Collections: {len(collections)}\n\n")

            # Database file statistics
            db_path = "./chroma_db"
            if os.path.exists(db_path):
                file_counts = {}

                for dirpath, dirnames, filenames in os.walk(db_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        file_size = os.path.getsize(filepath)
                        total_size += file_size

                        ext = os.path.splitext(filename)[1].lower()
                        file_counts[ext] = file_counts.get(ext, 0) + 1

                size_mb = total_size / (1024 * 1024)
                size_gb = size_mb / 1024

                text_widget.insert(tk.END, f"💾 Database Size: {size_mb:.2f} MB ({size_gb:.3f} GB)\n")
                text_widget.insert(tk.END, f"📁 Database Path: {os.path.abspath(db_path)}\n\n")

                if file_counts:
                    text_widget.insert(tk.END, "📋 File Type Breakdown:\n")
                    for ext, count in sorted(file_counts.items()):
                        text_widget.insert(tk.END, f"   • {ext or 'no extension'}: {count} files\n")

                text_widget.insert(tk.END, "\n")

            # Memory estimation
            if total_vectors > 0:
                avg_dimension = 512  # Estimate for face embeddings
                estimated_memory_mb = (total_vectors * avg_dimension * 4) / (1024 * 1024)
                text_widget.insert(tk.END, f"🧠 Estimated Memory Usage: {estimated_memory_mb:.2f} MB\n\n")

            # Performance metrics
            text_widget.insert(tk.END, "⚡ Performance Metrics:\n")
            text_widget.insert(tk.END, f"   • Average vectors per collection: {total_vectors / max(len(collections), 1):.0f}\n")
            text_widget.insert(tk.END, f"   • Storage efficiency: {(total_size / max(total_vectors, 1)) / 1024:.2f} KB per vector\n")

            text_widget.config(state=tk.DISABLED)

        except Exception as e:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, f"❌ Error generating statistics: {e}")
            text_widget.config(state=tk.DISABLED)

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
                self.stats_labels["metadata_loaded"].config(text=str(stats.get("metadata_loaded", 0)))
                self.stats_labels["metadata_missing"].config(text=str(stats.get("metadata_missing", 0)))
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
                               "metadata_loaded", "metadata_missing", "duplicates_skipped",
                               "errors", "remaining_files"]:
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
    required_files = ["4_embed_faces.py", "3_collect_faces.py"]
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