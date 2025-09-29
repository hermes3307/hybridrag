#!/usr/bin/env python3
"""
Face Similarity Search Test GUI
Downloads random test images and performs similarity search with visual results
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import requests
import os
import random
import threading
from PIL import Image, ImageTk
import io
import hashlib
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
import json

# Import our face processing modules
from face_database import FaceDatabase, FaceSearchInterface
from face_collector import FaceAnalyzer, FaceEmbedder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageDisplayWindow:
    """Separate window for displaying images"""

    def __init__(self, parent, title: str, image_path: str, metadata: Dict = None):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("700x800")

        # Keep reference to prevent garbage collection
        self.photo = None

        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image frame
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        try:
            # Load and display image
            pil_image = Image.open(image_path)
            original_size = pil_image.size

            # Calculate resize dimensions to fit in window while maintaining aspect ratio
            max_width, max_height = 650, 450
            width_ratio = max_width / original_size[0]
            height_ratio = max_height / original_size[1]
            scale_ratio = min(width_ratio, height_ratio)

            new_width = int(original_size[0] * scale_ratio)
            new_height = int(original_size[1] * scale_ratio)

            # Resize image
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(pil_image)

            # Create canvas for better image display
            canvas = tk.Canvas(image_frame, width=new_width, height=new_height,
                             highlightthickness=0, relief='ridge', bd=2)
            canvas.pack(pady=10)

            # Add image to canvas
            canvas.create_image(new_width//2, new_height//2, image=self.photo)

            # Add size info
            size_label = ttk.Label(image_frame,
                                 text=f"Size: {original_size[0]}×{original_size[1]} → {new_width}×{new_height}")
            size_label.pack(pady=5)

        except Exception as e:
            error_label = ttk.Label(image_frame, text=f"Error loading image: {e}")
            error_label.pack(pady=20)
            logger.error(f"Error displaying image {image_path}: {e}")

        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Image Information")
        info_frame.pack(fill=tk.BOTH, pady=(0, 10))

        # File path
        ttk.Label(info_frame, text=f"File: {os.path.basename(image_path)}").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, text=f"Path: {image_path}").pack(anchor=tk.W, padx=5, pady=2)

        # File size
        try:
            file_size = os.path.getsize(image_path)
            ttk.Label(info_frame, text=f"Size: {file_size:,} bytes ({file_size/1024:.1f} KB)").pack(anchor=tk.W, padx=5, pady=2)
        except:
            pass

        # Metadata
        if metadata:
            metadata_frame = ttk.LabelFrame(main_frame, text="Metadata")
            metadata_frame.pack(fill=tk.BOTH)

            # Create scrollable text widget for metadata
            metadata_text = scrolledtext.ScrolledText(metadata_frame, height=8, wrap=tk.WORD)
            metadata_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Format metadata nicely
            for key, value in metadata.items():
                metadata_text.insert(tk.END, f"{key}: {value}\n")

            metadata_text.config(state=tk.DISABLED)

class SimilaritySearchGUI:
    """GUI for testing face similarity search"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔍 Face Similarity Search Test")
        self.root.geometry("1000x800")

        # Initialize components
        self.face_db = None
        self.search_interface = None
        self.analyzer = FaceAnalyzer()
        self.embedder = FaceEmbedder()

        # Downloaded test files
        self.test_files = []
        self.current_query_file = None
        self.search_results = []

        # UI setup
        self.setup_ui()
        self.initialize_database()

    def setup_ui(self):
        """Set up the user interface"""
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Test Download Tab
        self.setup_download_tab()

        # Similarity Search Tab
        self.setup_search_tab()

        # Results Tab
        self.setup_results_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_download_tab(self):
        """Set up the test file download tab"""
        download_frame = ttk.Frame(self.notebook)
        self.notebook.add(download_frame, text="📥 Download Test Files")

        # Title
        title_label = ttk.Label(download_frame, text="📥 Download Random Test Images", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Download controls
        controls_frame = ttk.LabelFrame(download_frame, text="Download Controls")
        controls_frame.pack(fill=tk.X, padx=10, pady=10)

        # Number of files to download
        ttk.Label(controls_frame, text="Number of test files:").pack(side=tk.LEFT, padx=5)
        self.download_count = tk.StringVar(value="3")
        count_spin = ttk.Spinbox(controls_frame, from_=1, to=10, width=5, textvariable=self.download_count)
        count_spin.pack(side=tk.LEFT, padx=5)

        # Download button
        download_btn = ttk.Button(controls_frame, text="🔽 Download Random Faces",
                                command=self.download_test_files)
        download_btn.pack(side=tk.LEFT, padx=10)

        # Clear button
        clear_btn = ttk.Button(controls_frame, text="🗑️ Clear Test Files",
                             command=self.clear_test_files)
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Test files list
        files_frame = ttk.LabelFrame(download_frame, text="Downloaded Test Files")
        files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create treeview for file list
        columns = ("File", "Size", "Downloaded")
        self.files_tree = ttk.Treeview(files_frame, columns=columns, show="headings", height=10)

        # Configure columns
        self.files_tree.heading("File", text="File Name")
        self.files_tree.heading("Size", text="Size (KB)")
        self.files_tree.heading("Downloaded", text="Downloaded At")

        self.files_tree.column("File", width=300)
        self.files_tree.column("Size", width=100)
        self.files_tree.column("Downloaded", width=150)

        # Scrollbar for treeview
        files_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=files_scrollbar.set)

        # Pack treeview and scrollbar
        self.files_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Double-click to view image
        self.files_tree.bind("<Double-1>", self.view_test_file)

        # Buttons frame
        files_btn_frame = ttk.Frame(download_frame)
        files_btn_frame.pack(fill=tk.X, padx=10, pady=5)

        # View selected file button
        view_btn = ttk.Button(files_btn_frame, text="👁️ View Selected File",
                            command=self.view_selected_test_file)
        view_btn.pack(side=tk.LEFT, padx=5)

        # Use for search button
        use_search_btn = ttk.Button(files_btn_frame, text="🔍 Use for Search",
                                  command=self.use_for_search)
        use_search_btn.pack(side=tk.LEFT, padx=5)

    def setup_search_tab(self):
        """Set up the similarity search tab"""
        search_frame = ttk.Frame(self.notebook)
        self.notebook.add(search_frame, text="🔍 Similarity Search")

        # Title
        title_label = ttk.Label(search_frame, text="🔍 Face Similarity Search", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Query section
        query_frame = ttk.LabelFrame(search_frame, text="Query Image")
        query_frame.pack(fill=tk.X, padx=10, pady=10)

        # Query info and preview frame
        query_top_frame = ttk.Frame(query_frame)
        query_top_frame.pack(fill=tk.X, pady=5)

        # Current query file info
        self.query_info_var = tk.StringVar(value="No query image selected")
        query_info_label = ttk.Label(query_top_frame, textvariable=self.query_info_var)
        query_info_label.pack(side=tk.LEFT, pady=5)

        # Small preview canvas
        self.query_preview_canvas = tk.Canvas(query_top_frame, width=80, height=80,
                                            highlightthickness=1, relief='ridge', bd=1, bg='white')
        self.query_preview_canvas.pack(side=tk.RIGHT, padx=10, pady=5)
        self.query_preview_photo = None

        # Query buttons
        query_btn_frame = ttk.Frame(query_frame)
        query_btn_frame.pack(pady=5)

        # Select file button
        select_file_btn = ttk.Button(query_btn_frame, text="📁 Select Image File",
                                   command=self.select_query_file)
        select_file_btn.pack(side=tk.LEFT, padx=5)

        # Use random test file button
        random_btn = ttk.Button(query_btn_frame, text="🎲 Use Random Test File",
                              command=self.use_random_test_file)
        random_btn.pack(side=tk.LEFT, padx=5)

        # Preview query image button
        preview_btn = ttk.Button(query_btn_frame, text="👁️ Preview Query Image",
                               command=self.preview_query_image)
        preview_btn.pack(side=tk.LEFT, padx=5)

        # Search controls
        search_controls_frame = ttk.LabelFrame(search_frame, text="Search Parameters")
        search_controls_frame.pack(fill=tk.X, padx=10, pady=10)

        # Number of results
        ttk.Label(search_controls_frame, text="Number of results:").pack(side=tk.LEFT, padx=5)
        self.num_results = tk.StringVar(value="5")
        results_spin = ttk.Spinbox(search_controls_frame, from_=1, to=20, width=5, textvariable=self.num_results)
        results_spin.pack(side=tk.LEFT, padx=5)

        # Search button
        search_btn = ttk.Button(search_controls_frame, text="🔍 Find Similar Faces",
                              command=self.perform_similarity_search)
        search_btn.pack(side=tk.LEFT, padx=10)

        # Database info
        db_frame = ttk.LabelFrame(search_frame, text="Database Information")
        db_frame.pack(fill=tk.X, padx=10, pady=10)

        self.db_info_var = tk.StringVar(value="Initializing database...")
        db_info_label = ttk.Label(db_frame, textvariable=self.db_info_var)
        db_info_label.pack(pady=5)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(search_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

    def setup_results_tab(self):
        """Set up the search results tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 Search Results")

        # Title
        title_label = ttk.Label(results_frame, text="📊 Similarity Search Results", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Results summary
        self.results_summary_var = tk.StringVar(value="No search performed yet")
        summary_label = ttk.Label(results_frame, textvariable=self.results_summary_var)
        summary_label.pack(pady=5)

        # Results table
        results_table_frame = ttk.LabelFrame(results_frame, text="Similar Faces Found")
        results_table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create treeview for results
        result_columns = ("Rank", "Similarity", "File", "Age Group", "Skin Tone", "Quality")
        self.results_tree = ttk.Treeview(results_table_frame, columns=result_columns, show="headings", height=12)

        # Configure columns
        self.results_tree.heading("Rank", text="Rank")
        self.results_tree.heading("Similarity", text="Similarity %")
        self.results_tree.heading("File", text="File Name")
        self.results_tree.heading("Age Group", text="Age Group")
        self.results_tree.heading("Skin Tone", text="Skin Tone")
        self.results_tree.heading("Quality", text="Quality")

        self.results_tree.column("Rank", width=50)
        self.results_tree.column("Similarity", width=100)
        self.results_tree.column("File", width=200)
        self.results_tree.column("Age Group", width=100)
        self.results_tree.column("Skin Tone", width=100)
        self.results_tree.column("Quality", width=80)

        # Scrollbar for results
        results_scrollbar = ttk.Scrollbar(results_table_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)

        # Pack results treeview and scrollbar
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Double-click to view result image
        self.results_tree.bind("<Double-1>", self.view_result_image)

        # Results buttons
        results_btn_frame = ttk.Frame(results_frame)
        results_btn_frame.pack(fill=tk.X, padx=10, pady=5)

        # View selected result button
        view_result_btn = ttk.Button(results_btn_frame, text="👁️ View Selected Result",
                                   command=self.view_selected_result)
        view_result_btn.pack(side=tk.LEFT, padx=5)

        # Export results button
        export_btn = ttk.Button(results_btn_frame, text="💾 Export Results",
                              command=self.export_results)
        export_btn.pack(side=tk.LEFT, padx=5)

        # Clear results button
        clear_results_btn = ttk.Button(results_btn_frame, text="🗑️ Clear Results",
                                     command=self.clear_results)
        clear_results_btn.pack(side=tk.LEFT, padx=5)

    def initialize_database(self):
        """Initialize the face database"""
        try:
            self.status_var.set("Initializing database...")
            self.face_db = FaceDatabase()
            self.search_interface = FaceSearchInterface(self.face_db)

            # Get database stats
            stats = self.face_db.get_database_stats()
            total_faces = stats.get('total_faces', 0)

            self.db_info_var.set(f"Database loaded: {total_faces:,} face embeddings available")
            self.status_var.set(f"Ready - Database contains {total_faces:,} faces")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.db_info_var.set(f"Database error: {e}")
            self.status_var.set("Database initialization failed")
            messagebox.showerror("Database Error", f"Failed to initialize database:\n{e}")

    def download_test_files(self):
        """Download random test face images"""
        try:
            count = int(self.download_count.get())
            self.status_var.set("Downloading test files...")
            self.progress_var.set(0)

            # Run download in separate thread
            thread = threading.Thread(target=self._download_files_thread, args=(count,))
            thread.daemon = True
            thread.start()

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number of files to download")

    def _download_files_thread(self, count: int):
        """Download files in background thread"""
        try:
            # Create test directory if it doesn't exist
            test_dir = "test_images"
            os.makedirs(test_dir, exist_ok=True)

            downloaded_files = []

            for i in range(count):
                try:
                    # Update progress
                    progress = (i / count) * 100
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    self.root.after(0, lambda: self.status_var.set(f"Downloading file {i+1}/{count}..."))

                    # Download from ThisPersonDoesNotExist
                    response = requests.get("https://thispersondoesnotexist.com/", timeout=30)

                    if response.status_code == 200:
                        # Generate unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        hash_suffix = hashlib.md5(response.content).hexdigest()[:8]
                        filename = f"test_face_{timestamp}_{hash_suffix}.jpg"
                        file_path = os.path.join(test_dir, filename)

                        # Save file
                        with open(file_path, 'wb') as f:
                            f.write(response.content)

                        # Verify it's a valid image
                        try:
                            with Image.open(file_path) as img:
                                img.verify()

                            file_info = {
                                'path': file_path,
                                'filename': filename,
                                'size': len(response.content),
                                'downloaded': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            downloaded_files.append(file_info)

                        except Exception as img_error:
                            logger.warning(f"Downloaded file is not a valid image: {img_error}")
                            if os.path.exists(file_path):
                                os.remove(file_path)

                    # Small delay between downloads
                    import time
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Error downloading file {i+1}: {e}")
                    continue

            # Update UI with results
            self.root.after(0, lambda: self._update_download_results(downloaded_files))

        except Exception as e:
            logger.error(f"Error in download thread: {e}")
            self.root.after(0, lambda: messagebox.showerror("Download Error", f"Failed to download files:\n{e}"))
            self.root.after(0, lambda: self.status_var.set("Download failed"))

    def _update_download_results(self, downloaded_files):
        """Update UI with download results"""
        self.test_files.extend(downloaded_files)

        # Update treeview
        for file_info in downloaded_files:
            self.files_tree.insert("", tk.END, values=(
                file_info['filename'],
                f"{file_info['size']/1024:.1f}",
                file_info['downloaded']
            ))

        self.progress_var.set(100)
        self.status_var.set(f"Downloaded {len(downloaded_files)} test files successfully")

        if downloaded_files:
            messagebox.showinfo("Download Complete",
                              f"Successfully downloaded {len(downloaded_files)} test images!")

    def clear_test_files(self):
        """Clear all downloaded test files"""
        if not self.test_files:
            messagebox.showinfo("No Files", "No test files to clear")
            return

        result = messagebox.askyesno("Confirm Clear",
                                   f"Delete {len(self.test_files)} test files from disk?")
        if result:
            deleted_count = 0
            for file_info in self.test_files:
                try:
                    if os.path.exists(file_info['path']):
                        os.remove(file_info['path'])
                        deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting {file_info['path']}: {e}")

            # Clear UI
            self.test_files.clear()
            for item in self.files_tree.get_children():
                self.files_tree.delete(item)

            # Clear current query if it was a test file
            if self.current_query_file and self.current_query_file.startswith("test_images/"):
                self.current_query_file = None
                self.query_info_var.set("No query image selected")

            self.status_var.set(f"Deleted {deleted_count} test files")
            messagebox.showinfo("Files Cleared", f"Deleted {deleted_count} test files")

    def view_test_file(self, event):
        """View selected test file on double-click"""
        self.view_selected_test_file()

    def view_selected_test_file(self):
        """View the selected test file"""
        selection = self.files_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a test file to view")
            return

        # Get selected item
        item = self.files_tree.item(selection[0])
        filename = item['values'][0]

        # Find the file info
        file_info = None
        for f in self.test_files:
            if f['filename'] == filename:
                file_info = f
                break

        if file_info and os.path.exists(file_info['path']):
            # Create image display window
            ImageDisplayWindow(self.root, f"Test File: {filename}",
                             file_info['path'], file_info)
        else:
            messagebox.showerror("File Not Found", f"Test file not found: {filename}")

    def update_query_preview(self):
        """Update the small query preview"""
        if not self.current_query_file or not os.path.exists(self.current_query_file):
            # Clear preview
            self.query_preview_canvas.delete("all")
            self.query_preview_photo = None
            return

        try:
            # Load and resize image for preview
            pil_image = Image.open(self.current_query_file)
            pil_image.thumbnail((75, 75), Image.Resampling.LANCZOS)
            self.query_preview_photo = ImageTk.PhotoImage(pil_image)

            # Clear and update canvas
            self.query_preview_canvas.delete("all")
            self.query_preview_canvas.create_image(40, 40, image=self.query_preview_photo)

        except Exception as e:
            logger.error(f"Error updating query preview: {e}")
            # Clear preview on error
            self.query_preview_canvas.delete("all")
            self.query_preview_photo = None

    def use_for_search(self):
        """Use selected test file as query for search"""
        selection = self.files_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a test file to use for search")
            return

        # Get selected item
        item = self.files_tree.item(selection[0])
        filename = item['values'][0]

        # Find the file info
        file_info = None
        for f in self.test_files:
            if f['filename'] == filename:
                file_info = f
                break

        if file_info and os.path.exists(file_info['path']):
            self.current_query_file = file_info['path']
            self.query_info_var.set(f"Query: {filename}")
            self.status_var.set(f"Query image set to: {filename}")

            # Update preview
            self.update_query_preview()

            # Switch to search tab
            self.notebook.select(1)
        else:
            messagebox.showerror("File Not Found", f"Test file not found: {filename}")

    def select_query_file(self):
        """Select an image file for query"""
        file_path = filedialog.askopenfilename(
            title="Select Query Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_query_file = file_path
            filename = os.path.basename(file_path)
            self.query_info_var.set(f"Query: {filename}")
            self.status_var.set(f"Query image selected: {filename}")

            # Update preview
            self.update_query_preview()

    def use_random_test_file(self):
        """Use a random test file as query"""
        if not self.test_files:
            messagebox.showwarning("No Test Files", "Please download some test files first")
            return

        # Pick random test file
        file_info = random.choice(self.test_files)

        if os.path.exists(file_info['path']):
            self.current_query_file = file_info['path']
            self.query_info_var.set(f"Query: {file_info['filename']} (random)")
            self.status_var.set(f"Random query selected: {file_info['filename']}")

            # Update preview
            self.update_query_preview()
        else:
            messagebox.showerror("File Not Found", "Random test file not found")

    def preview_query_image(self):
        """Preview the current query image"""
        if not self.current_query_file:
            messagebox.showwarning("No Query", "Please select a query image first")
            return

        if os.path.exists(self.current_query_file):
            filename = os.path.basename(self.current_query_file)
            ImageDisplayWindow(self.root, f"Query Image: {filename}", self.current_query_file)
        else:
            messagebox.showerror("File Not Found", "Query image file not found")

    def perform_similarity_search(self):
        """Perform similarity search with current query"""
        if not self.current_query_file:
            messagebox.showwarning("No Query", "Please select a query image first")
            return

        if not os.path.exists(self.current_query_file):
            messagebox.showerror("File Not Found", "Query image file not found")
            return

        try:
            num_results = int(self.num_results.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number of results")
            return

        # Run search in background thread
        self.status_var.set("Performing similarity search...")
        self.progress_var.set(0)

        thread = threading.Thread(target=self._search_thread, args=(num_results,))
        thread.daemon = True
        thread.start()

    def _search_thread(self, num_results: int):
        """Perform search in background thread"""
        try:
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(20))
            self.root.after(0, lambda: self.status_var.set("Analyzing query image..."))

            # Use search interface to find similar faces
            results = self.search_interface.search_by_image(self.current_query_file, num_results)

            self.root.after(0, lambda: self.progress_var.set(80))

            if "error" in results:
                self.root.after(0, lambda: messagebox.showerror("Search Error", results["error"]))
                self.root.after(0, lambda: self.status_var.set("Search failed"))
                return

            # Process results
            search_results = results.get("results", {})

            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self._update_search_results(results, search_results))

        except Exception as e:
            logger.error(f"Error in search thread: {e}")
            self.root.after(0, lambda: messagebox.showerror("Search Error", f"Search failed:\n{e}"))
            self.root.after(0, lambda: self.status_var.set("Search failed"))

    def _update_search_results(self, full_results: Dict, search_results: Dict):
        """Update UI with search results"""
        self.search_results = []

        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        result_count = search_results.get("count", 0)

        if result_count == 0:
            self.results_summary_var.set("No similar faces found")
            self.status_var.set("Search completed - no results found")
            return

        # Process each result
        for i in range(result_count):
            try:
                face_id = search_results["ids"][i]
                similarity = 1 - search_results["distances"][i] if i < len(search_results.get("distances", [])) else 0

                # Get metadata
                metadata = search_results["metadatas"][i] if i < len(search_results.get("metadatas", [])) else {}

                # Extract file path from metadata
                file_path = metadata.get("file_path", "")
                filename = os.path.basename(file_path) if file_path else face_id

                # Get features
                age_group = metadata.get("estimated_age_group", "unknown")
                skin_tone = metadata.get("estimated_skin_tone", "unknown")
                quality = metadata.get("image_quality", "unknown")

                result_info = {
                    'rank': i + 1,
                    'face_id': face_id,
                    'similarity': similarity,
                    'file_path': file_path,
                    'filename': filename,
                    'metadata': metadata,
                    'age_group': age_group,
                    'skin_tone': skin_tone,
                    'quality': quality
                }

                self.search_results.append(result_info)

                # Add to treeview
                self.results_tree.insert("", tk.END, values=(
                    i + 1,
                    f"{similarity*100:.1f}%",
                    filename,
                    age_group,
                    skin_tone,
                    quality
                ))

            except Exception as e:
                logger.error(f"Error processing result {i}: {e}")
                continue

        # Update summary
        query_filename = os.path.basename(self.current_query_file)
        self.results_summary_var.set(f"Found {len(self.search_results)} similar faces for '{query_filename}'")
        self.status_var.set(f"Search completed - {len(self.search_results)} results found")

        # Switch to results tab
        self.notebook.select(2)

        messagebox.showinfo("Search Complete",
                          f"Found {len(self.search_results)} similar faces!\n"
                          f"Results displayed in the Results tab.")

    def view_result_image(self, event):
        """View selected result image on double-click"""
        self.view_selected_result()

    def view_selected_result(self):
        """View the selected search result image"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a search result to view")
            return

        # Get selected item
        item = self.results_tree.item(selection[0])
        rank = int(item['values'][0])

        # Find the result info
        result_info = None
        for r in self.search_results:
            if r['rank'] == rank:
                result_info = r
                break

        if result_info:
            file_path = result_info['file_path']
            if file_path and os.path.exists(file_path):
                title = f"Result #{rank}: {result_info['filename']} ({result_info['similarity']*100:.1f}% similar)"
                ImageDisplayWindow(self.root, title, file_path, result_info['metadata'])
            else:
                messagebox.showerror("File Not Found", f"Result image not found: {file_path}")
        else:
            messagebox.showerror("Error", "Could not find result information")

    def export_results(self):
        """Export search results to JSON file"""
        if not self.search_results:
            messagebox.showwarning("No Results", "No search results to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Search Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                export_data = {
                    'query_image': self.current_query_file,
                    'query_filename': os.path.basename(self.current_query_file) if self.current_query_file else "",
                    'search_timestamp': datetime.now().isoformat(),
                    'num_results': len(self.search_results),
                    'results': self.search_results
                }

                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

                messagebox.showinfo("Export Complete", f"Results exported to:\n{file_path}")
                self.status_var.set(f"Results exported to {os.path.basename(file_path)}")

            except Exception as e:
                logger.error(f"Error exporting results: {e}")
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")

    def clear_results(self):
        """Clear search results"""
        if not self.search_results:
            messagebox.showinfo("No Results", "No search results to clear")
            return

        # Clear results
        self.search_results.clear()

        # Clear treeview
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Update summary
        self.results_summary_var.set("No search performed yet")
        self.status_var.set("Search results cleared")

    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")

def main():
    """Main function"""
    print("🔍 Face Similarity Search Test GUI")
    print("=" * 50)

    try:
        # Create and run GUI
        app = SimilaritySearchGUI()
        app.run()

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()