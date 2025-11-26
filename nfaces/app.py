#!/usr/bin/env python3
"""
Unified Face Processing Application with Gradio
A modern web-based GUI that integrates all face processing operations:
- Download faces
- Process and embed into vectors
- Create HNSW index (via pgvector)
- Search faces
All in a single, unified interface with a separate configuration panel.
"""

import os
import sys
import json
import time
import gradio as gr
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image
import threading
from datetime import datetime

# Import existing backend modules
from core import (
    IntegratedFaceSystem,
    SystemConfig,
    SystemStats,
    FaceDownloader,
    FaceProcessor,
    FaceEmbedder
)
from pgvector_db import PgVectorDatabaseManager
from advanced_search import AdvancedSearchEngine, SearchQuery


class UnifiedFaceApp:
    """Unified application controller for Gradio interface"""

    def __init__(self):
        self.config = self.load_config()
        self.system = None
        self.db = None
        self.search_engine = None
        self.is_running = False
        self.system_stats = SystemStats()  # Use the core SystemStats class
        self.stats = {
            'downloads': {'total': 0, 'success': 0, 'errors': 0},
            'embeddings': {'total': 0, 'success': 0, 'errors': 0},
            'searches': {'total': 0}
        }
        self.stop_flag = threading.Event()
        # Shared log for real-time updates
        self.download_log = []
        self.log_lock = threading.Lock()
        self.download_active = False

    def safe_print(self, message: str):
        """Safely print to stdout, ignoring BrokenPipeError"""
        try:
            print(message, flush=True)
        except BrokenPipeError:
            # Client disconnected, simply stop trying to print
            pass

    def add_log(self, message: str):
        """Add a message to the download log (thread-safe)"""
        with self.log_lock:
            self.download_log.append(message)
        # Also print to stdout for terminal visibility
        self.safe_print(f"[DOWNLOAD] {message}")

    def get_logs(self) -> str:
        """Get all log messages as a single string (thread-safe)"""
        with self.log_lock:
            return "\n".join(self.download_log)

    def clear_logs(self):
        """Clear all log messages (thread-safe)"""
        with self.log_lock:
            self.download_log = []

    def load_config(self) -> SystemConfig:
        """Load configuration from system_config.json"""
        config_path = Path(__file__).parent / "system_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
                return SystemConfig(**data)
        return SystemConfig()

    def save_config(self, config_dict: dict):
        """Save configuration to system_config.json"""
        config_path = Path(__file__).parent / "system_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        self.config = SystemConfig(**config_dict)

    def initialize_system(self):
        """Initialize the face processing system"""
        if not self.system:
            # IntegratedFaceSystem expects a config file path, not a config object
            config_path = str(Path(__file__).parent / "system_config.json")
            self.system = IntegratedFaceSystem(config_path)
        if not self.db:
            # PgVectorDatabaseManager expects a SystemConfig object
            self.db = PgVectorDatabaseManager(self.config)
        if not self.search_engine:
            self.search_engine = AdvancedSearchEngine(self.db)

    def test_database_connection(self, host: str, port: int, db_name: str,
                                user: str, password: str) -> str:
        """Test database connection"""
        try:
            # Create temporary config for testing
            temp_config = SystemConfig(
                db_host=host,
                db_port=port,
                db_name=db_name,
                db_user=user,
                db_password=password
            )
            test_db = PgVectorDatabaseManager(temp_config)
            stats = test_db.get_statistics()
            test_db.close()
            return f"✅ Connection successful!\nFaces in database: {stats.get('total_faces', 0)}"
        except Exception as e:
            return f"❌ Connection failed: {str(e)}"

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        stats = {
            'database': 'Not connected',
            'total_faces': 0,
            'downloads': self.stats['downloads'],
            'embeddings': self.stats['embeddings'],
            'searches': self.stats['searches']['total']
        }

        # Initialize database if not already done
        if not self.db:
            try:
                self.db = PgVectorDatabaseManager(self.config)
                if self.db.initialize():
                    stats['database'] = 'Connected ✅'
                else:
                    stats['database'] = 'Connection failed ❌'
            except Exception as e:
                stats['database'] = f'Error: {str(e)[:30]}... ❌'
                return stats

        # Get database statistics
        if self.db and self.db.initialized:
            try:
                db_stats = self.db.get_statistics()
                stats['database'] = 'Connected ✅'
                stats['total_faces'] = db_stats.get('total_faces', 0)
            except Exception as e:
                stats['database'] = f'Error: {str(e)[:30]}... ❌'

        return stats

    def _download_worker(self, source: str, count: int, delay: float):
        """Background worker thread for downloading faces"""
        worker_start = time.time()
        try:
            print(f"[DEBUG] Worker thread started: source={source}, count={count}, delay={delay}", flush=True)
        except BrokenPipeError:
            pass
        try:
            # Set the download source in config (temporarily)
            original_source = self.config.download_source
            self.config.download_source = source.lower()
            try:
                print(f"[DEBUG] Config source set to: {self.config.download_source}", flush=True)
            except BrokenPipeError:
                pass

            # FaceDownloader only needs config and stats - no database required for downloading!
            init_start = time.time()
            try:
                print(f"[DEBUG] Creating FaceDownloader...", flush=True)
            except BrokenPipeError:
                pass
            self.add_log("⚙️ Initializing downloader (loading existing face hashes)...")
            downloader = FaceDownloader(self.config, self.system_stats)
            init_time = time.time() - init_start
            try:
                print(f"[DEBUG] FaceDownloader created successfully in {init_time:.2f}s", flush=True)
            except BrokenPipeError:
                pass
            self.add_log(f"✓ Downloader ready (init took {init_time:.2f}s)")

            results = []
            error_messages = []

            self.add_log(f"🚀 Starting download from {source}")
            self.add_log(f"📊 Target: {count} faces with {delay}s delay\n")

            for i in range(count):
                # Check stop flag at start of loop
                if self.stop_flag.is_set():
                    self.add_log(f"\n⏹️ Download stopped by user at {i}/{count}")
                    break

                try:
                    self.add_log(f"⬇️  [{i+1}/{count}] Downloading face...")

                    # Check stop flag before actual download
                    if self.stop_flag.is_set():
                        self.add_log(f"\n⏹️ Download stopped by user at {i}/{count}")
                        break

                    file_path = downloader.download_face()
                    if file_path:
                        results.append(file_path)
                        self.stats['downloads']['success'] += 1
                        self.add_log(f"   ✓ SUCCESS: Saved to {file_path}")
                        try:
                            print(f"✓ Downloaded face {i+1}/{count}: {file_path}")
                        except BrokenPipeError:
                            pass
                    else:
                        self.stats['downloads']['errors'] += 1
                        error_messages.append(f"Face {i+1}: No data returned")
                        self.add_log(f"   ✗ FAILED: No data returned")
                        try:
                            print(f"✗ Failed to download face {i+1}/{count}")
                        except BrokenPipeError:
                            pass
                    self.stats['downloads']['total'] += 1

                except Exception as e:
                    self.stats['downloads']['errors'] += 1
                    error_msg = f"Face {i+1}: {str(e)}"
                    error_messages.append(error_msg)
                    results.append(error_msg)
                    self.add_log(f"   ✗ ERROR: {str(e)}")
                    try:
                        print(f"✗ Error downloading face {i+1}/{count}: {str(e)}")
                    except BrokenPipeError:
                        pass

                # Only delay between downloads (not after the last one)
                if i < count - 1 and delay > 0:
                    # Check stop flag before delay
                    if self.stop_flag.is_set():
                        self.add_log(f"\n⏹️ Download stopped by user at {i+1}/{count}")
                        break
                    time.sleep(delay)

            # Count successes - results contains file paths (strings) for success, so count valid paths
            success_count = len([r for r in results if r and os.path.exists(r)])
            self.add_log(f"\n{'='*60}")
            self.add_log(f"📈 SUMMARY: {success_count}/{count} faces downloaded successfully")
            self.add_log(f"✅ Success: {success_count}")
            self.add_log(f"❌ Errors: {len(error_messages)}")

            if error_messages:
                self.add_log(f"\n⚠️ Errors ({len(error_messages)}):")
                for err in error_messages[:5]:
                    self.add_log(f"  • {err}")
                if len(error_messages) > 5:
                    self.add_log(f"  ... and {len(error_messages) - 5} more errors")

        except Exception as e:
            self.add_log(f"\n❌ CRITICAL ERROR: {str(e)}")
            try:
                print(f"❌ Critical error in download worker: {str(e)}")
            except BrokenPipeError:
                pass
        finally:
            # Restore original source
            self.config.download_source = original_source
            self.download_active = False

    def download_faces(self, source: str, count: int, delay: float):
        """Start downloading faces in background thread"""
        try:
            print(f"🔍 DEBUG: download_faces called with source={source}, count={count}, delay={delay}")
        except BrokenPipeError:
            pass

        if self.download_active:
            return "⚠️ Download already in progress", self.format_stats(), self.get_logs()

        try:
            print(f"[DEBUG] Clearing stop flag and logs...", flush=True)
        except BrokenPipeError:
            pass
        self.stop_flag.clear()
        self.clear_logs()
        self.download_active = True

        # Start download in background thread (don't initialize_system here as it blocks!)
        try:
            print(f"[DEBUG] Creating background thread...", flush=True)
        except BrokenPipeError:
            pass
        download_thread = threading.Thread(
            target=self._download_worker,
            args=(source, count, delay),
            daemon=True
        )
        try:
            print(f"[DEBUG] Starting thread...", flush=True)
        except BrokenPipeError:
            pass
        download_thread.start()
        try:
            print(f"[DEBUG] Thread started, returning...", flush=True)
        except BrokenPipeError:
            pass

        return "⏳ Download started...", self.format_stats(), self.get_logs()

    def get_download_status(self):
        """Poll function to get current download status and logs"""
        if self.download_active:
            status = "⏳ Downloading..."
        else:
            status = "✅ Ready"

        return status, self.format_stats(), self.get_logs()

    def stop_download_wrapper(self):
        """Wrapper for stop button that also returns current status"""
        self.stop_flag.set()
        return "⏹️ Stopping...", self.format_stats(), self.get_logs()

    def process_and_embed(self, batch_size: int, workers: int,
                         process_new_only: bool, progress=gr.Progress()) -> Tuple[str, str]:
        """Process images and create embeddings"""
        self.stop_flag.clear()
        self.initialize_system()

        try:
            faces_dir = Path(self.config.faces_dir)
            if not faces_dir.exists():
                return f"❌ Faces directory does not exist: {faces_dir}", self.format_stats()

            # Get list of image files
            image_files = list(faces_dir.glob("*.jpg")) + list(faces_dir.glob("*.png"))
            try:
                print(f"Found {len(image_files)} image files in {faces_dir}")
            except BrokenPipeError:
                pass

            if process_new_only:
                # Filter out already processed files
                processed = set()
                if self.db:
                    # Get list of processed files from database
                    try:
                        stats = self.db.get_statistics()
                        # This is simplified - you'd need to query actual file paths
                        processed = set()
                    except:
                        pass
                image_files = [f for f in image_files if f.name not in processed]
                try:
                    print(f"After filtering, {len(image_files)} files need processing")
                except BrokenPipeError:
                    pass

            total_files = len(image_files)
            if total_files == 0:
                return "ℹ️ No files to process", self.format_stats()

            # FaceProcessor expects SystemConfig, SystemStats, and db_manager
            processor = FaceProcessor(
                config=self.config,
                stats=self.system_stats,
                db_manager=self.db
            )

            success_count = 0
            error_count = 0
            error_messages = []

            for i, file_path in enumerate(image_files):
                if self.stop_flag.is_set():
                    try:
                        print(f"Processing stopped by user at {i}/{total_files}")
                    except BrokenPipeError:
                        pass
                    break

                progress((i + 1) / total_files,
                        desc=f"Processing {i+1}/{total_files}: {file_path.name}")

                try:
                    result = processor.process_face_file(str(file_path))
                    if result:
                        success_count += 1
                        self.stats['embeddings']['success'] += 1
                        try:
                            print(f"✓ Processed {i+1}/{total_files}: {file_path.name}")
                        except BrokenPipeError:
                            pass
                    else:
                        error_count += 1
                        self.stats['embeddings']['errors'] += 1
                        error_messages.append(f"{file_path.name}: Processing returned None")
                        try:
                            print(f"✗ Failed to process {file_path.name}")
                        except BrokenPipeError:
                            pass
                    self.stats['embeddings']['total'] += 1
                except Exception as e:
                    error_count += 1
                    self.stats['embeddings']['errors'] += 1
                    error_messages.append(f"{file_path.name}: {str(e)}")
                    try:
                        print(f"✗ Error processing {file_path.name}: {str(e)}")
                    except BrokenPipeError:
                        pass

            message = f"✅ Processed {success_count}/{total_files} faces successfully"
            if error_count > 0:
                message += f"\n\n⚠️ {error_count} errors occurred"
                if error_messages:
                    message += ":\n" + "\n".join(error_messages[:5])
                    if len(error_messages) > 5:
                        message += f"\n... and {len(error_messages) - 5} more errors"

            return message, self.format_stats()

        except Exception as e:
            error_detail = f"❌ Error during processing: {str(e)}\n\nDetails: {type(e).__name__}"
            try:
                print(f"Processing error: {str(e)}")
            except BrokenPipeError:
                pass
            import traceback
            try:
                traceback.print_exc()
            except BrokenPipeError:
                pass
            return error_detail, self.format_stats()

    def search_faces(self, query_image, top_k: int, search_mode: str,
                    sex_filter: str, age_filter: str, skin_tone_filter: str,
                    hair_color_filter: str, brightness_filter: str,
                    quality_filter: str) -> Tuple[List, str]:
        """Search for similar faces"""
        self.initialize_system()

        if not self.db or not self.search_engine:
            return [], "❌ Database not initialized"

        if query_image is None:
            return [], "⚠️ Please provide a query image"

        try:
            # Convert query image to embedding
            embedder = FaceEmbedder(model_name=self.config.embedding_model)

            # Handle different image input types
            if isinstance(query_image, str):
                query_img = Image.open(query_image)
            else:
                query_img = Image.fromarray(query_image)

            query_embedding = embedder.create_embedding(np.array(query_img))

            # Build metadata filters
            metadata_filters = {}
            if sex_filter != "Any":
                metadata_filters['sex'] = sex_filter.lower()
            if age_filter != "Any":
                metadata_filters['age_group'] = age_filter.lower().replace(' ', '_')
            if skin_tone_filter != "Any":
                metadata_filters['skin_tone'] = skin_tone_filter.lower()
            if hair_color_filter != "Any":
                metadata_filters['hair_color'] = hair_color_filter.lower()
            if brightness_filter != "Any":
                metadata_filters['brightness'] = brightness_filter.lower()
            if quality_filter != "Any":
                metadata_filters['quality'] = quality_filter.lower()

            # Perform search based on mode
            if search_mode == "Vector Search Only":
                results = self.db.search_faces(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    distance_metric='cosine'
                )
            elif search_mode == "Metadata Filter Only":
                search_query = SearchQuery(
                    metadata_filters=metadata_filters,
                    limit=top_k
                )
                results = self.search_engine.search(search_query)
            else:  # Hybrid
                search_query = SearchQuery(
                    query_embedding=query_embedding,
                    metadata_filters=metadata_filters,
                    limit=top_k,
                    distance_metric='cosine'
                )
                results = self.search_engine.search(search_query)

            self.stats['searches']['total'] += 1

            # Format results for Gradio gallery
            gallery_images = []
            for result in results:
                img_path = result.get('file_path', '')
                if os.path.exists(img_path):
                    distance = result.get('distance', 0)
                    gallery_images.append((img_path, f"Distance: {distance:.4f}"))

            message = f"✅ Found {len(gallery_images)} matching faces"
            return gallery_images, message

        except Exception as e:
            return [], f"❌ Search error: {str(e)}"

    def format_stats(self) -> str:
        """Format statistics for display"""
        stats = self.get_system_stats()
        return f"""
**System Statistics**
━━━━━━━━━━━━━━━━━━
📊 Database: {stats['database']}
👤 Total Faces: {stats['total_faces']}

📥 Downloads:
  • Total: {stats['downloads']['total']}
  • Success: {stats['downloads']['success']}
  • Errors: {stats['downloads']['errors']}

🔢 Embeddings:
  • Total: {stats['embeddings']['total']}
  • Success: {stats['embeddings']['success']}
  • Errors: {stats['embeddings']['errors']}

🔍 Searches: {stats['searches']}
        """

    def stop_operation(self):
        """Stop current operation"""
        self.stop_flag.set()
        return "⏹️ Stop signal sent"


def create_app():
    """Create and configure the Gradio interface"""

    app = UnifiedFaceApp()

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .stat-box {
        padding: 10px;
        border-radius: 5px;
        background: #f0f0f0;
    }
    """

    with gr.Blocks(title="Face Processing System") as interface:

        gr.Markdown("""
        # 🎭 Unified Face Processing System
        ### Download → Embed → Index → Search
        All face processing operations in one unified interface
        """)

        # Statistics display (always visible at top)
        with gr.Row():
            stats_display = gr.Markdown(app.format_stats(), elem_classes="stat-box")
            refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm")

        # Main operations in tabs
        with gr.Tabs() as tabs:

            # Tab 1: Download & Process Pipeline
            with gr.TabItem("📥 Download & Process"):
                gr.Markdown("### Complete Pipeline: Download faces and process them into embeddings")

                with gr.Row():
                    # Download section
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1️⃣ Download Faces")
                        download_source = gr.Dropdown(
                            choices=["ThisPersonDoesNotExist", "100k-faces"],
                            value="ThisPersonDoesNotExist",
                            label="Source"
                        )
                        download_count = gr.Slider(1, 100, value=10, step=1, label="Number of faces")
                        download_delay = gr.Slider(0.0, 2.0, value=0.1, step=0.1, label="Delay between downloads (seconds)")

                        with gr.Row():
                            download_btn = gr.Button("⬇️ Start Download", variant="primary")
                            stop_download_btn = gr.Button("⏹️ Stop", variant="stop")

                        download_status = gr.Textbox(label="Download Status", lines=2)
                        download_logs = gr.Textbox(label="📝 Detailed Download Logs", lines=15, max_lines=20, autoscroll=True)

                        # Timer for polling download status
                        download_timer = gr.Timer(0.5)

                    # Process section
                    with gr.Column(scale=1):
                        gr.Markdown("#### 2️⃣ Process & Embed")
                        process_batch_size = gr.Slider(10, 200, value=50, step=10, label="Batch size")
                        process_workers = gr.Slider(1, 8, value=4, step=1, label="Worker threads")
                        process_new_only = gr.Checkbox(label="Process new files only", value=True)

                        with gr.Row():
                            process_btn = gr.Button("⚙️ Start Processing", variant="primary")
                            stop_process_btn = gr.Button("⏹️ Stop", variant="stop")

                        process_status = gr.Textbox(label="Processing Status", lines=2)

                with gr.Row():
                    pipeline_btn = gr.Button("🚀 Run Complete Pipeline (Download + Process)",
                                            variant="primary", size="lg")

            # Tab 2: Search Interface
            with gr.TabItem("🔍 Search Faces"):
                gr.Markdown("### Search for similar faces using images or metadata filters")

                with gr.Row():
                    # Left column: Query input and filters
                    with gr.Column(scale=1):
                        query_image = gr.Image(label="Query Image", type="numpy")

                        with gr.Row():
                            capture_btn = gr.Button("📷 Use Webcam")
                            upload_btn = gr.UploadButton("📁 Upload Image", file_types=["image"])

                        top_k = gr.Slider(1, 50, value=10, step=1, label="Number of results")

                        search_mode = gr.Radio(
                            choices=["Vector Search Only", "Metadata Filter Only", "Hybrid Search"],
                            value="Hybrid Search",
                            label="Search Mode"
                        )

                        gr.Markdown("#### Metadata Filters")

                        with gr.Row():
                            sex_filter = gr.Dropdown(
                                choices=["Any", "Male", "Female"],
                                value="Any",
                                label="Sex"
                            )
                            age_filter = gr.Dropdown(
                                choices=["Any", "Child", "Young Adult", "Adult", "Senior"],
                                value="Any",
                                label="Age Group"
                            )

                        with gr.Row():
                            skin_tone_filter = gr.Dropdown(
                                choices=["Any", "Light", "Medium", "Dark"],
                                value="Any",
                                label="Skin Tone"
                            )
                            hair_color_filter = gr.Dropdown(
                                choices=["Any", "Black", "Brown", "Blonde", "Red", "Gray"],
                                value="Any",
                                label="Hair Color"
                            )

                        with gr.Row():
                            brightness_filter = gr.Dropdown(
                                choices=["Any", "Dark", "Normal", "Bright"],
                                value="Any",
                                label="Brightness"
                            )
                            quality_filter = gr.Dropdown(
                                choices=["Any", "Low", "Medium", "High"],
                                value="Any",
                                label="Quality"
                            )

                        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                        search_status = gr.Textbox(label="Search Status", lines=2)

                    # Right column: Results
                    with gr.Column(scale=2):
                        gr.Markdown("#### Search Results")
                        results_gallery = gr.Gallery(
                            label="Similar Faces",
                            columns=4,
                            rows=3,
                            height="auto",
                            object_fit="contain"
                        )

            # Tab 3: Configuration
            with gr.TabItem("⚙️ Configuration"):
                gr.Markdown("### System Configuration")
                gr.Markdown("Configure database connection and system settings. Changes are saved automatically.")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Database Settings")
                        cfg_db_host = gr.Textbox(label="Host", value=app.config.db_host)
                        cfg_db_port = gr.Number(label="Port", value=app.config.db_port)
                        cfg_db_name = gr.Textbox(label="Database", value=app.config.db_name)
                        cfg_db_user = gr.Textbox(label="User", value=app.config.db_user)
                        cfg_db_password = gr.Textbox(label="Password", value=app.config.db_password, type="password")

                        test_db_btn = gr.Button("🔌 Test Connection")
                        db_test_result = gr.Textbox(label="Connection Test Result", lines=2)

                    with gr.Column():
                        gr.Markdown("#### Application Settings")
                        cfg_faces_dir = gr.Textbox(label="Faces Directory", value=app.config.faces_dir)
                        cfg_embedding_model = gr.Dropdown(
                            choices=["statistical", "facenet", "arcface", "deepface", "vggface2", "openface"],
                            value=app.config.embedding_model,
                            label="Embedding Model"
                        )
                        cfg_download_source = gr.Dropdown(
                            choices=["thispersondoesnotexist", "100k-faces"],
                            value=app.config.download_source,
                            label="Default Download Source"
                        )
                        cfg_batch_size = gr.Slider(10, 200, value=app.config.batch_size, step=10, label="Default Batch Size")
                        cfg_max_workers = gr.Slider(1, 16, value=app.config.max_workers, step=1, label="Max Workers")

                        save_config_btn = gr.Button("💾 Save Configuration", variant="primary")
                        config_status = gr.Textbox(label="Configuration Status", lines=2)

        # Event handlers

        # Download handlers with polling for real-time updates
        download_btn.click(
            fn=app.download_faces,
            inputs=[download_source, download_count, download_delay],
            outputs=[download_status, stats_display, download_logs]
        ).then(
            fn=lambda: gr.Timer(active=True),
            outputs=[download_timer]
        )

        # Timer ticks - update status every 0.5 seconds
        download_timer.tick(
            fn=app.get_download_status,
            outputs=[download_status, stats_display, download_logs]
        )

        stop_download_btn.click(
            fn=app.stop_download_wrapper,
            outputs=[download_status, stats_display, download_logs]
        ).then(
            fn=lambda: gr.Timer(active=False),
            outputs=[download_timer]
        )

        # Process handlers
        process_btn.click(
            fn=app.process_and_embed,
            inputs=[process_batch_size, process_workers, process_new_only],
            outputs=[process_status, stats_display]
        )

        stop_process_btn.click(
            fn=app.stop_operation,
            outputs=[process_status]
        )

        # Pipeline handler (sequential download then process)
        def run_pipeline(source, count, delay, batch_size, workers, new_only):
            # Start download
            app.download_faces(source, count, delay)

            # Wait for download to complete
            import time
            while app.download_active:
                time.sleep(1)

            download_msg = "✅ Download complete"
            download_log = app.get_logs()
            stats1 = app.format_stats()

            # Then process
            process_msg, stats2 = app.process_and_embed(batch_size, workers, new_only)
            return download_msg, process_msg, stats2, download_log

        pipeline_btn.click(
            fn=run_pipeline,
            inputs=[download_source, download_count, download_delay,
                   process_batch_size, process_workers, process_new_only],
            outputs=[download_status, process_status, stats_display, download_logs]
        )

        # Search handlers
        search_btn.click(
            fn=app.search_faces,
            inputs=[query_image, top_k, search_mode, sex_filter, age_filter,
                   skin_tone_filter, hair_color_filter, brightness_filter, quality_filter],
            outputs=[results_gallery, search_status]
        ).then(
            fn=lambda: app.format_stats(),
            outputs=[stats_display]
        )

        # Upload button handler
        upload_btn.upload(
            fn=lambda file: file,
            inputs=[upload_btn],
            outputs=[query_image]
        )

        # Configuration handlers
        test_db_btn.click(
            fn=app.test_database_connection,
            inputs=[cfg_db_host, cfg_db_port, cfg_db_name, cfg_db_user, cfg_db_password],
            outputs=[db_test_result]
        )

        def save_configuration(host, port, db_name, user, password, faces_dir,
                              embedding_model, download_source, batch_size, max_workers):
            try:
                config_dict = {
                    'db_host': host,
                    'db_port': int(port),
                    'db_name': db_name,
                    'db_user': user,
                    'db_password': password,
                    'faces_dir': faces_dir,
                    'embedding_model': embedding_model,
                    'download_source': download_source,
                    'batch_size': int(batch_size),
                    'max_workers': int(max_workers),
                    'download_delay': app.config.download_delay
                }
                app.save_config(config_dict)
                # Reinitialize system with new config
                app.system = None
                app.db = None
                app.search_engine = None
                return "✅ Configuration saved successfully! System will reinitialize on next operation."
            except Exception as e:
                return f"❌ Error saving configuration: {str(e)}"

        save_config_btn.click(
            fn=save_configuration,
            inputs=[cfg_db_host, cfg_db_port, cfg_db_name, cfg_db_user, cfg_db_password,
                   cfg_faces_dir, cfg_embedding_model, cfg_download_source,
                   cfg_batch_size, cfg_max_workers],
            outputs=[config_status]
        )

        # Refresh stats
        refresh_stats_btn.click(
            fn=lambda: app.format_stats(),
            outputs=[stats_display]
        )

    return interface


if __name__ == "__main__":
    # Create and launch the app
    app = create_app()

    # Launch with options
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False  # Set to True to create a public link
    )
