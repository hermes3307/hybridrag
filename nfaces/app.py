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
    FaceEmbedder,
    FaceAnalyzer
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
        self.embed_log = []
        self.log_lock = threading.Lock()
        self.download_active = False
        self.embed_active = False

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

    def add_embed_log(self, message: str):
        """Add a message to the embed log (thread-safe)"""
        with self.log_lock:
            self.embed_log.append(message)
        # Also print to stdout for terminal visibility
        self.safe_print(f"[EMBED] {message}")

    def get_logs(self) -> str:
        """Get all log messages as a single string (thread-safe)"""
        with self.log_lock:
            return "\n".join(self.download_log)

    def get_embed_logs(self) -> str:
        """Get all embed log messages as a single string (thread-safe)"""
        with self.log_lock:
            return "\n".join(self.embed_log)

    def clear_logs(self):
        """Clear all log messages (thread-safe)"""
        with self.log_lock:
            self.download_log = []

    def clear_embed_logs(self):
        """Clear all embed log messages (thread-safe)"""
        with self.log_lock:
            self.embed_log = []

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

    def initialize_for_search(self):
        """Lightweight initialization for search operations only (no downloader)"""
        if not self.db:
            # Only initialize database connection, skip downloader
            self.db = PgVectorDatabaseManager(self.config)
            if not self.db.initialized:
                self.db.initialize()
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

            # Initialize the database connection
            if not test_db.initialize():
                return "❌ Connection failed: Could not initialize database"

            # Get statistics
            stats = test_db.get_statistics()

            # Get count for each embedding model
            conn = test_db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE embedding_facenet IS NOT NULL) as facenet_count,
                    COUNT(*) FILTER (WHERE embedding_arcface IS NOT NULL) as arcface_count,
                    COUNT(*) FILTER (WHERE embedding_vggface2 IS NOT NULL) as vggface2_count,
                    COUNT(*) FILTER (WHERE embedding_insightface IS NOT NULL) as insightface_count,
                    COUNT(*) FILTER (WHERE embedding_statistical IS NOT NULL) as statistical_count
                FROM faces
            """)
            model_counts = cursor.fetchone()
            cursor.close()
            test_db.return_connection(conn)
            test_db.close()

            # Format detailed response
            total_faces = stats.get('total_faces', 0)
            faces_with_embeddings = stats.get('faces_with_embeddings', 0)
            db_size = stats.get('database_size', 'Unknown')

            result = f"✅ Connection successful!\n"
            result += f"📊 Database: {db_name}\n"
            result += f"👤 Total faces: {total_faces:,}\n"
            result += f"🎯 Faces with embeddings: {faces_with_embeddings:,}\n"
            result += f"💾 Database size: {db_size}\n"
            result += f"\n🧠 Embeddings by Model:\n"

            if model_counts:
                if model_counts[0] > 0:
                    result += f"   • FaceNet: {model_counts[0]:,}\n"
                if model_counts[1] > 0:
                    result += f"   • ArcFace: {model_counts[1]:,}\n"
                if model_counts[2] > 0:
                    result += f"   • VGGFace2: {model_counts[2]:,}\n"
                if model_counts[3] > 0:
                    result += f"   • InsightFace: {model_counts[3]:,}\n"
                if model_counts[4] > 0:
                    result += f"   • Statistical: {model_counts[4]:,}\n"

                # Show if no embeddings at all
                if sum(model_counts) == 0:
                    result += f"   (No embeddings found)"
            else:
                result += f"   (Unable to retrieve model counts)"

            return result
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

    def _embed_worker(self, batch_size: int, workers: int, process_new_only: bool):
        """Background worker thread for embedding faces"""
        worker_start = time.time()
        try:
            print(f"[DEBUG] Embed worker thread started: batch_size={batch_size}, workers={workers}, process_new_only={process_new_only}", flush=True)
        except BrokenPipeError:
            pass

        try:
            faces_dir = Path(self.config.faces_dir)
            if not faces_dir.exists():
                self.add_embed_log(f"❌ Faces directory does not exist: {faces_dir}")
                return

            self.add_embed_log("⚙️ Initializing embedding system...")
            init_start = time.time()
            self.initialize_system()
            init_time = time.time() - init_start
            self.add_embed_log(f"✓ System initialized ({init_time:.2f}s)")

            # Get list of image files
            self.add_embed_log(f"📁 Scanning directory: {faces_dir}")
            image_files = list(faces_dir.glob("*.jpg")) + list(faces_dir.glob("*.png"))
            self.add_embed_log(f"✓ Found {len(image_files)} image files")

            if process_new_only:
                self.add_embed_log("🔍 Filtering to only new files (not in database)...")
                # Use processor's method to get only new files
                processor = FaceProcessor(
                    config=self.config,
                    stats=self.system_stats,
                    db_manager=self.db
                )
                new_files = processor.get_new_files_only()
                image_files = [Path(f) for f in new_files]
                self.add_embed_log(f"✓ After filtering: {len(image_files)} files need processing")

            total_files = len(image_files)
            if total_files == 0:
                self.add_embed_log("ℹ️ No files to process")
                return

            # FaceProcessor expects SystemConfig, SystemStats, and db_manager
            if not process_new_only:
                processor = FaceProcessor(
                    config=self.config,
                    stats=self.system_stats,
                    db_manager=self.db
                )

            success_count = 0
            error_count = 0
            duplicate_count = 0
            error_messages = []

            self.add_embed_log(f"🚀 Starting embedding process")
            self.add_embed_log(f"📊 Target: {total_files} faces\n")

            for i, file_path in enumerate(image_files):
                # Check stop flag
                if self.stop_flag.is_set():
                    self.add_embed_log(f"\n⏹️ Embedding stopped by user at {i}/{total_files}")
                    break

                file_name = file_path.name
                self.add_embed_log(f"🔢 [{i+1}/{total_files}] Processing: {file_name}")

                try:
                    # Track if this is a duplicate before processing
                    file_hash = processor._get_file_hash(str(file_path))
                    is_duplicate = processor.db_manager.hash_exists(file_hash)

                    result = processor.process_face_file(str(file_path))

                    if result:
                        if is_duplicate:
                            duplicate_count += 1
                            self.add_embed_log(f"   ⊘ DUPLICATE: Skipped (hash: {file_hash[:8]})")
                        else:
                            success_count += 1
                            self.stats['embeddings']['success'] += 1
                            self.add_embed_log(f"   ✓ SUCCESS: Embedded to database")
                    else:
                        error_count += 1
                        self.stats['embeddings']['errors'] += 1
                        error_msg = f"{file_name}: Processing returned None"
                        error_messages.append(error_msg)
                        self.add_embed_log(f"   ✗ FAILED: Processing returned None")

                    self.stats['embeddings']['total'] += 1

                except Exception as e:
                    error_count += 1
                    self.stats['embeddings']['errors'] += 1
                    error_msg = f"{file_name}: {str(e)}"
                    error_messages.append(error_msg)
                    self.add_embed_log(f"   ✗ ERROR: {str(e)}")

            # Summary
            self.add_embed_log(f"\n{'='*60}")
            self.add_embed_log(f"📈 SUMMARY: {success_count}/{total_files} faces embedded successfully")
            self.add_embed_log(f"✅ Success: {success_count}")
            self.add_embed_log(f"⊘ Duplicates: {duplicate_count}")
            self.add_embed_log(f"❌ Errors: {error_count}")

            if error_messages:
                self.add_embed_log(f"\n⚠️ Errors ({len(error_messages)}):")
                for err in error_messages[:5]:
                    self.add_embed_log(f"  • {err}")
                if len(error_messages) > 5:
                    self.add_embed_log(f"  ... and {len(error_messages) - 5} more errors")

        except Exception as e:
            self.add_embed_log(f"\n❌ CRITICAL ERROR: {str(e)}")
            import traceback
            self.add_embed_log(f"Traceback:\n{traceback.format_exc()}")
            try:
                print(f"❌ Critical error in embed worker: {str(e)}")
                traceback.print_exc()
            except BrokenPipeError:
                pass
        finally:
            self.embed_active = False

    def process_and_embed(self, batch_size: int, workers: int,
                         process_new_only: bool) -> Tuple[str, str, str]:
        """Start embedding process in background thread"""
        if self.embed_active:
            return "⚠️ Embedding already in progress", self.format_stats(), self.get_embed_logs()

        self.stop_flag.clear()
        self.clear_embed_logs()
        self.embed_active = True

        # Start embed in background thread
        embed_thread = threading.Thread(
            target=self._embed_worker,
            args=(batch_size, workers, process_new_only),
            daemon=True
        )
        embed_thread.start()

        return "⏳ Embedding started...", self.format_stats(), self.get_embed_logs()

    def get_embed_status(self):
        """Poll function to get current embed status and logs"""
        if self.embed_active:
            status = "⏳ Embedding..."
        else:
            status = "✅ Ready"

        return status, self.format_stats(), self.get_embed_logs()

    def stop_embed_wrapper(self):
        """Wrapper for stop button that also returns current status"""
        self.stop_flag.set()
        return "⏹️ Stopping...", self.format_stats(), self.get_embed_logs()

    def search_faces(self, query_image, top_k: int, search_mode: str,
                    sex_filter: str, age_filter: str, skin_tone_filter: str,
                    hair_color_filter: str, brightness_filter: str,
                    quality_filter: str) -> Tuple[List, str]:
        """Search for similar faces"""
        # Use lightweight initialization for search (skips downloader initialization)
        self.initialize_for_search()

        if not self.db or not self.search_engine:
            return [], "❌ Database not initialized"

        if query_image is None:
            return [], "⚠️ Please provide a query image"

        temp_file_path = None
        try:
            # Convert query image to embedding
            embedder = FaceEmbedder(model_name=self.config.embedding_model)
            analyzer = FaceAnalyzer()

            # Handle different image input types
            if isinstance(query_image, str):
                query_img = Image.open(query_image)
                query_image_path = query_image
            else:
                query_img = Image.fromarray(query_image)
                # Save temporary image for analysis
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    query_img.save(tmp.name)
                    query_image_path = tmp.name
                    temp_file_path = tmp.name

            # Analyze the query image to extract features
            features = analyzer.analyze_face(query_image_path)

            # Create embedding with both image_path and features
            query_embedding = embedder.create_embedding(query_image_path, features)

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
                    n_results=top_k,
                    distance_metric='cosine',
                    embedding_model=self.config.embedding_model
                )
            elif search_mode == "Metadata Filter Only":
                # Metadata-only search - use db manager directly
                results = self.db.search_by_metadata(
                    metadata_filter=metadata_filters,
                    n_results=top_k
                )
            else:  # Hybrid
                # Hybrid search - combine vector and metadata
                results = self.db.hybrid_search(
                    query_embedding=query_embedding,
                    metadata_filter=metadata_filters,
                    n_results=top_k,
                    embedding_model=self.config.embedding_model
                )

            self.stats['searches']['total'] += 1

            # Format results for Gradio gallery
            gallery_images = []
            for result in results:
                # file_path can be at top level or inside metadata
                img_path = result.get('file_path') or result.get('metadata', {}).get('file_path', '')
                if img_path and os.path.exists(img_path):
                    distance = result.get('distance', 0)
                    gallery_images.append((img_path, f"Distance: {distance:.4f}"))

            message = f"✅ Found {len(gallery_images)} matching faces (searched {len(results)} results)"
            return gallery_images, message

        except Exception as e:
            return [], f"❌ Search error: {str(e)}"
        finally:
            # Clean up temporary file if created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

    def detect_and_search_faces(self, query_image, top_k_per_face: int) -> Tuple[List, str]:
        """
        Advanced multi-face search: Detect all faces in image and search for each one

        Returns:
            Tuple of (results_list, status_message)
            results_list contains dicts with 'face_image', 'face_number', 'search_results'
        """
        self.initialize_for_search()

        if not self.db:
            return [], "❌ Database not initialized"

        if query_image is None:
            return [], "⚠️ Please provide a query image"

        temp_file_path = None
        detected_faces = []

        try:
            import cv2
            import tempfile

            # Convert query image to OpenCV format
            if isinstance(query_image, str):
                query_img = Image.open(query_image)
                query_image_path = query_image
            else:
                query_img = Image.fromarray(query_image)
                # Save temporary image
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    query_img.save(tmp.name)
                    query_image_path = tmp.name
                    temp_file_path = tmp.name

            # Convert to OpenCV format
            img_cv = cv2.imread(query_image_path)
            if img_cv is None:
                return [], "❌ Failed to load image"

            # Detect faces using Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                return [], "⚠️ No faces detected in the image"

            # Process each detected face
            analyzer = FaceAnalyzer()
            embedder = FaceEmbedder(model_name=self.config.embedding_model)

            all_results = []

            for idx, (x, y, w, h) in enumerate(faces, 1):
                # Extract face region with some padding
                padding = int(max(w, h) * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img_cv.shape[1], x + w + padding)
                y2 = min(img_cv.shape[0], y + h + padding)

                face_img = img_cv[y1:y2, x1:x2]

                # Save extracted face to temp file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as face_tmp:
                    cv2.imwrite(face_tmp.name, face_img)
                    face_path = face_tmp.name
                    detected_faces.append(face_path)

                # Analyze and create embedding for this face
                features = analyzer.analyze_face(face_path)
                face_embedding = embedder.create_embedding(face_path, features)

                # Search for similar faces
                search_results = self.db.search_faces(
                    query_embedding=face_embedding,
                    n_results=top_k_per_face,
                    distance_metric='cosine',
                    embedding_model=self.config.embedding_model
                )

                # Format search results
                gallery_images = []
                for result in search_results:
                    img_path = result.get('file_path') or result.get('metadata', {}).get('file_path', '')
                    if img_path and os.path.exists(img_path):
                        distance = result.get('distance', 0)
                        gallery_images.append((img_path, f"Distance: {distance:.4f}"))

                # Convert face image for display
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                all_results.append({
                    'face_number': idx,
                    'face_image': face_img_rgb,
                    'face_bbox': (x, y, w, h),
                    'search_results': gallery_images,
                    'num_results': len(gallery_images)
                })

            message = f"✅ Detected {len(faces)} face(s) and searched for each"
            return all_results, message

        except Exception as e:
            import traceback
            error_detail = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
            return [], error_detail
        finally:
            # Clean up temporary files
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            for face_path in detected_faces:
                try:
                    if os.path.exists(face_path):
                        os.unlink(face_path)
                except:
                    pass

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
                        embed_logs = gr.Textbox(label="📝 Detailed Embedding Logs", lines=15, max_lines=20, autoscroll=True)

                        # Timer for polling embed status
                        embed_timer = gr.Timer(0.5)

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

                        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                        search_status = gr.Textbox(label="Search Status", lines=2)

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

            # Tab 3: Advanced Multi-Face Search
            with gr.TabItem("🎯 Advanced Search"):
                gr.Markdown("### Multi-Face Detection & Search")
                gr.Markdown("Upload a photo with multiple people. The system will detect all faces and search for similar faces for each person.")

                with gr.Row():
                    # Left column: Input
                    with gr.Column(scale=1):
                        adv_query_image = gr.Image(label="Upload Photo with Multiple People", type="numpy")

                        with gr.Row():
                            adv_upload_btn = gr.UploadButton("📁 Upload Photo", file_types=["image"])

                        adv_top_k = gr.Slider(1, 20, value=5, step=1, label="Results per face")

                        adv_search_btn = gr.Button("🎯 Detect & Search All Faces", variant="primary", size="lg")
                        adv_search_status = gr.Textbox(label="Status", lines=2)

                    # Right column: Annotated image with detected faces
                    with gr.Column(scale=1):
                        gr.Markdown("#### Detected Faces Preview")
                        adv_detected_info = gr.Markdown("Upload an image to detect faces")

                # Results section - show each detected face and its matches
                gr.Markdown("---")
                gr.Markdown("### Search Results for Each Detected Face")

                # Person 1
                with gr.Row(visible=True) as adv_face1_row:
                    with gr.Column(scale=1):
                        adv_face1_image = gr.Image(label="👤 Person 1 - Detected Face", height=200, width=200, visible=False)
                    with gr.Column(scale=3):
                        adv_face1_results = gr.Gallery(label="Similar Faces", columns=5, rows=1, height=200, visible=False)

                # Person 2
                with gr.Row(visible=True) as adv_face2_row:
                    with gr.Column(scale=1):
                        adv_face2_image = gr.Image(label="👤 Person 2 - Detected Face", height=200, width=200, visible=False)
                    with gr.Column(scale=3):
                        adv_face2_results = gr.Gallery(label="Similar Faces", columns=5, rows=1, height=200, visible=False)

                # Person 3
                with gr.Row(visible=True) as adv_face3_row:
                    with gr.Column(scale=1):
                        adv_face3_image = gr.Image(label="👤 Person 3 - Detected Face", height=200, width=200, visible=False)
                    with gr.Column(scale=3):
                        adv_face3_results = gr.Gallery(label="Similar Faces", columns=5, rows=1, height=200, visible=False)

                # Person 4
                with gr.Row(visible=True) as adv_face4_row:
                    with gr.Column(scale=1):
                        adv_face4_image = gr.Image(label="👤 Person 4 - Detected Face", height=200, width=200, visible=False)
                    with gr.Column(scale=3):
                        adv_face4_results = gr.Gallery(label="Similar Faces", columns=5, rows=1, height=200, visible=False)

                # Person 5
                with gr.Row(visible=True) as adv_face5_row:
                    with gr.Column(scale=1):
                        adv_face5_image = gr.Image(label="👤 Person 5 - Detected Face", height=200, width=200, visible=False)
                    with gr.Column(scale=3):
                        adv_face5_results = gr.Gallery(label="Similar Faces", columns=5, rows=1, height=200, visible=False)

            # Tab 4: Configuration
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

        # Process handlers with polling for real-time updates
        process_btn.click(
            fn=app.process_and_embed,
            inputs=[process_batch_size, process_workers, process_new_only],
            outputs=[process_status, stats_display, embed_logs]
        ).then(
            fn=lambda: gr.Timer(active=True),
            outputs=[embed_timer]
        )

        # Timer ticks - update embed status every 0.5 seconds
        embed_timer.tick(
            fn=app.get_embed_status,
            outputs=[process_status, stats_display, embed_logs]
        )

        stop_process_btn.click(
            fn=app.stop_embed_wrapper,
            outputs=[process_status, stats_display, embed_logs]
        ).then(
            fn=lambda: gr.Timer(active=False),
            outputs=[embed_timer]
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
            process_msg, stats2, embed_log = app.process_and_embed(batch_size, workers, new_only)

            # Wait for embedding to complete
            while app.embed_active:
                time.sleep(1)

            # Get final embed status
            final_process_msg, final_stats, final_embed_log = app.get_embed_status()

            return download_msg, final_process_msg, final_stats, download_log, final_embed_log

        pipeline_btn.click(
            fn=run_pipeline,
            inputs=[download_source, download_count, download_delay,
                   process_batch_size, process_workers, process_new_only],
            outputs=[download_status, process_status, stats_display, download_logs, embed_logs]
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

        # Advanced search handlers
        def process_advanced_search(image, top_k):
            """Process multi-face detection and search"""
            if image is None:
                return (
                    "⚠️ Please upload an image",
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False)
                )

            results, status = app.detect_and_search_faces(image, top_k)

            if not results:
                return (
                    status,
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False)
                )

            # Prepare outputs for up to 5 faces
            outputs = [status]

            for i in range(5):
                if i < len(results):
                    face_data = results[i]
                    outputs.extend([
                        gr.update(value=face_data['face_image'], visible=True),  # face image
                        gr.update(value=face_data['search_results'], visible=True)  # search results gallery
                    ])
                else:
                    outputs.extend([
                        gr.update(visible=False),  # face image
                        gr.update(visible=False)   # search results gallery
                    ])

            return tuple(outputs)

        adv_search_btn.click(
            fn=process_advanced_search,
            inputs=[adv_query_image, adv_top_k],
            outputs=[
                adv_search_status,
                adv_face1_image, adv_face1_results,
                adv_face2_image, adv_face2_results,
                adv_face3_image, adv_face3_results,
                adv_face4_image, adv_face4_results,
                adv_face5_image, adv_face5_results
            ]
        )

        adv_upload_btn.upload(
            fn=lambda file: file,
            inputs=[adv_upload_btn],
            outputs=[adv_query_image]
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

    # Get absolute path to faces directory
    faces_dir = os.path.abspath("./faces")
    allowed_paths = [faces_dir]

    # Also include real path if it's a symlink
    if os.path.islink("./faces"):
        real_faces_dir = os.path.realpath("./faces")
        if real_faces_dir not in allowed_paths:
            allowed_paths.append(real_faces_dir)

    print(f"Gradio allowed_paths: {allowed_paths}")

    # Launch with options
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create a public link
        allowed_paths=allowed_paths  # Allow reading from faces directory
    )
