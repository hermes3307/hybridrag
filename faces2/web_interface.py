#!/usr/bin/env python3
"""
Web Interface for Integrated Face Processing System
Provides all features from the GUI in a web-based interface using Gradio
"""

import gradio as gr
import os
import time
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import threading

# Import the core backend
from core_backend import (
    IntegratedFaceSystem,
    SystemConfig,
    FaceAnalyzer,
    FaceEmbedder,
    CHROMADB_AVAILABLE,
    CV2_AVAILABLE
)

class WebInterface:
    """Web interface for the integrated face processing system"""

    def __init__(self):
        self.system = None
        self.download_thread = None
        self.is_downloading = False
        self.download_count = 0
        self.download_target = 0
        self.download_errors = 0
        self.download_duplicates = 0
        self.download_status_msg = ""
        self.downloaded_images = []  # Store downloaded image paths for gallery

    def initialize_system(self) -> str:
        """Initialize the face processing system"""
        try:
            self.system = IntegratedFaceSystem()
            if self.system.initialize():
                return "✅ System initialized successfully!"
            else:
                return "❌ Failed to initialize system. Check dependencies."
        except Exception as e:
            return f"❌ Error initializing system: {e}"

    def get_system_status(self) -> str:
        """Get system status information"""
        if not self.system:
            return "System not initialized"

        try:
            status = self.system.get_system_status()
            stats = status.get('statistics', {})
            db_info = status.get('database', {})

            status_text = f"""
**System Status**

**Database:**
- Status: {'Connected' if db_info else 'Disconnected'}
- Collection: {db_info.get('name', 'N/A')}
- Total Faces: {db_info.get('count', 0)}

**Statistics:**
- Downloads: {stats.get('download_success', 0)} (Duplicates: {stats.get('download_duplicates', 0)}, Errors: {stats.get('download_errors', 0)})
- Embeddings: {stats.get('embed_success', 0)} (Errors: {stats.get('embed_errors', 0)})
- Search Queries: {stats.get('search_queries', 0)}
- Download Rate: {stats.get('download_rate', 0):.2f}/sec
- Embed Rate: {stats.get('embed_rate', 0):.2f}/sec
- Elapsed Time: {stats.get('elapsed_time', 0):.1f}s

**Configuration:**
- Faces Directory: {status.get('faces_directory', 'N/A')}
- Faces Count: {status.get('faces_count', 0)}
"""
            return status_text
        except Exception as e:
            return f"Error getting status: {e}"

    # Download Functions
    def download_single_face(self) -> Tuple[Optional[str], str]:
        """Download a single face"""
        if not self.system:
            return None, "❌ System not initialized"

        try:
            file_path = self.system.downloader.download_face()
            if file_path:
                return file_path, f"✅ Downloaded: {os.path.basename(file_path)}"
            else:
                return None, "⚠️ No new face downloaded (duplicate or error)"
        except Exception as e:
            return None, f"❌ Download error: {e}"

    def start_continuous_download(self, num_faces: int):
        """Start downloading multiple faces with live progress updates and preview"""
        if not self.system:
            yield None, "❌ System not initialized", []
            return

        if self.is_downloading:
            yield None, "⚠️ Download already in progress", []
            return

        self.is_downloading = True
        self.download_count = 0
        self.download_target = num_faces
        self.download_errors = 0
        self.download_duplicates = 0
        self.downloaded_images = []  # Reset gallery

        start_time = time.time()
        yield None, f"▶️ Starting download of {num_faces} faces...", []

        for i in range(num_faces):
            if not self.is_downloading:
                yield None, f"⏹️ Download stopped at {self.download_count}/{num_faces}", self.downloaded_images
                break

            # Update status before download
            progress = ((i + 1) / num_faces) * 100
            yield None, f"⏳ Downloading {i+1}/{num_faces} ({progress:.0f}%)...", self.downloaded_images

            file_path = self.system.downloader.download_face()

            if file_path:
                self.download_count += 1
                elapsed = time.time() - start_time
                rate = self.download_count / elapsed if elapsed > 0 else 0

                # Add to gallery
                self.downloaded_images.append(file_path)

                # Load image for preview
                try:
                    image = Image.open(file_path)
                    image_array = np.array(image)
                    status_msg = f"✅ Downloaded {self.download_count}/{num_faces} | Rate: {rate:.2f}/sec | File: {os.path.basename(file_path)}"
                    yield image_array, status_msg, self.downloaded_images
                except Exception as e:
                    yield None, f"✅ Downloaded {self.download_count}/{num_faces} | Rate: {rate:.2f}/sec | Preview error: {e}", self.downloaded_images
            else:
                # Check if it was duplicate or error
                stats = self.system.stats.get_stats()
                self.download_duplicates = stats.get('download_duplicates', 0)
                self.download_errors = stats.get('download_errors', 0)
                yield None, f"⚠️ Progress {i+1}/{num_faces} | New: {self.download_count}, Duplicates: {self.download_duplicates}, Errors: {self.download_errors}", self.downloaded_images

            time.sleep(self.system.config.download_delay)

        # Final status
        self.is_downloading = False
        elapsed = time.time() - start_time

        if self.download_count == num_faces:
            yield None, f"🎉 Download complete! {self.download_count}/{num_faces} faces in {elapsed:.1f}s", self.downloaded_images
        else:
            yield None, f"✅ Download finished: {self.download_count} new, {self.download_duplicates} duplicates, {self.download_errors} errors in {elapsed:.1f}s", self.downloaded_images

    def get_download_status(self) -> str:
        """Get current download status"""
        if self.download_status_msg:
            return self.download_status_msg
        elif self.is_downloading:
            return f"⏳ Downloading... {self.download_count}/{self.download_target}"
        else:
            return "Ready to download"

    def stop_download(self) -> str:
        """Stop downloading"""
        if not self.is_downloading:
            return "⚠️ No download in progress"

        self.is_downloading = False
        return f"⏹️ Stopping download... ({self.download_count}/{self.download_target} completed)"

    def capture_from_camera(self) -> Tuple[Optional[np.ndarray], str]:
        """Capture image from camera (one-shot)"""
        if not CV2_AVAILABLE:
            return None, "❌ OpenCV not available. Install: pip install opencv-python"

        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return None, "❌ Cannot access camera"

            ret, frame = cap.read()
            cap.release()

            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Save to download directory
                if self.system:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"camera_capture_{timestamp}.jpg"
                    file_path = os.path.join(self.system.config.faces_dir, filename)
                    cv2.imwrite(file_path, frame)

                    return frame_rgb, f"✅ Captured and saved: {filename}"
                else:
                    return frame_rgb, "⚠️ Captured but system not initialized"
            else:
                return None, "❌ Failed to capture frame"
        except Exception as e:
            return None, f"❌ Camera error: {e}"

    def save_webcam_image(self, webcam_image: Optional[np.ndarray]) -> str:
        """Save image from live webcam component"""
        if webcam_image is None:
            return "❌ No image captured from webcam"

        if not self.system:
            return "❌ System not initialized"

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"webcam_{timestamp}.jpg"
            file_path = os.path.join(self.system.config.faces_dir, filename)

            # Webcam image is already RGB, convert to BGR for saving
            if CV2_AVAILABLE:
                webcam_bgr = cv2.cvtColor(webcam_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, webcam_bgr)
            else:
                # Fallback to PIL
                Image.fromarray(webcam_image).save(file_path)

            return f"✅ Webcam image saved: {filename}"
        except Exception as e:
            return f"❌ Error saving webcam image: {e}"

    # Search Functions
    def search_faces(self, query_image, search_mode: str, num_results: int,
                    sex: str, age_group: str, skin_tone: str, skin_color: str,
                    hair_color: str) -> Tuple[List, str]:
        """Search for similar faces"""
        if not self.system:
            return [], "❌ System not initialized"

        try:
            # Build metadata filter
            metadata_filter = {}
            if sex != "any":
                metadata_filter['sex'] = sex
            if age_group != "any":
                metadata_filter['age_group'] = age_group
            if skin_tone != "any":
                metadata_filter['skin_tone'] = skin_tone
            if skin_color != "any":
                metadata_filter['skin_color'] = skin_color
            if hair_color != "any":
                metadata_filter['hair_color'] = hair_color

            # Perform search based on mode
            if search_mode == "metadata":
                if not metadata_filter:
                    return [], "⚠️ Please select at least one metadata filter"

                results = self.system.db_manager.search_by_metadata(metadata_filter, num_results)
                message = f"✅ Metadata search: {len(results)} results"

            elif search_mode in ["vector", "hybrid"]:
                if query_image is None:
                    return [], "❌ Please provide a query image"

                # Save query image temporarily
                temp_path = os.path.join(self.system.config.faces_dir, "temp", "query_temp.jpg")
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                Image.fromarray(query_image).save(temp_path)

                # Create embedding
                analyzer = FaceAnalyzer()
                embedder = FaceEmbedder()
                features = analyzer.analyze_face(temp_path)
                embedding = embedder.create_embedding(temp_path, features)

                if search_mode == "hybrid" and metadata_filter:
                    results = self.system.db_manager.hybrid_search(embedding, metadata_filter, num_results)
                    message = f"✅ Hybrid search: {len(results)} results"
                else:
                    results = self.system.db_manager.search_faces(embedding, num_results)
                    message = f"✅ Vector search: {len(results)} results"

            # Convert results to image gallery
            gallery_images = []
            for result in results:
                metadata = result.get('metadata', {})
                file_path = metadata.get('file_path', '')

                if os.path.exists(file_path):
                    # Create caption with metadata
                    distance = result.get('distance', 0)
                    sex = metadata.get('sex', 'unknown')
                    age = metadata.get('estimated_age', 'unknown')
                    skin = metadata.get('skin_color', 'unknown')
                    hair = metadata.get('hair_color', 'unknown')

                    caption = f"Distance: {distance:.3f} | {sex}, {age}y, {skin} skin, {hair} hair"
                    gallery_images.append((file_path, caption))

            self.system.stats.increment_search_queries()
            return gallery_images, message

        except Exception as e:
            return [], f"❌ Search error: {e}"

    # Processing Functions
    def process_all_faces(self) -> str:
        """Process all faces for embedding"""
        if not self.system:
            return "❌ System not initialized"

        try:
            face_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                face_files.extend(Path(self.system.config.faces_dir).rglob(ext))

            processed = 0
            for file_path in face_files:
                if self.system.processor.process_face_file(str(file_path)):
                    processed += 1

            return f"✅ Processed {processed} faces out of {len(face_files)} total files"
        except Exception as e:
            return f"❌ Processing error: {e}"

    # Configuration Functions
    def initialize_vector_database(self, db_path: str, collection_name: str) -> str:
        """Initialize vector database"""
        if not self.system:
            return "❌ System not initialized"

        try:
            self.system.config.db_path = db_path
            self.system.config.collection_name = collection_name
            self.system.db_manager.config = self.system.config

            if self.system.db_manager.initialize():
                return f"✅ Vector database initialized!\n\nPath: {db_path}\nCollection: {collection_name}"
            else:
                return "❌ Failed to initialize database"
        except Exception as e:
            return f"❌ Error: {e}"

    def initialize_download_directory(self, faces_dir: str) -> str:
        """Initialize download directory"""
        try:
            os.makedirs(faces_dir, exist_ok=True)

            if self.system:
                self.system.config.faces_dir = faces_dir
                self.system.downloader.config.faces_dir = faces_dir

            # Count files
            existing_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                existing_files.extend(Path(faces_dir).rglob(ext))

            file_count = len(existing_files)
            total_size = sum(f.stat().st_size for f in existing_files if f.is_file())
            size_mb = total_size / (1024 * 1024)

            return f"✅ Directory initialized!\n\nPath: {faces_dir}\nFiles: {file_count}\nSize: {size_mb:.2f} MB"
        except Exception as e:
            return f"❌ Error: {e}"

    def save_configuration(self, faces_dir: str, db_path: str, collection_name: str,
                          download_delay: float, batch_size: int, max_workers: int) -> str:
        """Save configuration"""
        if not self.system:
            return "❌ System not initialized"

        try:
            self.system.config.faces_dir = faces_dir
            self.system.config.db_path = db_path
            self.system.config.collection_name = collection_name
            self.system.config.download_delay = download_delay
            self.system.config.batch_size = batch_size
            self.system.config.max_workers = max_workers

            self.system.config.save_to_file()

            return f"""✅ Configuration saved!

Faces Directory: {faces_dir}
Database Path: {db_path}
Collection: {collection_name}
Download Delay: {download_delay}s
Batch Size: {batch_size}
Max Workers: {max_workers}
"""
        except Exception as e:
            return f"❌ Error saving: {e}"

    def load_configuration(self) -> str:
        """Load configuration"""
        try:
            config = SystemConfig.from_file()

            if self.system:
                self.system.config = config

            return f"""✅ Configuration loaded!

Faces Directory: {config.faces_dir}
Database Path: {config.db_path}
Collection: {config.collection_name}
Download Delay: {config.download_delay}s
Batch Size: {config.batch_size}
Max Workers: {config.max_workers}
"""
        except Exception as e:
            return f"❌ Error loading: {e}"

def create_interface():
    """Create the Gradio interface"""
    web_app = WebInterface()

    # Initialize system on startup
    init_message = web_app.initialize_system()
    print(init_message)

    with gr.Blocks(title="Integrated Face Processing System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎭 Integrated Face Processing System")
        gr.Markdown("Complete face download, processing, and search system with camera support")

        # System Overview Tab
        with gr.Tab("📊 System Overview"):
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Status", variant="primary")
            status_output = gr.Markdown(value=web_app.get_system_status())
            refresh_btn.click(web_app.get_system_status, outputs=status_output)

        # Download Tab
        with gr.Tab("⬇️ Download Faces"):
            gr.Markdown("### Download faces from ThisPersonDoesNotExist.com or capture from camera")

            with gr.Row():
                with gr.Column(scale=2):
                    num_faces = gr.Slider(1, 100, value=10, step=1, label="Number of Faces to Download")

                    with gr.Row():
                        download_single_btn = gr.Button("📥 Download Single", variant="secondary")
                        start_download_btn = gr.Button("▶️ Start Download", variant="primary")
                        stop_download_btn = gr.Button("⏹️ Stop Download", variant="stop")

                    download_status = gr.Textbox(label="Status", lines=2)

                with gr.Column(scale=1):
                    camera_preview = gr.Image(label="Latest Downloaded Image")

            # Live webcam section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📹 Live Webcam Capture")
                    webcam_live = gr.Image(sources=["webcam"], label="Live Webcam", type="numpy")
                    capture_webcam_btn = gr.Button("📸 Capture from Webcam", variant="primary")
                    webcam_status = gr.Textbox(label="Webcam Status", lines=1)

            # Gallery for batch downloads
            download_gallery = gr.Gallery(
                label="Downloaded Images (Batch View)",
                columns=5,
                rows=2,
                height="auto",
                object_fit="contain",
                show_label=True
            )

            # Connect download buttons
            download_single_btn.click(
                web_app.download_single_face,
                outputs=[camera_preview, download_status]
            )

            start_download_btn.click(
                web_app.start_continuous_download,
                inputs=num_faces,
                outputs=[camera_preview, download_status, download_gallery]
            )

            stop_download_btn.click(
                web_app.stop_download,
                outputs=download_status
            )

            capture_webcam_btn.click(
                web_app.save_webcam_image,
                inputs=webcam_live,
                outputs=webcam_status
            )

        # Process & Embed Tab
        with gr.Tab("⚙️ Process & Embed"):
            gr.Markdown("### Create embeddings for downloaded faces")

            process_btn = gr.Button("🔄 Process All Faces", variant="primary", size="lg")
            process_status = gr.Textbox(label="Processing Status", lines=5)

            process_btn.click(
                web_app.process_all_faces,
                outputs=process_status
            )

        # Search Tab
        with gr.Tab("🔍 Search Faces"):
            gr.Markdown("### Search for similar faces using image and/or metadata filters")

            with gr.Row():
                with gr.Column(scale=2):
                    # Query image input
                    with gr.Row():
                        query_image = gr.Image(label="Query Image (Upload)", type="numpy")

                    # Live webcam for search
                    gr.Markdown("#### 📹 Or Use Live Webcam")
                    with gr.Row():
                        search_webcam = gr.Image(sources=["webcam"], label="Live Webcam for Search", type="numpy")
                        use_webcam_btn = gr.Button("📸 Use Webcam Image", variant="secondary")

                    # Search controls
                    search_mode = gr.Radio(
                        choices=["vector", "metadata", "hybrid"],
                        value="vector",
                        label="Search Mode",
                        info="Vector: Image similarity | Metadata: Attributes only | Hybrid: Both"
                    )

                    num_results = gr.Slider(1, 50, value=10, step=1, label="Number of Results")

                    # Metadata filters
                    with gr.Accordion("🎯 Metadata Filters", open=False):
                        with gr.Row():
                            sex_filter = gr.Dropdown(
                                choices=["any", "male", "female", "unknown"],
                                value="any",
                                label="Sex"
                            )
                            age_filter = gr.Dropdown(
                                choices=["any", "child", "young_adult", "adult", "middle_aged", "senior"],
                                value="any",
                                label="Age Group"
                            )

                        with gr.Row():
                            skin_tone_filter = gr.Dropdown(
                                choices=["any", "very_light", "light", "medium", "tan", "brown", "dark"],
                                value="any",
                                label="Skin Tone"
                            )
                            skin_color_filter = gr.Dropdown(
                                choices=["any", "light", "medium", "dark"],
                                value="any",
                                label="Skin Color"
                            )

                        with gr.Row():
                            hair_color_filter = gr.Dropdown(
                                choices=["any", "black", "dark_brown", "brown", "blonde", "red", "gray", "light_gray", "other"],
                                value="any",
                                label="Hair Color"
                            )

                    search_btn = gr.Button("🔍 Search Faces", variant="primary", size="lg")
                    search_status = gr.Textbox(label="Search Status", lines=2)

                with gr.Column(scale=1):
                    gr.Markdown("### Query Preview")
                    query_preview = gr.Image(label="Your Query Image", type="numpy")

            # Search results
            with gr.Row():
                search_results = gr.Gallery(
                    label="Search Results",
                    columns=5,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )

            # Connect search buttons
            # Copy webcam image to query image
            use_webcam_btn.click(
                lambda img: img,
                inputs=search_webcam,
                outputs=query_image
            )

            # Update preview when query image changes
            query_image.change(
                lambda img: img,
                inputs=query_image,
                outputs=query_preview
            )

            search_btn.click(
                web_app.search_faces,
                inputs=[query_image, search_mode, num_results, sex_filter, age_filter,
                       skin_tone_filter, skin_color_filter, hair_color_filter],
                outputs=[search_results, search_status]
            )

        # Configuration Tab
        with gr.Tab("⚙️ Configuration"):
            gr.Markdown("### System Configuration")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Database Configuration**")
                    db_path_input = gr.Textbox(value="./chroma_db", label="Database Path")
                    collection_name_input = gr.Textbox(value="faces", label="Collection Name")
                    init_db_btn = gr.Button("🗄️ Initialize Vector Database", variant="primary")
                    db_status = gr.Textbox(label="Database Status", lines=3)

                with gr.Column():
                    gr.Markdown("**Download Configuration**")
                    faces_dir_input = gr.Textbox(value="./faces", label="Faces Directory")
                    init_dir_btn = gr.Button("📁 Initialize Download Directory", variant="primary")
                    dir_status = gr.Textbox(label="Directory Status", lines=3)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**System Settings**")
                    download_delay_input = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Download Delay (seconds)")
                    batch_size_input = gr.Slider(1, 200, value=50, step=1, label="Batch Size")
                    max_workers_input = gr.Slider(1, 8, value=2, step=1, label="Max Workers")

                    with gr.Row():
                        save_config_btn = gr.Button("💾 Save Configuration", variant="primary")
                        load_config_btn = gr.Button("📂 Load Configuration", variant="secondary")

                    config_status = gr.Textbox(label="Configuration Status", lines=5)

            # Connect configuration buttons
            init_db_btn.click(
                web_app.initialize_vector_database,
                inputs=[db_path_input, collection_name_input],
                outputs=db_status
            )

            init_dir_btn.click(
                web_app.initialize_download_directory,
                inputs=faces_dir_input,
                outputs=dir_status
            )

            save_config_btn.click(
                web_app.save_configuration,
                inputs=[faces_dir_input, db_path_input, collection_name_input,
                       download_delay_input, batch_size_input, max_workers_input],
                outputs=config_status
            )

            load_config_btn.click(
                web_app.load_configuration,
                outputs=config_status
            )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
