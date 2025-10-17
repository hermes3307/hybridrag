#!/usr/bin/env python3
"""
Console Face Processing Application
Demonstrates the integrated face processing system without GUI
"""

import os
import sys
import json
import time
import hashlib
import threading
from datetime import datetime
from pathlib import Path

# Check dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("❌ requests not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("❌ PIL not available")

class ConsoleConfig:
    """Configuration management"""
    def __init__(self):
        self.faces_dir = "./faces"
        self.download_delay = 1.0
        self.config_file = "console_config.json"

    def save(self):
        data = {
            'faces_dir': self.faces_dir,
            'download_delay': self.download_delay
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Configuration saved to {self.config_file}")

    def load(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.faces_dir = data.get('faces_dir', './faces')
                self.download_delay = data.get('download_delay', 1.0)
            print(f"✅ Configuration loaded from {self.config_file}")
        else:
            print("ℹ️  Using default configuration")

class ConsoleDownloader:
    """Face downloader with console interface"""
    def __init__(self, config):
        self.config = config
        self.running = False
        self.downloaded_count = 0
        self.error_count = 0
        self.duplicate_count = 0
        self.downloaded_hashes = set()

        # Create faces directory
        os.makedirs(self.config.faces_dir, exist_ok=True)

        # Load existing hashes
        self._load_existing_hashes()

    def _load_existing_hashes(self):
        """Load hashes of existing images"""
        print(f"🔍 Scanning existing images in {self.config.faces_dir}...")
        count = 0
        for file_path in Path(self.config.faces_dir).rglob("*.jpg"):
            if file_path.is_file():
                try:
                    file_hash = self._get_file_hash(str(file_path))
                    if file_hash:
                        self.downloaded_hashes.add(file_hash)
                        count += 1
                except Exception:
                    pass
        print(f"📁 Found {count} existing images")

    def _get_file_hash(self, file_path):
        """Calculate hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def download_single_face(self):
        """Download a single face"""
        if not REQUESTS_AVAILABLE:
            return False, "requests module not available"

        try:
            print("📥 Downloading face...", end=" ", flush=True)

            response = requests.get(
                "https://thispersondoesnotexist.com/",
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30
            )
            response.raise_for_status()

            # Calculate hash for duplicate detection
            image_hash = hashlib.md5(response.content).hexdigest()

            # Check for duplicates
            if image_hash in self.downloaded_hashes:
                self.duplicate_count += 1
                print("🔄 (duplicate)")
                return False, "Duplicate image skipped"

            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"face_{timestamp}_{image_hash[:8]}.jpg"
            file_path = os.path.join(self.config.faces_dir, filename)

            # Save image
            with open(file_path, 'wb') as f:
                f.write(response.content)

            # Add to downloaded hashes
            self.downloaded_hashes.add(image_hash)

            self.downloaded_count += 1
            print(f"✅ {filename}")
            return True, f"Downloaded: {filename}"

        except Exception as e:
            self.error_count += 1
            print(f"❌ Error: {str(e)}")
            return False, f"Error: {str(e)}"

    def download_batch(self, count=5):
        """Download a batch of faces"""
        print(f"\n🚀 Starting batch download of {count} faces...")
        print("=" * 50)

        start_time = time.time()

        for i in range(count):
            print(f"[{i+1}/{count}] ", end="")
            success, message = self.download_single_face()

            if i < count - 1:  # Don't delay after last download
                time.sleep(self.config.download_delay)

        elapsed = time.time() - start_time

        print("\n" + "=" * 50)
        print(f"📊 Batch completed in {elapsed:.1f} seconds")
        print(f"✅ Downloaded: {self.downloaded_count}")
        print(f"🔄 Duplicates: {self.duplicate_count}")
        print(f"❌ Errors: {self.error_count}")

    def start_continuous_download(self):
        """Start continuous downloading"""
        print("\n🔄 Starting continuous download...")
        print("Press Ctrl+C to stop")
        print("=" * 50)

        self.running = True
        count = 0

        try:
            while self.running:
                count += 1
                print(f"[{count}] ", end="")
                success, message = self.download_single_face()
                time.sleep(self.config.download_delay)

        except KeyboardInterrupt:
            print("\n\n⏹️  Download stopped by user")
            self.running = False

    def show_stats(self):
        """Show download statistics"""
        print("\n📊 DOWNLOAD STATISTICS")
        print("=" * 30)
        print(f"✅ Successfully downloaded: {self.downloaded_count}")
        print(f"🔄 Duplicates skipped: {self.duplicate_count}")
        print(f"❌ Errors encountered: {self.error_count}")
        print(f"📁 Total unique images: {len(self.downloaded_hashes)}")
        print(f"📂 Save directory: {self.config.faces_dir}")

class FaceAnalyzer:
    """Simple face analysis"""
    def __init__(self):
        pass

    def analyze_images(self, directory):
        """Analyze all images in directory"""
        if not PIL_AVAILABLE:
            print("❌ PIL not available for image analysis")
            return

        print(f"\n🔍 Analyzing images in {directory}...")

        image_files = list(Path(directory).rglob("*.jpg"))
        if not image_files:
            print("📁 No images found")
            return

        print(f"📁 Found {len(image_files)} images")

        total_size = 0
        dimensions = []

        for i, file_path in enumerate(image_files):
            try:
                with Image.open(file_path) as img:
                    dimensions.append(img.size)
                    total_size += os.path.getsize(file_path)

                if (i + 1) % 10 == 0:
                    print(f"   Analyzed {i+1}/{len(image_files)} images...")

            except Exception as e:
                print(f"❌ Error analyzing {file_path}: {e}")

        if dimensions:
            avg_width = sum(d[0] for d in dimensions) / len(dimensions)
            avg_height = sum(d[1] for d in dimensions) / len(dimensions)

            print(f"\n📊 ANALYSIS RESULTS")
            print("=" * 25)
            print(f"📁 Total images: {len(image_files)}")
            print(f"💾 Total size: {total_size / (1024*1024):.1f} MB")
            print(f"📐 Average dimensions: {avg_width:.0f} x {avg_height:.0f}")
            print(f"🖼️  Most common size: {max(set(dimensions), key=dimensions.count)}")

class ConsoleFaceApp:
    """Main console application"""

    def __init__(self):
        self.config = ConsoleConfig()
        self.config.load()
        self.downloader = ConsoleDownloader(self.config)
        self.analyzer = FaceAnalyzer()

    def show_banner(self):
        """Show application banner"""
        print("=" * 60)
        print("🎭 INTEGRATED FACE PROCESSING SYSTEM - CONSOLE VERSION")
        print("=" * 60)
        print(f"📂 Faces directory: {self.config.faces_dir}")
        print(f"⏱️  Download delay: {self.config.download_delay}s")
        print(f"🌐 Requests available: {'✅' if REQUESTS_AVAILABLE else '❌'}")
        print(f"🖼️  PIL available: {'✅' if PIL_AVAILABLE else '❌'}")
        print()

    def show_menu(self):
        """Show main menu"""
        print("\n📋 MAIN MENU")
        print("=" * 20)
        print("1. Download single face")
        print("2. Download batch of faces")
        print("3. Start continuous download")
        print("4. Analyze existing images")
        print("5. Show download statistics")
        print("6. Configure settings")
        print("7. List downloaded faces")
        print("8. Open faces folder")
        print("9. Save configuration")
        print("0. Exit")
        print()

    def configure_settings(self):
        """Configure application settings"""
        print("\n⚙️  CONFIGURATION")
        print("=" * 20)

        # Faces directory
        print(f"Current faces directory: {self.config.faces_dir}")
        new_dir = input("Enter new directory (or press Enter to keep current): ").strip()
        if new_dir:
            self.config.faces_dir = new_dir
            self.downloader.config.faces_dir = new_dir
            os.makedirs(new_dir, exist_ok=True)
            print(f"✅ Faces directory updated to: {new_dir}")

        # Download delay
        print(f"Current download delay: {self.config.download_delay}s")
        try:
            new_delay = input("Enter new delay in seconds (or press Enter to keep current): ").strip()
            if new_delay:
                self.config.download_delay = float(new_delay)
                self.downloader.config.download_delay = float(new_delay)
                print(f"✅ Download delay updated to: {new_delay}s")
        except ValueError:
            print("❌ Invalid delay value")

    def list_faces(self):
        """List downloaded faces"""
        face_files = list(Path(self.config.faces_dir).rglob("*.jpg"))

        print(f"\n📁 DOWNLOADED FACES ({len(face_files)} files)")
        print("=" * 40)

        if not face_files:
            print("No faces found")
            return

        # Sort by modification time (newest first)
        face_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for i, file_path in enumerate(face_files[:20]):  # Show first 20
            stat = file_path.stat()
            size_kb = stat.st_size / 1024
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i+1:2d}. {file_path.name} ({size_kb:.1f} KB) - {mod_time}")

        if len(face_files) > 20:
            print(f"... and {len(face_files) - 20} more files")

    def open_faces_folder(self):
        """Open faces folder"""
        if os.path.exists(self.config.faces_dir):
            try:
                print(f"📂 Opening {self.config.faces_dir}...")
                os.system(f'xdg-open "{self.config.faces_dir}" &')
            except Exception as e:
                print(f"❌ Error opening folder: {e}")
                print(f"📂 Faces are saved in: {os.path.abspath(self.config.faces_dir)}")
        else:
            print("❌ Faces directory does not exist")

    def run(self):
        """Run the main application loop"""
        self.show_banner()

        while True:
            try:
                self.show_menu()
                choice = input("Enter your choice (0-9): ").strip()

                if choice == "1":
                    success, message = self.downloader.download_single_face()

                elif choice == "2":
                    try:
                        count = int(input("Enter number of faces to download (default 5): ") or "5")
                        self.downloader.download_batch(count)
                    except ValueError:
                        print("❌ Invalid number")

                elif choice == "3":
                    self.downloader.start_continuous_download()

                elif choice == "4":
                    self.analyzer.analyze_images(self.config.faces_dir)

                elif choice == "5":
                    self.downloader.show_stats()

                elif choice == "6":
                    self.configure_settings()

                elif choice == "7":
                    self.list_faces()

                elif choice == "8":
                    self.open_faces_folder()

                elif choice == "9":
                    self.config.save()

                elif choice == "0":
                    print("\n👋 Goodbye!")
                    break

                else:
                    print("❌ Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    app = ConsoleFaceApp()
    app.run()

if __name__ == "__main__":
    main()