#!/usr/bin/env python3
"""
Face Vector Database Pipeline GUI
A tabbed interface to run each step of the face collection and embedding pipeline
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import os
import json
import time
from datetime import datetime

class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Vector Database Pipeline")
        self.root.geometry("900x700")

        # Current working directory
        self.working_dir = os.getcwd()

        # Create main notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs for each step
        self.create_step1_tab()
        self.create_step2_tab()
        self.create_step3_tab()
        self.create_step4_tab()
        self.create_step5_tab()
        self.create_step6_tab()
        self.create_overview_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_step1_tab(self):
        """Step 1: ChromaDB Setup"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="1. Setup ChromaDB")

        # Title and description
        title = ttk.Label(frame, text="Step 1: ChromaDB Setup", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        desc = ttk.Label(frame, text="Install ChromaDB and create initial database structure",
                        font=("Arial", 10))
        desc.pack(pady=5)

        # What this step does
        info_frame = ttk.LabelFrame(frame, text="What this step does:")
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        info_text = """• Installs ChromaDB and required dependencies
• Creates persistent database in ./chroma_db/
• Sets up sample collection for testing
• Verifies installation is working correctly"""

        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(padx=10, pady=10)

        # Run button
        self.step1_button = ttk.Button(frame, text="🚀 Run Step 1: Setup ChromaDB",
                                      command=self.run_step1, style="Accent.TButton")
        self.step1_button.pack(pady=20)

        # Output area
        output_frame = ttk.LabelFrame(frame, text="Output:")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.step1_output = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD)
        self.step1_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_step2_tab(self):
        """Step 2: Database Verification"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="2. Verify Database")

        title = ttk.Label(frame, text="Step 2: Database Verification", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        desc = ttk.Label(frame, text="Verify ChromaDB installation and inspect database structure",
                        font=("Arial", 10))
        desc.pack(pady=5)

        info_frame = ttk.LabelFrame(frame, text="What this step does:")
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        info_text = """• Checks ChromaDB version and functionality
• Shows database collections and document counts
• Displays database file structure and storage usage
• Confirms everything is working correctly"""

        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(padx=10, pady=10)

        self.step2_button = ttk.Button(frame, text="🔍 Run Step 2: Verify Database",
                                      command=self.run_step2, style="Accent.TButton")
        self.step2_button.pack(pady=20)

        output_frame = ttk.LabelFrame(frame, text="Output:")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.step2_output = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD)
        self.step2_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_step3_tab(self):
        """Step 3: Face Collection"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="3. Collect Faces")

        title = ttk.Label(frame, text="Step 3: Face Data Collection", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        desc = ttk.Label(frame, text="Download synthetic faces and extract features",
                        font=("Arial", 10))
        desc.pack(pady=5)

        info_frame = ttk.LabelFrame(frame, text="What this step does:")
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        info_text = """• Downloads faces from ThisPersonDoesNotExist.com
• Extracts 143-dimensional embeddings per face
• Analyzes age groups, skin tones, image quality
• Saves processed data to face_data.json
• Takes a few minutes due to respectful rate limiting"""

        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(padx=10, pady=10)

        # Image count input
        count_frame = ttk.Frame(frame)
        count_frame.pack(pady=10)
        ttk.Label(count_frame, text="Number of images to download:").pack(side=tk.LEFT, padx=5)
        self.face_count_var = tk.StringVar(value="10")
        count_spinbox = ttk.Spinbox(count_frame, from_=1, to=100, width=10,
                                   textvariable=self.face_count_var)
        count_spinbox.pack(side=tk.LEFT, padx=5)

        # Progress bar for this step
        self.step3_progress = ttk.Progressbar(frame, mode='indeterminate')
        self.step3_progress.pack(pady=10)

        self.step3_button = ttk.Button(frame, text="🎭 Run Step 3: Collect Faces",
                                      command=self.run_step3, style="Accent.TButton")
        self.step3_button.pack(pady=20)

        output_frame = ttk.LabelFrame(frame, text="Output:")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.step3_output = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD)
        self.step3_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_step4_tab(self):
        """Step 4: Database Embedding"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="4. Embed to DB")

        title = ttk.Label(frame, text="Step 4: Embed to ChromaDB", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        desc = ttk.Label(frame, text="Store face embeddings in ChromaDB for semantic search",
                        font=("Arial", 10))
        desc.pack(pady=5)

        info_frame = ttk.LabelFrame(frame, text="What this step does:")
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        info_text = """• Loads processed face data from JSON file
• Creates dedicated faces collection in ChromaDB
• Stores 143D embeddings with metadata
• Database becomes ready for semantic search"""

        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(padx=10, pady=10)

        self.step4_button = ttk.Button(frame, text="💾 Run Step 4: Embed to Database",
                                      command=self.run_step4, style="Accent.TButton")
        self.step4_button.pack(pady=20)

        output_frame = ttk.LabelFrame(frame, text="Output:")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.step4_output = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD)
        self.step4_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_step5_tab(self):
        """Step 5: Database Inspection"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="5. Inspect DB")

        title = ttk.Label(frame, text="Step 5: Database Inspection", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        desc = ttk.Label(frame, text="Detailed analysis of vector database structure and performance",
                        font=("Arial", 10))
        desc.pack(pady=5)

        info_frame = ttk.LabelFrame(frame, text="What this step does:")
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        info_text = """• Analyzes vector dimensions, data types, memory usage
• Provides statistical analysis of embeddings
• Shows collection metadata and document samples
• Complete storage breakdown and optimization info"""

        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(padx=10, pady=10)

        self.step5_button = ttk.Button(frame, text="🔬 Run Step 5: Inspect Database",
                                      command=self.run_step5, style="Accent.TButton")
        self.step5_button.pack(pady=20)

        output_frame = ttk.LabelFrame(frame, text="Output:")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.step5_output = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD)
        self.step5_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_step6_tab(self):
        """Step 6: Search Testing"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="6. Test Search")

        title = ttk.Label(frame, text="Step 6: Semantic Search Test", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        desc = ttk.Label(frame, text="Test semantic search functionality with new face",
                        font=("Arial", 10))
        desc.pack(pady=5)

        info_frame = ttk.LabelFrame(frame, text="What this step does:")
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        info_text = """• Downloads new test face for comparison
• Performs similarity search against database
• Tests feature-based filtering (age, skin tone)
• Measures search performance and accuracy
• Validates complete pipeline functionality"""

        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(padx=10, pady=10)

        # Custom search section
        custom_frame = ttk.LabelFrame(frame, text="Custom Image Search:")
        custom_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(custom_frame, text="📁 Browse Image",
                  command=self.browse_image).pack(side=tk.LEFT, padx=5, pady=5)

        self.custom_image_var = tk.StringVar(value="No image selected")
        ttk.Label(custom_frame, textvariable=self.custom_image_var).pack(side=tk.LEFT, padx=5)

        self.step6_button = ttk.Button(frame, text="🧪 Run Step 6: Test Search",
                                      command=self.run_step6, style="Accent.TButton")
        self.step6_button.pack(pady=20)

        output_frame = ttk.LabelFrame(frame, text="Output:")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.step6_output = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD)
        self.step6_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_overview_tab(self):
        """Overview tab showing pipeline status"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Overview")

        title = ttk.Label(frame, text="Pipeline Overview", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # Pipeline status
        status_frame = ttk.LabelFrame(frame, text="Pipeline Status:")
        status_frame.pack(fill=tk.X, padx=20, pady=10)

        self.status_text = scrolledtext.ScrolledText(status_frame, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Quick actions
        actions_frame = ttk.LabelFrame(frame, text="Quick Actions:")
        actions_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(actions_frame, text="🔄 Refresh Status",
                  command=self.refresh_status).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(actions_frame, text="📁 Open Faces Folder",
                  command=self.open_faces_folder).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(actions_frame, text="🔧 Interactive Search",
                  command=self.open_interactive_search).pack(side=tk.LEFT, padx=5, pady=5)

        # System info
        info_frame = ttk.LabelFrame(frame, text="System Information:")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.info_text = scrolledtext.ScrolledText(info_frame, height=8, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initialize status
        self.refresh_status()

    def run_command_in_thread(self, command, output_widget, step_name):
        """Run a command in a separate thread and update the output widget"""
        def run():
            try:
                self.status_var.set(f"Running {step_name}...")
                output_widget.delete(1.0, tk.END)
                output_widget.insert(tk.END, f"Starting {step_name}...\n\n")
                output_widget.update()

                # Run the command
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.working_dir,
                    shell=True
                )

                # Read output in real-time
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        output_widget.insert(tk.END, output)
                        output_widget.see(tk.END)
                        output_widget.update()

                # Get return code
                return_code = process.poll()

                if return_code == 0:
                    output_widget.insert(tk.END, f"\n✅ {step_name} completed successfully!\n")
                    self.status_var.set(f"{step_name} completed successfully")
                else:
                    output_widget.insert(tk.END, f"\n❌ {step_name} failed with return code {return_code}\n")
                    self.status_var.set(f"{step_name} failed")

                output_widget.see(tk.END)
                self.refresh_status()

            except Exception as e:
                output_widget.insert(tk.END, f"\n❌ Error running {step_name}: {str(e)}\n")
                self.status_var.set(f"Error in {step_name}")

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

    def run_step1(self):
        """Run Step 1: ChromaDB Setup"""
        if hasattr(self, 'step3_progress'):
            self.step3_progress.stop()
        self.run_command_in_thread("./1_setup_chromadb.sh", self.step1_output, "Step 1")

    def run_step2(self):
        """Run Step 2: Database Verification"""
        self.run_command_in_thread("./2_check_chromadb.sh", self.step2_output, "Step 2")

    def run_step3(self):
        """Run Step 3: Face Collection"""
        self.step3_progress.start()
        face_count = self.face_count_var.get()
        command = f"python3 face_collector.py --count {face_count}"
        self.run_command_in_thread(command, self.step3_output, "Step 3")

    def run_step4(self):
        """Run Step 4: Database Embedding"""
        self.run_command_in_thread("./4_embed_to_chromadb.sh", self.step4_output, "Step 4")

    def run_step5(self):
        """Run Step 5: Database Inspection"""
        self.run_command_in_thread("./5_inspect_database.sh", self.step5_output, "Step 5")

    def run_step6(self):
        """Run Step 6: Search Testing"""
        self.run_command_in_thread("./6_test_search.sh", self.step6_output, "Step 6")

    def browse_image(self):
        """Browse for custom image to search"""
        filename = filedialog.askopenfilename(
            title="Select Image for Search",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        if filename:
            self.custom_image_var.set(os.path.basename(filename))
            # TODO: Implement custom image search

    def refresh_status(self):
        """Refresh the pipeline status"""
        status = []

        # Check database
        if os.path.exists("chroma_db"):
            status.append("✅ ChromaDB database exists")
        else:
            status.append("❌ ChromaDB database not found")

        # Check faces
        if os.path.exists("faces"):
            face_count = len([f for f in os.listdir("faces") if f.endswith('.jpg')])
            status.append(f"✅ {face_count} face images collected")
        else:
            status.append("❌ No faces collected yet")

        # Check face data
        if os.path.exists("face_data.json"):
            try:
                with open("face_data.json", 'r') as f:
                    data = json.load(f)
                status.append(f"✅ {len(data)} face records processed")
            except:
                status.append("⚠️ Face data file exists but couldn't be read")
        else:
            status.append("❌ Face data not processed yet")

        # Update status display
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.status_text.insert(tk.END, "\n".join(status))

        # Update system info
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"Working Directory: {self.working_dir}\n")
        self.info_text.insert(tk.END, f"Python Version: {subprocess.check_output(['python3', '--version'], text=True).strip()}\n")

        # Database size
        if os.path.exists("chroma_db"):
            try:
                size_output = subprocess.check_output(['du', '-sh', 'chroma_db'], text=True)
                self.info_text.insert(tk.END, f"Database Size: {size_output.split()[0]}\n")
            except:
                pass

    def open_faces_folder(self):
        """Open the faces folder in file manager"""
        if os.path.exists("faces"):
            if os.name == 'nt':  # Windows
                os.startfile("faces")
            elif os.name == 'posix':  # macOS/Linux
                subprocess.run(['open', 'faces'])
        else:
            messagebox.showwarning("Warning", "Faces folder not found. Run Step 3 first.")

    def open_interactive_search(self):
        """Open the interactive search interface"""
        try:
            # Open in a new terminal window for proper interactivity
            if sys.platform == "darwin":  # macOS
                subprocess.Popen(['open', '-a', 'Terminal.app', '--args', 'python3', 'face_database.py'])
            elif sys.platform == "linux":
                # Try common Linux terminal emulators
                for terminal in ['gnome-terminal', 'xterm', 'konsole']:
                    try:
                        subprocess.Popen([terminal, '--', 'python3', 'face_database.py'])
                        break
                    except FileNotFoundError:
                        continue
                else:
                    messagebox.showwarning("Warning", "Could not find terminal emulator. Run 'python3 face_database.py' manually.")
            else:  # Windows
                subprocess.Popen(['cmd', '/c', 'start', 'python3', 'face_database.py'])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open interactive search: {str(e)}")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()

    # Configure style
    style = ttk.Style()

    # Create and configure the application
    app = PipelineGUI(root)

    # Center the window
    root.geometry("900x700+100+50")

    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main()