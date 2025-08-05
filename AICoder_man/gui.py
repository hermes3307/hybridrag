#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Coder GUI - Manual-based Professional AI Code Generation System
Tkinter-based graphical user interface
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import tkinter.font as tkfont
import asyncio
import threading
import json
import os
import sys
import logging
from datetime import datetime

# Import main system
try:
    from main import AICoderSystem, CodeGenerationResult
except ImportError as e:
    print(f"main.py file not found: {e}")
    print("Please ensure main.py is in the same directory.")
    sys.exit(1)

# Enhanced GUI Log Handler with Progress Tracking
class GUILogHandler(logging.Handler):
    """Custom log handler that displays logs in GUI with enhanced color coding and progress tracking"""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setup_color_tags()
        self.progress_indicators = {}
    
    def setup_color_tags(self):
        """Setup enhanced color tags for different log levels and progress states"""
        # Standard log levels with enhanced colors
        self.text_widget.tag_config('ERROR', foreground='#FF4444', background='#2D1B1B', font=('Consolas', 10, 'bold'))
        self.text_widget.tag_config('WARNING', foreground='#FFA500', background='#2D2416', font=('Consolas', 10))
        self.text_widget.tag_config('INFO', foreground='#4CAF50', background='#1B2D1B', font=('Consolas', 10))
        self.text_widget.tag_config('DEBUG', foreground='#9E9E9E', background='#1E1E1E', font=('Consolas', 9))
        
        # Progress and status indicators
        self.text_widget.tag_config('PROGRESS', foreground='#03DAC6', background='#1A2B2B', font=('Consolas', 10, 'bold'))
        self.text_widget.tag_config('SUCCESS', foreground='#66BB6A', background='#1B2D1B', font=('Consolas', 10, 'bold'))
        self.text_widget.tag_config('CRITICAL', foreground='#FFFFFF', background='#B71C1C', font=('Consolas', 10, 'bold'))
        
        # Component-specific colors
        self.text_widget.tag_config('MANUAL_PROC', foreground='#9C27B0', font=('Consolas', 10))
        self.text_widget.tag_config('VECTOR_DB', foreground='#2196F3', font=('Consolas', 10))
        self.text_widget.tag_config('CLAUDE_API', foreground='#FF9800', font=('Consolas', 10))
        self.text_widget.tag_config('GUI_EVENT', foreground='#607D8B', font=('Consolas', 10))
    
    def emit(self, record):
        try:
            if not self.text_widget or not self.text_widget.winfo_exists():
                return
                
            # Format message with timestamp and detailed info
            timestamp = self.formatTime(record, '%H:%M:%S')
            level = record.levelname
            name = record.name.split('.')[-1]  # Get last part of logger name
            msg = record.getMessage()
            
            # Create enhanced message format
            formatted_msg = f"[{timestamp}] {level:8} | {name:12} | {msg}"
            
            # Determine tag based on content and level
            tag = self.determine_tag(record, msg)
            
            # Add progress indicators for certain operations
            if self.is_progress_message(msg):
                formatted_msg = self.add_progress_indicator(formatted_msg, msg)
            
            # Insert with appropriate styling
            self.text_widget.insert(tk.END, formatted_msg + '\n', tag)
            self.text_widget.see(tk.END)
            
            # Auto-scroll and update display
            self.text_widget.update_idletasks()
            
        except Exception as e:
            # Fallback to prevent GUI crashes
            try:
                self.text_widget.insert(tk.END, f"[LOG ERROR] {str(e)}\n", 'ERROR')
            except:
                pass  # Ultimate fallback
    
    def determine_tag(self, record, msg):
        """Determine appropriate tag based on log level and message content"""
        level = record.levelname
        msg_lower = msg.lower()
        
        # Critical errors
        if level == 'CRITICAL' or 'critical' in msg_lower or 'fatal' in msg_lower:
            return 'CRITICAL'
        
        # Component-specific tagging
        if 'manual' in msg_lower and ('processing' in msg_lower or 'parsing' in msg_lower):
            return 'MANUAL_PROC'
        elif 'vector' in msg_lower or 'chroma' in msg_lower or 'embedding' in msg_lower:
            return 'VECTOR_DB'
        elif 'claude' in msg_lower or 'api' in msg_lower:
            return 'CLAUDE_API'
        elif 'gui' in msg_lower or 'button' in msg_lower or 'click' in msg_lower:
            return 'GUI_EVENT'
        
        # Progress indicators
        elif any(word in msg_lower for word in ['uploading', 'processing', 'generating', 'saving']):
            return 'PROGRESS'
        elif any(word in msg_lower for word in ['completed', 'success', 'finished', 'done']):
            return 'SUCCESS'
        
        # Default to log level
        return level
    
    def is_progress_message(self, msg):
        """Check if message is a progress update"""
        progress_keywords = ['uploading', 'processing', 'generating', 'parsing', 'storing', 'searching']
        return any(keyword in msg.lower() for keyword in progress_keywords)
    
    def add_progress_indicator(self, formatted_msg, msg):
        """Add visual progress indicators to progress messages"""
        msg_lower = msg.lower()
        
        if 'uploading' in msg_lower:
            return f"📤 {formatted_msg}"
        elif 'processing' in msg_lower or 'parsing' in msg_lower:
            return f"⚙️ {formatted_msg}"
        elif 'generating' in msg_lower:
            return f"🚀 {formatted_msg}"
        elif 'storing' in msg_lower or 'saving' in msg_lower:
            return f"💾 {formatted_msg}"
        elif 'searching' in msg_lower:
            return f"🔍 {formatted_msg}"
        else:
            return f"▶️ {formatted_msg}"

class AICoderGUI:
    """Main AI Coder GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AI Coder - Manual-based Code Generation System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set fonts
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=10)
        self.root.option_add("*Font", default_font)
        
        # System initialization
        self.system = None
        self.is_processing = False
        self.manual_upload_thread = None
        self.code_generation_thread = None
        
        # Configuration
        self.config_file = "ai_coder_config.json"
        self.config = self.load_config()
        
        # Generated code history
        self.code_history = []
        self.code_history_file = "code_generation_history.json"
        self.load_code_history()
        
        # Manual management
        self.uploaded_manuals = []
        self.manual_stats = {}
        
        # Setup UI
        self.setup_ui()
        self.setup_logging()
        
        # Initialize system
        self.root.after(1000, self.initialize_system)
    
    def setup_ui(self):
        """Setup the main user interface"""
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="AI Coder - Manual-based Code Generation", 
                               font=("Segoe UI", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Setup tabs
        self.setup_manual_management_tab()
        self.setup_code_generation_tab()
        self.setup_code_validation_tab()
        self.setup_manual_search_tab()
        self.setup_settings_tab()
        self.setup_logs_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Please initialize system")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor='w')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
    
    def setup_manual_management_tab(self):
        """Setup manual management tab"""
        
        manual_frame = ttk.Frame(self.notebook)
        self.notebook.add(manual_frame, text="📚 Manual Management")
        
        # Left panel - Upload and controls
        left_panel = ttk.LabelFrame(manual_frame, text="Manual Upload", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Upload section
        ttk.Label(left_panel, text="Select Manual Files:").pack(anchor='w')
        
        file_frame = ttk.Frame(left_panel)
        file_frame.pack(fill=tk.X, pady=5)
        
        # Change to listbox for multiple files
        self.manual_files_listbox = tk.Listbox(file_frame, height=4)
        self.manual_files_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Store selected files
        self.selected_files = []
        
        buttons_frame = ttk.Frame(file_frame)
        buttons_frame.pack(side=tk.RIGHT, padx=(5, 0))
        
        browse_btn = ttk.Button(buttons_frame, text="Browse", command=self.browse_manual_file)
        browse_btn.pack(pady=(0, 2))
        
        clear_btn = ttk.Button(buttons_frame, text="Clear", command=self.clear_selected_files)
        clear_btn.pack()
        
        # Manual type
        ttk.Label(left_panel, text="Manual Type:").pack(anchor='w', pady=(10, 0))
        self.manual_type_var = tk.StringVar(value="custom")
        type_combo = ttk.Combobox(left_panel, textvariable=self.manual_type_var, 
                                 values=["altibase", "database", "api_reference", "administration", "custom"])
        type_combo.pack(fill=tk.X, pady=5)
        
        # Version
        ttk.Label(left_panel, text="Version:").pack(anchor='w')
        self.manual_version_var = tk.StringVar(value="1.0")
        version_entry = ttk.Entry(left_panel, textvariable=self.manual_version_var)
        version_entry.pack(fill=tk.X, pady=5)
        
        # Upload button
        self.upload_btn = ttk.Button(left_panel, text="📤 Upload Manual", 
                                    command=self.upload_manual, style="Accent.TButton")
        self.upload_btn.pack(pady=10, fill=tk.X)
        
        # Upload progress
        self.upload_progress = ttk.Progressbar(left_panel, mode='indeterminate')
        self.upload_progress.pack(fill=tk.X, pady=5)
        
        # Vector DB status
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(left_panel, text="Vector Database Status:", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        
        self.db_chunks_var = tk.StringVar(value="0")
        self.db_collection_var = tk.StringVar(value="Not initialized")
        
        ttk.Label(left_panel, text="Total Chunks:").pack(anchor='w')
        ttk.Label(left_panel, textvariable=self.db_chunks_var, foreground='blue').pack(anchor='w')
        
        ttk.Label(left_panel, text="Collection:").pack(anchor='w', pady=(5, 0))
        ttk.Label(left_panel, textvariable=self.db_collection_var, foreground='blue').pack(anchor='w')
        
        # Right panel - Manual list and details
        right_panel = ttk.LabelFrame(manual_frame, text="Uploaded Manuals", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Manual list
        list_frame = ttk.Frame(right_panel)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for manual list
        columns = ("Type", "Version", "File", "Chunks", "Status")
        self.manual_tree = ttk.Treeview(list_frame, columns=columns, show='tree headings', height=15)
        
        # Configure columns
        self.manual_tree.heading('#0', text='Manual Name')
        self.manual_tree.column('#0', width=200)
        
        for col in columns:
            self.manual_tree.heading(col, text=col)
            self.manual_tree.column(col, width=100)
        
        # Scrollbars for treeview
        tree_scroll_y = ttk.Scrollbar(list_frame, orient='vertical', command=self.manual_tree.yview)
        tree_scroll_x = ttk.Scrollbar(list_frame, orient='horizontal', command=self.manual_tree.xview)
        self.manual_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        
        self.manual_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Manual details
        details_frame = ttk.LabelFrame(right_panel, text="Manual Details", padding=5)
        details_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.manual_details_text = scrolledtext.ScrolledText(details_frame, height=8, wrap=tk.WORD)
        self.manual_details_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind selection event with error handling
        self.manual_tree.bind('<<TreeviewSelect>>', self.on_manual_select_safe)
    
    def setup_code_generation_tab(self):
        """Setup code generation tab"""
        
        code_frame = ttk.Frame(self.notebook)
        self.notebook.add(code_frame, text="🚀 Code Generation")
        
        # Top panel - Input controls
        input_panel = ttk.LabelFrame(code_frame, text="Code Generation Parameters", padding=10)
        input_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # First row - Task and Language
        row1 = ttk.Frame(input_panel)
        row1.pack(fill=tk.X, pady=5)
        
        ttk.Label(row1, text="Task Description:").pack(side=tk.LEFT)
        self.task_var = tk.StringVar()
        task_entry = ttk.Entry(row1, textvariable=self.task_var, width=50)
        task_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        
        ttk.Label(row1, text="Language:").pack(side=tk.LEFT)
        self.language_var = tk.StringVar(value="python")
        language_combo = ttk.Combobox(row1, textvariable=self.language_var, width=15,
                                     values=["python", "sql", "javascript", "java", "cpp", "c", "csharp", "go", "rust"])
        language_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Second row - Manual context
        row2 = ttk.Frame(input_panel)
        row2.pack(fill=tk.X, pady=5)
        
        ttk.Label(row2, text="Manual Type:").pack(side=tk.LEFT)
        self.gen_manual_type_var = tk.StringVar()
        manual_combo = ttk.Combobox(row2, textvariable=self.gen_manual_type_var, width=15,
                                   values=["", "altibase", "database", "api_reference", "administration", "custom"])
        manual_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(row2, text="Version:").pack(side=tk.LEFT)
        self.gen_version_var = tk.StringVar()
        version_entry = ttk.Entry(row2, textvariable=self.gen_version_var, width=10)
        version_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(row2, text="Style:").pack(side=tk.LEFT)
        self.style_var = tk.StringVar(value="professional")
        style_combo = ttk.Combobox(row2, textvariable=self.style_var, width=15,
                                  values=["professional", "compact", "detailed", "educational"])
        style_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Third row - Specifications
        row3 = ttk.Frame(input_panel)
        row3.pack(fill=tk.X, pady=5)
        
        ttk.Label(row3, text="Additional Specifications:").pack(anchor='w')
        self.specifications_text = scrolledtext.ScrolledText(row3, height=3, wrap=tk.WORD)
        self.specifications_text.pack(fill=tk.X, pady=(5, 0))
        
        # Generate button
        self.generate_btn = ttk.Button(input_panel, text="🚀 Generate Code", 
                                      command=self.generate_code, style="Accent.TButton")
        self.generate_btn.pack(pady=10)
        
        # Progress bar
        self.generation_progress = ttk.Progressbar(input_panel, mode='indeterminate')
        self.generation_progress.pack(fill=tk.X, pady=5)
        
        # Bottom panel - Results
        results_panel = ttk.LabelFrame(code_frame, text="Generated Code", padding=10)
        results_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Code display
        code_display_frame = ttk.Frame(results_panel)
        code_display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.code_text = scrolledtext.ScrolledText(code_display_frame, wrap=tk.NONE, 
                                                  font=('Consolas', 10))
        self.code_text.pack(fill=tk.BOTH, expand=True)
        
        # Code info panel
        info_panel = ttk.Frame(results_panel)
        info_panel.pack(fill=tk.X, pady=(10, 0))
        
        # Confidence and references
        self.confidence_var = tk.StringVar(value="Confidence: N/A")
        self.references_var = tk.StringVar(value="Manual References: 0")
        
        ttk.Label(info_panel, textvariable=self.confidence_var, foreground='blue').pack(side=tk.LEFT)
        ttk.Label(info_panel, textvariable=self.references_var, foreground='green').pack(side=tk.LEFT, padx=(20, 0))
        
        # Action buttons
        button_frame = ttk.Frame(info_panel)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="💾 Save Code", command=self.save_generated_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="📋 Copy Code", command=self.copy_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="🔍 Validate", command=self.validate_current_code).pack(side=tk.LEFT, padx=2)
    
    def setup_code_validation_tab(self):
        """Setup code validation tab"""
        
        validation_frame = ttk.Frame(self.notebook)
        self.notebook.add(validation_frame, text="✅ Code Validation")
        
        # Input panel
        input_panel = ttk.LabelFrame(validation_frame, text="Code to Validate", padding=10)
        input_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls
        controls_frame = ttk.Frame(input_panel)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(controls_frame, text="Language:").pack(side=tk.LEFT)
        self.val_language_var = tk.StringVar(value="python")
        lang_combo = ttk.Combobox(controls_frame, textvariable=self.val_language_var, width=15,
                                 values=["python", "sql", "javascript", "java", "cpp", "c", "csharp", "go", "rust"])
        lang_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(controls_frame, text="Manual Type:").pack(side=tk.LEFT)
        self.val_manual_type_var = tk.StringVar()
        manual_combo = ttk.Combobox(controls_frame, textvariable=self.val_manual_type_var, width=15,
                                   values=["", "altibase", "database", "api_reference", "administration", "custom"])
        manual_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Button(controls_frame, text="✅ Validate Code", 
                  command=self.validate_code).pack(side=tk.RIGHT)
        
        # Code input
        self.validation_code_text = scrolledtext.ScrolledText(input_panel, height=15, 
                                                             font=('Consolas', 10), wrap=tk.NONE)
        self.validation_code_text.pack(fill=tk.BOTH, expand=True)
        
        # Results panel
        results_panel = ttk.LabelFrame(validation_frame, text="Validation Results", padding=10)
        results_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Validation scores
        scores_frame = ttk.Frame(results_panel)
        scores_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.compliance_var = tk.StringVar(value="Compliance: N/A")
        self.syntax_var = tk.StringVar(value="Syntax: N/A")
        self.compatibility_var = tk.StringVar(value="API Compatibility: N/A")
        
        ttk.Label(scores_frame, textvariable=self.compliance_var, foreground='blue').pack(side=tk.LEFT)
        ttk.Label(scores_frame, textvariable=self.syntax_var, foreground='green').pack(side=tk.LEFT, padx=(20, 0))
        ttk.Label(scores_frame, textvariable=self.compatibility_var, foreground='purple').pack(side=tk.LEFT, padx=(20, 0))
        
        # Suggestions
        self.validation_results_text = scrolledtext.ScrolledText(results_panel, height=8, wrap=tk.WORD)
        self.validation_results_text.pack(fill=tk.X)
    
    def setup_manual_search_tab(self):
        """Setup manual search tab"""
        
        search_frame = ttk.Frame(self.notebook)
        self.notebook.add(search_frame, text="🔍 Manual Search")
        
        # Search panel
        search_panel = ttk.LabelFrame(search_frame, text="Search Manual Content", padding=10)
        search_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Search controls
        controls_frame = ttk.Frame(search_panel)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(controls_frame, text="Search Query:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(controls_frame, textvariable=self.search_var, width=40)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        
        ttk.Label(controls_frame, text="Manual Type:").pack(side=tk.LEFT)
        self.search_manual_type_var = tk.StringVar()
        search_manual_combo = ttk.Combobox(controls_frame, textvariable=self.search_manual_type_var, width=15,
                                          values=["", "altibase", "database", "api_reference", "administration", "custom"])
        search_manual_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(controls_frame, text="🔍 Search", command=self.search_manuals).pack(side=tk.RIGHT)
        
        # Bind Enter key to search
        search_entry.bind('<Return>', lambda _: self.search_manuals())
        
        # Results panel
        results_panel = ttk.LabelFrame(search_frame, text="Search Results", padding=10)
        results_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results tree
        search_results_frame = ttk.Frame(results_panel)
        search_results_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Relevance", "Manual Type", "Version", "Section")
        self.search_tree = ttk.Treeview(search_results_frame, columns=columns, show='tree headings', height=10)
        
        self.search_tree.heading('#0', text='Content Preview')
        self.search_tree.column('#0', width=300)
        
        for col in columns:
            self.search_tree.heading(col, text=col)
            self.search_tree.column(col, width=100)
        
        search_scroll_y = ttk.Scrollbar(search_results_frame, orient='vertical', command=self.search_tree.yview)
        self.search_tree.configure(yscrollcommand=search_scroll_y.set)
        
        self.search_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        search_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Content display
        content_panel = ttk.LabelFrame(results_panel, text="Selected Content", padding=5)
        content_panel.pack(fill=tk.X, pady=(10, 0))
        
        self.search_content_text = scrolledtext.ScrolledText(content_panel, height=10, wrap=tk.WORD)
        self.search_content_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind selection event with error handling
        self.search_tree.bind('<<TreeviewSelect>>', self.on_search_select_safe)
    
    def setup_settings_tab(self):
        """Setup settings tab"""
        
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # API Settings
        api_panel = ttk.LabelFrame(settings_frame, text="API Configuration", padding=10)
        api_panel.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(api_panel, text="Claude API Key:").pack(anchor='w')
        self.api_key_var = tk.StringVar(value=self.config.get('claude_api_key', ''))
        api_entry = ttk.Entry(api_panel, textvariable=self.api_key_var, show='*', width=60)
        api_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(api_panel, text="Vector DB Path:").pack(anchor='w', pady=(10, 0))
        self.db_path_var = tk.StringVar(value=self.config.get('vector_db_path', './manual_db'))
        
        db_frame = ttk.Frame(api_panel)
        db_frame.pack(fill=tk.X, pady=5)
        
        db_entry = ttk.Entry(db_frame, textvariable=self.db_path_var)
        db_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(db_frame, text="Browse", command=self.browse_db_path).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Generation Settings
        gen_panel = ttk.LabelFrame(settings_frame, text="Code Generation Settings", padding=10)
        gen_panel.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(gen_panel, text="Default Language:").pack(anchor='w')
        self.default_language_var = tk.StringVar(value=self.config.get('default_language', 'python'))
        default_lang_combo = ttk.Combobox(gen_panel, textvariable=self.default_language_var, width=20,
                                         values=["python", "sql", "javascript", "java", "cpp", "c", "csharp", "go", "rust"])
        default_lang_combo.pack(anchor='w', pady=5)
        
        ttk.Label(gen_panel, text="Default Style:").pack(anchor='w', pady=(10, 0))
        self.default_style_var = tk.StringVar(value=self.config.get('default_style', 'professional'))
        default_style_combo = ttk.Combobox(gen_panel, textvariable=self.default_style_var, width=20,
                                          values=["professional", "compact", "detailed", "educational"])
        default_style_combo.pack(anchor='w', pady=5)
        
        # Auto-save settings
        auto_save_var = tk.BooleanVar(value=self.config.get('auto_save_generated_code', True))
        ttk.Checkbutton(gen_panel, text="Auto-save generated code", variable=auto_save_var).pack(anchor='w', pady=5)
        
        # Save settings button
        ttk.Button(settings_frame, text="💾 Save Settings", command=self.save_settings).pack(pady=20)
    
    def setup_logs_tab(self):
        """Setup logs tab"""
        
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="📋 Logs")
        
        # Log controls
        controls_frame = ttk.Frame(logs_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="🗑️ Clear Logs", command=self.clear_logs).pack(side=tk.LEFT)
        
        auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Auto-scroll", variable=auto_scroll_var).pack(side=tk.LEFT, padx=(10, 0))
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_logging(self):
        """Setup enhanced logging to display in GUI with color coding and progress tracking"""
        try:
            # Create custom handler with enhanced formatting
            gui_handler = GUILogHandler(self.log_text)
            gui_handler.setLevel(logging.DEBUG)  # Show all levels
            
            # Enhanced formatter with more details
            formatter = logging.Formatter(
                '%(name)s | %(funcName)s:%(lineno)d | %(message)s'
            )
            gui_handler.setFormatter(formatter)
            
            # Configure root logger
            root_logger = logging.getLogger()
            
            # Remove existing handlers to avoid duplicates
            for handler in root_logger.handlers[:]:
                if isinstance(handler, GUILogHandler):
                    root_logger.removeHandler(handler)
            
            root_logger.addHandler(gui_handler)
            root_logger.setLevel(logging.DEBUG)
            
            # Set up component-specific loggers
            logging.getLogger('main').setLevel(logging.DEBUG)
            logging.getLogger('gui').setLevel(logging.DEBUG)
            logging.getLogger('manual_processor').setLevel(logging.DEBUG)
            logging.getLogger('vector_db').setLevel(logging.DEBUG)
            logging.getLogger('claude_client').setLevel(logging.DEBUG)
            
            # Log initialization
            logging.info("Enhanced logging system initialized with color coding and progress tracking")
            logging.debug("Debug level logging enabled for detailed progress monitoring")
            
        except Exception as e:
            print(f"Failed to setup enhanced logging: {e}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)
    
    def initialize_system(self):
        """Initialize the AI Coder system with detailed progress logging"""
        try:
            logging.info("🚀 Starting AI Coder system initialization...")
            self.status_var.set("Initializing AI Coder system...")
            
            # Step 1: Validate configuration
            logging.debug("📋 Validating system configuration...")
            claude_api_key = self.api_key_var.get() if self.api_key_var.get() else None
            vector_db_path = self.db_path_var.get()
            
            if claude_api_key:
                logging.info("🔑 Claude API key configured")
            else:
                logging.warning("⚠️ Claude API key not configured - system will run in test mode")
            
            logging.info(f"📁 Vector DB path: {vector_db_path}")
            
            # Step 2: Initialize core system
            logging.info("⚙️ Initializing AI Coder core system...")
            self.system = AICoderSystem(claude_api_key=claude_api_key, vector_db_path=vector_db_path)
            logging.info("✅ AI Coder core system initialized successfully")
            
            # Step 3: Initialize GUI components
            logging.debug("🎨 Initializing GUI components...")
            if not hasattr(self, 'selected_files'):
                self.selected_files = []
                logging.debug("📋 Initialized selected_files list")
            
            # Step 4: Update UI with system stats
            logging.debug("📊 Updating system statistics...")
            self.update_system_stats()
            
            # Step 5: Final status update
            success_msg = "AI Coder system initialized successfully"
            logging.info(f"🎉 {success_msg}")
            self.status_var.set(success_msg)
            
        except Exception as e:
            error_msg = f"Failed to initialize system: {e}"
            logging.error(f"❌ {error_msg}")
            logging.debug(f"💥 Initialization error details: {str(e)}", exc_info=True)
            messagebox.showerror("Initialization Error", error_msg)
            self.status_var.set("System initialization failed")
    
    def update_system_stats(self):
        """Update system statistics in UI with detailed logging"""
        if self.system:
            try:
                logging.debug("📊 Updating system statistics...")
                stats = self.system.get_system_stats()
                
                # Update vector DB stats
                db_stats = stats.get('vector_db', {})
                total_chunks = db_stats.get('total_chunks', 0)
                collection_name = db_stats.get('collection_name', 'unknown')
                
                self.db_chunks_var.set(str(total_chunks))
                self.db_collection_var.set(collection_name)
                
                # Log detailed stats
                logging.info(f"📁 Vector DB: {total_chunks} chunks in collection '{collection_name}'")
                logging.info(f"🤖 Claude API requests: {stats.get('claude_requests', 0)}")
                
                if stats.get('test_mode', False):
                    logging.warning("⚠️ System running in test mode (API key not configured)")
                else:
                    logging.info("🔑 System running with API key configured")
                
                logging.debug("✅ System statistics updated successfully")
                
            except Exception as e:
                error_msg = f"Failed to update system stats: {e}"
                logging.error(f"❌ {error_msg}")
                logging.debug("System stats update error details", exc_info=True)
    
    def browse_manual_file(self):
        """Browse for multiple manual files"""
        try:
            # Log the browser action
            logging.info("Opening file browser for manual selection")
            
            # Simplified file types for better macOS compatibility
            file_types = [
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("HTML files", "*.html"),
                ("HTML files", "*.htm"),
                ("Markdown files", "*.md"),
                ("Text files", "*.txt"),
                ("All files", "*")
            ]
            
            # Ensure the widget exists before proceeding
            if not hasattr(self, 'manual_files_listbox') or not self.manual_files_listbox.winfo_exists():
                logging.error("Manual files listbox not properly initialized")
                messagebox.showerror("Error", "GUI component not properly initialized. Please restart the application.")
                return
            
            # Initialize selected_files if not exists
            if not hasattr(self, 'selected_files'):
                self.selected_files = []
                logging.warning("selected_files was not initialized, creating empty list")
            
            # Use a more compatible approach for macOS
            try:
                filenames = filedialog.askopenfilenames(
                    title="Select Manual Files",
                    filetypes=file_types,
                    parent=self.root
                )
            except Exception as dialog_error:
                logging.warning(f"File dialog error, trying without filetypes: {dialog_error}")
                # Fallback to no file type restrictions
                filenames = filedialog.askopenfilenames(
                    title="Select Manual Files",
                    parent=self.root
                )
            
            if filenames:
                logging.info(f"Selected {len(filenames)} files for upload")
                
                # Add new files to the selection
                added_count = 0
                for filename in filenames:
                    try:
                        if filename not in self.selected_files:
                            # Verify file exists and is readable
                            if not os.path.exists(filename):
                                logging.warning(f"Selected file does not exist: {filename}")
                                continue
                                
                            if not os.access(filename, os.R_OK):
                                logging.warning(f"Selected file is not readable: {filename}")
                                continue
                            
                            self.selected_files.append(filename)
                            display_name = os.path.basename(filename)
                            self.manual_files_listbox.insert(tk.END, display_name)
                            added_count += 1
                            logging.debug(f"Added file to selection: {display_name}")
                        else:
                            logging.debug(f"File already selected: {os.path.basename(filename)}")
                    except Exception as e:
                        logging.error(f"Error processing selected file {filename}: {e}")
                        continue
                
                if added_count > 0:
                    logging.info(f"Successfully added {added_count} new files to selection")
                    self.status_var.set(f"Added {added_count} files to selection")
                    
                    # Auto-detect manual type from first filename
                    if self.selected_files:
                        first_filename_lower = os.path.basename(self.selected_files[0]).lower()
                        if 'altibase' in first_filename_lower:
                            self.manual_type_var.set('altibase')
                            logging.info("Auto-detected manual type: altibase")
                        elif any(word in first_filename_lower for word in ['sql', 'database']):
                            self.manual_type_var.set('database')
                            logging.info("Auto-detected manual type: database")
                        elif 'api' in first_filename_lower:
                            self.manual_type_var.set('api_reference')
                            logging.info("Auto-detected manual type: api_reference")
                else:
                    logging.warning("No new files were added to selection")
                    self.status_var.set("No new files added (already selected or invalid)")
                    
            else:
                logging.info("File selection cancelled by user")
                self.status_var.set("File selection cancelled")
                
        except Exception as e:
            error_msg = f"Error in file browser: {e}"
            logging.error(error_msg)
            messagebox.showerror("File Browser Error", error_msg)
            self.status_var.set("File browser error occurred")
    
    def clear_selected_files(self):
        """Clear all selected files with error handling"""
        try:
            if hasattr(self, 'selected_files'):
                self.selected_files.clear()
                logging.info("🗑️ Cleared selected files list")
            else:
                self.selected_files = []
                logging.warning("⚠️ selected_files was not initialized, created empty list")
            
            if hasattr(self, 'manual_files_listbox') and self.manual_files_listbox.winfo_exists():
                self.manual_files_listbox.delete(0, tk.END)
                logging.debug("🗑️ Cleared manual files listbox")
            
            self.status_var.set("File selection cleared")
            
        except Exception as e:
            error_msg = f"Error clearing selected files: {e}"
            logging.error(f"❌ {error_msg}")
            messagebox.showerror("Clear Files Error", error_msg)
    
    def upload_manual(self):
        """Upload multiple manual files to system"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return
        
        if not self.selected_files:
            messagebox.showerror("Error", "Please select manual files")
            return
        
        # Check if all files exist
        for file_path in self.selected_files:
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"File does not exist: {os.path.basename(file_path)}")
                return
        
        # Disable upload button and show progress
        self.upload_btn.config(state='disabled')
        self.upload_progress.start()
        self.status_var.set(f"Uploading {len(self.selected_files)} manual files...")
        
        # Run upload in separate thread with detailed progress
        def upload_thread():
            try:
                manual_type = self.manual_type_var.get()
                version = self.manual_version_var.get()
                total_files = len(self.selected_files)
                
                logging.info(f"📤 Starting batch upload of {total_files} manual files")
                logging.info(f"🏷️ Manual type: {manual_type}, Version: {version}")
                
                uploaded_files = []
                failed_files = []
                
                for i, file_path in enumerate(self.selected_files):
                    try:
                        file_name = os.path.basename(file_path)
                        progress_msg = f"Uploading file {i+1}/{total_files}: {file_name}"
                        
                        # Update status in main thread
                        self.root.after(0, lambda msg=progress_msg: self.status_var.set(msg))
                        
                        # Log detailed progress
                        logging.info(f"📤 [{i+1}/{total_files}] Processing: {file_name}")
                        file_size = self.get_file_size(file_path)
                        logging.debug(f"📊 File size: {file_size}")
                        
                        # Perform the upload
                        success = self.system.upload_manual(file_path, manual_type, version)
                        
                        if success:
                            uploaded_files.append(file_path)
                            logging.info(f"✅ [{i+1}/{total_files}] Successfully uploaded: {file_name}")
                        else:
                            failed_files.append(file_path)
                            logging.error(f"❌ [{i+1}/{total_files}] Failed to upload: {file_name}")
                            
                    except Exception as e:
                        error_msg = f"Failed to upload {os.path.basename(file_path)}: {e}"
                        logging.error(f"💥 [{i+1}/{total_files}] {error_msg}")
                        logging.debug(f"Upload error details for {file_path}", exc_info=True)
                        failed_files.append(file_path)
                
                # Update UI in main thread
                self.root.after(0, lambda: self.upload_multiple_complete(uploaded_files, failed_files, manual_type, version))
                
            except Exception as e:
                error_msg = f"Upload failed: {e}"
                self.root.after(0, lambda: self.upload_error(error_msg))
        
        self.manual_upload_thread = threading.Thread(target=upload_thread)
        self.manual_upload_thread.start()
    
    def upload_multiple_complete(self, uploaded_files, failed_files, manual_type, version):
        """Handle multiple file upload completion"""
        self.upload_progress.stop()
        self.upload_btn.config(state='normal')
        
        # Add uploaded files to manual list
        for file_path in uploaded_files:
            manual_name = os.path.basename(file_path)
            self.manual_tree.insert('', 'end', text=manual_name, 
                                   values=(manual_type, version, file_path, "Processing...", "✅ Uploaded"))
        
        # Update system stats
        self.update_system_stats()
        
        # Clear selected files
        self.selected_files.clear()
        self.manual_files_listbox.delete(0, tk.END)
        
        # Show results message
        if uploaded_files and not failed_files:
            self.status_var.set(f"All {len(uploaded_files)} manuals uploaded successfully")
            messagebox.showinfo("Success", f"All {len(uploaded_files)} manual files uploaded successfully!")
        elif uploaded_files and failed_files:
            self.status_var.set(f"{len(uploaded_files)} manuals uploaded, {len(failed_files)} failed")
            failed_names = [os.path.basename(f) for f in failed_files]
            messagebox.showwarning("Partial Success", 
                                 f"Uploaded: {len(uploaded_files)} files\n"
                                 f"Failed: {len(failed_files)} files\n\n"
                                 f"Failed files:\n" + "\n".join(failed_names))
        else:
            self.status_var.set("All manual uploads failed")
            messagebox.showerror("Error", "Failed to upload any manual files. Check logs for details.")
    
    def upload_complete(self, success, file_path, manual_type, version):
        """Handle single upload completion (legacy method for compatibility)"""
        self.upload_progress.stop()
        self.upload_btn.config(state='normal')
        
        if success:
            # Add to manual list
            manual_name = os.path.basename(file_path)
            self.manual_tree.insert('', 'end', text=manual_name, 
                                   values=(manual_type, version, file_path, "Processing...", "✅ Uploaded"))
            
            self.status_var.set(f"Manual uploaded successfully: {manual_name}")
            messagebox.showinfo("Success", f"Manual uploaded successfully:\n{manual_name}")
            
            # Update system stats
            self.update_system_stats()
            
        else:
            self.status_var.set("Manual upload failed")
            messagebox.showerror("Error", "Failed to upload manual. Check logs for details.")
    
    def upload_error(self, error_msg):
        """Handle upload error"""
        self.upload_progress.stop()
        self.upload_btn.config(state='normal')
        self.status_var.set("Manual upload failed")
        messagebox.showerror("Upload Error", error_msg)
    
    def on_manual_select_safe(self, _event):
        """Handle manual selection in tree with error handling
        
        Args:
            _event: Tkinter event object (unused but required by callback signature)
        """
        try:
            self.on_manual_select()
        except Exception as e:
            logging.error(f"❌ Error in manual selection: {e}")
    
    def on_manual_select(self):
        """Handle manual selection in tree"""
        try:
            selection = self.manual_tree.selection()
            if selection:
                item = self.manual_tree.item(selection[0])
                manual_name = item['text']
                values = item['values']
                
                logging.debug(f"📋 Selected manual: {manual_name}")
                
                if values and len(values) >= 5:
                    manual_type, version, file_path, chunks, status = values
                    
                    details = f"""Manual Details:
                    
Name: {manual_name}
Type: {manual_type}
Version: {version}
File Path: {file_path}
Chunks: {chunks}
Status: {status}

File Information:
Size: {self.get_file_size(file_path)}
Modified: {self.get_file_modified(file_path)}
"""
                    
                    if hasattr(self, 'manual_details_text') and self.manual_details_text.winfo_exists():
                        self.manual_details_text.delete(1.0, tk.END)
                        self.manual_details_text.insert(1.0, details)
                else:
                    logging.warning("⚠️ Manual selection has incomplete data")
        except Exception as e:
            logging.error(f"❌ Error handling manual selection: {e}")
    
    def get_file_size(self, file_path):
        """Get file size in human readable format"""
        try:
            size = os.path.getsize(file_path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"
        except:
            return "Unknown"
    
    def get_file_modified(self, file_path):
        """Get file modification time"""
        try:
            mtime = os.path.getmtime(file_path)
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except:
            return "Unknown"
    
    def generate_code(self):
        """Generate code based on user input"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return
        
        # Check if Claude API is configured
        if self.system.claude_client.test_mode:
            result = messagebox.askyesno(
                "API Not Configured", 
                "Claude API key is not configured. Code generation requires a valid API key.\n\n"
                "Would you like to go to Settings to configure your API key?"
            )
            if result:
                self.notebook.select(4)  # Switch to Settings tab
            return
        
        task = self.task_var.get().strip()
        if not task:
            messagebox.showerror("Error", "Please enter a task description")
            return
        
        # Disable generate button and show progress
        self.generate_btn.config(state='disabled')
        self.generation_progress.start()
        self.status_var.set("Generating code...")
        
        # Clear previous results
        self.code_text.delete(1.0, tk.END)
        self.confidence_var.set("Confidence: Generating...")
        self.references_var.set("Manual References: Generating...")
        
        # Run generation in separate thread with detailed progress
        def generation_thread():
            try:
                language = self.language_var.get()
                manual_type = self.gen_manual_type_var.get() or None
                version = self.gen_version_var.get() or None
                style = self.style_var.get()
                specifications = self.specifications_text.get(1.0, tk.END).strip()
                
                # Log generation parameters
                logging.info(f"🚀 Starting code generation process")
                logging.info(f"📝 Task: {task}")
                logging.info(f"🔥 Language: {language}")
                if manual_type:
                    logging.info(f"📚 Manual type: {manual_type}")
                if version:
                    logging.info(f"🏷️ Version: {version}")
                logging.info(f"🎨 Style: {style}")
                if specifications:
                    logging.debug(f"📜 Specifications: {specifications[:100]}...")
                
                # Run async function in thread
                logging.debug("⚙️ Setting up async event loop...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                logging.info("🔍 Searching for relevant manual content...")
                result = loop.run_until_complete(
                    self.system.generate_code(
                        task=task,
                        language=language,
                        manual_type=manual_type,
                        version=version,
                        specifications=specifications,
                        style=style
                    )
                )
                
                loop.close()
                logging.info("✅ Code generation completed successfully")
                
                # Update UI in main thread
                self.root.after(0, lambda: self.generation_complete(result))
                
            except Exception as e:
                error_msg = str(e)
                if "API key not configured" in error_msg or "Claude API not available" in error_msg:
                    self.root.after(0, lambda: self.api_not_configured_error())
                else:
                    full_error = f"Code generation failed: {e}"
                    self.root.after(0, lambda: self.generation_error(full_error))
        
        self.code_generation_thread = threading.Thread(target=generation_thread)
        self.code_generation_thread.start()
    
    def generation_complete(self, result: CodeGenerationResult):
        """Handle code generation completion"""
        self.generation_progress.stop()
        self.generate_btn.config(state='normal')
        
        # Display generated code
        self.code_text.delete(1.0, tk.END)
        self.code_text.insert(1.0, result.code)
        
        # Update info
        self.confidence_var.set(f"Confidence: {result.confidence_score:.2f}")
        self.references_var.set(f"Manual References: {len(result.manual_references)}")
        
        # Store current result for saving
        self.current_code_result = result
        
        # Add to history
        self.code_history.append({
            'timestamp': result.generated_at,
            'task': self.task_var.get(),
            'language': result.language,
            'confidence': result.confidence_score,
            'references_count': len(result.manual_references),
            'code_preview': result.code[:200] + "..." if len(result.code) > 200 else result.code
        })
        
        self.save_code_history()
        
        self.status_var.set("Code generation completed successfully")
    
    def generation_error(self, error_msg):
        """Handle code generation error"""
        self.generation_progress.stop()
        self.generate_btn.config(state='normal')
        self.status_var.set("Code generation failed")
        messagebox.showerror("Generation Error", error_msg)
        
        self.confidence_var.set("Confidence: Error")
        self.references_var.set("Manual References: Error")
    
    def api_not_configured_error(self):
        """Handle API not configured error"""
        self.generation_progress.stop()
        self.generate_btn.config(state='normal')
        self.status_var.set("API key required for code generation")
        
        result = messagebox.askyesno(
            "API Configuration Required", 
            "Claude API key is required for code generation.\n\n"
            "The system is currently using test mode which only returns simple templates.\n\n"
            "Would you like to configure your API key now?"
        )
        
        if result:
            self.notebook.select(4)  # Switch to Settings tab
        
        self.confidence_var.set("Confidence: N/A (No API)")
        self.references_var.set("Manual References: N/A (No API)")
    
    def save_generated_code(self):
        """Save generated code to file"""
        if not hasattr(self, 'current_code_result') or not self.current_code_result:
            messagebox.showerror("Error", "No code to save")
            return
        
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return
        
        try:
            saved_path = self.system.save_generated_code(self.current_code_result)
            if saved_path:
                messagebox.showinfo("Success", f"Code saved to:\n{saved_path}")
                self.status_var.set(f"Code saved to: {os.path.basename(saved_path)}")
            else:
                messagebox.showerror("Error", "Failed to save code")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save code: {e}")
    
    def copy_code(self):
        """Copy generated code to clipboard"""
        code = self.code_text.get(1.0, tk.END).strip()
        if code:
            self.root.clipboard_clear()
            self.root.clipboard_append(code)
            self.status_var.set("Code copied to clipboard")
        else:
            messagebox.showerror("Error", "No code to copy")
    
    def validate_current_code(self):
        """Validate currently generated code"""
        code = self.code_text.get(1.0, tk.END).strip()
        if not code:
            messagebox.showerror("Error", "No code to validate")
            return
        
        # Switch to validation tab and populate
        self.notebook.select(2)  # Validation tab
        self.validation_code_text.delete(1.0, tk.END)
        self.validation_code_text.insert(1.0, code)
        
        # Set language
        if hasattr(self, 'current_code_result') and self.current_code_result:
            self.val_language_var.set(self.current_code_result.language)
        
        # Trigger validation
        self.validate_code()
    
    def validate_code(self):
        """Validate code in validation tab"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return
        
        code = self.validation_code_text.get(1.0, tk.END).strip()
        if not code:
            messagebox.showerror("Error", "No code to validate")
            return
        
        try:
            language = self.val_language_var.get()
            manual_type = self.val_manual_type_var.get() or None
            
            result = self.system.validate_code(code, language, manual_type)
            
            # Update validation results
            self.compliance_var.set(f"Compliance: {result.compliance_score}/100")
            self.syntax_var.set(f"Syntax: {result.syntax_score}/10")
            self.compatibility_var.set(f"API Compatibility: {result.api_compatibility}/10")
            
            # Display suggestions
            self.validation_results_text.delete(1.0, tk.END)
            
            results_text = f"Validation Status: {'✅ Valid' if result.is_valid else '❌ Invalid'}\n\n"
            
            if result.suggestions:
                results_text += "Suggestions:\n"
                for i, suggestion in enumerate(result.suggestions, 1):
                    results_text += f"{i}. [{suggestion['type'].upper()}] {suggestion['description']}\n"
                    if 'manual_reference' in suggestion:
                        results_text += f"   Reference: {suggestion['manual_reference']}\n"
                    results_text += "\n"
            else:
                results_text += "No suggestions - code looks good!\n"
            
            if result.manual_references:
                results_text += f"\nManual References Checked:\n"
                for ref in result.manual_references:
                    results_text += f"- {ref}\n"
            
            self.validation_results_text.insert(1.0, results_text)
            
            self.status_var.set("Code validation completed")
            
        except Exception as e:
            error_msg = f"Validation failed: {e}"
            messagebox.showerror("Validation Error", error_msg)
            self.status_var.set("Code validation failed")
    
    def search_manuals(self):
        """Search manual content"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return
        
        query = self.search_var.get().strip()
        if not query:
            messagebox.showerror("Error", "Please enter a search query")
            return
        
        try:
            manual_type = self.search_manual_type_var.get() or None
            
            results = self.system.vector_db.search_manual_content(
                query=query,
                manual_type=manual_type,
                top_k=20
            )
            
            # Clear previous results
            for item in self.search_tree.get_children():
                self.search_tree.delete(item)
            
            # Add results to tree
            for result in results['results']:
                preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                
                self.search_tree.insert('', 'end', text=preview, values=(
                    f"{result['relevance_score']:.2f}",
                    result['manual_type'],
                    result['version'],
                    result['section']
                ))
            
            self.status_var.set(f"Found {results['total_results']} search results")
            
        except Exception as e:
            error_msg = f"Search failed: {e}"
            messagebox.showerror("Search Error", error_msg)
            self.status_var.set("Manual search failed")
    
    def on_search_select_safe(self, _event):
        """Handle search result selection with error handling
        
        Args:
            _event: Tkinter event object (unused but required by callback signature)
        """
        try:
            self.on_search_select()
        except Exception as e:
            logging.error(f"❌ Error in search selection: {e}")
    
    def on_search_select(self):
        """Handle search result selection"""
        try:
            selection = self.search_tree.selection()
            if selection and self.system:
                item_index = self.search_tree.index(selection[0])
                
                logging.debug(f"🔍 Selected search result at index {item_index}")
                
                # Get the corresponding result from last search
                query = self.search_var.get().strip()
                manual_type = self.search_manual_type_var.get() or None
                
                results = self.system.vector_db.search_manual_content(
                    query=query,
                    manual_type=manual_type,
                    top_k=20
                )
                
                if item_index < len(results['results']):
                    result = results['results'][item_index]
                    
                    # Display full content
                    content = f"Section: {result['section']}\n"
                    content += f"Manual Type: {result['manual_type']}\n"
                    content += f"Version: {result['version']}\n"
                    content += f"Relevance: {result['relevance_score']:.2f}\n"
                    content += f"Source: {result['source_file']}\n"
                    content += "-" * 50 + "\n\n"
                    content += result['content']
                    
                    if hasattr(self, 'search_content_text') and self.search_content_text.winfo_exists():
                        self.search_content_text.delete(1.0, tk.END)
                        self.search_content_text.insert(1.0, content)
                        logging.debug("📄 Displayed search result content")
                else:
                    logging.warning(f"⚠️ Search result index {item_index} out of range")
                    
        except Exception as e:
            logging.error(f"❌ Error handling search result selection: {e}")
    
    def browse_db_path(self):
        """Browse for vector database path"""
        directory = filedialog.askdirectory(title="Select Vector Database Directory")
        if directory:
            self.db_path_var.set(directory)
    
    def save_settings(self):
        """Save application settings"""
        self.config = {
            'claude_api_key': self.api_key_var.get(),
            'vector_db_path': self.db_path_var.get(),
            'default_language': self.default_language_var.get(),
            'default_style': self.default_style_var.get(),
            'auto_save_generated_code': True  # This would come from checkbox
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            messagebox.showinfo("Success", "Settings saved successfully")
            self.status_var.set("Settings saved")
            
            # Reinitialize system with new settings
            self.initialize_system()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def load_config(self):
        """Load application configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
        
        return {
            'claude_api_key': '',
            'vector_db_path': './manual_db',
            'default_language': 'python',
            'default_style': 'professional',
            'auto_save_generated_code': True
        }
    
    def save_code_history(self):
        """Save code generation history"""
        try:
            with open(self.code_history_file, 'w') as f:
                json.dump(self.code_history, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save code history: {e}")
    
    def load_code_history(self):
        """Load code generation history"""
        try:
            if os.path.exists(self.code_history_file):
                with open(self.code_history_file, 'r') as f:
                    self.code_history = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load code history: {e}")
            self.code_history = []
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
        self.status_var.set("Logs cleared")
    
def main():
    """Main function to run the GUI with enhanced error handling"""
    try:
        # Check tkinter availability
        try:
            import tkinter as tk
            print("✅ tkinter is available")
        except ImportError as e:
            print(f"❌ tkinter not available: {e}")
            print("Please install tkinter or use a Python distribution that includes it.")
            return
        
        print("🚀 Starting AI Coder GUI...")
        root = tk.Tk()
        
        # Set up error handling for tkinter
        def handle_tk_error(exc, val, tb):
            print(f"Tkinter error: {exc.__name__}: {val}")
            import traceback
            traceback.print_exception(exc, val, tb)
        
        root.report_callback_exception = handle_tk_error
        
        # Initialize application
        AICoderGUI(root)
        print("✅ AI Coder GUI initialized successfully")
        
        # Start main loop
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required dependencies are installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Failed to start AI Coder GUI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()