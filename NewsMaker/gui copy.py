#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AI News Writer GUI
Professional news generation system with advanced features
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import asyncio
import threading
import json
import os
from datetime import datetime, timedelta
import logging
import sys
import traceback
import tkinter.font as tkfont
from ttkthemes import ThemedTk
import csv
import html

# main.py에서 필요한 클래스들 임포트
try:
    from main import AINewsWriterSystem
except ImportError as e:
    print(f"main.py 파일을 찾을 수 없습니다: {e}")
    print("main.py가 같은 폴더에 있는지 확인해주세요.")
    sys.exit(1)

# GUI용 로깅 핸들러
class GUILogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        # Setup color tags
        self.text_widget.tag_config('ERROR', foreground='red')
        self.text_widget.tag_config('WARNING', foreground='orange')
        self.text_widget.tag_config('INFO', foreground='green')
        self.text_widget.tag_config('DEBUG', foreground='gray')
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if self.text_widget and self.text_widget.winfo_exists():
                level = record.levelname
                tag = level if level in ('ERROR', 'WARNING', 'INFO', 'DEBUG') else None
                self.text_widget.insert(tk.END, msg + '\n', tag)
                self.text_widget.see(tk.END)
                self.text_widget.update()
        except Exception:
            pass  # GUI가 닫혔을 때 오류 방지

class EnhancedNewsWriterGUI:
    def __init__(self, root):
        # Set a global font
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=10)
        root.option_add("*Font", default_font)
        # Set window icon placeholder (if you have an .ico file, set it here)
        # root.iconbitmap('app_icon.ico')
        # Add status bar
        self.statusbar_var = tk.StringVar(value="Ready.")
        self.root = root
        self.root.title("AI News Writer Pro - 전문 뉴스 자동 생성 시스템")
        self.root.geometry("1069x768")
        self.root.minsize(900, 600)
        
        # 시스템 인스턴스
        self.system = None
        self.collection_thread = None
        self.is_collecting = False
        self.collected_news = []  # 수집된 뉴스 저장
        self.saved_articles_count = 0
        self.auto_scroll = True  # <-- Fix: Ensure auto_scroll is always initialized
        
        # 뉴스 저장 디렉토리
        self.news_directory = "collected_news"
        if not os.path.exists(self.news_directory):
            os.makedirs(self.news_directory)
        
        self.news_history_file = "generated_news_history.json"
        self.news_history = []
        self.load_news_history()
        
        # Initialize vector stats variables to prevent attribute errors
        self.vector_total_chunks_var = tk.StringVar(value="0")
        self.vector_collection_name_var = tk.StringVar(value="unknown")
        self.vector_last_update_var = tk.StringVar(value="N/A")
        self.vector_avg_relevance_var = tk.StringVar(value="N/A")
        
        self.setup_ui()
        self.setup_logging()
        self.load_config()
        # Add status bar at the bottom
        self.statusbar = ttk.Label(root, textvariable=self.statusbar_var, relief=tk.SUNKEN, anchor='w', padding=4)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 시작 시 자동으로 시스템 초기화
        self.root.after(1000, self.auto_initialize_system)
        
    def setup_ui(self):
        """UI 구성 (벡터DB 탭 추가)"""
        # PanedWindow로 상단(탭)과 하단(로그) 분할
        paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=6, showhandle=True)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 상단 프레임(탭)
        top_frame = ttk.Frame(paned)
        paned.add(top_frame, stretch='always', minsize=350)

        # 하단 프레임(로그)
        bottom_frame = ttk.Frame(paned)
        paned.add(bottom_frame, stretch='always', minsize=120)

        # 메인 노트북 (탭)
        self.notebook = ttk.Notebook(top_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 탭 생성 순서: 설정, 뉴스 수집, 뉴스 작성, 벡터DB, 최근 생성 뉴스/프롬프트
        self.setup_config_tab(self.notebook)
        self.setup_collection_tab(self.notebook)
        self.setup_writing_tab(self.notebook)
        self.setup_vector_stats_tab(self.notebook)  # 벡터DB 탭 (moved here)
        self.setup_history_tab(self.notebook)

        # 벡터DB 탭 선택 시 로그에 내용 표시 (다이얼로그 X)
        def on_tab_changed(event):
            selected_tab = event.widget.select()
            tab_text = event.widget.tab(selected_tab, "text")
            
            if "벡터DB" in tab_text:
                # Show vector DB content in log panel instead of dialog
                try:
                    content = self.get_vector_db_log_content()
                    self.log_text.insert(tk.END, content + '\n')
                    self.log_text.see(tk.END)
                except Exception as e:
                    self.log_text.insert(tk.END, f"[벡터DB 표시 오류] {e}\n")
                    self.log_text.see(tk.END)
            elif "뉴스 수집" in tab_text:
                # 뉴스 수집 탭이 선택되면 저장된 뉴스 로드
                if self.system and hasattr(self, 'headlines_tree'):
                    self.load_saved_headlines()
        self.notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

        # 하단 로그 프레임 (bottom_frame에)
        self.setup_bottom_log_frame(bottom_frame)

        # Add menu bar
        self.menubar = tk.Menu(self.root)
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=file_menu)
        edit_menu = tk.Menu(self.menubar, tearoff=0)
        edit_menu.add_command(label="Copy", command=lambda: self.root.event_generate('<<Copy>>'))
        edit_menu.add_command(label="Paste", command=lambda: self.root.event_generate('<<Paste>>'))
        self.menubar.add_cascade(label="Edit", menu=edit_menu)
        help_menu = tk.Menu(self.menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: tk.messagebox.showinfo("About", "AI News Writer Pro\nProfessional News Generation System\nPowered by ttkthemes"))
        self.menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=self.menubar)

        # Add padding and style to all frames and widgets
        style = ttk.Style()
        style.theme_use('arc')  # Use a modern theme from ttkthemes
        style.configure('TFrame', padding=8)
        style.configure('TLabel', padding=4)
        style.configure('TButton', padding=6)
        style.configure('TNotebook.Tab', padding=[12, 6])
        style.configure('TEntry', padding=4)
        style.configure('TCombobox', padding=4)
        style.configure('Treeview', rowheight=24)

        # Add tooltips to important buttons/fields
        self.add_tooltips()

    def get_vector_db_log_content(self):
        # Return a comprehensive summary string of the vector DB content for log panel
        try:
            if not self.system or not self.system.db_manager:
                return "[벡터DB] 시스템이 초기화되지 않았습니다"
            
            stats = self.system.db_manager.get_collection_stats()
            if not stats or stats.get("total_chunks", 0) == 0:
                return "[벡터DB] 데이터 없음"
            
            # Get detailed collection data for enhanced summary
            collection_data = self.system.db_manager.collection.get(
                include=['documents', 'metadatas'], 
                limit=10  # Get top 10 for preview
            )
            
            lines = ["=== 벡터 데이터베이스 요약 ==="]
            lines.append(f"📊 총 청크 수: {stats.get('total_chunks', 0):,}")
            lines.append(f"🗂️ 컬렉션명: {stats.get('collection_name', 'unknown')}")
            lines.append(f"🔢 임베딩 차원: {stats.get('embedding_dimension', 'unknown')}")
            
            # Add topic distribution summary
            if collection_data and 'metadatas' in collection_data:
                topics_count = {}
                for metadata in collection_data['metadatas'][:10]:  # Sample first 10
                    if metadata and 'topics' in metadata:
                        try:
                            import json
                            topics_raw = metadata['topics']
                            topics = json.loads(topics_raw) if isinstance(topics_raw, str) else topics_raw
                            if isinstance(topics, list):
                                for topic in topics[:2]:  # Take first 2 topics
                                    topics_count[topic] = topics_count.get(topic, 0) + 1
                        except:
                            continue
                
                if topics_count:
                    lines.append(f"🏷️ 주요 토픽: {', '.join(list(topics_count.keys())[:3])}")
            
            lines.append("")
            lines.append("=== 최근 벡터 데이터베이스 내용 (상위 10개) ===")
            
            # Add content preview
            if collection_data and 'documents' in collection_data:
                for i, (doc_id, document) in enumerate(zip(
                    collection_data.get('ids', []), 
                    collection_data.get('documents', [])
                )):
                    if i >= 10:  # Limit to 10 items
                        break
                    
                    # Get metadata for this document
                    metadata = collection_data['metadatas'][i] if i < len(collection_data.get('metadatas', [])) else {}
                    
                    # Extract topic info
                    topic_info = "N/A"
                    if metadata and 'topics' in metadata:
                        try:
                            import json
                            topics_raw = metadata['topics']
                            topics = json.loads(topics_raw) if isinstance(topics_raw, str) else topics_raw
                            if isinstance(topics, list) and topics:
                                topic_info = ', '.join(topics[:2])
                        except:
                            pass
                    
                    # Content preview (first 100 chars)
                    content_preview = document[:100] + "..." if len(document) > 100 else document
                    
                    lines.append(f"📄 ID: {str(doc_id)[:20]}")
                    lines.append(f"   토픽: {topic_info}")
                    lines.append(f"   내용: {content_preview}")
                    lines.append("")
            
            return '\n'.join(lines)
            
        except Exception as e:
            import traceback
            return f"[벡터DB 상태 조회 오류] {e}\n상세: {traceback.format_exc()[:200]}"

    def reload_previous_news(self):
        """이전 뉴스 불러오기 (NEW FUNCTION)"""
        try:
            # 저장된 뉴스 파일들 스캔
            news_files = []
            if os.path.exists(self.news_directory):
                for filename in os.listdir(self.news_directory):
                    if filename.endswith('.txt') and filename.startswith('news_'):
                        filepath = os.path.join(self.news_directory, filename)
                        try:
                            # 파일 정보 읽기
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # 제목 추출
                            lines = content.split('\n')
                            title = "제목 없음"
                            for line in lines:
                                if line.startswith('제목:'):
                                    title = line.replace('제목:', '').strip()
                                    break
                            
                            # 파일 정보 저장
                            file_stat = os.stat(filepath)
                            news_files.append({
                                'filename': filename,
                                'filepath': filepath,
                                'title': title,
                                'size': file_stat.st_size,
                                'modified': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                            })
                        except Exception as e:
                            logging.warning(f"파일 읽기 실패 {filename}: {e}")
            
            if not news_files:
                messagebox.showinfo("알림", "불러올 이전 뉴스 파일이 없습니다.")
                return
            
            # 뉴스 선택 창 열기
            self.show_news_selection_dialog(news_files)
            
        except Exception as e:
            messagebox.showerror("오류", f"이전 뉴스 불러오기 실패: {e}")
            logging.error(f"이전 뉴스 불러오기 실패: {e}")

    def show_news_selection_dialog(self, news_files):
        """뉴스 선택 다이얼로그 (NEW FUNCTION)"""
        # 새 창 생성
        selection_window = tk.Toplevel(self.root)
        selection_window.title("이전 뉴스 선택")
        selection_window.geometry("800x600")
        selection_window.transient(self.root)
        selection_window.grab_set()
        
        # 상단 안내
        info_frame = ttk.Frame(selection_window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text="벡터 데이터베이스에 추가할 뉴스를 선택하세요 (다중 선택 가능)", 
                font=("", 10, "bold")).pack()
        
        # 버튼 프레임 (상단으로 이동)
        button_frame = ttk.Frame(selection_window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="전체 선택", 
                command=lambda: self.select_all_news(news_tree, selected_items)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="전체 해제", 
                command=lambda: self.deselect_all_news(news_tree, selected_items)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="선택된 뉴스 벡터DB 추가", 
                command=lambda: self.process_selected_news(selection_window, selected_items)).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="취소", 
                command=selection_window.destroy).pack(side=tk.RIGHT, padx=5)

        # 뉴스 목록 (체크박스 포함)
        list_frame = ttk.Frame(selection_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 트리뷰로 뉴스 목록 표시
        columns = ('select', 'title', 'filename', 'size', 'modified')
        news_tree = ttk.Treeview(list_frame, columns=columns, show='tree headings', height=20)
        
        news_tree.heading('#0', text='번호')
        news_tree.heading('select', text='선택')
        news_tree.heading('title', text='제목')
        news_tree.heading('filename', text='파일명')
        news_tree.heading('size', text='크기')
        news_tree.heading('modified', text='수정일')
        
        news_tree.column('#0', width=50, minwidth=50)
        news_tree.column('select', width=50, minwidth=50)
        news_tree.column('title', width=300, minwidth=200)
        news_tree.column('filename', width=200, minwidth=150)
        news_tree.column('size', width=80, minwidth=60)
        news_tree.column('modified', width=120, minwidth=100)
        
        # 스크롤바
        news_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=news_tree.yview)
        news_tree.configure(yscrollcommand=news_scrollbar.set)
        
        # 뉴스 파일 목록 추가
        selected_items = {}
        for i, news_file in enumerate(news_files):
            item_id = news_tree.insert('', 'end', 
                text=str(i+1),
                values=('☐', news_file['title'][:50] + '...', news_file['filename'], 
                    f"{news_file['size']} bytes", news_file['modified'])
            )
            selected_items[item_id] = {'selected': False, 'data': news_file}
        
        # 클릭 이벤트로 체크박스 토글
        def toggle_selection(event):
            item = news_tree.selection()[0] if news_tree.selection() else None
            if item and item in selected_items:
                current_values = list(news_tree.item(item, 'values'))
                if selected_items[item]['selected']:
                    current_values[0] = '☐'
                    selected_items[item]['selected'] = False
                else:
                    current_values[0] = '☑'
                    selected_items[item]['selected'] = True
                news_tree.item(item, values=current_values)
        
        news_tree.bind('<Double-1>', toggle_selection)
        
        news_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        news_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def select_all_news(self, news_tree, selected_items):
        """전체 뉴스 선택 (NEW FUNCTION)"""
        for item_id in selected_items:
            selected_items[item_id]['selected'] = True
            current_values = list(news_tree.item(item_id, 'values'))
            current_values[0] = '☑'
            news_tree.item(item_id, values=current_values)

    def deselect_all_news(self, news_tree, selected_items):
        """전체 뉴스 선택 해제 (NEW FUNCTION)"""
        for item_id in selected_items:
            selected_items[item_id]['selected'] = False
            current_values = list(news_tree.item(item_id, 'values'))
            current_values[0] = '☐'
            news_tree.item(item_id, values=current_values)

    def process_selected_news(self, selection_window, selected_items):
        """선택된 뉴스를 벡터DB에 추가 (NEW FUNCTION)"""
        # 선택된 항목들 수집
        selected_files = []
        for item_id, item_data in selected_items.items():
            if item_data['selected']:
                selected_files.append(item_data['data'])
        
        if not selected_files:
            messagebox.showwarning("경고", "선택된 뉴스가 없습니다.")
            return
        
        # 확인 다이얼로그
        if not messagebox.askyesno("확인", f"선택된 {len(selected_files)}개 뉴스를 벡터 데이터베이스에 추가하시겠습니까?"):
            return
        
        selection_window.destroy()
        
        # 별도 스레드에서 처리
        def process_worker():
            try:
                processed_count = 0
                
                for news_file in selected_files:
                    try:
                        # 파일 내용 읽기
                        with open(news_file['filepath'], 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # 간단한 파싱으로 기사 정보 추출
                        article_info = self.parse_saved_news_file(content, news_file['filename'])
                        
                        if article_info:
                            # 벡터DB에 추가
                            success = asyncio.run(self.add_news_to_vector_db(article_info))
                            if success:
                                processed_count += 1
                                
                            # UI 업데이트
                            self.root.after(0, lambda: logging.info(f"처리 완료: {news_file['filename']}"))
                        
                    except Exception as e:
                        self.root.after(0, lambda e=e: logging.error(f"파일 처리 실패: {e}"))
                
                # 완료 메시지
                self.root.after(0, lambda: messagebox.showinfo("완료", f"{processed_count}개 뉴스가 벡터 데이터베이스에 추가되었습니다."))
                self.root.after(0, lambda: self.refresh_vector_stats())
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("오류", f"벡터DB 추가 실패: {e}"))
        
        threading.Thread(target=process_worker, daemon=True).start()

    def add_selected_to_vector(self):
        """선택된 헤드라인 뉴스를 벡터DB에 추가 (NEW FUNCTION)"""
        selection = self.headlines_tree.selection()
        if not selection:
            messagebox.showwarning("경고", "벡터DB에 추가할 뉴스를 선택해주세요.")
            return
        
        selected_news = []
        for item in selection:
            try:
                index = int(self.headlines_tree.item(item, "text")) - 1
                if 0 <= index < len(self.collected_news):
                    selected_news.append(self.collected_news[index])
            except (ValueError, IndexError):
                continue
        
        if not selected_news:
            messagebox.showwarning("경고", "유효한 뉴스가 선택되지 않았습니다.")
            return
        
        if messagebox.askyesno("확인", f"선택된 {len(selected_news)}개 뉴스를 벡터 데이터베이스에 추가하시겠습니까?"):
            def add_worker():
                try:
                    processed_count = 0
                    for news in selected_news:
                        success = asyncio.run(self.add_news_to_vector_db(news))
                        if success:
                            processed_count += 1
                    
                    self.root.after(0, lambda: messagebox.showinfo("완료", f"{processed_count}개 뉴스가 벡터 데이터베이스에 추가되었습니다."))
                    self.root.after(0, lambda: self.refresh_vector_stats())
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("오류", f"벡터DB 추가 실패: {e}"))
            
            threading.Thread(target=add_worker, daemon=True).start()

    def refresh_vector_stats(self):
        """벡터 데이터베이스 통계 새로고침 (NEW FUNCTION)"""
        try:
            if not self.system:
                return
            
            stats = self.system.get_system_stats()
            db_stats = stats.get('database', {})
            
            # 기본 통계 업데이트
            self.vector_total_chunks_var.set(str(db_stats.get('total_chunks', 0)))
            self.vector_collection_name_var.set(db_stats.get('collection_name', 'unknown'))
            self.vector_last_update_var.set(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # 상세 통계 조회
            try:
                all_data = self.system.db_manager.collection.get(include=['metadatas'])
                if all_data.get('metadatas'):
                    relevance_scores = []
                    for metadata in all_data['metadatas']:
                        relevance = metadata.get('relevance_score', 0)
                        if isinstance(relevance, (int, float)):
                            relevance_scores.append(relevance)
                    
                    if relevance_scores:
                        avg_relevance = sum(relevance_scores) / len(relevance_scores)
                        self.vector_avg_relevance_var.set(f"{avg_relevance:.1f}/10")
                    else:
                        self.vector_avg_relevance_var.set("N/A")
                else:
                    self.vector_avg_relevance_var.set("N/A")
            except:
                self.vector_avg_relevance_var.set("N/A")
            
            logging.info("벡터 데이터베이스 통계 새로고침 완료")
            
        except Exception as e:
            logging.error(f"벡터 통계 새로고침 실패: {e}")

    def export_vector_db(self):
        """벡터 데이터베이스 내보내기 (FIXED VERSION)"""
        try:
            if not self.system:
                messagebox.showwarning("경고", "시스템이 초기화되지 않았습니다(E).")
                return
            
            # 저장할 파일 선택 (FIXED PARAMETERS!)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"vector_db_export_{timestamp}.json"
            
            file_path = filedialog.asksaveasfilename(
                title="벡터 데이터베이스 내보내기",
                initialvalue=default_filename,  # ✅ FIXED: Correct parameter name
                defaultextension=".json",
                filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")]
            )
            
            if not file_path:
                return
            
            # 데이터 조회 시작 메시지
            messagebox.showinfo("알림", "데이터를 내보내는 중입니다. 잠시만 기다려주세요...")
            
            try:
                # ✅ FIXED: Proper get() method usage
                all_data = self.system.db_manager.collection.get(
                    include=['documents', 'metadatas']
                )
                
                if not all_data.get('documents'):
                    messagebox.showinfo("알림", "내보낼 데이터가 없습니다.")
                    return
                
                # IDs는 항상 반환됨
                ids = all_data.get('ids', [])
                documents = all_data.get('documents', [])
                metadatas = all_data.get('metadatas', [])
                
                # 내보내기 데이터 구성
                export_data = {
                    "export_info": {
                        "timestamp": datetime.now().isoformat(),
                        "total_items": len(documents),
                        "collection_name": self.system.db_manager.collection.name,
                        "export_version": "1.0"
                    },
                    "data": []
                }
                
                # 안전한 데이터 처리
                for i in range(len(documents)):
                    try:
                        item = {
                            "id": ids[i] if i < len(ids) else f"item_{i}",
                            "document": documents[i],
                            "metadata": metadatas[i] if i < len(metadatas) else {},
                            "index": i
                        }
                        export_data["data"].append(item)
                    except Exception as item_error:
                        logging.warning(f"항목 {i} 내보내기 실패: {item_error}")
                        # 오류 항목도 기록
                        error_item = {
                            "id": f"error_item_{i}",
                            "document": f"ERROR: {str(item_error)}",
                            "metadata": {"error": True, "original_index": i},
                            "index": i
                        }
                        export_data["data"].append(error_item)
                
                # 파일로 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                success_msg = f"""벡터 데이터베이스 내보내기 완료!

파일: {file_path}
총 항목: {len(export_data['data'])}개
파일 크기: {os.path.getsize(file_path):,} bytes"""
                
                messagebox.showinfo("완료", success_msg)
                logging.info(f"벡터DB 내보내기 완료: {file_path}")
                
            except Exception as data_error:
                messagebox.showerror("데이터 오류", f"데이터 조회 실패: {data_error}")
                logging.error(f"벡터DB 데이터 조회 실패: {data_error}")
                
        except Exception as e:
            messagebox.showerror("오류", f"벡터DB 내보내기 실패: {e}")
            logging.error(f"벡터DB 내보내기 실패: {e}")

    def view_collection_contents(self):
        """컬렉션 내용 보기 (ENHANCED WITH CONTENT VIEWING & EMBEDDINGS)"""
        try:
            # Check if system is initialized
            if not self.system:
                logging.info("시스템이 초기화되지 않았습니다(C).")
                # messagebox.showwarning("경고", "시스템이 초기화되지 않았습니다. 설정 탭에서 시스템을 초기화해주세요.")
                return
            
            # Check if vector_tree exists
            if not hasattr(self, 'vector_tree') or self.vector_tree is None:
                logging.error("vector_tree가 초기화되지 않았습니다.")
                messagebox.showerror("오류", "벡터 트리 뷰가 초기화되지 않았습니다.")
                return
            
            # Clear existing items
            for item in self.vector_tree.get_children():
                self.vector_tree.delete(item)
            self.vector_full_data = []
            
            # Check collection count
            try:
                collection_count = self.system.db_manager.collection.count()
                if collection_count == 0:
                    logging.info("컬렉션이 비어있습니다.")
                    messagebox.showinfo("알림", "벡터 데이터베이스가 비어있습니다.")
                    return
                logging.info(f"컬렉션에 {collection_count}개 항목이 있습니다.")
            except Exception as e:
                logging.error(f"컬렉션 카운트 조회 실패: {e}")
                messagebox.showerror("오류", f"컬렉션 카운트 조회 실패: {e}")
                return
            
            # Get collection data
            try:
                all_data = self.system.db_manager.collection.get(
                    include=['documents', 'metadatas', 'embeddings']
                )
                ids = all_data.get('ids', [])
                documents = all_data.get('documents', [])
                metadatas = all_data.get('metadatas', [])
                embeddings = all_data.get('embeddings', [])
                
                logging.info(f"데이터 조회 완료: {len(documents)}개 문서, {len(metadatas)}개 메타데이터")
                
            except Exception as e:
                logging.error(f"컬렉션 데이터 조회 실패: {e}")
                messagebox.showerror("오류", f"컬렉션 데이터 조회 실패: {e}")
                return
            
            if not documents:
                logging.info("컬렉션에 문서가 없습니다.")
                messagebox.showinfo("알림", "벡터 데이터베이스에 저장된 문서가 없습니다.")
                return
            
            # Process and display documents
            max_display = min(300, len(documents))
            processed_count = 0
            
            for i in range(max_display):
                try:
                    doc = documents[i]
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    embedding = embeddings[i] if i < len(embeddings) else []
                    doc_id = ids[i] if i < len(ids) else f"unknown_{i}"
                    
                    full_item_data = {
                        'id': doc_id,
                        'document': doc,
                        'metadata': metadata,
                        'embedding': embedding,
                        'index': i
                    }
                    self.vector_full_data.append(full_item_data)
                    
                    # Process topics
                    try:
                        topics_raw = metadata.get('topics', '[]')
                        if isinstance(topics_raw, str):
                            topics = json.loads(topics_raw)
                        else:
                            topics = topics_raw if isinstance(topics_raw, list) else []
                        topics_str = ', '.join(topics[:2]) if topics else 'N/A'
                    except:
                        topics_str = str(metadata.get('topics', 'N/A'))[:20]
                    
                    # Process content preview - Enhanced for better readability
                    try:
                        # Extract meaningful content preview (first 80 chars)
                        content_preview = doc[:80] + "..." if len(doc) > 80 else doc
                        content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')
                        # Clean up extra spaces
                        content_preview = ' '.join(content_preview.split())
                    except:
                        content_preview = "내용 없음"
                    
                    # Process relevance score
                    try:
                        relevance = metadata.get('relevance_score', 'N/A')
                        if isinstance(relevance, (int, float)):
                            relevance_str = f"{relevance}/10"
                        else:
                            relevance_str = str(relevance)
                    except:
                        relevance_str = "N/A"
                    
                    # Process date
                    date_str = metadata.get('date', metadata.get('created_at', 'N/A'))
                    if isinstance(date_str, str) and 'T' in date_str:
                        date_str = date_str.split('T')[0]
                    
                    # Insert into tree
                    self.vector_tree.insert('', 'end',
                        text=str(i+1),
                        values=(
                            doc_id[:15] + "..." if len(str(doc_id)) > 15 else str(doc_id),
                            content_preview,
                            topics_str,
                            relevance_str,
                            str(date_str)
                        )
                    )
                    processed_count += 1
                    
                except Exception as item_error:
                    logging.warning(f"항목 {i} 처리 실패: {item_error}")
                    error_data = {
                        'id': f'error_{i}',
                        'document': f'처리 오류: {str(item_error)}',
                        'metadata': {'error': True},
                        'embedding': [],
                        'index': i
                    }
                    self.vector_full_data.append(error_data)
                    self.vector_tree.insert('', 'end',
                        text=str(i+1),
                        values=(f"error_{i}", "처리 오류", "N/A", "N/A", "N/A")
                    )
            
            # Bind double-click event
            self.vector_tree.bind('<Double-1>', self.on_vector_item_double_click)
            
            # Update comprehensive statistics
            self.vector_total_chunks_var.set(f"{len(documents):,}")
            if hasattr(self.system.db_manager, 'collection'):
                self.vector_collection_name_var.set(self.system.db_manager.collection.name)
            
            # Calculate average relevance score
            relevance_scores = []
            for metadata in metadatas:
                if metadata and 'relevance_score' in metadata:
                    try:
                        score = float(metadata['relevance_score'])
                        if 0 <= score <= 10:  # Valid relevance score range
                            relevance_scores.append(score)
                    except (ValueError, TypeError):
                        continue
            
            if relevance_scores:
                avg_relevance = sum(relevance_scores) / len(relevance_scores)
                self.vector_avg_relevance_var.set(f"{avg_relevance:.1f}/10")
            else:
                self.vector_avg_relevance_var.set("N/A")
            
            # Get last update time (most recent date from metadata)
            dates = []
            for metadata in metadatas:
                if metadata:
                    date_str = metadata.get('date', metadata.get('created_at', ''))
                    if date_str and isinstance(date_str, str):
                        dates.append(date_str)
            
            if dates:
                # Sort dates and get the most recent
                try:
                    sorted_dates = sorted(dates, reverse=True)
                    recent_date = sorted_dates[0]
                    if 'T' in recent_date:
                        recent_date = recent_date.split('T')[0]
                    self.vector_last_update_var.set(recent_date)
                except:
                    self.vector_last_update_var.set("N/A")
            else:
                self.vector_last_update_var.set("N/A")
            
            logging.info(f"컬렉션 내용 표시 완료: {processed_count}개 항목 처리됨 (전체 {len(documents)}개)")
            
            if len(documents) > 300:
                messagebox.showinfo("알림", f"총 {len(documents)}개 항목 중 처음 300개만 표시됩니다.\n\n💡 팁: 항목을 더블클릭하면 전체 내용을 볼 수 있습니다.")
            else:
                messagebox.showinfo("표시 완료", f"총 {len(documents)}개 항목이 표시되었습니다.\n\n💡 팁: 항목을 더블클릭하면 전체 내용을 볼 수 있습니다.")
                
        except Exception as e:
            error_msg = f"컬렉션 내용 조회 실패: {e}"
            logging.error(error_msg)
            messagebox.showerror("오류", error_msg)
            # Print full traceback for debugging
            import traceback
            logging.error(f"전체 오류 정보: {traceback.format_exc()}")

    def on_vector_item_double_click(self, event):
        """벡터 아이템 더블클릭 이벤트 (NEW FUNCTION)"""
        try:
            # 선택된 항목 가져오기
            selection = self.vector_tree.selection()
            if not selection:
                return
            
            item = selection[0]
            item_text = self.vector_tree.item(item, "text")
            
            try:
                # 인덱스 추출
                index = int(item_text) - 1
                if 0 <= index < len(self.vector_full_data):
                    self.show_vector_content_detail(self.vector_full_data[index])
                else:
                    messagebox.showwarning("오류", "해당 항목의 데이터를 찾을 수 없습니다.")
            except (ValueError, IndexError):
                messagebox.showerror("오류", "항목 인덱스를 파싱할 수 없습니다.")
                
        except Exception as e:
            logging.error(f"벡터 아이템 클릭 처리 실패: {e}")

    def show_vector_content_detail(self, item_data, current_index=None):
        """벡터 컨텐츠 상세 보기 창 (ENHANCED: shows embedding vector, navigation)"""
        try:
            if current_index is None:
                # fallback for direct calls
                current_index = item_data.get('index', 0)
            detail_window = tk.Toplevel(self.root)
            detail_window.title(f"벡터 데이터 상세 보기 - {item_data['id'][:30]}...")
            detail_window.geometry("900x700")
            detail_window.transient(self.root)
            menubar = tk.Menu(detail_window)
            detail_window.config(menu=menubar)
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="파일", menu=file_menu)
            file_menu.add_command(label="내용 저장", command=lambda: self.save_vector_content(item_data))
            file_menu.add_command(label="클립보드 복사", command=lambda: self.copy_vector_content_to_clipboard(item_data))
            file_menu.add_command(label="메타데이터 내보내기", command=lambda: self.export_metadata(item_data))
            info_frame = ttk.LabelFrame(detail_window, text="기본 정보", padding=10)
            info_frame.pack(fill=tk.X, padx=10, pady=5)
            info_grid = ttk.Frame(info_frame)
            info_grid.pack(fill=tk.X)
            ttk.Label(info_grid, text="ID:", font=("", 9, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5)
            ttk.Label(info_grid, text=str(item_data['id'])).grid(row=0, column=1, sticky=tk.W, padx=5)
            metadata = item_data['metadata']
            ttk.Label(info_grid, text="관련도:", font=("", 9, "bold")).grid(row=0, column=2, sticky=tk.W, padx=15)
            relevance = metadata.get('relevance_score', 'N/A')
            ttk.Label(info_grid, text=f"{relevance}/10" if isinstance(relevance, (int, float)) else str(relevance)).grid(row=0, column=3, sticky=tk.W, padx=5)
            ttk.Label(info_grid, text="날짜:", font=("", 9, "bold")).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            date_str = metadata.get('date', metadata.get('created_at', 'N/A'))
            ttk.Label(info_grid, text=str(date_str)).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
            ttk.Label(info_grid, text="청크 타입:", font=("", 9, "bold")).grid(row=1, column=2, sticky=tk.W, padx=15, pady=2)
            chunk_type = metadata.get('chunk_type', 'N/A')
            ttk.Label(info_grid, text=str(chunk_type)).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
            ttk.Label(info_grid, text="토픽:", font=("", 9, "bold")).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            try:
                topics = json.loads(metadata.get('topics', '[]')) if isinstance(metadata.get('topics'), str) else metadata.get('topics', [])
                topics_text = ', '.join(topics) if topics else 'N/A'
            except:
                topics_text = str(metadata.get('topics', 'N/A'))
            ttk.Label(info_grid, text=topics_text).grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)
            ttk.Label(info_grid, text="키워드:", font=("", 9, "bold")).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
            try:
                keywords = json.loads(metadata.get('keywords', '[]')) if isinstance(metadata.get('keywords'), str) else metadata.get('keywords', [])
                keywords_text = ', '.join(keywords) if keywords else 'N/A'
            except:
                keywords_text = str(metadata.get('keywords', 'N/A'))
            ttk.Label(info_grid, text=keywords_text).grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)
            # Embedding vector display
            emb_frame = ttk.LabelFrame(detail_window, text="임베딩 벡터 (Embedding Vector)", padding=10)
            emb_frame.pack(fill=tk.X, padx=10, pady=5)
            emb = item_data.get('embedding', [])
            emb_str = ', '.join([f"{x:.3f}" for x in emb[:10]]) + (" ..." if len(emb) > 10 else "")
            emb_label = ttk.Label(emb_frame, text=emb_str, wraplength=800, foreground="blue")
            emb_label.pack(anchor=tk.W)
            def toggle_embedding():
                if emb_label.cget('text').endswith('...'):
                    emb_label.config(text=', '.join([f"{x:.3f}" for x in emb]))
                else:
                    emb_label.config(text=emb_str)
            if len(emb) > 10:
                ttk.Button(emb_frame, text="전체 보기/접기", command=toggle_embedding).pack(anchor=tk.W, pady=2)
            content_frame = ttk.LabelFrame(detail_window, text="전체 내용", padding=10)
            content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            content_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, font=("맑은 고딕", 10))
            content_text.pack(fill=tk.BOTH, expand=True)
            full_content = f"""=== 벡터 데이터베이스 저장 내용 ===\n\n{item_data['document']}\n\n=== 메타데이터 정보 ===\n"""
            for key, value in metadata.items():
                if key in ['topics', 'keywords', 'company_mentions']:
                    try:
                        if isinstance(value, str):
                            parsed_value = json.loads(value)
                            full_content += f"{key}: {', '.join(parsed_value) if parsed_value else 'N/A'}\n"
                        else:
                            full_content += f"{key}: {', '.join(value) if value else 'N/A'}\n"
                    except:
                        full_content += f"{key}: {str(value)}\n"
                else:
                    full_content += f"{key}: {str(value)}\n"
            content_text.insert(1.0, full_content)
            content_text.config(state=tk.DISABLED)
            button_frame = ttk.Frame(detail_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            ttk.Button(button_frame, text="내용 저장", command=lambda: self.save_vector_content(item_data)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="클립보드 복사", command=lambda: self.copy_vector_content_to_clipboard(item_data)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="닫기", command=detail_window.destroy).pack(side=tk.RIGHT, padx=5)
            # --- Navigation controls ---
            nav_frame = ttk.Frame(detail_window)
            nav_frame.pack(fill=tk.X, padx=10, pady=5)
            def go_to_index(new_index):
                if 0 <= new_index < len(self.vector_full_data):
                    detail_window.destroy()
                    self.show_vector_content_detail(self.vector_full_data[new_index], new_index)
            prev_btn = ttk.Button(nav_frame, text="⬅️ 이전", command=lambda: go_to_index(current_index-1))
            next_btn = ttk.Button(nav_frame, text="다음 ➡️", command=lambda: go_to_index(current_index+1))
            prev_btn.pack(side=tk.LEFT, padx=5)
            next_btn.pack(side=tk.LEFT, padx=5)
            ttk.Label(nav_frame, text=f"{current_index+1} / {len(self.vector_full_data)}").pack(side=tk.LEFT, padx=10)
            ttk.Label(nav_frame, text="이동:").pack(side=tk.LEFT)
            jump_var = tk.StringVar()
            jump_entry = ttk.Entry(nav_frame, textvariable=jump_var, width=5)
            jump_entry.pack(side=tk.LEFT)
            def jump_to():
                try:
                    idx = int(jump_var.get()) - 1
                    go_to_index(idx)
                except:
                    pass
            ttk.Button(nav_frame, text="Go", command=jump_to).pack(side=tk.LEFT, padx=2)
        except Exception as e:
            messagebox.showerror("오류", f"상세 내용 표시 실패: {e}")
            logging.error(f"벡터 내용 상세 표시 실패: {e}")

    def save_vector_content(self, item_data):
        """벡터 내용 저장 (NEW FUNCTION)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_id = "".join(c for c in str(item_data['id']) if c.isalnum() or c in (' ', '-', '_'))[:30]
            default_filename = f"vector_content_{safe_id}_{timestamp}.txt"
            
            file_path = filedialog.asksaveasfilename(
                title="벡터 내용 저장",
                initialvalue=default_filename,
                defaultextension=".txt",
                filetypes=[("텍스트 파일", "*.txt"), ("JSON 파일", "*.json"), ("모든 파일", "*.*")]
            )
            
            if file_path:
                content = f"""벡터 데이터베이스 내용
===================
ID: {item_data['id']}
인덱스: {item_data['index']}
저장 시간: {datetime.now().isoformat()}

=== 문서 내용 ===
{item_data['document']}

=== 메타데이터 ===
{json.dumps(item_data['metadata'], indent=2, ensure_ascii=False)}
"""
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                messagebox.showinfo("완료", f"벡터 내용이 저장되었습니다.\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("오류", f"저장 실패: {e}")

    def copy_vector_content_to_clipboard(self, item_data):
        """벡터 내용 클립보드 복사 (NEW FUNCTION)"""
        try:
            content = item_data['document']
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            messagebox.showinfo("완료", "벡터 내용이 클립보드에 복사되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"클립보드 복사 실패: {e}")

    def export_metadata(self, item_data):
        """메타데이터 내보내기 (NEW FUNCTION)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_id = "".join(c for c in str(item_data['id']) if c.isalnum() or c in (' ', '-', '_'))[:30]
            default_filename = f"vector_metadata_{safe_id}_{timestamp}.json"
            
            file_path = filedialog.asksaveasfilename(
                title="메타데이터 내보내기",
                initialvalue=default_filename,
                defaultextension=".json",
                filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")]
            )
            
            if file_path:
                export_data = {
                    "id": item_data['id'],
                    "index": item_data['index'],
                    "document_preview": item_data['document'][:200] + "..." if len(item_data['document']) > 200 else item_data['document'],
                    "document_length": len(item_data['document']),
                    "metadata": item_data['metadata'],
                    "export_timestamp": datetime.now().isoformat()
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("완료", f"메타데이터가 내보내졌습니다.\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("오류", f"메타데이터 내보내기 실패: {e}")
            
    def refresh_vector_stats(self):
        """벡터 데이터베이스 통계 새로고침 (ALSO FIXED)"""
        try:
            if not self.system:
                return
            
            # 기본 통계 업데이트
            try:
                collection_count = self.system.db_manager.collection.count()
                self.vector_total_chunks_var.set(str(collection_count))
            except Exception as e:
                logging.error(f"컬렉션 카운트 조회 실패: {e}")
                self.vector_total_chunks_var.set("오류")
            
            self.vector_collection_name_var.set(self.system.db_manager.collection.name)
            self.vector_last_update_var.set(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # 상세 통계 조회 (안전하게)
            try:
                # ✅ FIXED: Proper get() method usage
                all_data = self.system.db_manager.collection.get(include=['metadatas'])
                
                if all_data.get('metadatas'):
                    relevance_scores = []
                    for metadata in all_data['metadatas']:
                        relevance = metadata.get('relevance_score', 0)
                        if isinstance(relevance, (int, float)) and relevance > 0:
                            relevance_scores.append(relevance)
                    
                    if relevance_scores:
                        avg_relevance = sum(relevance_scores) / len(relevance_scores)
                        self.vector_avg_relevance_var.set(f"{avg_relevance:.1f}/10")
                    else:
                        self.vector_avg_relevance_var.set("N/A")
                else:
                    self.vector_avg_relevance_var.set("N/A")
                    
            except Exception as e:
                logging.error(f"상세 통계 조회 실패: {e}")
                self.vector_avg_relevance_var.set("오류")
            
            logging.info("벡터 데이터베이스 통계 새로고침 완료")
            
        except Exception as e:
            logging.error(f"벡터 통계 새로고침 실패: {e}")
 
    def clear_vector_db(self):
        """벡터 데이터베이스 초기화 (NEW FUNCTION)"""
        if not self.system:
            messagebox.showwarning("경고", "시스템이 초기화되지 않았습니다(CV).")
            return
        
        if messagebox.askyesno("경고", "벡터 데이터베이스의 모든 데이터가 삭제됩니다. 계속하시겠습니까?"):
            try:
                # 컬렉션 삭제 후 재생성
                self.system.db_manager.client.delete_collection("enhanced_news_collection")
                self.system.db_manager.collection = self.system.db_manager.client.get_or_create_collection(
                    name="enhanced_news_collection",
                    metadata={"description": "Enhanced AI News Writer 뉴스 컬렉션"}
                )
                
                # UI 업데이트
                self.refresh_vector_stats()
                self.view_collection_contents()
                
                messagebox.showinfo("완료", "벡터 데이터베이스가 초기화되었습니다.")
                logging.info("벡터 데이터베이스 초기화 완료")
                
            except Exception as e:
                messagebox.showerror("오류", f"벡터DB 초기화 실패: {e}")
                logging.error(f"벡터DB 초기화 실패: {e}")

    def show_vector_status(self):
        """벡터DB 상태 표시 (NEW FUNCTION)"""
        self.notebook.select(3)  # 벡터DB 탭으로 이동 (0:설정, 1:뉴스수집, 2:뉴스작성, 3:벡터DB)
        self.refresh_vector_stats()

    def repair_vector_db(self):
        """벡터DB 복구 (NEW METHOD)"""
        try:
            if not self.system or not self.system.db_manager:
                messagebox.showwarning("경고", "시스템이 초기화되지 않았습니다.")
                return
            
            if messagebox.askyesno("확인", "벡터 데이터베이스를 복구하시겠습니까?\n\n이 작업은 손상된 컬렉션을 재생성합니다."):
                try:
                    db_manager = self.system.db_manager
                    
                    # Check if collection exists and is accessible
                    try:
                        collection_count = db_manager.collection.count()
                        logging.info(f"기존 컬렉션 확인: {collection_count}개 항목")
                    except Exception as e:
                        logging.warning(f"기존 컬렉션 접근 실패: {e}")
                        collection_count = 0
                    
                    # Backup existing data if possible
                    old_data = None
                    if collection_count > 0:
                        try:
                            old_data = db_manager.collection.get(include=['documents', 'metadatas'])
                            logging.info(f"기존 데이터 백업: {len(old_data.get('documents', []))}개 항목")
                        except Exception as e:
                            logging.warning(f"데이터 백업 실패: {e}")
                    
                    # Delete and recreate collection
                    try:
                        db_manager.client.delete_collection("enhanced_news_collection")
                        logging.info("기존 컬렉션 삭제 완료")
                    except Exception as e:
                        logging.warning(f"컬렉션 삭제 실패: {e}")
                    
                    # Create new collection
                    db_manager.collection = db_manager.client.create_collection(
                        name="enhanced_news_collection",
                        metadata={"description": "Enhanced AI News Writer 뉴스 컬렉션 (복구됨)"},
                        embedding_function=db_manager.embedding_function
                    )
                    logging.info("새 컬렉션 생성 완료")
                    
                    # Restore data if available
                    if old_data and old_data.get('documents'):
                        restored_count = 0
                        for i, doc in enumerate(old_data['documents']):
                            try:
                                metadata = old_data['metadatas'][i] if i < len(old_data['metadatas']) else {}
                                # Create new embedding for the document
                                embedding = db_manager.embedding_function([doc])[0] if db_manager.embedding_function else []
                                db_manager.collection.add(
                                    documents=[doc],
                                    metadatas=[metadata],
                                    embeddings=[embedding]
                                )
                                restored_count += 1
                            except Exception as e:
                                logging.warning(f"항목 {i} 복원 실패: {e}")
                        
                        logging.info(f"데이터 복원 완료: {restored_count}개 항목")
                        messagebox.showinfo("완료", f"벡터 데이터베이스 복구 완료\n\n복원된 항목: {restored_count}개")
                    else:
                        messagebox.showinfo("완료", "벡터 데이터베이스 복구 완료\n\n새 컬렉션이 생성되었습니다.")
                    
                    # Refresh the display
                    self.view_collection_contents()
                    
                except Exception as e:
                    messagebox.showerror("오류", f"벡터DB 복구 실패: {e}")
                    logging.error(f"벡터DB 복구 실패: {e}")
                    import traceback
                    logging.error(f"전체 오류 정보: {traceback.format_exc()}")
        except Exception as e:
            messagebox.showerror("오류", f"벡터DB 복구 중 오류 발생: {e}")
            logging.error(f"벡터DB 복구 중 오류 발생: {e}")

    def load_saved_headlines(self):
        """저장된 뉴스 헤드라인 로드"""
        try:
            # 기존 트리뷰 항목 초기화
            for item in self.headlines_tree.get_children():
                self.headlines_tree.delete(item)
            
            # collected_news 목록 초기화
            self.collected_news.clear()
            
            # 뉴스 디렉토리에서 파일 로드
            news_files = []
            if os.path.exists(self.news_directory):
                for filename in os.listdir(self.news_directory):
                    if filename.endswith('.txt') and filename.startswith('news_'):
                        filepath = os.path.join(self.news_directory, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # 뉴스 정보 파싱
                            news_info = self.parse_saved_news_file(content, filename)
                            if news_info:
                                self.collected_news.append(news_info)
                                
                        except Exception as e:
                            logging.warning(f"뉴스 파일 로드 실패 {filename}: {e}")
            
            # 수집된 뉴스를 최신순으로 정렬
            self.collected_news.sort(key=lambda x: x.get('pub_date', ''), reverse=True)
            
            # 트리뷰에 표시
            for i, news in enumerate(self.collected_news):
                title = news.get('title', 'No Title')
                pub_date = news.get('pub_date', 'N/A')
                description = news.get('description', '')[:100] + '...' if news.get('description', '') else 'No Description'
                
                self.headlines_tree.insert('', 'end',
                    text=str(i+1),
                    values=(title, description, pub_date)
                )
            
            # 통계 업데이트
            self.total_articles_var.set(str(len(self.collected_news)))
            self.saved_articles_var.set(str(len(self.collected_news)))
            
            logging.info(f"저장된 뉴스 {len(self.collected_news)}개를 로드했습니다.")
            
        except Exception as e:
            logging.error(f"저장된 뉴스 로드 실패: {e}")
            messagebox.showerror("오류", f"저장된 뉴스 로드 실패: {e}")

    def parse_saved_news_file(self, content: str, filename: str) -> dict:
        """저장된 뉴스 파일 파싱 (NEW FUNCTION)"""
        try:
            lines = content.split('\n')
            
            # 기본 정보 추출
            news_info = {
                'title': '',
                'link': '',
                'description': '',
                'pub_date': '',
                'content': '',
                'filename': filename
            }
            
            # 라인별 파싱
            current_section = None
            content_lines = []
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('제목:'):
                    news_info['title'] = line.replace('제목:', '').strip()
                elif line.startswith('링크:'):
                    news_info['link'] = line.replace('링크:', '').strip()
                elif line.startswith('발행일:'):
                    news_info['pub_date'] = line.replace('발행일:', '').strip()
                elif line.startswith('설명:'):
                    current_section = 'description'
                    continue
                elif line.startswith('본문:'):
                    current_section = 'content'
                    continue
                elif line.startswith('수집 정보:'):
                    break  # 수집 정보 이후는 무시
                elif current_section == 'description' and line:
                    news_info['description'] += line + ' '
                elif current_section == 'content' and line:
                    content_lines.append(line)
            
            news_info['content'] = '\n'.join(content_lines)
            news_info['description'] = news_info['description'].strip()
            
            # 필수 정보 확인
            if not news_info['title']:
                news_info['title'] = f"제목 없음 - {filename}"
            
            return news_info
            
        except Exception as e:
            logging.error(f"뉴스 파일 파싱 실패 {filename}: {e}")
            return None

    async def add_news_to_vector_db(self, news_info: dict) -> bool:
        """뉴스를 벡터 데이터베이스에 추가 (NEW FUNCTION)"""
        try:
            if not self.system:
                return False
            
            company_name = self.company_var.get()
            
            # NewsArticle 객체 생성
            from main import NewsArticle
            article = NewsArticle(
                title=news_info.get('title', ''),
                link=news_info.get('link', ''),
                description=news_info.get('description', ''),
                pub_date=news_info.get('pub_date', ''),
                content=news_info.get('content', '')
            )
            
            # 간단한 수집 및 저장 (기존 함수 사용)
            success = await self.system.news_collector.collect_and_store_news(company_name, article)
            
            return success
            
        except Exception as e:
            logging.error(f"벡터DB 추가 실패: {e}")
            return False

    # 초기화 시 벡터 통계 자동 로드
    def auto_initialize_system(self):
        """시작 시 자동 시스템 초기화 (벡터 통계 포함)"""
        try:
            # 설정값 가져오기
            claude_key = self.claude_key_var.get().strip()
            naver_id = self.naver_id_var.get().strip()
            naver_secret = self.naver_secret_var.get().strip()
            
            # 시스템 초기화
            self.system = AINewsWriterSystem(
                claude_api_key=claude_key if claude_key else None,
                naver_client_id=naver_id if naver_id else None,
                naver_client_secret=naver_secret if naver_secret else None
            )
            
            # 상태 업데이트 (경고 다이얼로그 없이)
            status_parts = []
            if claude_key:
                status_parts.append("Claude API ✅")
            else:
                status_parts.append("Claude API ❌ (테스트 모드)")
            
            if naver_id and naver_secret:
                status_parts.append("네이버 API ✅")
            else:
                status_parts.append("네이버 API ❌ (테스트 모드)")
            
            # 상태바 업데이트
            self.status_var.set("시스템 자동 초기화 완료 - " + " | ".join(status_parts))
            self.status_label_widget.config(foreground="green")
            logging.info("시스템이 자동으로 초기화되었습니다.")
            
            # 3초 후 벡터 통계 업데이트
            self.root.after(3000, self.refresh_vector_stats)
              
            # 네이버 API 자동 테스트 (5초 후)
            if naver_id and naver_secret:
                self.root.after(5000, self.test_naver_api)
            
        except Exception as e:
            logging.error(f"시스템 자동 초기화 실패: {e}")
            self.status_var.set("API 키를 설정하고 시스템을 초기화해주세요.")
            self.status_label_widget.config(foreground="red")
            self.status_var.set("자동 초기화 실패 - 수동으로 초기화해주세요.")
            self.status_label_widget.config(foreground="red")


    def setup_config_tab(self, parent):
        """설정 탭 (분할: 기본/고급)"""
        config_frame = ttk.Frame(parent)
        parent.add(config_frame, text="🔧 설정")
        config_notebook = ttk.Notebook(config_frame)
        config_notebook.pack(fill=tk.BOTH, expand=True)
        basic_frame = ttk.Frame(config_notebook)
        config_notebook.add(basic_frame, text="기본 설정")
        advanced_frame = ttk.Frame(config_notebook)
        config_notebook.add(advanced_frame, text="고급 설정")
        # --- BASIC SETTINGS ---
        basic_canvas = tk.Canvas(basic_frame)
        basic_scrollbar = ttk.Scrollbar(basic_frame, orient="vertical", command=basic_canvas.yview)
        basic_scrollable = ttk.Frame(basic_canvas)
        basic_scrollable.bind(
            "<Configure>",
            lambda e: basic_canvas.configure(scrollregion=basic_canvas.bbox("all"))
        )
        basic_canvas.create_window((0, 0), window=basic_scrollable, anchor="nw")
        basic_canvas.configure(yscrollcommand=basic_scrollbar.set)
        basic_canvas.pack(side="left", fill="both", expand=True)
        basic_scrollbar.pack(side="right", fill="y")
        # 회사 및 키워드 설정 프레임
        company_frame = ttk.LabelFrame(basic_scrollable, text="대상 회사 및 키워드 설정", padding=10)
        company_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(company_frame, text="회사명:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.company_var = tk.StringVar(value="알티베이스")
        ttk.Entry(company_frame, textvariable=self.company_var, width=40).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(company_frame, text="추가 키워드 (,구분):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.additional_keywords_var = tk.StringVar(value="데이터베이스, DBMS, 오라클")
        keyword_entry = ttk.Entry(company_frame, textvariable=self.additional_keywords_var, width=60)
        keyword_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        help_label = ttk.Label(company_frame, text="※ 회사명과 추가 키워드를 조합하여 더 정확한 뉴스를 검색합니다", foreground="gray", font=("", 8))
        help_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        # 로그 레벨 설정 프레임
        loglevel_frame = ttk.LabelFrame(basic_scrollable, text="로그 레벨", padding=10)
        loglevel_frame.pack(fill=tk.X, padx=10, pady=5)
        self.log_level_var = tk.StringVar(value="INFO")
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        loglevel_combo = ttk.Combobox(loglevel_frame, textvariable=self.log_level_var, values=log_levels, state="readonly", width=10)
        loglevel_combo.pack(side=tk.LEFT, padx=5)
        loglevel_combo.bind("<<ComboboxSelected>>", self.on_log_level_change)
        ttk.Label(loglevel_frame, text="(실시간 변경 가능)").pack(side=tk.LEFT, padx=5)
        # --- ADVANCED SETTINGS ---
        adv_canvas = tk.Canvas(advanced_frame)
        adv_scrollbar = ttk.Scrollbar(advanced_frame, orient="vertical", command=adv_canvas.yview)
        adv_scrollable = ttk.Frame(adv_canvas)
        adv_scrollable.bind(
            "<Configure>",
            lambda e: adv_canvas.configure(scrollregion=adv_canvas.bbox("all"))
        )
        adv_canvas.create_window((0, 0), window=adv_scrollable, anchor="nw")
        adv_canvas.configure(yscrollcommand=adv_scrollbar.set)
        adv_canvas.pack(side="left", fill="both", expand=True)
        adv_scrollbar.pack(side="right", fill="y")

        # API 키 설정 프레임 (moved)
        api_frame = ttk.LabelFrame(adv_scrollable, text="API 키 설정", padding=10)
        api_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(api_frame, text="Claude API Key:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.claude_key_var = tk.StringVar()
        claude_entry = ttk.Entry(api_frame, textvariable=self.claude_key_var, show="*", width=60)
        claude_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(api_frame, text="네이버 Client ID:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.naver_id_var = tk.StringVar()
        naver_id_entry = ttk.Entry(api_frame, textvariable=self.naver_id_var, width=60)
        naver_id_entry.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(api_frame, text="네이버 Client Secret:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.naver_secret_var = tk.StringVar()
        naver_secret_entry = ttk.Entry(api_frame, textvariable=self.naver_secret_var, show="*", width=60)
        naver_secret_entry.grid(row=2, column=1, padx=5, pady=2)
        btn_frame = ttk.Frame(api_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="설정 저장", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="설정 불러오기", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="시스템 초기화", command=self.initialize_system).pack(side=tk.LEFT, padx=5)
        
        # 상태 표시 프레임
        status_frame = ttk.LabelFrame(adv_scrollable, text="시스템 상태", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        self.status_var = tk.StringVar(value="시스템이 초기화되지 않았습니다(S).")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground="green")
        status_label.pack()
        self.status_label_widget = status_label
        # 뉴스 저장 위치 (moved)
        newsdir_frame = ttk.LabelFrame(adv_scrollable, text="뉴스 저장 폴더", padding=10)
        newsdir_frame.pack(fill=tk.X, padx=10, pady=5)
        self.news_dir_var = tk.StringVar(value=self.news_directory)
        ttk.Entry(newsdir_frame, textvariable=self.news_dir_var, width=40).grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Button(newsdir_frame, text="폴더 선택", command=self.select_news_directory).grid(row=0, column=1, padx=5, pady=2)
        # 고급 설정 프레임 (RAG 등)
        advanced_frame_inner = ttk.LabelFrame(adv_scrollable, text="고급 설정", padding=10)
        advanced_frame_inner.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(advanced_frame_inner, text="RAG 참조 뉴스 개수:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.rag_news_count_var = tk.IntVar(value=5)
        ttk.Spinbox(advanced_frame_inner, from_=5, to=20, textvariable=self.rag_news_count_var, width=10).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(advanced_frame_inner, text="개").grid(row=0, column=2, sticky=tk.W)

    def on_log_level_change(self, event=None):
        """로그 레벨 변경 핸들러"""
        level = self.log_level_var.get()
        import logging
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level, logging.INFO))
        logging.info(f"로그 레벨이 {level}로 변경되었습니다.")
    
    def setup_collection_tab(self, parent):
        """뉴스 수집 탭"""
        collection_frame = ttk.Frame(parent)
        parent.add(collection_frame, text="📰 뉴스 수집")
        
        collection_frame.columnconfigure(0, weight=1, uniform="col")
        collection_frame.columnconfigure(1, weight=1, uniform="col")
        collection_frame.rowconfigure(0, weight=1)
        
        left_frame = ttk.Frame(collection_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        right_frame = ttk.Frame(collection_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_frame.grid_propagate(True)
        
        # 좌측: 수집 설정 및 제어
        # API 상태 체크 프레임
        api_check_frame = ttk.LabelFrame(left_frame, text="API 상태 확인", padding=10)
        api_check_frame.pack(fill=tk.X, pady=5)
        
        status_check_frame = ttk.Frame(api_check_frame)
        status_check_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(status_check_frame, text="네이버 API 테스트", command=self.test_naver_api).pack(side=tk.LEFT, padx=5)
        self.api_status_var = tk.StringVar(value="API 상태를 확인해주세요")
        ttk.Label(status_check_frame, textvariable=self.api_status_var).pack(side=tk.LEFT, padx=10)
        
        # 자동 수집 프레임 (개선됨)
        auto_frame = ttk.LabelFrame(left_frame, text="자동 뉴스 수집", padding=10)
        auto_frame.pack(fill=tk.X, pady=5)
        
        # 수집 설정
        settings_frame = ttk.Frame(auto_frame)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="수집 기간:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.days_var = tk.IntVar(value=365)  # 기본값을 365일로 변경
        days_spinbox = ttk.Spinbox(settings_frame, from_=1, to=730, textvariable=self.days_var, width=10)
        days_spinbox.grid(row=0, column=1, padx=5)
        ttk.Label(settings_frame, text="일 (최대 2년)").grid(row=0, column=2, sticky=tk.W)
        
        # 수집할 기사 수
        ttk.Label(settings_frame, text="수집할 기사 수:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_articles_var = tk.IntVar(value=50)
        ttk.Spinbox(settings_frame, from_=10, to=200, textvariable=self.max_articles_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(settings_frame, text="개").grid(row=1, column=2, sticky=tk.W, pady=2)
        
        # 버튼
        button_frame = ttk.Frame(auto_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.collect_btn = ttk.Button(button_frame, text="뉴스 수집 시작", command=self.start_collection)
        self.collect_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="수집 중지", command=self.stop_collection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 진행 상황
        self.progress = ttk.Progressbar(auto_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # 통계 프레임
        stats_frame = ttk.LabelFrame(left_frame, text="수집 통계", padding=10)
        stats_frame.pack(fill=tk.X, pady=5)
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, pady=5)
        
        # 한 줄에 모두 표시
        ttk.Label(stats_grid, text="총 수집:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.total_articles_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.total_articles_var, foreground="blue").grid(row=0, column=1, sticky=tk.W, padx=2)
        
        ttk.Label(stats_grid, text="| 관련도 높음:").grid(row=0, column=2, sticky=tk.W, padx=2)
        self.relevant_articles_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.relevant_articles_var, foreground="green").grid(row=0, column=3, sticky=tk.W, padx=2)
        
        ttk.Label(stats_grid, text="| 로컬 저장:").grid(row=0, column=4, sticky=tk.W, padx=2)
        self.saved_articles_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.saved_articles_var, foreground="purple").grid(row=0, column=5, sticky=tk.W, padx=2)
        
        ttk.Label(stats_grid, text="| DB 저장:").grid(row=0, column=6, sticky=tk.W, padx=2)
        self.db_saved_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.db_saved_var, foreground="red").grid(row=0, column=7, sticky=tk.W, padx=2)
        
        # 수동 입력 프레임
        manual_frame = ttk.LabelFrame(left_frame, text="수동 뉴스 입력", padding=10)
        manual_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(manual_frame, text="뉴스 내용:").pack(anchor=tk.W)
        self.manual_text = scrolledtext.ScrolledText(manual_frame, height=4, wrap=tk.WORD)
        self.manual_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        manual_btn_frame = ttk.Frame(manual_frame)
        manual_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(manual_btn_frame, text="수동 뉴스 추가", command=self.add_manual_news).pack(side=tk.LEFT, padx=5)
        ttk.Button(manual_btn_frame, text="파일에서 불러오기", command=self.load_news_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(manual_btn_frame, text="내용 지우기", command=lambda: self.manual_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        
        # 우측: 수집된 뉴스 헤드라인 (개선됨)
        headlines_frame = ttk.LabelFrame(right_frame, text="수집된 뉴스 헤드라인", padding=10)
        headlines_frame.pack(fill=tk.BOTH, expand=True)
        
        # 헤드라인 리스트 (개선된 표시)
        headlines_scroll_frame = ttk.Frame(headlines_frame)
        headlines_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        # 트리뷰 사용하여 더 많은 정보 표시
        columns = ('title', 'preview', 'date')  # Changed order to match display
        self.headlines_tree = ttk.Treeview(headlines_scroll_frame, columns=columns, show='tree headings', height=5)
        
        self.headlines_tree.heading('#0', text='번호')
        self.headlines_tree.heading('title', text='제목')
        self.headlines_tree.heading('preview', text='미리보기')
        self.headlines_tree.heading('date', text='날짜')
        
        self.headlines_tree.column('#0', width=50, minwidth=50)
        self.headlines_tree.column('title', width=300, minwidth=200)
        self.headlines_tree.column('preview', width=300, minwidth=150)
        self.headlines_tree.column('date', width=100, minwidth=80)
        
        scrollbar_headlines = ttk.Scrollbar(headlines_scroll_frame, orient=tk.VERTICAL, command=self.headlines_tree.yview)
        self.headlines_tree.configure(yscrollcommand=scrollbar_headlines.set)
        
        self.headlines_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_headlines.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 헤드라인 버튼 (ENHANCED with reload functionality)
        headlines_btn_frame = ttk.Frame(headlines_frame)
        headlines_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(headlines_btn_frame, text="새로고침", command=self.refresh_headlines).pack(side=tk.LEFT, padx=5)
        ttk.Button(headlines_btn_frame, text="선택 기사 보기", command=self.view_selected_article).pack(side=tk.LEFT, padx=5)
        ttk.Button(headlines_btn_frame, text="헤드라인 지우기", command=self.clear_headlines).pack(side=tk.LEFT, padx=5)
        ttk.Button(headlines_btn_frame, text="저장된 파일 열기", command=self.open_news_directory).pack(side=tk.LEFT, padx=5)
        
        # NEW: Reload previous news functionality
        reload_btn_frame = ttk.Frame(headlines_frame)
        reload_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(reload_btn_frame, text="🔄 이전 뉴스 불러오기", command=self.reload_previous_news).pack(side=tk.LEFT, padx=5)
        ttk.Button(reload_btn_frame, text="✅ 선택 뉴스 벡터DB 추가", command=self.add_selected_to_vector).pack(side=tk.LEFT, padx=5)
        ttk.Button(reload_btn_frame, text="📊 벡터DB 상태", command=self.show_vector_status).pack(side=tk.LEFT, padx=5)
        # 수동 뉴스 입력 버튼 추가
        ttk.Button(reload_btn_frame, text="✍️ 수동 뉴스 입력", command=self.show_manual_news_popup).pack(side=tk.LEFT, padx=5)

    def setup_vector_stats_tab(self, parent):
        """벡터 데이터베이스 통계 탭 (NEW)"""
        vector_frame = ttk.Frame(parent)
        parent.add(vector_frame, text="📊 벡터DB")
        
        # --- Search Bar ---
        search_frame = ttk.Frame(vector_frame)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(search_frame, text="검색:").pack(side=tk.LEFT)
        self.vector_search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.vector_search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=5)
        def do_search(*args):
            query = self.vector_search_var.get().strip().lower()
            if not hasattr(self, 'vector_full_data') or not self.vector_full_data:
                return
            if not query:
                self._refresh_vector_tree(self.vector_full_data)
                return
            filtered = [item for item in self.vector_full_data if query in str(item.get('id', '')).lower() or query in str(item.get('document', '')).lower()]
            self._refresh_vector_tree(filtered)
        search_entry.bind('<Return>', do_search)
        ttk.Button(search_frame, text="검색", command=do_search).pack(side=tk.LEFT, padx=2)
        def clear_search():
            self.vector_search_var.set("")
            if hasattr(self, 'vector_full_data'):
                self._refresh_vector_tree(self.vector_full_data)
        ttk.Button(search_frame, text="초기화", command=clear_search).pack(side=tk.LEFT, padx=2)
        
        # Initialize StringVars for vector stats
        self.vector_total_chunks_var = tk.StringVar(value="0")
        self.vector_collection_name_var = tk.StringVar(value="unknown")
        self.vector_last_update_var = tk.StringVar(value="N/A")
        self.vector_avg_relevance_var = tk.StringVar(value="N/A")
        
        # 상단 통계 요약
        stats_summary_frame = ttk.LabelFrame(vector_frame, text="벡터 데이터베이스 요약", padding=10)
        stats_summary_frame.pack(fill=tk.X, padx=10, pady=5)
        stats_grid = ttk.Frame(stats_summary_frame)
        stats_grid.pack(fill=tk.X, pady=5)
        
        # Row 0: Total chunks and Collection name
        ttk.Label(stats_grid, text="총 청크 수:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Label(stats_grid, textvariable=self.vector_total_chunks_var, foreground="blue").grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(stats_grid, text="컬렉션명:").grid(row=0, column=2, sticky=tk.W, padx=15)
        ttk.Label(stats_grid, textvariable=self.vector_collection_name_var, foreground="green").grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Row 1: Last update and Average relevance
        ttk.Label(stats_grid, text="마지막 업데이트:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(stats_grid, textvariable=self.vector_last_update_var, foreground="gray").grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(stats_grid, text="평균 관련도:").grid(row=1, column=2, sticky=tk.W, padx=15, pady=2)
        ttk.Label(stats_grid, textvariable=self.vector_avg_relevance_var, foreground="purple").grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # --- Vector Tree View ---
        tree_frame = ttk.LabelFrame(vector_frame, text="벡터 데이터베이스 내용", padding=10)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create Treeview with scrollbars
        tree_container = ttk.Frame(tree_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)
        
        # Create the Treeview
        columns = ('ID', '내용 미리보기', '토픽', '관련도', '날짜')
        self.vector_tree = ttk.Treeview(tree_container, columns=columns, show='headings', height=15)
        
        # Configure column headings
        self.vector_tree.heading('ID', text='ID')
        self.vector_tree.heading('내용 미리보기', text='내용 미리보기')
        self.vector_tree.heading('토픽', text='토픽')
        self.vector_tree.heading('관련도', text='관련도')
        self.vector_tree.heading('날짜', text='날짜')
        
        # Configure column widths - Enhanced for better content display
        self.vector_tree.column('ID', width=120, minwidth=80)
        self.vector_tree.column('내용 미리보기', width=400, minwidth=250)  # Increased for better content preview
        self.vector_tree.column('토픽', width=180, minwidth=120)  # Increased for topic display
        self.vector_tree.column('관련도', width=80, minwidth=60)
        self.vector_tree.column('날짜', width=100, minwidth=80)
        
        # Create scrollbars
        v_scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.vector_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=self.vector_tree.xview)
        self.vector_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack the tree and scrollbars
        self.vector_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 벡터DB 탭 선택 시 컬렉션내용보기만 표시, 팝업 없이
        def on_tab_selected(event=None):
            self.view_collection_contents()
        parent.bind("<Visibility>", on_tab_selected)

        # --- Admin Buttons ---
        admin_btn_frame = ttk.Frame(vector_frame)
        admin_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(admin_btn_frame, text="🗑️ 벡터DB 초기화", command=self.clear_vector_db).pack(side=tk.LEFT, padx=5)
        ttk.Button(admin_btn_frame, text="🔧 벡터DB 복구", command=self.repair_vector_db).pack(side=tk.LEFT, padx=5)
        ttk.Button(admin_btn_frame, text="📊 통계 보기", command=self.show_vector_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(admin_btn_frame, text="ℹ️ DB 정보", command=self.show_vector_db_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(admin_btn_frame, text="🏥 상태 확인", command=lambda: messagebox.showinfo("ChromaDB 상태", self.check_chromadb_health())).pack(side=tk.LEFT, padx=5)

    def setup_writing_tab(self, parent):
        """뉴스 작성 탭 (개선됨)"""
        writing_frame = ttk.Frame(parent)
        parent.add(writing_frame, text="✍️ 뉴스 작성")
        left_writing_frame = ttk.Frame(writing_frame)
        left_writing_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        right_writing_frame = ttk.Frame(writing_frame)
        right_writing_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        input_frame = ttk.LabelFrame(left_writing_frame, text="뉴스 작성 입력", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(input_frame, text="토픽:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.topic_var = tk.StringVar(value="알티베이스, 3년간 단계별 기술 로드맵 발표...2027년 차세대 클러스터 출시")
        topic_entry = ttk.Entry(input_frame, textvariable=self.topic_var, width=50)
        topic_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        topic_entry.configure(takefocus=False)  # 포커스 해제
        ttk.Label(input_frame, text="키워드:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.keywords_var = tk.StringVar(value="알티베이스, 차세대 기술 로드맵, 클러스터")
        ttk.Entry(input_frame, textvariable=self.keywords_var, width=50).grid(row=1, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        ttk.Label(input_frame, text="스타일:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.style_var = tk.StringVar(value="기업 보도형")
        style_combo = ttk.Combobox(input_frame, textvariable=self.style_var, 
                                  values=["기업 보도형", "분석형", "인터뷰형", "발표형", "기술 리뷰형"], width=20)
        style_combo.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(input_frame, text="뉴스 길이:").grid(row=3, column=0, sticky=tk.W, pady=2)
        length_frame = ttk.Frame(input_frame)
        length_frame.grid(row=3, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        self.length_type_var = tk.StringVar(value="줄 수")
        ttk.Radiobutton(length_frame, text="줄 수", variable=self.length_type_var, value="줄 수").pack(side=tk.LEFT)
        ttk.Radiobutton(length_frame, text="단어 수", variable=self.length_type_var, value="단어 수").pack(side=tk.LEFT, padx=10)
        self.length_count_var = tk.IntVar(value=50)
        ttk.Spinbox(length_frame, from_=10, to=500, textvariable=self.length_count_var, width=10).pack(side=tk.LEFT, padx=10)
        ttk.Label(input_frame, text="주요 사실:").grid(row=4, column=0, sticky=tk.NW, pady=2)
        self.facts_text = scrolledtext.ScrolledText(input_frame, height=12, width=60, foreground="#888888")  # gray tone
        self.facts_text.grid(row=4, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        self.facts_text.insert(1.0, "알티베이스는 국내외 시장의 급변하는 요구사항과 AI 기술 발전 추세를 반영해 4대 핵심 기술을 중심으로 한 중장기 기술 로드맵을 수립했다고 7일 밝혔다. 이번 로드맵은 △인메모리 고성능 기술 △멀티 데이터모델 △AI 에이전트·벡터 데이터베이스 기능 △차세대 클러스터 기술 등으로 구성되며, AI 시대에 맞는 데이터베이스 기술 혁신을 목표로 한다.")
        rag_frame = ttk.LabelFrame(input_frame, text="RAG 참조 설정", padding=5)
        rag_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=10, sticky=tk.EW)
        self.use_rag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rag_frame, text="RAG 참조 사용", variable=self.use_rag_var).pack(side=tk.LEFT)
        ttk.Label(rag_frame, text="참조 뉴스 개수:").pack(side=tk.LEFT, padx=10)
        self.rag_count_var = tk.IntVar(value=5)
        ttk.Spinbox(rag_frame, from_=5, to=20, textvariable=self.rag_count_var, width=8).pack(side=tk.LEFT)
        generate_frame = ttk.Frame(input_frame)
        generate_frame.grid(row=6, column=0, columnspan=3, pady=15)
        self.generate_btn = ttk.Button(generate_frame, text="🚀 뉴스 생성", command=self.generate_news)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(generate_frame, text="📋 템플릿 불러오기", command=self.load_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(generate_frame, text="💾 템플릿 저장", command=self.save_template).pack(side=tk.LEFT, padx=5)
        result_frame = ttk.LabelFrame(right_writing_frame, text="생성된 뉴스", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, font=("맑은 고딕", 10), height=5)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        result_btn_frame = ttk.Frame(result_frame)
        result_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(result_btn_frame, text="📁 파일로 저장", command=self.save_news).pack(side=tk.LEFT, padx=5)
        ttk.Button(result_btn_frame, text="📋 클립보드 복사", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(result_btn_frame, text="🔄 다시 생성", command=self.regenerate_news).pack(side=tk.LEFT, padx=5)
        ttk.Button(result_btn_frame, text="❌ 결과 지우기", command=lambda: self.result_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        quality_frame = ttk.LabelFrame(result_frame, text="품질 평가", padding=5)
        quality_frame.pack(fill=tk.X, pady=5)
        self.quality_var = tk.StringVar(value="뉴스를 생성하면 품질 평가가 표시됩니다.")
        ttk.Label(quality_frame, textvariable=self.quality_var, foreground="gray").pack()

    def setup_bottom_log_frame(self, parent):
        """하단 로그 프레임 (모든 탭에서 보이도록)"""
        # 구분선
        separator = ttk.Separator(parent, orient='horizontal')
        separator.pack(fill=tk.X, pady=5)
        
        # 로그 프레임
        log_frame = ttk.LabelFrame(parent, text="📋 시스템 로그", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 로그 텍스트 (높이를 8로 늘림)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        
        # 버튼 삭제: 로그 지우기, 로그 저장, 자동 스크롤

    def setup_logging(self):
        """로깅 설정"""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        gui_handler = GUILogHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(gui_handler)
        # 로그 레벨을 설정값에 따라 적용
        level = getattr(logging, getattr(self, 'log_level_var', None).get() if hasattr(self, 'log_level_var') else 'INFO', logging.INFO)
        root_logger.setLevel(level)
        
    def toggle_auto_scroll(self):
        """자동 스크롤 토글"""
        self.auto_scroll = not self.auto_scroll
        status = "켜짐" if self.auto_scroll else "꺼짐"
        logging.info(f"자동 스크롤: {status}")
        
    def select_news_directory(self):
        """뉴스 저장 디렉토리 선택"""
        directory = filedialog.askdirectory(title="뉴스 저장 폴더 선택")
        if directory:
            self.news_dir_var.set(directory)
            self.news_directory = directory
    
    def test_naver_api(self):
        """네이버 API 테스트 (개선된 검색 쿼리 포함)"""
        if not self.system:
            self.api_status_var.set("❌ 시스템이 초기화되지 않았습니다(N).")
            return
        
        def api_test_worker():
            try:
                # 회사명과 추가 키워드를 포함한 테스트 검색
                company = self.company_var.get()
                additional_keywords = self.additional_keywords_var.get()
                
                # 테스트 쿼리 생성
                test_query = f"{company}"
                if additional_keywords:
                    first_keyword = additional_keywords.split(',')[0].strip()
                    test_query = f"{company} {first_keyword}"
                
                logging.info(f"네이버 API 테스트 중: '{test_query}'")
                test_articles = self.system.naver_api.search_news(test_query, display=1)
                
                if test_articles:
                    if self.system.naver_api.test_mode:
                        status = "⚠️ 테스트 모드 (더미 데이터)"
                    else:
                        status = f"✅ 네이버 API 정상 작동 (쿼리: {test_query})"
                else:
                    status = "❌ API 응답 없음"
                    
                self.root.after(0, lambda: self.api_status_var.set(status))
                self.root.after(0, lambda: logging.info(f"네이버 API 테스트 결과: {status}"))
                
            except Exception as e:
                error_status = f"❌ API 오류: {str(e)[:50]}"
                self.root.after(0, lambda: self.api_status_var.set(error_status))
                self.root.after(0, lambda: logging.error(f"네이버 API 테스트 실패: {e}"))
        
        self.api_status_var.set("🔄 API 테스트 중...")
        threading.Thread(target=api_test_worker, daemon=True).start()
        
    def save_config(self):
        """설정 저장"""
        config = {
            "claude_api_key": self.claude_key_var.get(),
            "naver_client_id": self.naver_id_var.get(),
            "naver_client_secret": self.naver_secret_var.get(),
            "company_name": self.company_var.get(),
            "additional_keywords": self.additional_keywords_var.get(),
            "rag_news_count": self.rag_news_count_var.get(),
            "news_directory": self.news_dir_var.get()
        }
        
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("성공", "설정이 저장되었습니다.")
            logging.info("설정 저장 완료")
        except Exception as e:
            messagebox.showerror("오류", f"설정 저장 실패: {e}")
            logging.error(f"설정 저장 실패: {e}")
    
    def load_config(self):
        """설정 불러오기"""
        # .env 파일 우선 확인
        if os.path.exists(".env"):
            try:
                from dotenv import load_dotenv
                load_dotenv()
                self.claude_key_var.set(os.getenv('CLAUDE_API_KEY', ''))
                self.naver_id_var.set(os.getenv('NAVER_CLIENT_ID', ''))
                self.naver_secret_var.set(os.getenv('NAVER_CLIENT_SECRET', ''))
                logging.info(".env 파일에서 설정 로드")
            except ImportError:
                logging.warning("python-dotenv 패키지가 설치되지 않았습니다.")
        
        # config.json 파일 확인
        elif os.path.exists("config.json"):
            try:
                with open("config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.claude_key_var.set(config.get("claude_api_key", ""))
                self.naver_id_var.set(config.get("naver_client_id", ""))
                self.naver_secret_var.set(config.get("naver_client_secret", ""))
                self.company_var.set(config.get("company_name", "알티베이스"))
                self.additional_keywords_var.set(config.get("additional_keywords", "데이터베이스, DBMS, 오라클"))
                self.rag_news_count_var.set(config.get("rag_news_count", 10))
                self.news_dir_var.set(config.get("news_directory", self.news_directory))
                logging.info("config.json에서 설정 로드")
            except Exception as e:
                logging.warning(f"설정 불러오기 실패: {e}")
    
    def initialize_system(self):
        """시스템 초기화"""
        try:
            claude_key = self.claude_key_var.get().strip()
            naver_id = self.naver_id_var.get().strip()
            naver_secret = self.naver_secret_var.get().strip()
            
            self.system = AINewsWriterSystem(
                claude_api_key=claude_key if claude_key else None,
                naver_client_id=naver_id if naver_id else None,
                naver_client_secret=naver_secret if naver_secret else None
            )
            
            # 상태 업데이트
            status_parts = []
            if claude_key:
                status_parts.append("Claude API ✅")
            else:
                status_parts.append("Claude API ❌ (테스트 모드)")
            
            if naver_id and naver_secret:
                status_parts.append("네이버 API ✅")
            else:
                status_parts.append("네이버 API ❌ (테스트 모드)")
            
            self.status_var.set("시스템 초기화 완료 - " + " | ".join(status_parts))
            self.status_label_widget.config(foreground="green")
            
            messagebox.showinfo("성공", "시스템이 초기화되었습니다.")
            logging.info("AI News Writer Pro 시스템 초기화 완료")
            
        except Exception as e:
            self.status_var.set(f"시스템 초기화 실패: {e}")
            self.status_label_widget.config(foreground="red")
            messagebox.showerror("오류", f"시스템 초기화 실패: {e}")
            logging.error(f"시스템 초기화 실패: {e}")
    
    def get_enhanced_search_queries(self):
        """개선된 검색 쿼리 생성 (회사명 + 추가 키워드 조합)"""
        company = self.company_var.get().strip()
        additional_keywords = self.additional_keywords_var.get().strip()
        
        queries = [company]  # 기본 회사명 검색
        
        if additional_keywords:
            keywords = [k.strip() for k in additional_keywords.split(',')]
            # 회사명과 각 키워드 조합
            for keyword in keywords:
                queries.append(f"{company} {keyword}")
            
            # 회사명과 여러 키워드 조합
            if len(keywords) >= 2:
                queries.append(f"{company} {keywords[0]} {keywords[1]}")
        
        # 기본 추가 검색어
        queries.extend([
            f"{company} 신제품",
            f"{company} 발표",
            f"{company} 기술"
        ])
        
        return list(set(queries))  # 중복 제거
    
    def check_system_health(self):
        """시스템 상태 확인"""
        if not self.system:
            return False, "시스템이 초기화되지 않았습니다(S)."
        try:
            # 네이버 API 키 확인
            naver_id = self.naver_id_var.get().strip()
            naver_secret = self.naver_secret_var.get().strip()
            if not naver_id or not naver_secret:
                return False, "네이버 API 키가 설정되지 않았습니다."
            # 회사명 확인
            company = self.company_var.get().strip()
            if not company:
                return False, "회사명이 설정되지 않았습니다."
            return True, "시스템이 정상입니다."
        except Exception as e:
            return False, f"시스템 상태 확인 실패: {e}"

    def check_chromadb_health(self):
        """ChromaDB 상태 상세 확인 (NEW METHOD)"""
        try:
            if not self.system or not self.system.db_manager:
                return "시스템이 초기화되지 않았습니다."
            
            db_manager = self.system.db_manager
            health_report = []
            
            # Check client connection
            try:
                client = db_manager.client
                health_report.append("✅ ChromaDB 클라이언트 연결: 정상")
            except Exception as e:
                health_report.append(f"❌ ChromaDB 클라이언트 연결 실패: {e}")
                return "\n".join(health_report)
            
            # Check collection existence
            try:
                collection = db_manager.collection
                health_report.append(f"✅ 컬렉션 존재: {collection.name}")
            except Exception as e:
                health_report.append(f"❌ 컬렉션 접근 실패: {e}")
                return "\n".join(health_report)
            
            # Check collection count
            try:
                count = collection.count()
                health_report.append(f"✅ 컬렉션 항목 수: {count}개")
            except Exception as e:
                health_report.append(f"❌ 컬렉션 카운트 실패: {e}")
            
            # Check data retrieval
            try:
                if count > 0:
                    sample_data = collection.get(limit=1, include=['documents', 'metadatas'])
                    if sample_data.get('documents'):
                        health_report.append("✅ 데이터 조회: 정상")
                    else:
                        health_report.append("⚠️ 데이터 조회: 문서 없음")
                else:
                    health_report.append("ℹ️ 데이터 조회: 빈 컬렉션")
            except Exception as e:
                health_report.append(f"❌ 데이터 조회 실패: {e}")
            
            # Check embedding function
            try:
                embedding_function = db_manager.embedding_function
                if embedding_function:
                    health_report.append("✅ 임베딩 함수: 정상")
                else:
                    health_report.append("⚠️ 임베딩 함수: 없음")
            except Exception as e:
                health_report.append(f"❌ 임베딩 함수 확인 실패: {e}")
            
            # Check available collections
            try:
                collections = [col.name for col in client.list_collections()]
                health_report.append(f"✅ 사용 가능한 컬렉션: {', '.join(collections)}")
            except Exception as e:
                health_report.append(f"❌ 컬렉션 목록 조회 실패: {e}")
            
            return "\n".join(health_report)
            
        except Exception as e:
            return f"ChromaDB 상태 확인 중 오류: {e}"

    def simple_store_news(self, company, article):
        """간단한 동기 뉴스 저장 (asyncio 없이)"""
        try:
            # 본문이 없으면 제목+설명 사용
            news_content = article.content if article.content else f"{article.title}\n{article.description}"
            if len(news_content.strip()) < 50:
                logging.warning(f"뉴스 내용이 너무 짧음: {article.title}")
                return False
            # 간단한 텍스트 기반 관련도 검사
            full_text = f"{article.title} {article.description} {news_content}".lower()
            company_lower = company.lower()
            # 기본 관련도 계산
            relevance_score = 0
            # 회사명 언급 횟수
            company_mentions = full_text.count(company_lower)
            if company_mentions == 0:
                logging.info(f"회사명 없음: {article.title}")
                return False
            relevance_score += min(4, company_mentions)  # 최대 4점
            # 제목에 회사명 있으면 보너스
            if company_lower in article.title.lower():
                relevance_score += 3
            # 중요 키워드 체크
            important_keywords = ["출시", "발표", "개발", "계약", "파트너십", "투자", "실적"]
            for keyword in important_keywords:
                if keyword in full_text:
                    relevance_score += 1
            # 최종 관련도 (1-10)
            final_relevance = max(1, min(10, relevance_score))
            # 관련도 5점 미만 제외
            if final_relevance < 5:
                logging.info(f"관련도 부족 ({final_relevance}): {article.title}")
                return False
            # 간단한 청킹
            chunks = []
            # 제목 청크
            chunks.append({
                'chunk_id': 1,
                'content': article.title,
                'topics': [],
                'keywords': [company],
                'chunk_type': "제목"
            })
            # 본문 청크들 (문단별 분할)
            paragraphs = [p.strip() for p in news_content.split('\n') if len(p.strip()) > 30]
            for i, paragraph in enumerate(paragraphs[:3]):  # 최대 3개 문단
                chunks.append({
                    'chunk_id': i + 2,
                    'content': paragraph,
                    'topics': [],
                    'keywords': [company],
                    'chunk_type': "본문"
                })
            # 간단한 메타데이터 생성
            metadata = {
                'relevance_score': final_relevance,
                'topics': ["기업뉴스"],
                'keywords': [company, "뉴스"],
                'summary': article.description[:100] if article.description else article.title,
                'sentiment': "중립",
                'importance': final_relevance,
                'company_mentions': [company],
                'date': self._convert_pub_date(article.pub_date),
                'source': article.link
            }
            # 벡터 DB에 저장 (동기 방식)
            for chunk in chunks:
                embedding = [0.1] * 768  # 더미 임베딩
                self.system.db_manager.store_news_chunk(chunk, metadata, embedding)
            logging.info(f"간단 저장 완료: {article.title[:50]}... ({len(chunks)}개 청크, 관련도: {final_relevance})")
            return True
        except Exception as e:
            logging.error(f"뉴스 저장 실패: {e}")
            return False

    def _convert_pub_date(self, pub_date: str) -> str:
        """날짜 형식 변환"""
        try:
            from datetime import datetime
            dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
            return dt.strftime("%Y-%m-%d")
        except:
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d")

    def start_collection(self):
        """뉴스 수집 시작 (개선됨 - 안정성 향상)"""
        # 시스템 상태 확인
        is_healthy, health_msg = self.check_system_health()
        if not is_healthy:
            messagebox.showwarning("시스템 오류", health_msg)
            return
        
        if self.is_collecting:
            messagebox.showwarning("경고", "이미 수집이 진행 중입니다.")
            return
        
        # UI 상태 업데이트
        self.is_collecting = True
        self.collect_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress.start()
        
        # 수집 통계 초기화
        self.update_statistics(0, 0, 0, 0)
        self.clear_headlines()
        
        # GUI 응답성 유지를 위한 주기적 업데이트
        def update_gui():
            if self.is_collecting:
                self.root.update_idletasks()
                self.root.after(100, update_gui)  # 100ms마다 GUI 업데이트
        
        # GUI 업데이트 시작
        update_gui()
        
        # 별도 스레드에서 수집 실행
        def collection_worker():
            import traceback
            total_collected = 0
            relevant_collected = 0
            saved_collected = 0
            db_saved_collected = 0
            loop = None
            
            try:
                company = self.company_var.get()
                days = self.days_var.get()
                max_articles = self.max_articles_var.get()
                
                logging.info(f"{company} 뉴스 수집 시작 (최근 {days}일, 최대 {max_articles}개)")
                
                # asyncio 루프 실행 (타임아웃 설정)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # 개선된 검색 쿼리 사용
                search_queries = self.get_enhanced_search_queries()
                logging.info(f"검색 쿼리: {search_queries}")
                
                articles_per_query = max(1, max_articles // len(search_queries))
                
                for query_idx, query in enumerate(search_queries):
                    if not self.is_collecting or total_collected >= max_articles:
                        break
                        
                    try:
                        logging.info(f"검색 중: '{query}' ({query_idx + 1}/{len(search_queries)})")
                        
                        # 네이버 뉴스 검색 (타임아웃 설정)
                        try:
                            articles = self.system.naver_api.search_news(query, display=min(10, articles_per_query))
                            if not articles:
                                logging.warning(f"'{query}' 검색 결과 없음")
                                continue
                            total_collected += len(articles)
                        except Exception as search_error:
                            logging.error(f"검색 실패 ({query}): {search_error}")
                            continue
                        
                        for article_idx, article in enumerate(articles):
                            if not self.is_collecting or saved_collected >= max_articles:
                                break
                                
                            try:
                                # 날짜 필터링
                                if self.system.news_collector._is_recent_article(article.pub_date, days):
                                    # 로컬 파일로 저장
                                    saved_filename = self.save_article_to_file(article, query_idx + 1, article_idx + 1)
                                    
                                    # 수집된 뉴스 정보 저장
                                    article_info = {
                                        'title': article.title,
                                        'link': article.link,
                                        'description': article.description,
                                        'pub_date': article.pub_date,
                                        'content': article.content,
                                        'filename': saved_filename,
                                        'query': query
                                    }
                                    self.collected_news.append(article_info)
                                    
                                    # UI 업데이트 (개선된 헤드라인 표시)
                                    self.root.after(0, lambda info=article_info: self.add_enhanced_headline(info))
                                    
                                    saved_collected += 1
                                    
                                    # DB 저장 시도 (타임아웃 설정)
                                    try:
                                        success = loop.run_until_complete(
                                            asyncio.wait_for(
                                                self.system.news_collector.collect_and_store_news(company, article),
                                                timeout=30.0  # 30초 타임아웃
                                            )
                                        )
                                        if success:
                                            db_saved_collected += 1
                                            relevant_collected += 1
                                    except asyncio.TimeoutError:
                                        logging.warning(f"DB 저장 타임아웃: {article.title}")
                                    except Exception as e:
                                        logging.warning(f"DB 저장 실패: {e}")
                                    
                                    # 통계 업데이트
                                    self.root.after(0, lambda: self.update_statistics(
                                        total_collected, relevant_collected, saved_collected, db_saved_collected
                                    ))
                                
                                # API 호출 제한 (타임아웃 설정)
                                try:
                                    loop.run_until_complete(asyncio.wait_for(asyncio.sleep(1), timeout=5.0))
                                except asyncio.TimeoutError:
                                    logging.warning("API 호출 제한 타임아웃")
                                
                            except Exception as article_error:
                                logging.error(f"기사 처리 실패: {article_error}\n{traceback.format_exc()}")
                                continue
                        
                        # 쿼리 간 딜레이 (타임아웃 설정)
                        try:
                            loop.run_until_complete(asyncio.wait_for(asyncio.sleep(2), timeout=10.0))
                        except asyncio.TimeoutError:
                            logging.warning("쿼리 간 딜레이 타임아웃")
                        
                    except Exception as e:
                        logging.error(f"뉴스 수집 중 오류 ({query}): {e}\n{traceback.format_exc()}")
                        continue
                
                # 수집 완료 처리
                self.root.after(0, lambda: self.collection_complete(saved_collected, db_saved_collected))
                
            except Exception as e:
                logging.error(f"수집 작업자 오류: {e}\n{traceback.format_exc()}")
                self.root.after(0, lambda: self.collection_error(str(e)))
            finally:
                # 리소스 정리
                if loop and not loop.is_closed():
                    try:
                        # 남은 작업 취소
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        
                        # 루프 정리
                        loop.run_until_complete(asyncio.sleep(0))
                        loop.close()
                    except Exception as cleanup_error:
                        logging.error(f"루프 정리 오류: {cleanup_error}\n{traceback.format_exc()}")
        
        self.collection_thread = threading.Thread(target=collection_worker, daemon=True)
        self.collection_thread.start()
    
    def save_article_to_file(self, article, query_num, article_num):
        """기사를 로컬 파일로 저장"""
        try:
            # 파일명 생성 (안전한 파일명으로 변환)
            safe_title = "".join(c for c in article.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title[:50]  # 길이 제한
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_{timestamp}_q{query_num}_a{article_num}_{safe_title}.txt"
            filepath = os.path.join(self.news_directory, filename)
            
            # 기사 내용 구성
            content = f"""뉴스 기사 정보
===================
제목: {article.title}
링크: {article.link}
발행일: {article.pub_date}
설명: {article.description}

본문:
{article.content}

수집 정보:
- 수집 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 쿼리 번호: {query_num}
- 기사 번호: {article_num}
"""
            
            # 파일 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logging.info(f"기사 저장 완료: {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"기사 파일 저장 실패: {e}")
            return None
    
    def add_enhanced_headline(self, article_info):
        """개선된 헤드라인 추가 (제목, 날짜, 미리보기 포함)"""
        try:
            # 날짜 형식 변환
            pub_date = article_info['pub_date']
            try:
                date_obj = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
                formatted_date = date_obj.strftime("%m/%d")
            except:
                formatted_date = pub_date[:10] if len(pub_date) > 10 else pub_date
            
            # 미리보기 텍스트 (설명의 앞부분)
            preview = article_info['description'][:100] + "..." if len(article_info['description']) > 100 else article_info['description']
            
            # 트리뷰에 추가
            item_id = self.headlines_tree.insert('', 'end', 
                text=str(len(self.collected_news)),
                values=(article_info['title'], preview, formatted_date)  # Reordered to match column order
            )
            
            # 자동 스크롤
            if self.auto_scroll:
                self.headlines_tree.see(item_id)
                
        except Exception as e:
            logging.error(f"헤드라인 추가 실패: {e}")
    
    def stop_collection(self):
        """뉴스 수집 중지 (개선됨)"""
        try:
            self.is_collecting = False
            self.collect_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.progress.stop()
            
            # 스레드가 실행 중이면 잠시 대기
            if self.collection_thread and self.collection_thread.is_alive():
                logging.info("뉴스 수집 중지 요청됨. 완료 대기 중...")
                self.collection_thread.join(timeout=5.0)  # 최대 5초 대기
                
                if self.collection_thread.is_alive():
                    logging.warning("수집 스레드가 5초 내에 종료되지 않았습니다.")
                else:
                    logging.info("뉴스 수집이 안전하게 중지되었습니다.")
            else:
                logging.info("뉴스 수집이 중지되었습니다.")
                
        except Exception as e:
            logging.error(f"수집 중지 중 오류: {e}")
            # 최소한의 복구
            try:
                self.is_collecting = False
                self.collect_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.progress.stop()
            except:
                pass
    
    def collection_complete(self, saved_count, db_saved_count):
        """수집 완료 처리"""
        self.is_collecting = False
        self.collect_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()
        
        message = f"뉴스 수집 완료!\n로컬 파일: {saved_count}개\nDB 저장: {db_saved_count}개"
        messagebox.showinfo("완료", message)
        logging.info(f"뉴스 수집 완료: 로컬 {saved_count}개, DB {db_saved_count}개")
    
    def collection_error(self, error_msg):
        """수집 오류 처리 (개선됨)"""
        try:
            self.is_collecting = False
            self.collect_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.progress.stop()
            
            # 오류 메시지 개선
            if "timeout" in error_msg.lower() or "타임아웃" in error_msg:
                display_msg = "네트워크 연결이 느리거나 불안정합니다. 잠시 후 다시 시도해주세요."
            elif "api" in error_msg.lower():
                display_msg = "API 호출에 실패했습니다. API 키와 설정을 확인해주세요."
            else:
                display_msg = f"뉴스 수집 중 오류가 발생했습니다: {error_msg}"
            
            messagebox.showerror("수집 오류", display_msg)
            logging.error(f"뉴스 수집 실패: {error_msg}")
            
            # 부분적으로 수집된 결과가 있으면 알림
            if self.collected_news:
                partial_count = len(self.collected_news)
                messagebox.showinfo("부분 완료", f"오류가 발생했지만 {partial_count}개의 기사가 수집되었습니다.")
                
        except Exception as e:
            logging.error(f"오류 처리 중 추가 오류: {e}")
            # 최소한의 복구 시도
            try:
                self.is_collecting = False
                self.collect_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.progress.stop()
            except:
                pass
    
    def update_statistics(self, total, relevant, saved, db_saved):
        """통계 업데이트"""
        self.total_articles_var.set(str(total))
        self.relevant_articles_var.set(str(relevant))
        self.saved_articles_var.set(str(saved))
        self.db_saved_var.set(str(db_saved))
    
    def clear_headlines(self):
        """헤드라인 지우기"""
        for item in self.headlines_tree.get_children():
            self.headlines_tree.delete(item)
        self.collected_news.clear()
    
    def refresh_headlines(self):
        """헤드라인 새로고침"""
        self.clear_headlines()
        # 수집된 뉴스 다시 표시
        for article_info in self.collected_news:
            self.add_enhanced_headline(article_info)
    
    def view_selected_article(self):
        """선택된 기사 보기"""
        selection = self.headlines_tree.selection()
        if selection:
            item = selection[0]
            item_text = self.headlines_tree.item(item, "text")
            try:
                index = int(item_text) - 1
                if 0 <= index < len(self.collected_news):
                    article = self.collected_news[index]
                    
                    # 새 창으로 기사 내용 표시
                    article_window = tk.Toplevel(self.root)
                    article_window.title(f"기사 내용 - {article['title'][:50]}...")
                    article_window.geometry("900x700")
                    
                    # 메뉴바 추가
                    menubar = tk.Menu(article_window)
                    article_window.config(menu=menubar)
                    
                    file_menu = tk.Menu(menubar, tearoff=0)
                    menubar.add_cascade(label="파일", menu=file_menu)
                    file_menu.add_command(label="파일로 저장", command=lambda: self.save_article_content(article))
                    file_menu.add_command(label="클립보드 복사", command=lambda: self.copy_article_to_clipboard(article))
                    
                    # 기사 내용 표시
                    content_text = scrolledtext.ScrolledText(article_window, wrap=tk.WORD, font=("맑은 고딕", 10))
                    content_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    article_content = f"""제목: {article['title']}
발행일: {article['pub_date']}
링크: {article['link']}
검색 쿼리: {article.get('query', 'N/A')}
로컬 파일: {article.get('filename', 'N/A')}

설명:
{article['description']}

본문:
{article['content']}
"""
                    content_text.insert(1.0, article_content)
                    content_text.config(state=tk.DISABLED)  # 읽기 전용
            except (ValueError, IndexError):
                messagebox.showerror("오류", "기사 정보를 찾을 수 없습니다.")
        else:
            messagebox.showwarning("선택 없음", "보려는 기사를 선택해주세요.")
    
    def open_news_directory(self):
        """뉴스 저장 디렉토리 열기"""
        import subprocess
        import platform
        
        try:
            if platform.system() == "Windows":
                os.startfile(self.news_directory)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", self.news_directory])
            else:  # Linux
                subprocess.call(["xdg-open", self.news_directory])
        except Exception as e:
            messagebox.showerror("오류", f"폴더를 열 수 없습니다: {e}")
    
    def add_manual_news(self):
        """수동 뉴스 추가"""
        if not self.system:
            messagebox.showwarning("경고", "먼저 시스템을 초기화해주세요.")
            return
        
        news_content = self.manual_text.get(1.0, tk.END).strip()
        if not news_content:
            messagebox.showwarning("경고", "뉴스 내용을 입력해주세요.")
            return
        
        def manual_worker():
            try:
                company = self.company_var.get()
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(
                    self.system.collect_manual_news(company, news_content)
                )
                loop.close()
                
                if success:
                    self.root.after(0, lambda: messagebox.showinfo("성공", "수동 뉴스가 추가되었습니다."))
                    self.root.after(0, lambda: self.manual_text.delete(1.0, tk.END))
                    self.root.after(0, lambda: self.update_statistics(
                        int(self.total_articles_var.get()) + 1,
                        int(self.relevant_articles_var.get()) + 1,
                        int(self.saved_articles_var.get()),
                        int(self.db_saved_var.get()) + 1
                    ))
                else:
                    self.root.after(0, lambda: messagebox.showerror("실패", "수동 뉴스 추가에 실패했습니다."))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("오류", f"수동 뉴스 추가 오류: {e}"))
        
        threading.Thread(target=manual_worker, daemon=True).start()
    
    def load_news_file(self):
        """파일에서 뉴스 불러오기"""
        file_path = filedialog.askopenfilename(
            title="뉴스 파일 선택",
            filetypes=[("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.manual_text.delete(1.0, tk.END)
                self.manual_text.insert(1.0, content)
                logging.info(f"파일 로드 완료: {file_path}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 읽기 실패: {e}")
    
    def generate_news(self):
        """뉴스 생성 (개선된 RAG 포함, Claude API 분석 정보 표시)"""
        if not self.system:
            messagebox.showwarning("경고", "먼저 시스템을 초기화해주세요.")
            return
        topic = self.topic_var.get().strip()
        keywords_str = self.keywords_var.get().strip()
        user_facts = self.facts_text.get(1.0, tk.END).strip()
        style = self.style_var.get()
        if not topic or not keywords_str or not user_facts:
            messagebox.showwarning("경고", "모든 필드를 입력해주세요.")
            return
        keywords = [k.strip() for k in keywords_str.split(",")]
        length_type = self.length_type_var.get()
        length_count = self.length_count_var.get()
        # Add: get original news content from a new input field or from loaded file
        try:
            original_news = self.original_news_text.get(1.0, tk.END).strip()
        except Exception:
            original_news = ""
        # If not present, fallback to user_facts
        if not original_news:
            original_news = user_facts
        def generation_worker():
            try:
                import html
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                if self.use_rag_var.get():
                    rag_count = self.rag_count_var.get()
                    search_query = f"{topic} {' '.join(keywords)}"
                    logging.info(f"RAG 검색 중: '{search_query}' (상위 {rag_count}개)")
                    search_results = self.system.db_manager.search_relevant_news(search_query, n_results=rag_count)
                    reference_materials = self.system.news_writer._build_comprehensive_reference_materials(search_results)
                else:
                    reference_materials = "참고 자료를 사용하지 않습니다."
                # Compose enhanced_user_facts with original news
                enhanced_user_facts = f"{user_facts}\n\n[원본 뉴스]\n{original_news}\n\n[생성 설정]\n- 스타일: {style}\n- 길이: {length_count} {length_type}"
                self.root.after(0, lambda: self.update_generation_status("뉴스 생성 중..."))
                prompt = self.system.news_writer.get_full_generation_prompt(
                    topic, keywords, enhanced_user_facts, reference_materials, f"{length_count}{length_type}"
                )
                # Claude API 호출 및 분석 정보 수집
                result = loop.run_until_complete(
                    self.system.write_news(
                        topic, keywords, enhanced_user_facts, style,
                        length_specification=f"{length_count}{length_type}",
                        use_rag=self.use_rag_var.get(),
                        rag_count=self.rag_count_var.get()
                    )
                )
                news = result.get("response", "")
                analysis_info = (
                    f"[Claude API 분석 정보]\n"
                    f"- 모델명: {result.get('model', 'N/A')}\n"
                    f"- 경과 시간: {result.get('elapsed', 0):.2f}초\n"
                    f"- 프롬프트 길이: {len(result.get('prompt', ''))}자\n"
                    f"- 프롬프트 내용 (아래 참조):\n"
                    f"-----------------------------\n"
                    f"{html.escape(result.get('prompt', '')[:2000])}"
                )
                # 뉴스 본문은 분석 정보 아래에 표시
                self.root.after(0, lambda: self.show_generated_news(news, analysis_info))
                self.root.after(0, lambda: self.evaluate_news_quality(news))
                
                # 생성된 뉴스와 프롬프트를 히스토리에 저장
                meta = {
                    'topic': topic,
                    'keywords': ', '.join(keywords),
                    'style': style,
                    'length': f"{length_count}{length_type}",
                    'use_rag': self.use_rag_var.get(),
                    'model': result.get('model', 'N/A'),
                    'elapsed': result.get('elapsed', 0)
                }
                self.root.after(0, lambda: self.add_news_history(news, prompt, meta))
                loop.close()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("오류", f"뉴스 생성 오류: {e}"))
                self.root.after(0, lambda: logging.error(f"뉴스 생성 오류: {e}"))
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, "뉴스 생성 중입니다. 잠시만 기다려주세요...\n\n")
        self.generate_btn.config(state=tk.DISABLED)
        threading.Thread(target=generation_worker, daemon=True).start()

    def show_generated_news(self, news, analysis_info=None):
        """생성된 뉴스 및 Claude API 분석 정보 표시"""
        self.result_text.delete(1.0, tk.END)
        if analysis_info:
            self.result_text.insert(tk.END, analysis_info + "\n\n")
        self.result_text.insert(tk.END, "[생성된 뉴스 본문]\n" + news)
        self.generate_btn.config(state=tk.NORMAL)
        logging.info("뉴스 생성 완료 (분석 정보 포함)")
    
    def evaluate_news_quality(self, news):
        """뉴스 품질 평가"""
        try:
            # 간단한 품질 지표 계산
            lines = len([line for line in news.split('\n') if line.strip()])
            words = len(news.split())
            chars = len(news)
            
            # 구조적 요소 확인
            has_title = "제목:" in news
            has_lead = "리드:" in news or "요약:" in news
            has_body = lines > 5
            has_conclusion = "결론:" in news or "전망:" in news
            
            structure_score = sum([has_title, has_lead, has_body, has_conclusion]) * 25
            length_score = min(100, (words / 200) * 100) if words > 0 else 0
            
            overall_score = (structure_score + length_score) / 2
            
            quality_text = f"품질 평가: {overall_score:.0f}점 | 줄수: {lines} | 단어수: {words} | 글자수: {chars}"
            
            if structure_score < 75:
                quality_text += " | ⚠️ 구조 개선 필요"
            if length_score < 50:
                quality_text += " | ⚠️ 길이 부족"
            
            self.quality_var.set(quality_text)
            
        except Exception as e:
            self.quality_var.set(f"품질 평가 오류: {e}")
    
    def regenerate_news(self):
        """뉴스 다시 생성"""
        self.generate_news()
    
    def load_template(self):
        """템플릿 불러오기"""
        file_path = filedialog.askopenfilename(
            title="템플릿 파일 선택",
            filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    template = json.load(f)
                
                self.topic_var.set(template.get("topic", ""))
                self.keywords_var.set(template.get("keywords", ""))
                self.style_var.set(template.get("style", "기업 보도형"))
                self.length_type_var.set(template.get("length_type", "줄 수"))
                self.length_count_var.set(template.get("length_count", 100))
                
                self.facts_text.delete(1.0, tk.END)
                self.facts_text.insert(1.0, template.get("user_facts", ""))
                
                messagebox.showinfo("성공", "템플릿이 로드되었습니다.")
                logging.info(f"템플릿 로드: {file_path}")
                
            except Exception as e:
                messagebox.showerror("오류", f"템플릿 로드 실패: {e}")
    
    def save_template(self):
        """템플릿 저장"""
        template = {
            "topic": self.topic_var.get(),
            "keywords": self.keywords_var.get(),
            "style": self.style_var.get(),
            "length_type": self.length_type_var.get(),
            "length_count": self.length_count_var.get(),
            "user_facts": self.facts_text.get(1.0, tk.END).strip()
        }
        
        file_path = filedialog.asksaveasfilename(
            title="템플릿 저장",
            defaultextension=".json",
            filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(template, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("성공", "템플릿이 저장되었습니다.")
                logging.info(f"템플릿 저장: {file_path}")
            except Exception as e:
                messagebox.showerror("오류", f"템플릿 저장 실패: {e}")
    
    def save_news(self):
        """뉴스를 파일로 저장"""
        content = self.result_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("경고", "저장할 뉴스가 없습니다.")
            return
        
        # 기본 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"generated_news_{timestamp}.txt"
        
        file_path = filedialog.asksaveasfilename(
            title="뉴스 저장",
            initialvalue=default_name,
            defaultextension=".txt",
            filetypes=[("텍스트 파일", "*.txt"), ("Word 문서", "*.docx"), ("모든 파일", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("성공", "뉴스가 저장되었습니다.")
                logging.info(f"뉴스 저장: {file_path}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 실패: {e}")
    
    def copy_to_clipboard(self):
        """클립보드에 복사"""
        content = self.result_text.get(1.0, tk.END).strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            messagebox.showinfo("성공", "클립보드에 복사되었습니다.")
            logging.info("클립보드 복사 완료")
        else:
            messagebox.showwarning("경고", "복사할 내용이 없습니다.")
    
    def save_article_content(self, article):
        """기사 내용을 파일로 저장"""
        content = f"""제목: {article['title']}
발행일: {article['pub_date']}
링크: {article['link']}

설명:
{article['description']}

본문:
{article['content']}
"""
        
        safe_title = "".join(c for c in article['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
        default_name = f"article_{safe_title}.txt"
        
        file_path = filedialog.asksaveasfilename(
            title="기사 저장",
            initialvalue=default_name,
            defaultextension=".txt",
            filetypes=[("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("성공", "기사가 저장되었습니다.")
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 실패: {e}")
    
    def copy_article_to_clipboard(self, article):
        """기사 내용을 클립보드에 복사"""
        content = f"{article['title']}\n\n{article['description']}\n\n{article['content']}"
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        messagebox.showinfo("성공", "기사가 클립보드에 복사되었습니다.")
    
    def save_log(self):
        """로그 저장"""
        content = self.log_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("경고", "저장할 로그가 없습니다.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"news_system_log_{timestamp}.log"
        
        file_path = filedialog.asksaveasfilename(
            title="로그 저장",
            initialvalue=default_name,
            defaultextension=".log",
            filetypes=[("로그 파일", "*.log"), ("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("성공", "로그가 저장되었습니다.")
                logging.info(f"로그 저장: {file_path}")
            except Exception as e:
                messagebox.showerror("오류", f"로그 저장 실패: {e}")

    def show_manual_news_popup(self):
        """수동 뉴스 입력 팝업"""
        popup = tk.Toplevel(self.root)
        popup.title("수동 뉴스 입력")
        popup.geometry("600x400")
        popup.transient(self.root)
        
        ttk.Label(popup, text="뉴스 내용:").pack(anchor=tk.W, padx=10, pady=5)
        manual_text = scrolledtext.ScrolledText(popup, height=10, wrap=tk.WORD)
        manual_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        btn_frame = ttk.Frame(popup)
        btn_frame.pack(fill=tk.X, pady=10)
        
        def save_manual():
            content = manual_text.get(1.0, tk.END).strip()
            if not content:
                messagebox.showwarning("경고", "뉴스 내용을 입력해주세요.")
                return
            self.add_manual_news_content(content)
            popup.destroy()
        
        ttk.Button(btn_frame, text="저장", command=save_manual).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="취소", command=popup.destroy).pack(side=tk.LEFT, padx=5)

    def add_manual_news_content(self, content):
        """수동 뉴스 입력 실제 저장 로직 (기존 add_manual_news와 유사)"""
        if not self.system:
            messagebox.showwarning("경고", "먼저 시스템을 초기화해주세요.")
            return
        def manual_worker():
            try:
                company = self.company_var.get()
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(
                    self.system.collect_manual_news(company, content)
                )
                loop.close()
                if success:
                    self.root.after(0, lambda: messagebox.showinfo("성공", "수동 뉴스가 추가되었습니다."))
                else:
                    self.root.after(0, lambda: messagebox.showerror("실패", "수동 뉴스 추가에 실패했습니다."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("오류", f"수동 뉴스 추가 오류: {e}"))
        import threading
        threading.Thread(target=manual_worker, daemon=True).start()

    def setup_history_tab(self, parent):
        """최근 생성 뉴스/프롬프트 탭 (comprehensive upgrade)"""
        history_frame = ttk.Frame(parent)
        parent.add(history_frame, text="🕑 최근 생성 뉴스/프롬프트")

        # Top: Search/filter and sort
        top_frame = ttk.Frame(history_frame)
        top_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(top_frame, text="검색:").pack(side=tk.LEFT)
        self.history_search_var = tk.StringVar()
        search_entry = ttk.Entry(top_frame, textvariable=self.history_search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=2)
        search_entry.bind('<KeyRelease>', lambda e: self.refresh_history_list())
        ttk.Label(top_frame, text="정렬:").pack(side=tk.LEFT, padx=(10,0))
        self.history_sort_var = tk.StringVar(value="날짜 내림차순")
        sort_combo = ttk.Combobox(top_frame, textvariable=self.history_sort_var, values=["날짜 내림차순", "날짜 오름차순", "토픽"], state="readonly", width=12)
        sort_combo.pack(side=tk.LEFT, padx=2)
        sort_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_history_list())
        ttk.Button(top_frame, text="내보내기", command=self.export_history).pack(side=tk.RIGHT, padx=2)
        ttk.Button(top_frame, text="가져오기", command=self.import_history).pack(side=tk.RIGHT, padx=2)

        # Create a horizontal PanedWindow for 50/50 split
        paned = tk.PanedWindow(history_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        # Left: Treeview for history (50%)
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, stretch='always')
        
        columns = ("timestamp", "topic", "type")
        self.history_tree = ttk.Treeview(left_frame, columns=columns, show="headings", selectmode="browse")
        self.history_tree.heading("timestamp", text="날짜/시간")
        self.history_tree.heading("topic", text="토픽")
        self.history_tree.heading("type", text="유형")
        self.history_tree.column("timestamp", width=140, anchor="center")
        self.history_tree.column("topic", width=200, anchor="w")
        self.history_tree.column("type", width=60, anchor="center")
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.history_tree.bind('<<TreeviewSelect>>', self.on_history_select)
        # Alternating row colors
        self.history_tree.tag_configure('oddrow', background='#f0f0f0')
        self.history_tree.tag_configure('evenrow', background='#e0e0e0')
        # Scrollbar for list
        tree_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Empty state label
        self.history_empty_label = ttk.Label(left_frame, text="뉴스를 생성하면 여기에 표시됩니다!", foreground="gray")
        self.history_empty_label.place(relx=0.5, rely=0.5, anchor="center")
        self.history_empty_label.lower(self.history_tree)

        # Right: Content display (50%)
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, stretch='always')
        
        # Content display with vertical split
        content_paned = tk.PanedWindow(right_frame, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=3)
        content_paned.pack(fill=tk.BOTH, expand=True)
        
        # Top: News content
        news_frame = ttk.Frame(content_paned)
        content_paned.add(news_frame, stretch='always')
        ttk.Label(news_frame, text="📰 생성된 뉴스", font=("맑은 고딕", 10, "bold")).pack(anchor="w", padx=5, pady=2)
        self.history_news_text = scrolledtext.ScrolledText(news_frame, font=("맑은 고딕", 9), foreground="gray", wrap=tk.WORD)
        self.history_news_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        # Bottom: Prompt content
        prompt_frame = ttk.Frame(content_paned)
        content_paned.add(prompt_frame, stretch='always')
        ttk.Label(prompt_frame, text="🔧 사용된 프롬프트", font=("맑은 고딕", 10, "bold")).pack(anchor="w", padx=5, pady=2)
        self.history_prompt_text = scrolledtext.ScrolledText(prompt_frame, font=("Consolas", 8), foreground="gray", wrap=tk.WORD)
        self.history_prompt_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        # Action buttons at the bottom of right frame
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.btn_copy_news = ttk.Button(btn_frame, text="📰 뉴스 복사", command=self.copy_history_news)
        self.btn_copy_news.pack(side=tk.LEFT, padx=2)
        self.btn_copy_prompt = ttk.Button(btn_frame, text="🔧 프롬프트 복사", command=self.copy_history_prompt)
        self.btn_copy_prompt.pack(side=tk.LEFT, padx=2)
        self.btn_regen = ttk.Button(btn_frame, text="🔄 재생성", command=self.regenerate_from_history)
        self.btn_regen.pack(side=tk.LEFT, padx=2)
        self.btn_delete = ttk.Button(btn_frame, text="🗑️ 삭제", command=self.delete_history_entry)
        self.btn_delete.pack(side=tk.LEFT, padx=2)
        # Tooltips (simple)
        self.btn_copy_prompt.tooltip = "Copy the full prompt to clipboard"
        self.btn_copy_news.tooltip = "Copy the generated news to clipboard"
        self.btn_regen.tooltip = "Refill the writing tab with this prompt"
        self.btn_delete.tooltip = "Delete this entry from history"

        # Keyboard shortcuts
        self.root.bind_all('<Control-c>', lambda e: self.copy_history_prompt())
        self.root.bind_all('<Control-r>', lambda e: self.regenerate_from_history())
        self.root.bind_all('<Delete>', lambda e: self.delete_history_entry())

        self.refresh_history_list()

    def refresh_history_list(self):
        try:
            # Save selection
            selected = self.history_tree.selection()
            selected_id = selected[0] if selected else None
            # Filter and sort
            search = self.history_search_var.get().strip().lower() if hasattr(self, 'history_search_var') else ''
            sort = self.history_sort_var.get() if hasattr(self, 'history_sort_var') else '날짜 내림차순'
            filtered = [entry for entry in self.news_history if (search in entry['meta'].get('topic', '').lower() or search in entry['prompt'].lower() or search in entry['news'].lower())]
            if sort == '날짜 내림차순':
                filtered.sort(key=lambda x: x['timestamp'], reverse=True)
            elif sort == '날짜 오름차순':
                filtered.sort(key=lambda x: x['timestamp'])
            elif sort == '토픽':
                filtered.sort(key=lambda x: x['meta'].get('topic', ''))
            self.history_tree.delete(*self.history_tree.get_children())
            if not filtered:
                self.history_empty_label.lift(self.history_tree)
                self.history_news_text.config(state=tk.NORMAL)
                self.history_news_text.delete(1.0, tk.END)
                self.history_news_text.config(state=tk.DISABLED)
                self.history_prompt_text.config(state=tk.NORMAL)
                self.history_prompt_text.delete(1.0, tk.END)
                self.history_prompt_text.config(state=tk.DISABLED)
                for btn in [self.btn_copy_prompt, self.btn_copy_news, self.btn_regen, self.btn_delete]:
                    btn.state(['disabled'])
                return
            else:
                self.history_empty_label.lower(self.history_tree)
            for i, entry in enumerate(filtered):
                topic = entry['meta'].get('topic', '(제목 없음)')
                ts = entry['timestamp']
                ntype = '뉴스'  # 모든 항목은 뉴스 생성 기록
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                self.history_tree.insert('', 'end', iid=str(i), values=(ts, topic, ntype), tags=(tag,))
            # Restore selection
            if selected_id and selected_id in self.history_tree.get_children():
                self.history_tree.selection_set(selected_id)
                self.history_tree.see(selected_id)
            else:
                self.history_tree.selection_set(self.history_tree.get_children()[0])
            for btn in [self.btn_copy_prompt, self.btn_copy_news, self.btn_regen, self.btn_delete]:
                btn.state(['!disabled'])
            self.statusbar_var.set(f"히스토리 {len(filtered)}건 표시됨.")
        except Exception as e:
            logging.error(f"히스토리 리스트 갱신 오류: {e}")
            self.history_tree.delete(*self.history_tree.get_children())
            self.history_empty_label.lift(self.history_tree)
            self.statusbar_var.set(f"Error: {e}")

    def on_history_select(self, event=None):
        sel = self.history_tree.selection()
        if not sel:
            for btn in [self.btn_copy_prompt, self.btn_copy_news, self.btn_regen, self.btn_delete]:
                btn.state(['disabled'])
            return
        try:
            idx = int(sel[0])
            if idx < 0 or idx >= len(self.news_history):
                for btn in [self.btn_copy_prompt, self.btn_copy_news, self.btn_regen, self.btn_delete]:
                    btn.state(['disabled'])
                return
            entry = self.news_history[idx]
            self.history_news_text.config(state=tk.NORMAL)
            self.history_news_text.delete(1.0, tk.END)
            self.history_news_text.insert(1.0, entry["news"])
            self.history_news_text.config(state=tk.DISABLED)
            self.history_prompt_text.config(state=tk.NORMAL)
            self.history_prompt_text.delete(1.0, tk.END)
            self.history_prompt_text.insert(1.0, entry["prompt"])
            self.history_prompt_text.config(state=tk.DISABLED)
            for btn in [self.btn_copy_prompt, self.btn_copy_news, self.btn_regen, self.btn_delete]:
                btn.state(['!disabled'])
        except (ValueError, IndexError):
            for btn in [self.btn_copy_prompt, self.btn_copy_news, self.btn_regen, self.btn_delete]:
                btn.state(['disabled'])

    def export_history(self):
        try:
            file_path = filedialog.asksaveasfilename(title="히스토리 내보내기", defaultextension=".csv", filetypes=[("CSV 파일", "*.csv"), ("JSON 파일", "*.json")])
            if not file_path:
                return
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.news_history, f, ensure_ascii=False, indent=2)
            else:
                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "topic", "type", "news", "prompt"])
                    for entry in self.news_history:
                        writer.writerow([entry['timestamp'], entry['meta'].get('topic', ''), entry['meta'].get('type', ''), entry['news'], entry['prompt']])
            self.statusbar_var.set("히스토리 내보내기 완료.")
        except Exception as e:
            messagebox.showerror("오류", f"히스토리 내보내기 실패: {e}")

    def import_history(self):
        try:
            file_path = filedialog.askopenfilename(title="히스토리 가져오기", filetypes=[("CSV 파일", "*.csv"), ("JSON 파일", "*.json")])
            if not file_path:
                return
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.news_history = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self.news_history = []
                    for row in reader:
                        self.news_history.append({
                            "timestamp": row.get("timestamp", ""),
                            "news": row.get("news", ""),
                            "prompt": row.get("prompt", ""),
                            "meta": {"topic": row.get("topic", ""), "type": row.get("type", "일반")}
                        })
            self.save_news_history()
            self.refresh_history_list()
            self.statusbar_var.set("히스토리 가져오기 완료.")
        except Exception as e:
            messagebox.showerror("오류", f"히스토리 가져오기 실패: {e}")

    def load_news_history(self):
        try:
            with open(self.news_history_file, "r", encoding="utf-8") as f:
                self.news_history = json.load(f)
        except Exception:
            self.news_history = []

    def save_news_history(self):
        try:
            with open(self.news_history_file, "w", encoding="utf-8") as f:
                json.dump(self.news_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"뉴스 히스토리 저장 실패: {e}")

    def add_news_history(self, news, prompt, meta):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "news": news,
            "prompt": prompt,
            "meta": meta
        }
        self.news_history.insert(0, entry)
        self.news_history = self.news_history[:20]  # Keep only 20 most recent
        self.save_news_history()
        self.refresh_history_list()

    def delete_history_entry(self):
        sel = self.history_tree.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
            if 0 <= idx < len(self.news_history):
                del self.news_history[idx]
                self.save_news_history()
                self.refresh_history_list()
        except (ValueError, IndexError) as e:
            messagebox.showerror("오류", f"삭제할 수 없습니다: {e}")

    def add_tooltips(self):
        try:
            import tooltip
        except ImportError:
            # Simple fallback if tooltip module is not available
            def add_tooltip(widget, text):
                pass
        else:
            def add_tooltip(widget, text):
                tooltip.ToolTip(widget, text)
        # Example: add_tooltip(self.generate_btn, "뉴스 생성")
        # Add more tooltips as needed
        pass

    def copy_history_prompt(self):
        """Copy the selected prompt from history to clipboard"""
        sel = self.history_tree.selection()
        if not sel:
            messagebox.showwarning("경고", "선택된 항목이 없습니다.")
            return
        try:
            idx = int(sel[0])
            if idx < 0 or idx >= len(self.news_history):
                messagebox.showerror("오류", "유효하지 않은 히스토리 인덱스입니다.")
                return
            prompt = self.news_history[idx].get("prompt", "")
            if prompt:
                self.root.clipboard_clear()
                self.root.clipboard_append(prompt)
                logging.info("프롬프트가 클립보드에 복사되었습니다.")
            else:
                messagebox.showwarning("경고", "복사할 프롬프트가 없습니다.")
        except (ValueError, IndexError) as e:
            messagebox.showerror("오류", f"복사할 수 없습니다: {e}")

    def copy_history_news(self):
        """Copy the selected news content from history to clipboard"""
        sel = self.history_tree.selection()
        if not sel:
            messagebox.showwarning("경고", "선택된 항목이 없습니다.")
            return
        try:
            idx = int(sel[0])
            if idx < 0 or idx >= len(self.news_history):
                messagebox.showerror("오류", "유효하지 않은 히스토리 인덱스입니다.")
                return
            news = self.news_history[idx].get("news", "")
            if news:
                self.root.clipboard_clear()
                self.root.clipboard_append(news)
                logging.info("뉴스가 클립보드에 복사되었습니다.")
            else:
                messagebox.showwarning("경고", "복사할 뉴스가 없습니다.")
        except (ValueError, IndexError) as e:
            messagebox.showerror("오류", f"복사할 수 없습니다: {e}")

    def regenerate_from_history(self):
        """Refill the writing tab with the selected prompt and related fields from history"""
        sel = self.history_tree.selection()
        if not sel:
            messagebox.showwarning("경고", "선택된 항목이 없습니다.")
            return
        try:
            idx = int(sel[0])
            if idx < 0 or idx >= len(self.news_history):
                messagebox.showerror("오류", "유효하지 않은 히스토리 인덱스입니다.")
                return
            entry = self.news_history[idx]
            prompt = entry.get("prompt", "")
            meta = entry.get("meta", {})
            topic = meta.get("topic", "")
            keywords = meta.get("keywords", "")
            style = meta.get("style", "기업 보도형")
            length_type = meta.get("length_type", "줄 수")
            length_count = meta.get("length_count", 100)
            user_facts = meta.get("user_facts", prompt)
            self.topic_var.set(topic)
            self.keywords_var.set(keywords)
            self.style_var.set(style)
            self.length_type_var.set(length_type)
            self.length_count_var.set(length_count)
            self.facts_text.delete(1.0, tk.END)
            self.facts_text.insert(1.0, user_facts)
            self.notebook.select(2)
            messagebox.showinfo("완료", "프롬프트와 관련 정보가 작성 탭에 채워졌습니다.")
        except (ValueError, IndexError) as e:
            messagebox.showerror("오류", f"재생성할 수 없습니다: {e}")

    def show_vector_db_info(self):
        """Show vector DB backend/path/config info in the log file."""
        try:
            db_manager = self.system.db_manager if self.system else None
            if not db_manager:
                logging.warning("시스템이 초기화되지 않았습니다(V).")
                return
            
            # Check if collection exists and is accessible
            try:
                collection_count = db_manager.collection.count()
                collection_name = getattr(db_manager.collection, 'name', 'N/A')
                collection_status = "정상"
            except Exception as e:
                collection_count = "접근 불가"
                collection_name = "N/A"
                collection_status = f"오류: {str(e)}"
            
            info = f"=== 벡터DB 정보 ===\n"
            info += f"DB Path: {getattr(db_manager, 'db_path', 'N/A')}\n"
            info += f"Collection Name: {collection_name}\n"
            info += f"Collection Status: {collection_status}\n"
            info += f"Embedding Dimension: {getattr(db_manager, '_embedding_dimension', 'N/A')}\n"
            info += f"Backend: ChromaDB\n"
            info += f"Collection Count: {collection_count}\n"
            
            # Add additional diagnostic information
            try:
                client = getattr(db_manager, 'client', None)
                if client:
                    collections = [col.name for col in client.list_collections()]
                    info += f"Available Collections: {', '.join(collections) if collections else 'None'}\n"
            except Exception as e:
                info += f"Collections List Error: {str(e)}\n"
            
            info += f"==================\n"
            
            # Log the information instead of showing popup
            logging.info(info)
            
        except Exception as e:
            logging.error(f"DB 정보 표시 실패: {e}")
            import traceback
            logging.error(f"전체 오류 정보: {traceback.format_exc()}")

    def _refresh_vector_tree(self, data):
        """Refresh the vector tree with the given data list."""
        for item in self.vector_tree.get_children():
            self.vector_tree.delete(item)
        for i, item_data in enumerate(data):
            doc_id = item_data.get('id', f'unknown_{i}')
            doc = item_data.get('document', '')
            metadata = item_data.get('metadata', {})
            try:
                topics_raw = metadata.get('topics', '[]')
                if isinstance(topics_raw, str):
                    topics = json.loads(topics_raw)
                else:
                    topics = topics_raw if isinstance(topics_raw, list) else []
                topics_str = ', '.join(topics[:2]) if topics else 'N/A'
            except:
                topics_str = str(metadata.get('topics', 'N/A'))[:20]
            try:
                content_preview = doc[:50] + "..." if len(doc) > 50 else doc
                content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')
            except:
                content_preview = "내용 없음"
            try:
                relevance = metadata.get('relevance_score', 'N/A')
                if isinstance(relevance, (int, float)):
                    relevance_str = f"{relevance}/10"
                else:
                    relevance_str = str(relevance)
            except:
                relevance_str = "N/A"
            date_str = metadata.get('date', metadata.get('created_at', 'N/A'))
            if isinstance(date_str, str) and 'T' in date_str:
                date_str = date_str.split('T')[0]
            self.vector_tree.insert('', 'end',
                text=str(i+1),
                values=(
                    doc_id[:15] + "..." if len(str(doc_id)) > 15 else str(doc_id),
                    content_preview,
                    topics_str,
                    relevance_str,
                    str(date_str)
                )
            )


def main():
    """GUI 메인 함수"""
    try:
        # NOTE: Requires 'pip install ttkthemes'
        root = ThemedTk(theme="arc")
        app = EnhancedNewsWriterGUI(root)
        
        # 창 닫기 이벤트 처리
        def on_closing():
            if app.is_collecting:
                if messagebox.askokcancel("종료", "뉴스 수집이 진행 중입니다. 정말 종료하시겠습니까?"):
                    app.stop_collection()
                    root.destroy()
            else:
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # 시작 메시지
        print("="*60)
        print("🚀 AI News Writer Pro - 전문 뉴스 생성 시스템")
        print("="*60)
        print("✅ 향상된 기능:")
        print("   • 회사명 + 추가 키워드 조합 검색")
        print("   • 12개월(365일) 기본 수집 기간")
        print("   • 로컬 파일 자동 저장")
        print("   • 개선된 RAG (10개 뉴스 참조)")
        print("   • 헤드라인 + 미리보기 표시")
        print("   • 뉴스 길이 조절 (줄 수/단어 수)")
        print("   • 하단 통합 로그 표시")
        print("="*60)
        print("💡 설정 탭에서 API 키를 입력하고 시스템을 초기화해주세요.")
        print("💡 뉴스 수집 후 저장된 파일은 'collected_news' 폴더에서 확인할 수 있습니다.")
        
        root.mainloop()
        
    except Exception as e:
        print(f"GUI 시작 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()