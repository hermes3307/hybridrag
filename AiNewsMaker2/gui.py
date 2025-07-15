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
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if self.text_widget and self.text_widget.winfo_exists():
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)
                self.text_widget.update()
        except:
            pass  # GUI가 닫혔을 때 오류 방지

class EnhancedNewsWriterGUI:
    def __init__(self, root):
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
        
        # 뉴스 저장 디렉토리
        self.news_directory = "collected_news"
        if not os.path.exists(self.news_directory):
            os.makedirs(self.news_directory)
        
        self.setup_ui()
        self.setup_logging()
        self.load_config()
        
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

        # 탭 생성 (벡터DB 탭 추가)
        self.setup_config_tab(self.notebook)
        self.setup_collection_tab(self.notebook)
        self.setup_writing_tab(self.notebook)
        self.setup_vector_stats_tab(self.notebook)  # NEW: Vector database statistics tab

        # 하단 로그 프레임 (bottom_frame에)
        self.setup_bottom_log_frame(bottom_frame)

        # 벡터DB 탭 선택 시 자동으로 컬렉션 내용 미리보기 표시
        def on_tab_changed(event):
            selected_tab = event.widget.select()
            tab_text = event.widget.tab(selected_tab, "text")
            if "벡터DB" in tab_text:
                self.view_collection_contents()
        self.notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

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
        
        # 버튼 프레임
        button_frame = ttk.Frame(selection_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="전체 선택", 
                command=lambda: self.select_all_news(news_tree, selected_items)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="전체 해제", 
                command=lambda: self.deselect_all_news(news_tree, selected_items)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="선택된 뉴스 벡터DB 추가", 
                command=lambda: self.process_selected_news(selection_window, selected_items)).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="취소", 
                command=selection_window.destroy).pack(side=tk.RIGHT, padx=5)

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
                messagebox.showwarning("경고", "시스템이 초기화되지 않았습니다.")
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
        """컬렉션 내용 보기 (ENHANCED WITH CONTENT VIEWING)"""
        try:
            if not self.system:
                messagebox.showwarning("경고", "시스템이 초기화되지 않았습니다.")
                return
            
            # 기존 항목 지우기
            for item in self.vector_tree.get_children():
                self.vector_tree.delete(item)
            
            # 컬렉션 데이터 저장 (전체 내용 보기를 위해)
            self.vector_full_data = []
            
            # 먼저 컬렉션이 비어있는지 확인
            try:
                collection_count = self.system.db_manager.collection.count()
                if collection_count == 0:
                    logging.info("컬렉션이 비어있습니다.")
                    messagebox.showinfo("알림", "벡터 데이터베이스가 비어있습니다.")
                    return
            except Exception as e:
                logging.error(f"컬렉션 카운트 조회 실패: {e}")
            
            # 컬렉션 데이터 조회
            try:
                all_data = self.system.db_manager.collection.get(
                    include=['documents', 'metadatas']
                )
                
                ids = all_data.get('ids', [])
                documents = all_data.get('documents', [])
                metadatas = all_data.get('metadatas', [])
                
            except Exception as e:
                logging.error(f"컬렉션 데이터 조회 실패: {e}")
                messagebox.showerror("오류", f"컬렉션 데이터 조회 실패: {e}")
                return
            
            if not documents:
                logging.info("컬렉션에 문서가 없습니다.")
                messagebox.showinfo("알림", "벡터 데이터베이스에 저장된 문서가 없습니다.")
                return
            
            # 트리뷰에 데이터 추가 및 전체 데이터 저장
            max_display = min(300, len(documents))  # 300개로 확장
            
            for i in range(max_display):
                try:
                    doc = documents[i]
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    doc_id = ids[i] if i < len(ids) else f"unknown_{i}"
                    
                    # ✅ NEW: 전체 데이터 저장 (클릭시 보기용)
                    full_item_data = {
                        'id': doc_id,
                        'document': doc,
                        'metadata': metadata,
                        'index': i
                    }
                    self.vector_full_data.append(full_item_data)
                    
                    # 토픽 파싱 (안전하게)
                    try:
                        topics_raw = metadata.get('topics', '[]')
                        if isinstance(topics_raw, str):
                            topics = json.loads(topics_raw)
                        else:
                            topics = topics_raw if isinstance(topics_raw, list) else []
                        topics_str = ', '.join(topics[:2]) if topics else 'N/A'
                    except:
                        topics_str = str(metadata.get('topics', 'N/A'))[:20]
                    
                    # 내용 미리보기
                    try:
                        content_preview = doc[:50] + "..." if len(doc) > 50 else doc
                        content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')
                    except:
                        content_preview = "내용 없음"
                    
                    # 관련도 처리
                    try:
                        relevance = metadata.get('relevance_score', 'N/A')
                        if isinstance(relevance, (int, float)):
                            relevance_str = f"{relevance}/10"
                        else:
                            relevance_str = str(relevance)
                    except:
                        relevance_str = "N/A"
                    
                    # 날짜 처리
                    date_str = metadata.get('date', metadata.get('created_at', 'N/A'))
                    if isinstance(date_str, str) and 'T' in date_str:
                        date_str = date_str.split('T')[0]
                    
                    # 트리뷰에 추가
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
                    
                except Exception as item_error:
                    logging.warning(f"항목 {i} 처리 실패: {item_error}")
                    # 오류 데이터도 저장
                    error_data = {
                        'id': f'error_{i}',
                        'document': f'처리 오류: {str(item_error)}',
                        'metadata': {'error': True},
                        'index': i
                    }
                    self.vector_full_data.append(error_data)
                    
                    # 오류가 있는 항목은 기본값으로 표시
                    self.vector_tree.insert('', 'end',
                        text=str(i+1),
                        values=(f"error_{i}", "처리 오류", "N/A", "N/A", "N/A")
                    )
            
            # ✅ NEW: 더블클릭 이벤트 핸들러 바인딩
            self.vector_tree.bind('<Double-1>', self.on_vector_item_double_click)
            
            logging.info(f"컬렉션 내용 표시 완료: {max_display}개 항목 (전체 {len(documents)}개)")
            
            if len(documents) > 300:
                messagebox.showinfo("알림", f"총 {len(documents)}개 항목 중 처음 300개만 표시됩니다.\n\n💡 팁: 항목을 더블클릭하면 전체 내용을 볼 수 있습니다.")
            else:
                messagebox.showinfo("표시 완료", f"총 {len(documents)}개 항목이 표시되었습니다.\n\n💡 팁: 항목을 더블클릭하면 전체 내용을 볼 수 있습니다.")
            
        except Exception as e:
            error_msg = f"컬렉션 내용 조회 실패: {e}"
            messagebox.showerror("오류", error_msg)
            logging.error(error_msg)

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

    def show_vector_content_detail(self, item_data):
        """벡터 컨텐츠 상세 보기 창 (NEW FUNCTION)"""
        try:
            # 새 창 생성
            detail_window = tk.Toplevel(self.root)
            detail_window.title(f"벡터 데이터 상세 보기 - {item_data['id'][:30]}...")
            detail_window.geometry("900x700")
            detail_window.transient(self.root)
            
            # 메뉴바 추가
            menubar = tk.Menu(detail_window)
            detail_window.config(menu=menubar)
            
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="파일", menu=file_menu)
            file_menu.add_command(label="내용 저장", command=lambda: self.save_vector_content(item_data))
            file_menu.add_command(label="클립보드 복사", command=lambda: self.copy_vector_content_to_clipboard(item_data))
            file_menu.add_command(label="메타데이터 내보내기", command=lambda: self.export_metadata(item_data))
            
            # 상단 정보 프레임
            info_frame = ttk.LabelFrame(detail_window, text="기본 정보", padding=10)
            info_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # 기본 정보 표시
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
            
            # 토픽 및 키워드
            ttk.Label(info_grid, text="토픽:", font=("", 9, "bold")).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            try:
                topics = json.loads(metadata.get('topics', '[]')) if isinstance(metadata.get('topics'), str) else metadata.get('topics', [])
                topics_text = ', '.join(topics) if topics else 'N/A'
            except:
                topics_text = str(metadata.get('topics', 'N/A'))
            ttk.Label(info_grid, text=topics_text).grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)
            
            # 키워드
            ttk.Label(info_grid, text="키워드:", font=("", 9, "bold")).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
            try:
                keywords = json.loads(metadata.get('keywords', '[]')) if isinstance(metadata.get('keywords'), str) else metadata.get('keywords', [])
                keywords_text = ', '.join(keywords) if keywords else 'N/A'
            except:
                keywords_text = str(metadata.get('keywords', 'N/A'))
            ttk.Label(info_grid, text=keywords_text).grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)
            
            # 내용 프레임
            content_frame = ttk.LabelFrame(detail_window, text="전체 내용", padding=10)
            content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # 스크롤 가능한 텍스트 위젯
            content_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, font=("맑은 고딕", 10))
            content_text.pack(fill=tk.BOTH, expand=True)
            
            # 전체 내용 표시
            full_content = f"""=== 벡터 데이터베이스 저장 내용 ===

{item_data['document']}

=== 메타데이터 정보 ===
"""
            
            # 메타데이터를 보기 좋게 정리
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
            content_text.config(state=tk.DISABLED)  # 읽기 전용
            
            # 하단 버튼
            button_frame = ttk.Frame(detail_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Button(button_frame, text="내용 저장", command=lambda: self.save_vector_content(item_data)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="클립보드 복사", command=lambda: self.copy_vector_content_to_clipboard(item_data)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="닫기", command=detail_window.destroy).pack(side=tk.RIGHT, padx=5)
            
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
            messagebox.showwarning("경고", "시스템이 초기화되지 않았습니다.")
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
            claude_key = self.claude_key_var.get().strip()
            naver_id = self.naver_id_var.get().strip()
            naver_secret = self.naver_secret_var.get().strip()
            
            # API 키가 있으면 자동 초기화
            if claude_key or (naver_id and naver_secret):
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
                
                self.status_var.set("시스템 자동 초기화 완료 - " + " | ".join(status_parts))
                self.status_label_widget.config(foreground="green")
                logging.info("시스템이 자동으로 초기화되었습니다.")
                
                # 벡터 데이터베이스 통계 자동 로드 (NEW)
                self.root.after(3000, self.refresh_vector_stats)  # 3초 후 통계 로드
                
                # 네이버 API 자동 테스트
                if naver_id and naver_secret:
                    self.root.after(2000, self.test_naver_api)
            else:
                self.status_var.set("API 키를 설정하고 시스템을 초기화해주세요.")
                self.status_label_widget.config(foreground="green")
                logging.info("API 키가 설정되지 않아 수동 초기화가 필요합니다.")
                
        except Exception as e:
            logging.error(f"자동 시스템 초기화 실패: {e}")
            self.status_var.set("자동 초기화 실패 - 수동으로 초기화해주세요.")
            self.status_label_widget.config(foreground="red")


    def setup_config_tab(self, parent):
        """설정 탭"""
        config_frame = ttk.Frame(parent)
        parent.add(config_frame, text="🔧 설정")
        
        # 스크롤 가능한 프레임
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # API 키 설정 프레임
        api_frame = ttk.LabelFrame(scrollable_frame, text="API 키 설정", padding=10)
        api_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Claude API Key
        ttk.Label(api_frame, text="Claude API Key:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.claude_key_var = tk.StringVar()
        claude_entry = ttk.Entry(api_frame, textvariable=self.claude_key_var, show="*", width=60)
        claude_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # Naver Client ID
        ttk.Label(api_frame, text="네이버 Client ID:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.naver_id_var = tk.StringVar()
        naver_id_entry = ttk.Entry(api_frame, textvariable=self.naver_id_var, width=60)
        naver_id_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # Naver Client Secret
        ttk.Label(api_frame, text="네이버 Client Secret:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.naver_secret_var = tk.StringVar()
        naver_secret_entry = ttk.Entry(api_frame, textvariable=self.naver_secret_var, show="*", width=60)
        naver_secret_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # 버튼 프레임
        btn_frame = ttk.Frame(api_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="설정 저장", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="설정 불러오기", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="시스템 초기화", command=self.initialize_system).pack(side=tk.LEFT, padx=5)
        
        # 상태 표시 프레임
        status_frame = ttk.LabelFrame(scrollable_frame, text="시스템 상태", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_var = tk.StringVar(value="시스템이 초기화되지 않았습니다.")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground="green")
        status_label.pack()
        self.status_label_widget = status_label  # 상태 라벨 위젯 참조 저장
        
        # 회사 및 키워드 설정 프레임 (개선됨)
        company_frame = ttk.LabelFrame(scrollable_frame, text="대상 회사 및 키워드 설정", padding=10)
        company_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 회사명
        ttk.Label(company_frame, text="회사명:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.company_var = tk.StringVar(value="알티베이스")
        ttk.Entry(company_frame, textvariable=self.company_var, width=40).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        # 추가 키워드 (새로 추가)
        ttk.Label(company_frame, text="추가 키워드 (,구분):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.additional_keywords_var = tk.StringVar(value="데이터베이스, DBMS, 오라클")
        keyword_entry = ttk.Entry(company_frame, textvariable=self.additional_keywords_var, width=60)
        keyword_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # 키워드 도움말
        help_label = ttk.Label(company_frame, text="※ 회사명과 추가 키워드를 조합하여 더 정확한 뉴스를 검색합니다", 
                              foreground="gray", font=("", 8))
        help_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # 고급 설정 프레임
        advanced_frame = ttk.LabelFrame(scrollable_frame, text="고급 설정", padding=10)
        advanced_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # RAG 관련 뉴스 개수 설정
        ttk.Label(advanced_frame, text="RAG 참조 뉴스 개수:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.rag_news_count_var = tk.IntVar(value=15)
        ttk.Spinbox(advanced_frame, from_=5, to=20, textvariable=self.rag_news_count_var, width=10).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(advanced_frame, text="개").grid(row=0, column=2, sticky=tk.W)
        
        # 뉴스 저장 위치
        ttk.Label(advanced_frame, text="뉴스 저장 폴더:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.news_dir_var = tk.StringVar(value=self.news_directory)
        ttk.Entry(advanced_frame, textvariable=self.news_dir_var, width=40).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Button(advanced_frame, text="폴더 선택", command=self.select_news_directory).grid(row=1, column=2, padx=5, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
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
        self.manual_text = scrolledtext.ScrolledText(manual_frame, height=8, wrap=tk.WORD)
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
        columns = ('title', 'date', 'preview')
        self.headlines_tree = ttk.Treeview(headlines_scroll_frame, columns=columns, show='tree headings', height=5)
        
        self.headlines_tree.heading('#0', text='번호')
        self.headlines_tree.heading('title', text='제목')
        self.headlines_tree.heading('date', text='날짜')
        self.headlines_tree.heading('preview', text='미리보기')
        
        self.headlines_tree.column('#0', width=50, minwidth=50)
        self.headlines_tree.column('title', width=300, minwidth=200)
        self.headlines_tree.column('date', width=100, minwidth=80)
        self.headlines_tree.column('preview', width=200, minwidth=150)
        
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
        
        # 상단 통계 요약
        stats_summary_frame = ttk.LabelFrame(vector_frame, text="벡터 데이터베이스 요약", padding=10)
        stats_summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 통계 정보 표시
        stats_grid = ttk.Frame(stats_summary_frame)
        stats_grid.pack(fill=tk.X, pady=5)
        
        ttk.Label(stats_grid, text="총 청크 수:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.vector_total_chunks_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.vector_total_chunks_var, foreground="blue").grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(stats_grid, text="컬렉션명:").grid(row=0, column=2, sticky=tk.W, padx=15)
        self.vector_collection_name_var = tk.StringVar(value="unknown")
        ttk.Label(stats_grid, textvariable=self.vector_collection_name_var, foreground="green").grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(stats_grid, text="마지막 업데이트:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.vector_last_update_var = tk.StringVar(value="N/A")
        ttk.Label(stats_grid, textvariable=self.vector_last_update_var, foreground="purple").grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="평균 관련도:").grid(row=1, column=2, sticky=tk.W, padx=15, pady=2)
        self.vector_avg_relevance_var = tk.StringVar(value="N/A")
        ttk.Label(stats_grid, textvariable=self.vector_avg_relevance_var, foreground="red").grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # 제어 버튼
        control_frame = ttk.Frame(stats_summary_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="📊 통계 새로고침", command=self.refresh_vector_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🔍 컬렉션 내용 보기", command=self.view_collection_contents).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🗑️ 벡터DB 초기화", command=self.clear_vector_db).pack(side=tk.LEFT, padx=5)
        # 삭제: 벡터DB 내보내기 버튼
        # 삭제: 임베딩 차원 확인 버튼
        
        # 컬렉션 내용 표시
        content_frame = ttk.LabelFrame(vector_frame, text="컬렉션 내용 미리보기", padding=10)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 트리뷰로 벡터DB 내용 표시
        columns = ('id', 'content_preview', 'topics', 'relevance', 'date')
        self.vector_tree = ttk.Treeview(content_frame, columns=columns, show='tree headings', height=5)
        
        self.vector_tree.heading('#0', text='순번')
        self.vector_tree.heading('id', text='청크 ID')
        self.vector_tree.heading('content_preview', text='내용 미리보기')
        self.vector_tree.heading('topics', text='토픽')
        self.vector_tree.heading('relevance', text='관련도')
        self.vector_tree.heading('date', text='날짜')
        
        self.vector_tree.column('#0', width=50, minwidth=50)
        self.vector_tree.column('id', width=100, minwidth=80)
        self.vector_tree.column('content_preview', width=300, minwidth=200)
        self.vector_tree.column('topics', width=100, minwidth=80)
        self.vector_tree.column('relevance', width=80, minwidth=60)
        self.vector_tree.column('date', width=100, minwidth=80)
        
        # 스크롤바
        vector_scrollbar = ttk.Scrollbar(content_frame, orient=tk.VERTICAL, command=self.vector_tree.yview)
        self.vector_tree.configure(yscrollcommand=vector_scrollbar.set)
        
        self.vector_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vector_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ✅ ADD: Embedding diagnostics button
        ttk.Button(control_frame, text="🔧 임베딩 차원 확인", command=self.fix_embedding_dimensions).pack(side=tk.LEFT, padx=5)

    # ✅ ENHANCED: Add collection reset functionality to GUI
    def fix_embedding_dimensions(self):
        """임베딩 차원 문제 해결 (NEW GUI METHOD)"""
        if not self.system:
            messagebox.showwarning("경고", "시스템이 초기화되지 않았습니다.")
            return
        
        # 현재 임베딩 정보 확인
        embed_info = self.system.db_manager.get_embedding_info()
        
        if embed_info.get("sample_embedding_dim") != 768:
            # 임베딩 차원 불일치 - 재설정 제안
            if messagebox.askyesno("임베딩 차원 불일치", 
                                f"현재 임베딩 차원: {embed_info.get('sample_embedding_dim')}\n"
                                f"필요한 차원: 768\n\n"
                                f"벡터 데이터베이스를 768차원으로 재설정하시겠습니까?\n"
                                f"(기존 데이터는 보존되지만 새로운 임베딩으로 변환됩니다)"):
                
                try:
                    success = self.system.db_manager.reset_collection_with_proper_embeddings()
                    if success:
                        messagebox.showinfo("완료", "벡터 데이터베이스가 768차원으로 재설정되었습니다.")
                        self.refresh_vector_stats()
                    else:
                        messagebox.showerror("실패", "벡터 데이터베이스 재설정에 실패했습니다.")
                except Exception as e:
                    messagebox.showerror("오류", f"재설정 중 오류 발생: {e}")
        else:
            messagebox.showinfo("정상", "임베딩 차원이 정상입니다 (768차원).")
            
    def setup_writing_tab(self, parent):
        """뉴스 작성 탭 (개선됨)"""
        writing_frame = ttk.Frame(parent)
        parent.add(writing_frame, text="✍️ 뉴스 작성")
        
        # 좌우 분할
        left_writing_frame = ttk.Frame(writing_frame)
        left_writing_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_writing_frame = ttk.Frame(writing_frame)
        right_writing_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 좌측: 입력 설정
        input_frame = ttk.LabelFrame(left_writing_frame, text="뉴스 작성 입력", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # 토픽
        ttk.Label(input_frame, text="토픽:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.topic_var = tk.StringVar(value="알티베이스, 3년간 단계별 기술 로드맵 발표...2027년 차세대 클러스터 출시")
        ttk.Entry(input_frame, textvariable=self.topic_var, width=50).grid(row=0, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        
        # 키워드
        ttk.Label(input_frame, text="키워드:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.keywords_var = tk.StringVar(value="알티베이스, 차세대 기술 로드맵, 클러스터")
        ttk.Entry(input_frame, textvariable=self.keywords_var, width=50).grid(row=1, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        
        # 스타일
        ttk.Label(input_frame, text="스타일:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.style_var = tk.StringVar(value="기업 보도형")
        style_combo = ttk.Combobox(input_frame, textvariable=self.style_var, 
                                  values=["기업 보도형", "분석형", "인터뷰형", "발표형", "기술 리뷰형"], width=20)
        style_combo.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        
        # 뉴스 길이 설정 (새로 추가)
        ttk.Label(input_frame, text="뉴스 길이:").grid(row=3, column=0, sticky=tk.W, pady=2)
        
        length_frame = ttk.Frame(input_frame)
        length_frame.grid(row=3, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        
        self.length_type_var = tk.StringVar(value="줄 수")
        ttk.Radiobutton(length_frame, text="줄 수", variable=self.length_type_var, value="줄 수").pack(side=tk.LEFT)
        ttk.Radiobutton(length_frame, text="단어 수", variable=self.length_type_var, value="단어 수").pack(side=tk.LEFT, padx=10)
        
        self.length_count_var = tk.IntVar(value=100)  # 기본값 100줄
        ttk.Spinbox(length_frame, from_=10, to=500, textvariable=self.length_count_var, width=10).pack(side=tk.LEFT, padx=10)
        
        # 사용자 사실
        ttk.Label(input_frame, text="주요 사실:").grid(row=4, column=0, sticky=tk.NW, pady=2)
        self.facts_text = scrolledtext.ScrolledText(input_frame, height=12, width=60)  # 12줄로 확장
        self.facts_text.grid(row=4, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        self.facts_text.insert(1.0, "알티베이스는 국내외 시장의 급변하는 요구사항과 AI 기술 발전 추세를 반영해 4대 핵심 기술을 중심으로 한 중장기 기술 로드맵을 수립했다고 7일 밝혔다. 이번 로드맵은 △인메모리 고성능 기술 △멀티 데이터모델 △AI 에이전트·벡터 데이터베이스 기능 △차세대 클러스터 기술 등으로 구성되며, AI 시대에 맞는 데이터베이스 기술 혁신을 목표로 한다.")
        
        # RAG 설정
        rag_frame = ttk.LabelFrame(input_frame, text="RAG 참조 설정", padding=5)
        rag_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=10, sticky=tk.EW)
        
        self.use_rag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rag_frame, text="RAG 참조 사용", variable=self.use_rag_var).pack(side=tk.LEFT)
        
        ttk.Label(rag_frame, text="참조 뉴스 개수:").pack(side=tk.LEFT, padx=10)
        self.rag_count_var = tk.IntVar(value=15)
        ttk.Spinbox(rag_frame, from_=5, to=20, textvariable=self.rag_count_var, width=8).pack(side=tk.LEFT)
        
        # 생성 버튼
        generate_frame = ttk.Frame(input_frame)
        generate_frame.grid(row=6, column=0, columnspan=3, pady=15)
        
        self.generate_btn = ttk.Button(generate_frame, text="🚀 뉴스 생성", command=self.generate_news)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(generate_frame, text="📋 템플릿 불러오기", command=self.load_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(generate_frame, text="💾 템플릿 저장", command=self.save_template).pack(side=tk.LEFT, padx=5)
        
        # 우측: 결과 표시
        result_frame = ttk.LabelFrame(right_writing_frame, text="생성된 뉴스", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, font=("맑은 고딕", 10), height=5)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 결과 버튼
        result_btn_frame = ttk.Frame(result_frame)
        result_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(result_btn_frame, text="📁 파일로 저장", command=self.save_news).pack(side=tk.LEFT, padx=5)
        ttk.Button(result_btn_frame, text="📋 클립보드 복사", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(result_btn_frame, text="🔄 다시 생성", command=self.regenerate_news).pack(side=tk.LEFT, padx=5)
        ttk.Button(result_btn_frame, text="❌ 결과 지우기", command=lambda: self.result_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        
        # 품질 평가 표시
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
        log_frame.pack(fill=tk.X, pady=5)
        
        # 로그 텍스트 (높이를 8로 늘림)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8, font=("Consolas", 9))
        self.log_text.pack(fill=tk.X, pady=(5,0))
        
        # 버튼 삭제: 로그 지우기, 로그 저장, 자동 스크롤

    def setup_logging(self):
        """로깅 설정"""
        # 기존 핸들러 제거
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # GUI 핸들러 추가
        gui_handler = GUILogHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(gui_handler)
        root_logger.setLevel(logging.INFO)
        
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
            self.api_status_var.set("❌ 시스템이 초기화되지 않았습니다")
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
    
    def start_collection(self):
        """뉴스 수집 시작 (개선됨)"""
        if not self.system:
            messagebox.showwarning("경고", "먼저 시스템을 초기화해주세요.")
            return
        
        if self.is_collecting:
            messagebox.showwarning("경고", "이미 수집이 진행 중입니다.")
            return
        
        self.is_collecting = True
        self.collect_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress.start()
        
        # 별도 스레드에서 수집 실행
        def collection_worker():
            total_collected = 0
            relevant_collected = 0
            saved_collected = 0
            db_saved_collected = 0
            
            try:
                company = self.company_var.get()
                days = self.days_var.get()
                max_articles = self.max_articles_var.get()
                
                logging.info(f"{company} 뉴스 수집 시작 (최근 {days}일, 최대 {max_articles}개)")
                
                # 수집 통계 초기화
                self.root.after(0, lambda: self.update_statistics(0, 0, 0, 0))
                self.root.after(0, lambda: self.clear_headlines())
                
                # asyncio 루프 실행
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
                        
                        # 네이버 뉴스 검색
                        articles = self.system.naver_api.search_news(query, display=min(10, articles_per_query))
                        total_collected += len(articles)
                        
                        for article_idx, article in enumerate(articles):
                            if not self.is_collecting or saved_collected >= max_articles:
                                break
                                
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
                                
                                # DB 저장 시도
                                try:
                                    success = loop.run_until_complete(
                                        self.system.news_collector.collect_and_store_news(company, article)
                                    )
                                    if success:
                                        db_saved_collected += 1
                                        relevant_collected += 1
                                except Exception as e:
                                    logging.warning(f"DB 저장 실패: {e}")
                                
                                # 통계 업데이트
                                self.root.after(0, lambda: self.update_statistics(
                                    total_collected, relevant_collected, saved_collected, db_saved_collected
                                ))
                            
                            # API 호출 제한
                            loop.run_until_complete(asyncio.sleep(1))
                        
                        # 쿼리 간 딜레이
                        loop.run_until_complete(asyncio.sleep(2))
                        
                    except Exception as e:
                        logging.error(f"뉴스 수집 중 오류 ({query}): {e}")
                        
                loop.close()
                
                # 수집 완료 처리
                self.root.after(0, lambda: self.collection_complete(saved_collected, db_saved_collected))
                
            except Exception as e:
                self.root.after(0, lambda: self.collection_error(str(e)))
        
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
                values=(article_info['title'], formatted_date, preview)
            )
            
            # 자동 스크롤
            if self.auto_scroll:
                self.headlines_tree.see(item_id)
                
        except Exception as e:
            logging.error(f"헤드라인 추가 실패: {e}")
    
    def stop_collection(self):
        """뉴스 수집 중지"""
        self.is_collecting = False
        self.collect_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()
        logging.info("뉴스 수집이 중지되었습니다.")
    
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
        """수집 오류 처리"""
        self.is_collecting = False
        self.collect_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()
        
        messagebox.showerror("오류", f"뉴스 수집 실패: {error_msg}")
        logging.error(f"뉴스 수집 실패: {error_msg}")
    
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
        """뉴스 생성 (개선된 RAG 포함)"""
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
        
        # 뉴스 길이 설정
        length_type = self.length_type_var.get()
        length_count = self.length_count_var.get()
        
        def generation_worker():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # RAG 사용 여부 확인
                if self.use_rag_var.get():
                    # 개선된 RAG: 관련 뉴스 10개 찾기
                    rag_count = self.rag_count_var.get()
                    search_query = f"{topic} {' '.join(keywords)}"
                    
                    logging.info(f"RAG 검색 중: '{search_query}' (상위 {rag_count}개)")
                    search_results = self.system.db_manager.search_relevant_news(search_query, n_results=rag_count)
                    
                    # 참고 자료 구성 (전체 내용 포함)
                    reference_materials = self.build_enhanced_reference_materials(search_results)
                else:
                    reference_materials = "참고 자료를 사용하지 않습니다."
                
                # 길이 설정을 포함한 사용자 사실 업데이트
                enhanced_user_facts = f"{user_facts}\n\n[생성 설정]\n- 스타일: {style}\n- 길이: {length_count} {length_type}"
                
                # 뉴스 생성
                self.root.after(0, lambda: self.update_generation_status("뉴스 생성 중..."))
                
                news = loop.run_until_complete(
                    self.system.write_news(
                        topic, keywords, enhanced_user_facts, style,
                        length_specification=f"{length_count}{length_type}",  # 예: "10줄", "100줄"
                        use_rag=self.use_rag_var.get(),
                        rag_count=self.rag_count_var.get()
                    )
                )
                
                # 길이 조정 (필요시)
                if news and length_type == "줄 수":
                    news = self.adjust_news_length_by_lines(news, length_count)
                elif news and length_type == "단어 수":
                    news = self.adjust_news_length_by_words(news, length_count)
                
                loop.close()
                
                if news:
                    self.root.after(0, lambda: self.show_generated_news(news))
                    self.root.after(0, lambda: self.evaluate_news_quality(news))
                else:
                    self.root.after(0, lambda: messagebox.showerror("실패", "뉴스 생성에 실패했습니다."))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("오류", f"뉴스 생성 오류: {e}"))
                self.root.after(0, lambda: logging.error(f"뉴스 생성 오류: {e}"))
        
        # 생성 중 표시
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, "뉴스 생성 중입니다. 잠시만 기다려주세요...\n\n")
        self.generate_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=generation_worker, daemon=True).start()
    
    def build_enhanced_reference_materials(self, search_results):
        """개선된 참고 자료 구성 (전체 내용 포함)"""
        if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
            return "관련 참고 자료가 없습니다."
        
        materials = []
        documents = search_results['documents'][0]
        metadatas = search_results.get('metadatas', [[]])[0] if search_results.get('metadatas') else []
        
        for i, doc in enumerate(documents[:10]):  # 최대 10개
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # 메타데이터에서 추가 정보 추출
            source = metadata.get('source', f'참고자료 {i+1}')
            date = metadata.get('date', 'N/A')
            importance = metadata.get('importance', 'N/A')
            
            # 전체 내용 포함
            material = f"""=== {source} ({date}) ===
중요도: {importance}
내용: {doc}

"""
            materials.append(material)
        
        reference_text = "\n".join(materials) if materials else "관련 참고 자료가 없습니다."
        logging.info(f"RAG 참고 자료 구성 완료: {len(materials)}개 문서")
        
        return reference_text
    
    def adjust_news_length_by_lines(self, news, target_lines):
        """줄 수 기준으로 뉴스 길이 조정"""
        lines = news.split('\n')
        current_lines = len([line for line in lines if line.strip()])
        
        if current_lines < target_lines:
            # 길이 부족시 확장 요청
            expansion_note = f"\n\n[편집자 주: 현재 {current_lines}줄입니다. {target_lines}줄로 확장이 필요합니다.]"
            return news + expansion_note
        elif current_lines > target_lines * 1.2:  # 20% 이상 초과시만 축약
            # 주요 섹션만 유지하여 축약
            shortened_lines = lines[:int(target_lines * 0.8)]
            return '\n'.join(shortened_lines) + f"\n\n[편집됨: {target_lines}줄로 축약]"
        
        return news
    
    def adjust_news_length_by_words(self, news, target_words):
        """단어 수 기준으로 뉴스 길이 조정"""
        words = news.split()
        current_words = len(words)
        
        if current_words < target_words:
            expansion_note = f"\n\n[편집자 주: 현재 {current_words}단어입니다. {target_words}단어로 확장이 필요합니다.]"
            return news + expansion_note
        elif current_words > target_words * 1.2:
            shortened_words = words[:int(target_words * 0.9)]
            return ' '.join(shortened_words) + f"\n\n[편집됨: {target_words}단어로 축약]"
        
        return news
    
    def update_generation_status(self, status):
        """생성 상태 업데이트"""
        current_content = self.result_text.get(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, f"{status}\n\n{current_content}")
    
    def show_generated_news(self, news):
        """생성된 뉴스 표시"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, news)
        self.generate_btn.config(state=tk.NORMAL)
        logging.info("뉴스 생성 완료")
    
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


def main():
    """GUI 메인 함수"""
    try:
        root = tk.Tk()
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