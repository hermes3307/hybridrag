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
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
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
        """UI 구성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 상단 탭 프레임
        tab_frame = ttk.Frame(main_frame)
        tab_frame.pack(fill=tk.BOTH, expand=True)
        
        # 메인 노트북 (탭)
        self.notebook = ttk.Notebook(tab_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 탭 생성
        self.setup_config_tab(self.notebook)
        self.setup_collection_tab(self.notebook)
        self.setup_writing_tab(self.notebook)
        
        # 하단 로그 프레임 (모든 탭에서 보이도록)
        self.setup_bottom_log_frame(main_frame)
        
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
        status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground="red")
        status_label.pack()
        
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
        self.rag_news_count_var = tk.IntVar(value=10)
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
        
        # 좌우 분할 프레임
        left_frame = ttk.Frame(collection_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(collection_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
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
        
        ttk.Label(stats_grid, text="총 수집:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.total_articles_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.total_articles_var, foreground="blue").grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(stats_grid, text="관련도 높음:").grid(row=0, column=2, sticky=tk.W, padx=15)
        self.relevant_articles_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.relevant_articles_var, foreground="green").grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(stats_grid, text="로컬 저장:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.saved_articles_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.saved_articles_var, foreground="purple").grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="DB 저장:").grid(row=1, column=2, sticky=tk.W, padx=15, pady=2)
        self.db_saved_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.db_saved_var, foreground="red").grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
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
        self.headlines_tree = ttk.Treeview(headlines_scroll_frame, columns=columns, show='tree headings', height=15)
        
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
        
        # 헤드라인 버튼
        headlines_btn_frame = ttk.Frame(headlines_frame)
        headlines_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(headlines_btn_frame, text="새로고침", command=self.refresh_headlines).pack(side=tk.LEFT, padx=5)
        ttk.Button(headlines_btn_frame, text="선택 기사 보기", command=self.view_selected_article).pack(side=tk.LEFT, padx=5)
        ttk.Button(headlines_btn_frame, text="헤드라인 지우기", command=self.clear_headlines).pack(side=tk.LEFT, padx=5)
        ttk.Button(headlines_btn_frame, text="저장된 파일 열기", command=self.open_news_directory).pack(side=tk.LEFT, padx=5)
        
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
        self.topic_var = tk.StringVar(value="기업 신제품 출시")
        ttk.Entry(input_frame, textvariable=self.topic_var, width=50).grid(row=0, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        
        # 키워드
        ttk.Label(input_frame, text="키워드:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.keywords_var = tk.StringVar(value="알티베이스, HyperDB, 인메모리, 성능향상")
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
        self.facts_text = scrolledtext.ScrolledText(input_frame, height=6, width=60)
        self.facts_text.grid(row=4, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        self.facts_text.insert(1.0, "알티베이스가 HyperDB 3.0을 출시했고, 기존 대비 30% 성능이 향상되었다")
        
        # RAG 설정
        rag_frame = ttk.LabelFrame(input_frame, text="RAG 참조 설정", padding=5)
        rag_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=10, sticky=tk.EW)
        
        self.use_rag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rag_frame, text="RAG 참조 사용", variable=self.use_rag_var).pack(side=tk.LEFT)
        
        ttk.Label(rag_frame, text="참조 뉴스 개수:").pack(side=tk.LEFT, padx=10)
        self.rag_count_var = tk.IntVar(value=10)
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
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, font=("맑은 고딕", 10))
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
        
        # 로그 텍스트 (높이를 줄여서 하단에 배치)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=6, font=("Consolas", 9))
        self.log_text.pack(fill=tk.X, pady=5)
        
        # 로그 버튼
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(log_btn_frame, text="로그 지우기", command=lambda: self.log_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_btn_frame, text="로그 저장", command=self.save_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_btn_frame, text="자동 스크롤", command=self.toggle_auto_scroll).pack(side=tk.LEFT, padx=5)
        
        self.auto_scroll = True
        
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
        
    def auto_initialize_system(self):
        """시작 시 자동 시스템 초기화"""
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
                logging.info("시스템이 자동으로 초기화되었습니다.")
                
                # 네이버 API 자동 테스트
                if naver_id and naver_secret:
                    self.root.after(2000, self.test_naver_api)
            else:
                self.status_var.set("API 키를 설정하고 시스템을 초기화해주세요.")
                logging.info("API 키가 설정되지 않아 수동 초기화가 필요합니다.")
                
        except Exception as e:
            logging.error(f"자동 시스템 초기화 실패: {e}")
            self.status_var.set("자동 초기화 실패 - 수동으로 초기화해주세요.")

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
            
            messagebox.showinfo("성공", "시스템이 초기화되었습니다.")
            logging.info("AI News Writer Pro 시스템 초기화 완료")
            
        except Exception as e:
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
                    self.system.write_news(topic, keywords, enhanced_user_facts, style)
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