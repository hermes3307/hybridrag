import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import datetime
import random
import threading
import webbrowser
import requests
import json
import time
import pandas as pd
from urllib.parse import urlencode
import base64

# --- Data Structures from Original GUI ---
BUILDING_TYPES = {
    "residential": {
        "display": "주거용 부동산",
        "subcategories": {
            "apartments": {"display": "아파트", "code": "A01"},
            "residential_commercial": {"display": "주상복합", "code": "A03"},
            "reconstruction": {"display": "재건축", "code": "A04"},
            "officetels": {"display": "오피스텔", "code": "A02"},
            "villas": {"display": "빌라", "code": "B01"},
            "detached_houses": {"display": "단독/다가구", "code": "B02"},
            "onerooms": {"display": "원룸", "code": "C01"},
        }
    },
    "commercial": {
        "display": "상업용 부동산",
        "subcategories": {
            "retail_spaces": {"display": "상가", "code": "SG"},
            "offices": {"display": "사무실", "code": "SMS"},
            "factories_warehouses": {"display": "공장/창고", "code": "GJCG"},
            "buildings": {"display": "건물", "code": "GM"},
            "knowledge_centers": {"display": "지식산업센터", "code": "APTHGJ"}
        }
    },
    "land": {
        "display": "토지",
        "subcategories": {
            "land": {"display": "토지", "code": "TJ"}
        }
    }
}

DEAL_TYPES = {
    'sale': {'display': '매매', 'code': 'A1'},
    'jeonse': {'display': '전세', 'code': 'B1'},
    'rent': {'display': '월세', 'code': 'B2'},
}

# Seoul districts with their geographic coordinates for coordinate-based search
SEOUL_COORDINATES = {
    "강남구": {"lat": 37.5173, "lng": 127.0473, "cortarNo": "1168000000"},
    "강동구": {"lat": 37.5301, "lng": 127.1238, "cortarNo": "1174000000"},
    "강북구": {"lat": 37.6397, "lng": 127.0256, "cortarNo": "1130500000"},
    "강서구": {"lat": 37.5509, "lng": 126.8495, "cortarNo": "1150000000"},
    "관악구": {"lat": 37.4784, "lng": 126.9516, "cortarNo": "1162000000"},
    "광진구": {"lat": 37.5385, "lng": 127.0823, "cortarNo": "1121500000"},
    "구로구": {"lat": 37.4955, "lng": 126.8866, "cortarNo": "1153000000"},
    "금천구": {"lat": 37.4563, "lng": 126.8956, "cortarNo": "1154500000"},
    "노원구": {"lat": 37.6542, "lng": 127.0568, "cortarNo": "1135000000"},
    "도봉구": {"lat": 37.6688, "lng": 127.0471, "cortarNo": "1132000000"},
    "동대문구": {"lat": 37.5744, "lng": 127.0396, "cortarNo": "1123000000"},
    "동작구": {"lat": 37.5124, "lng": 126.9393, "cortarNo": "1159000000"},
    "마포구": {"lat": 37.5637, "lng": 126.9084, "cortarNo": "1144000000"},
    "서대문구": {"lat": 37.5791, "lng": 126.9368, "cortarNo": "1141000000"},
    "서초구": {"lat": 37.4837, "lng": 127.0324, "cortarNo": "1165000000"},
    "성동구": {"lat": 37.5634, "lng": 127.0365, "cortarNo": "1120000000"},
    "성북구": {"lat": 37.5894, "lng": 127.0167, "cortarNo": "1129000000"},
    "송파구": {"lat": 37.5145, "lng": 127.1059, "cortarNo": "1171000000"},
    "양천구": {"lat": 37.5170, "lng": 126.8665, "cortarNo": "1147000000"},
    "영등포구": {"lat": 37.5264, "lng": 126.8962, "cortarNo": "1156000000"},
    "용산구": {"lat": 37.5324, "lng": 126.9910, "cortarNo": "1117000000"},
    "은평구": {"lat": 37.6026, "lng": 126.9291, "cortarNo": "1138000000"},
    "종로구": {"lat": 37.5735, "lng": 126.9791, "cortarNo": "1111000000"},
    "중구": {"lat": 37.5640, "lng": 126.9970, "cortarNo": "1114000000"},
    "중랑구": {"lat": 37.6063, "lng": 127.0925, "cortarNo": "1126000000"}
}

REGIONS = {
    "서울특별시": list(SEOUL_COORDINATES.keys()),
    "경기도": [
        "수원시", "성남시", "안양시", "안산시", "용인시", "부천시", "평택시", "화성시",
        "시흥시", "광명시", "김포시", "군포시", "하남시", "오산시", "양주시", "구리시"
    ],
    "인천광역시": [
        "중구", "동구", "미추홀구", "연수구", "남동구", "부평구", "계양구", "서구", "강화군", "옹진군"
    ]
}

# ==============================================================================
# MODULE 1: ADVANCED REGION FINDER WITH MULTIPLE STRATEGIES
# ==============================================================================
class AdvancedRegionFinder:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://land.naver.com/',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        })
        self.region_cache = {}
        self.session_initialized = False

    def _init_session(self):
        """Initialize session by visiting the main page"""
        if self.session_initialized:
            return True
        try:
            response = self.session.get('https://land.naver.com/', timeout=10)
            if response.status_code == 200:
                self.session_initialized = True
                time.sleep(1)
                return True
        except Exception as e:
            print(f"Session initialization failed: {e}")
        return False

    def get_coordinates_for_address(self, address: str):
        """Get coordinates for Seoul districts directly"""
        parts = address.split()
        if len(parts) >= 2 and parts[0] in ["서울특별시", "서울시", "서울"]:
            district = parts[1]
            if district in SEOUL_COORDINATES:
                return SEOUL_COORDINATES[district]
        return None

# ==============================================================================
# MODULE 2: ENHANCED PROPERTY SCRAPER WITH MULTIPLE ENDPOINT STRATEGIES
# ==============================================================================
class AdvancedNaverLandScraper:
    def __init__(self, region_finder: AdvancedRegionFinder):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://land.naver.com/',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        self.region_finder = region_finder
        self.endpoints = [
            'https://new.land.naver.com/api/articles',
            'https://m.land.naver.com/cluster/ajax/articleList',
            'https://fin.land.naver.com/front-api/v1/article/list'
        ]
        self.current_endpoint = 0

    def _establish_session(self):
        """Establish a valid session with cookies"""
        try:
            # Visit main page
            main_response = self.session.get('https://land.naver.com/', timeout=10)
            if main_response.status_code != 200:
                return False
            
            # Visit new land page
            new_response = self.session.get('https://new.land.naver.com/', timeout=10)
            if new_response.status_code != 200:
                return False
                
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Session establishment failed: {e}")
            return False

    def _try_coordinate_search(self, coordinates, criteria):
        """Try coordinate-based search (mobile endpoint strategy)"""
        lat, lng = coordinates['lat'], coordinates['lng']
        
        # Create coordinate bounds (roughly 2km x 2km area)
        delta = 0.009  # Approximately 1km
        bounds = {
            'btm': lat - delta,
            'lft': lng - delta,
            'top': lat + delta,
            'rgt': lng + delta
        }
        
        mobile_url = 'https://m.land.naver.com/cluster/ajax/articleList'
        params = {
            'rletTpCd': criteria['building_code'],
            'tradTpCd': criteria['deal_type_code'],
            'z': '14',  # Zoom level
            'cortarNo': coordinates.get('cortarNo', ''),
            **bounds
        }
        
        try:
            response = self.session.get(mobile_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'body' in data and data['body']:
                    return data['body']
        except Exception as e:
            print(f"Coordinate search failed: {e}")
        
        return None

    def _try_traditional_search(self, criteria):
        """Try traditional cortarNo-based search"""
        endpoint = self.endpoints[self.current_endpoint]
        params = {
            'cortarNo': criteria.get('cortar_no', '1100000000'),  # Default to Seoul
            'order': 'rank',
            'realEstateType': criteria['building_code'],
            'tradeType': criteria['deal_type_code'],
            'page': 1
        }
        
        # Add price filters if specified
        if criteria.get('price_max', 0) > 0:
            params['priceMax'] = criteria['price_max']
        if criteria.get('min_area', 0) > 0:
            params['areaMin'] = criteria['min_area']
        if criteria.get('max_area', 0) > 0:
            params['areaMax'] = criteria['max_area']
        
        try:
            response = self.session.get(endpoint, params=params, timeout=15)
            print(f"Trying endpoint: {endpoint}")
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'body' in data or 'articleList' in data:
                    return data
            elif response.status_code == 401:
                print("401 Unauthorized - rotating endpoint")
                self.current_endpoint = (self.current_endpoint + 1) % len(self.endpoints)
                
        except Exception as e:
            print(f"Traditional search failed: {e}")
        
        return None

    def scrape_region_advanced(self, criteria, max_attempts=3):
        """Advanced scraping with multiple strategies"""
        print(f"Starting advanced scrape for {criteria['address']}")
        
        # Establish session first
        if not self._establish_session():
            print("Failed to establish session, using mock data")
            return self._generate_enhanced_mock_data(criteria), False
        
        all_articles = []
        success = False
        
        # Strategy 1: Try coordinate-based search for Seoul districts
        coordinates = self.region_finder.get_coordinates_for_address(criteria['address'])
        if coordinates:
            print("Trying coordinate-based search...")
            coord_result = self._try_coordinate_search(coordinates, criteria)
            if coord_result:
                all_articles.extend(coord_result)
                success = True
                print(f"Coordinate search successful: {len(coord_result)} articles")
        
        # Strategy 2: Try traditional endpoint rotation
        if not success:
            print("Trying traditional endpoint rotation...")
            for attempt in range(max_attempts):
                result = self._try_traditional_search(criteria)
                if result:
                    articles = result.get('articleList', result.get('body', []))
                    if articles:
                        all_articles.extend(articles)
                        success = True
                        print(f"Traditional search successful: {len(articles)} articles")
                        break
                
                # Wait before retry
                time.sleep(random.uniform(3, 6))
                self.current_endpoint = (self.current_endpoint + 1) % len(self.endpoints)
        
        # Strategy 3: Use enhanced mock data if all else fails
        if not success or not all_articles:
            print("All API attempts failed, generating enhanced mock data")
            return self._generate_enhanced_mock_data(criteria), False
        
        return all_articles, True

    def _generate_enhanced_mock_data(self, criteria):
        """Generate very realistic mock data based on actual Naver patterns"""
        mock_articles = []
        
        # Get location info
        location_parts = criteria['address'].split()
        district = location_parts[-1] if location_parts else '강남구'
        
        # Realistic property complexes in Seoul
        property_complexes = {
            "강남구": ["트리마제", "래미안", "아크로리버파크", "자이", "센트럴아이파크"],
            "서초구": ["반포자이", "서초래미안", "아크로비스타", "트리마제더클래스"],
            "송파구": ["잠실래미안", "롯데캐슬", "헬리오시티", "위브더제니스"],
            "default": ["힐스테이트", "아이파크", "푸르지오", "센트럴파크", "리버파크"]
        }
        
        complexes = property_complexes.get(district, property_complexes["default"])
        
        # Generate 12-20 realistic listings
        for i in range(random.randint(12, 20)):
            # Realistic pricing based on deal type and district
            if criteria['deal_type_code'] == 'A1':  # 매매
                if district in ["강남구", "서초구", "송파구"]:
                    price = f"{random.randint(80, 200)}억"
                else:
                    price = f"{random.randint(40, 120)}억"
            elif criteria['deal_type_code'] == 'B1':  # 전세
                if district in ["강남구", "서초구", "송파구"]:
                    price = f"{random.randint(50, 120)}억"
                else:
                    price = f"{random.randint(25, 80)}억"
            else:  # 월세
                deposit = random.randint(2000, 10000)
                monthly = random.randint(150, 400)
                price = f"{deposit}/{monthly}"
            
            # Realistic areas
            area_m2 = random.randint(59, 145)
            area_pyeong = round(area_m2 / 3.3, 1)
            
            # Floor information
            current_floor = random.randint(1, 25)
            total_floor = random.randint(current_floor + 1, 30)
            
            article = {
                'articleNo': f'demo_{i+1:04d}',
                'articleName': f"{random.choice(complexes)} {random.randint(1, 15)}단지",
                'realEstateTypeName': BUILDING_TYPES[criteria.get('category', 'residential')]['subcategories'][criteria.get('subcategory', 'apartments')]['display'],
                'tradeTypeName': next(v['display'] for k, v in DEAL_TYPES.items() if v['code'] == criteria['deal_type_code']),
                'dealOrWarrantPrc': price,
                'areaName': f"{area_m2}㎡",
                'area2': area_pyeong,
                'floorInfo': f"{current_floor}/{total_floor}",
                'articleFeatureDesc': random.choice([
                    '남향, 역세권, 초등학교 도보 5분',
                    '동남향, 대형마트 인근, 지하철 3분',
                    '남서향, 한강뷰, 학군 우수',
                    '북향, 조용한 주거지역, 공원 인접',
                    '동향, 교통 편리, 커뮤니티 시설 완비'
                ]),
                'realtorName': f"부동산플러스 {random.choice(['강남', '서초', '송파', '역삼', '논현'])}점",
                'lat': 37.5 + random.uniform(-0.1, 0.1),
                'lng': 127.0 + random.uniform(-0.1, 0.1),
                'direction': random.choice(['남향', '남동향', '동향', '서향', '북향']),
                'buildYear': random.randint(1995, 2023)
            }
            mock_articles.append(article)
        
        return mock_articles

# ==============================================================================
# MODULE 3: ENHANCED DATA PARSER
# ==============================================================================
def parse_articles_advanced(articles: list):
    """Enhanced parser that handles various response formats"""
    parsed_list = []
    if not articles:
        return parsed_list
    
    for article in articles:
        # Handle different API response formats
        article_data = article
        if isinstance(article, dict) and 'article' in article:
            article_data = article['article']
        
        parsed_item = {
            'listing_id': article_data.get('articleNo', article_data.get('articleId', f'unknown_{random.randint(1000, 9999)}')),
            'complex_name': article_data.get('articleName', article_data.get('complexName', 'Unknown Complex')),
            'property_type': article_data.get('realEstateTypeName', article_data.get('rletTpNm', 'Unknown Type')),
            'transaction_type': article_data.get('tradeTypeName', article_data.get('tradTpNm', 'Unknown Transaction')),
            'price': article_data.get('dealOrWarrantPrc', article_data.get('prc', 'Price Unknown')),
            'supply_area_m2': article_data.get('areaName', article_data.get('spc', 'Area Unknown')),
            'dedicated_area_m2': article_data.get('area2', article_data.get('spc2', 'N/A')),
            'floor': article_data.get('floorInfo', article_data.get('flrInfo', 'Floor Unknown')),
            'description': article_data.get('articleFeatureDesc', article_data.get('atclFetrDesc', 'No description')),
            'agent_name': article_data.get('realtorName', article_data.get('rltrNm', 'Unknown Agent')),
            'direction': article_data.get('direction', article_data.get('drct', 'Unknown')),
            'build_year': article_data.get('buildYear', article_data.get('bldY', 'Unknown'))
        }
        parsed_list.append(parsed_item)
    
    return parsed_list

# ==============================================================================
# MODULE 4: ENHANCED GUI WITH STATUS MONITORING
# ==============================================================================
class AdvancedNaverRealEstateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Naver Real Estate Scraper - Enhanced Multi-Strategy Version")
        self.root.geometry("1400x900")
        
        self.region_finder = AdvancedRegionFinder()
        self.scraper = AdvancedNaverLandScraper(self.region_finder)
        
        self.selected_category = tk.StringVar()
        self.selected_subcategory = tk.StringVar()
        self.selected_city = tk.StringVar()
        self.selected_district = tk.StringVar()
        self.selected_deal_type = tk.StringVar()
        
        self.max_price = tk.StringVar(value="0")
        self.max_jeonse = tk.StringVar(value="0")
        self.max_deposit = tk.StringVar(value="0")
        self.max_monthly = tk.StringVar(value="0")
        
        self.min_area = tk.StringVar(value="0")
        self.max_area = tk.StringVar(value="0")
        
        self.results_data = []
        self.last_search_real_data = False
        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        search_frame = ttk.Frame(notebook)
        notebook.add(search_frame, text="Search Criteria")
        
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="API Status & Info")
        
        self.create_search_tab(search_frame)
        self.create_results_tab(results_frame)
        self.create_status_tab(status_frame)

    def create_search_tab(self, parent):
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Header with disclaimer
        disclaimer_frame = ttk.LabelFrame(scrollable_frame, text="⚠️ Important Notice", padding=10)
        disclaimer_frame.pack(fill='x', padx=5, pady=5)
        disclaimer_text = tk.Text(disclaimer_frame, height=4, wrap=tk.WORD, bg='#fff3cd', fg='#856404')
        disclaimer_text.pack(fill='x')
        disclaimer_text.insert('1.0', 
            "This tool uses advanced techniques to access Naver Land data, but success is not guaranteed due to anti-bot measures. "
            "Real API access violates Naver's Terms of Service. For production use, consider official government APIs (data.go.kr) or commercial real estate data providers. "
            "This is for educational/research purposes only.")
        disclaimer_text.config(state='disabled')
        
        category_frame = ttk.LabelFrame(scrollable_frame, text="Property Category", padding=10)
        category_frame.pack(fill='x', padx=5, pady=5)
        for i, (cat_key, cat_data) in enumerate(BUILDING_TYPES.items()):
            ttk.Radiobutton(category_frame, text=cat_data['display'], variable=self.selected_category, value=cat_key, command=self.on_category_change).grid(row=i, column=0, sticky='w')

        self.subcategory_frame = ttk.LabelFrame(scrollable_frame, text="Property Type", padding=10)
        self.subcategory_frame.pack(fill='x', padx=5, pady=5)

        location_frame = ttk.LabelFrame(scrollable_frame, text="Location", padding=10)
        location_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(location_frame, text="City:").grid(row=0, column=0, sticky='w')
        city_combo = ttk.Combobox(location_frame, textvariable=self.selected_city, values=list(REGIONS.keys()))
        city_combo.grid(row=0, column=1, sticky='ew')
        city_combo.bind('<<ComboboxSelected>>', self.on_city_change)
        ttk.Label(location_frame, text="District:").grid(row=1, column=0, sticky='w')
        self.district_combo = ttk.Combobox(location_frame, textvariable=self.selected_district)
        self.district_combo.grid(row=1, column=1, sticky='ew')
        location_frame.columnconfigure(1, weight=1)

        deal_frame = ttk.LabelFrame(scrollable_frame, text="Deal Type", padding=10)
        deal_frame.pack(fill='x', padx=5, pady=5)
        for i, (deal_key, deal_data) in enumerate(DEAL_TYPES.items()):
            ttk.Radiobutton(deal_frame, text=deal_data['display'], variable=self.selected_deal_type, value=deal_key, command=self.on_deal_type_change).grid(row=i, column=0, sticky='w')

        self.price_frame = ttk.LabelFrame(scrollable_frame, text="Price Filters (0 for no max)", padding=10)
        self.price_frame.pack(fill='x', padx=5, pady=5)

        area_frame = ttk.LabelFrame(scrollable_frame, text="Area Filters (m², 0 for no limit)", padding=10)
        area_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(area_frame, text="Min Area:").grid(row=0, column=0)
        ttk.Entry(area_frame, textvariable=self.min_area, width=10).grid(row=0, column=1)
        ttk.Label(area_frame, text="Max Area:").grid(row=0, column=2)
        ttk.Entry(area_frame, textvariable=self.max_area, width=10).grid(row=0, column=3)

        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill='x', padx=5, pady=20)
        ttk.Button(button_frame, text="🔍 Advanced Search", command=self.search_properties, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame, text="🧪 Test API Status", command=self.test_api_status).pack(side='left', padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_results_tab(self, parent):
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status bar
        status_bar = ttk.Frame(results_frame)
        status_bar.pack(fill='x', pady=5)
        self.results_info = ttk.Label(status_bar, text="No search performed yet", font=('Segoe UI', 10))
        self.results_info.pack(side='left')
        
        self.data_source_label = ttk.Label(status_bar, text="", font=('Segoe UI', 10, 'italic'))
        self.data_source_label.pack(side='right')
        
        columns = ('ID', 'Complex', 'Type', 'Deal', 'Price', 'Area', 'Floor', 'Direction', 'Built', 'Agent')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=22)
        
        # Configure columns
        column_widths = {'ID': 80, 'Complex': 180, 'Type': 80, 'Deal': 60, 'Price': 100, 'Area': 80, 'Floor': 60, 'Direction': 60, 'Built': 60, 'Agent': 150}
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=column_widths.get(col, 100), anchor='w')

        tree_scroll_y = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        tree_scroll_x = ttk.Scrollbar(results_frame, orient='horizontal', command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.results_tree.bind('<Double-1>', self.on_property_double_click)
        
        self.results_tree.pack(side='left', fill='both', expand=True)
        tree_scroll_y.pack(side='right', fill='y')
        tree_scroll_x.pack(side='bottom', fill='x')
        
        export_frame = ttk.Frame(parent)
        export_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(export_frame, text="📄 Export to CSV", command=self.export_to_csv).pack(side='left')
        ttk.Button(export_frame, text="🔄 Refresh Data", command=self.refresh_search).pack(side='left', padx=5)
        ttk.Label(export_frame, text="💡 Double-click property for details", foreground='blue').pack(side='right')

    def create_status_tab(self, parent):
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # API Status Section
        api_status_frame = ttk.LabelFrame(status_frame, text="API Status Monitor", padding=10)
        api_status_frame.pack(fill='x', pady=5)
        
        self.api_status_text = tk.Text(api_status_frame, height=8, wrap=tk.WORD, bg='#f8f9fa')
        self.api_status_text.pack(fill='x')
        self.api_status_text.insert('1.0', "Click 'Test API Status' to check current API availability...")
        
        # Technical Info Section
        tech_info_frame = ttk.LabelFrame(status_frame, text="Technical Information", padding=10)
        tech_info_frame.pack(fill='both', expand=True, pady=5)
        
        tech_info = tk.Text(tech_info_frame, wrap=tk.WORD, bg='#f8f9fa')
        tech_info.pack(fill='both', expand=True)
        
        tech_content = """
🔧 ADVANCED SCRAPING STRATEGIES IMPLEMENTED:

1. MULTI-ENDPOINT ROTATION:
   • Primary: https://new.land.naver.com/api/articles
   • Mobile: https://m.land.naver.com/cluster/ajax/articleList  
   • Front-API: https://fin.land.naver.com/front-api/v1/article/list
   
2. COORDINATE-BASED SEARCH:
   • Uses geographic coordinates for Seoul districts
   • Creates 2km x 2km search boundaries
   • Bypasses traditional region code requirements
   
3. ENHANCED SESSION MANAGEMENT:
   • Establishes valid browser sessions with cookies
   • Visits multiple pages to simulate real user behavior
   • Maintains session state across requests
   
4. INTELLIGENT FALLBACK SYSTEM:
   • Real API → Coordinate Search → Mock Data
   • Automatic endpoint rotation on failures
   • Enhanced mock data based on real patterns
   
5. ANTI-DETECTION MEASURES:
   • Realistic browser headers and user agents
   • Variable request timing (3-6 second delays)
   • Proper referer and CORS headers
   
⚠️ LIMITATIONS & LEGAL NOTES:
   • Naver has NO official public API
   • All methods reverse-engineer internal APIs
   • Success depends on current anti-bot measures
   • Terms of Service violations possible
   • For production: Use government APIs (data.go.kr)
   
📊 DATA SOURCES PRIORITY:
   1. Live Naver Land API (if accessible)
   2. Enhanced realistic mock data (for demo)
   3. Clearly labeled test data
   
🔍 SEARCH STRATEGIES:
   • Seoul districts: Coordinate-based search preferred
   • Other regions: Traditional cortarNo search
   • Multiple retries with different endpoints
   • Intelligent error handling and recovery
        """
        
        tech_info.insert('1.0', tech_content)
        tech_info.config(state='disabled')

    def on_category_change(self):
        for widget in self.subcategory_frame.winfo_children():
            widget.destroy()
        category = self.selected_category.get()
        if category in BUILDING_TYPES:
            subcategories = BUILDING_TYPES[category]['subcategories']
            for i, (subcat_key, subcat_data) in enumerate(subcategories.items()):
                ttk.Radiobutton(self.subcategory_frame, text=f"{subcat_data['display']}", variable=self.selected_subcategory, value=subcat_key).grid(row=i, column=0, sticky='w')

    def on_city_change(self, event=None):
        city = self.selected_city.get()
        if city in REGIONS:
            self.district_combo['values'] = REGIONS[city]
            self.selected_district.set("")

    def on_deal_type_change(self):
        for widget in self.price_frame.winfo_children():
            widget.destroy()
        deal_type = self.selected_deal_type.get()
        if deal_type == 'sale':
            ttk.Label(self.price_frame, text="Max Sale Price (억원):").grid(row=0, column=0)
            ttk.Entry(self.price_frame, textvariable=self.max_price, width=15).grid(row=0, column=1)
        elif deal_type == 'jeonse':
            ttk.Label(self.price_frame, text="Max Jeonse (억원):").grid(row=0, column=0)
            ttk.Entry(self.price_frame, textvariable=self.max_jeonse, width=15).grid(row=0, column=1)
        elif deal_type == 'rent':
            ttk.Label(self.price_frame, text="Max Deposit (만원):").grid(row=0, column=0)
            ttk.Entry(self.price_frame, textvariable=self.max_deposit, width=15).grid(row=0, column=1)
            ttk.Label(self.price_frame, text="Max Monthly (만원):").grid(row=0, column=2)
            ttk.Entry(self.price_frame, textvariable=self.max_monthly, width=15).grid(row=0, column=3)

    def validate_inputs(self):
        if not all([self.selected_category.get(), self.selected_subcategory.get(), self.selected_city.get(), self.selected_district.get(), self.selected_deal_type.get()]):
            messagebox.showerror("Error", "Please fill all category, location, and deal type fields.")
            return False
        return True

    def test_api_status(self):
        """Test current API status and update status display"""
        self.api_status_text.delete('1.0', tk.END)
        self.api_status_text.insert('1.0', "Testing API endpoints...\n\n")
        
        def run_test():
            results = []
            
            # Test session establishment
            self.api_status_text.insert(tk.END, "1. Testing session establishment...\n")
            self.root.update()
            
            session_ok = self.scraper._establish_session()
            if session_ok:
                results.append("✅ Session establishment: SUCCESS")
            else:
                results.append("❌ Session establishment: FAILED")
            
            # Test endpoints
            test_criteria = {
                'building_code': 'A01',
                'deal_type_code': 'A1',
                'address': '서울특별시 강남구'
            }
            
            self.api_status_text.insert(tk.END, "2. Testing API endpoints...\n")
            self.root.update()
            
            for i, endpoint in enumerate(self.scraper.endpoints, 1):
                try:
                    response = self.scraper.session.get(f"{endpoint}?cortarNo=1168000000&realEstateType=A01&tradeType=A1", timeout=10)
                    if response.status_code == 200:
                        results.append(f"✅ Endpoint {i}: {endpoint} - SUCCESS")
                    elif response.status_code == 401:
                        results.append(f"🔒 Endpoint {i}: {endpoint} - UNAUTHORIZED")
                    else:
                        results.append(f"⚠️ Endpoint {i}: {endpoint} - STATUS {response.status_code}")
                except Exception as e:
                    results.append(f"❌ Endpoint {i}: {endpoint} - ERROR: {str(e)[:50]}")
                
                time.sleep(2)
            
            # Test coordinate search
            self.api_status_text.insert(tk.END, "3. Testing coordinate search...\n")
            self.root.update()
            
            coordinates = self.scraper.region_finder.get_coordinates_for_address('서울특별시 강남구')
            if coordinates:
                coord_result = self.scraper._try_coordinate_search(coordinates, test_criteria)
                if coord_result:
                    results.append("✅ Coordinate search: SUCCESS")
                else:
                    results.append("❌ Coordinate search: FAILED")
            
            # Update status display
            self.api_status_text.delete('1.0', tk.END)
            status_report = f"API Status Test Results ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n\n"
            status_report += "\n".join(results)
            status_report += "\n\n📝 Recommendation: "
            
            success_count = sum(1 for r in results if "SUCCESS" in r)
            if success_count >= 2:
                status_report += "API access is working! Try real search."
            elif success_count == 1:
                status_report += "Limited API access. Results may be mixed."
            else:
                status_report += "API blocked. Will use enhanced mock data."
            
            self.api_status_text.insert('1.0', status_report)
        
        # Run test in thread to avoid blocking UI
        test_thread = threading.Thread(target=run_test)
        test_thread.daemon = True
        test_thread.start()

    def search_properties(self):
        if not self.validate_inputs():
            return
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Advanced Search in Progress...")
        progress_window.geometry("400x200")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Progress content
        ttk.Label(progress_window, text="🔍 Running Advanced Multi-Strategy Search", font=('Segoe UI', 12, 'bold')).pack(pady=10)
        
        self.progress_text = tk.Text(progress_window, height=8, width=50)
        self.progress_text.pack(pady=10, padx=10, fill='both', expand=True)
        
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, fill='x', expand=True, padx=10)
        progress_bar.start()
        
        def update_progress(message):
            self.progress_text.insert(tk.END, f"{message}\n")
            self.progress_text.see(tk.END)
            progress_window.update()
        
        search_thread = threading.Thread(target=self._perform_advanced_search, args=(progress_window, update_progress))
        search_thread.daemon = True
        search_thread.start()

    def _perform_advanced_search(self, progress_window, update_progress):
        try:
            update_progress("🚀 Initializing advanced search...")
            
            criteria = self._build_criteria()
            update_progress(f"📍 Target: {criteria['address']}")
            update_progress("🔄 Trying multiple endpoints and strategies...")
            
            raw_articles, api_working = self.scraper.scrape_region_advanced(criteria)
            
            update_progress(f"📊 Found {len(raw_articles)} listings")
            update_progress("🔧 Parsing and formatting data...")
            
            parsed_results = parse_articles_advanced(raw_articles)
            
            update_progress("✅ Search completed successfully!")
            time.sleep(1)
            
            self.root.after(0, self._update_results, parsed_results, progress_window, api_working)
        except Exception as e:
            self.root.after(0, self._handle_search_error, str(e), progress_window)

    def _build_criteria(self):
        category = self.selected_category.get()
        subcategory = self.selected_subcategory.get()
        deal_type = self.selected_deal_type.get()

        criteria = {
            'address': f"{self.selected_city.get()} {self.selected_district.get()}",
            'building_code': BUILDING_TYPES[category]['subcategories'][subcategory]['code'],
            'deal_type_code': DEAL_TYPES[deal_type]['code'],
            'category': category,
            'subcategory': subcategory,
            'min_area': int(self.min_area.get() or 0),
            'max_area': int(self.max_area.get() or 0),
            'price_min': 0,
            'price_max': 0,
        }
        
        if deal_type == 'sale':
            max_p = int(self.max_price.get() or 0)
            if max_p > 0: criteria['price_max'] = max_p * 100000000
        elif deal_type == 'jeonse':
            max_j = int(self.max_jeonse.get() or 0)
            if max_j > 0: criteria['price_max'] = max_j * 100000000
        
        return criteria

    def _update_results(self, results, progress_window, api_working=True):
        progress_window.destroy()
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        self.results_data = results
        self.last_search_real_data = api_working
        
        if not results:
            self.results_info.config(text="No properties found matching your criteria.")
            self.data_source_label.config(text="")
            return
        
        # Update status messages
        if api_working:
            self.results_info.config(text=f"✅ Found {len(results)} properties from live API")
            self.data_source_label.config(text="🔗 Real Data", foreground='green')
        else:
            self.results_info.config(text=f"⚠️ API unavailable - Showing {len(results)} demo properties")
            self.data_source_label.config(text="🧪 Demo Data", foreground='orange')
        
        # Populate results
        for item in results:
            self.results_tree.insert('', 'end', values=(
                item.get('listing_id', 'N/A'),
                item.get('complex_name', 'N/A'),
                item.get('property_type', 'N/A'),
                item.get('transaction_type', 'N/A'),
                item.get('price', 'N/A'),
                item.get('supply_area_m2', 'N/A'),
                item.get('floor', 'N/A'),
                item.get('direction', 'N/A'),
                item.get('build_year', 'N/A'),
                item.get('agent_name', 'N/A')
            ))

    def _handle_search_error(self, error_message, progress_window):
        progress_window.destroy()
        messagebox.showerror("Search Error", f"An error occurred during the advanced search:\n{error_message}")

    def refresh_search(self):
        """Refresh the last search"""
        if hasattr(self, 'results_data') and self.results_data:
            self.search_properties()
        else:
            messagebox.showinfo("No Previous Search", "Please perform a search first before refreshing.")

    def on_property_double_click(self, event):
        selection = self.results_tree.selection()
        if not selection:
            return
        
        selected_item_id = selection[0]
        item_index = self.results_tree.index(selected_item_id)

        if item_index < len(self.results_data):
            property_data = self.results_data[item_index]
            listing_id = property_data.get('listing_id')
            
            if listing_id and not listing_id.startswith('demo_') and self.last_search_real_data:
                # Real listing from API
                property_url = f"https://land.naver.com/article/{listing_id}"
                if messagebox.askyesno("Open in Browser", f"Open details for listing {listing_id} in your web browser?"):
                    webbrowser.open(property_url)
            else:
                # Demo/mock data - show detailed popup
                self._show_property_details(property_data)

    def _show_property_details(self, property_data):
        """Show detailed property information in a popup"""
        details_window = tk.Toplevel(self.root)
        details_window.title("Property Details")
        details_window.geometry("500x600")
        details_window.transient(self.root)
        
        # Create scrollable frame
        canvas = tk.Canvas(details_window)
        scrollbar = ttk.Scrollbar(details_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Property details
        ttk.Label(scrollable_frame, text="🏠 Property Details", font=('Segoe UI', 14, 'bold')).pack(pady=10)
        
        details = [
            ("Complex Name", property_data.get('complex_name', 'N/A')),
            ("Property Type", property_data.get('property_type', 'N/A')),
            ("Transaction Type", property_data.get('transaction_type', 'N/A')),
            ("Price", property_data.get('price', 'N/A')),
            ("Supply Area", property_data.get('supply_area_m2', 'N/A')),
            ("Dedicated Area", property_data.get('dedicated_area_m2', 'N/A')),
            ("Floor", property_data.get('floor', 'N/A')),
            ("Direction", property_data.get('direction', 'N/A')),
            ("Build Year", property_data.get('build_year', 'N/A')),
            ("Agent", property_data.get('agent_name', 'N/A')),
            ("Description", property_data.get('description', 'N/A'))
        ]
        
        for label, value in details:
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill='x', padx=20, pady=2)
            ttk.Label(frame, text=f"{label}:", font=('Segoe UI', 10, 'bold')).pack(side='left')
            ttk.Label(frame, text=str(value), font=('Segoe UI', 10)).pack(side='left', padx=(10, 0))
        
        if not self.last_search_real_data:
            notice_frame = ttk.Frame(scrollable_frame)
            notice_frame.pack(fill='x', padx=20, pady=20)
            notice_label = ttk.Label(notice_frame, text="ℹ️ This is demonstration data for interface testing.", 
                                   foreground='orange', font=('Segoe UI', 10, 'italic'))
            notice_label.pack()
        
        ttk.Button(scrollable_frame, text="Close", command=details_window.destroy).pack(pady=20)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def export_to_csv(self):
        if not self.results_data:
            messagebox.showwarning("No Data", "No results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"naver_listings_advanced_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if not filename:
            return
            
        try:
            df = pd.DataFrame(self.results_data)
            # Add metadata
            df['search_timestamp'] = datetime.datetime.now().isoformat()
            df['data_source'] = 'Live API' if self.last_search_real_data else 'Demo Data'
            df['scraper_version'] = 'Advanced Multi-Strategy v2.0'
            
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            messagebox.showinfo("Export Successful", f"Data exported to {filename}\n\nIncluded {len(self.results_data)} properties with metadata.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedNaverRealEstateGUI(root)
    root.mainloop()