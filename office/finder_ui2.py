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

REGIONS = {
    "서울특별시": [
        "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
        "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구",
        "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구",
        "종로구", "중구", "중랑구"
    ],
    "경기도": [
        "수원시", "성남시", "안양시", "안산시", "용인시", "부천시", "평택시", "화성시",
        "시흥시", "광명시", "김포시", "군포시", "하남시", "오산시", "양주시", "구리시"
    ],
    "인천광역시": [
        "중구", "동구", "미추홀구", "연수구", "남동구", "부평구", "계양구", "서구", "강화군", "옹진군"
    ]
}

# ==============================================================================
# MODULE 1: REGION FINDER (From Technical Guide)
# ==============================================================================
class RegionFinder:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'Referer': 'https://new.land.naver.com/'
        })
        self.region_cache = {}

    def _fetch_regions(self, cortar_no="0000000000"):
        if cortar_no in self.region_cache:
            return self.region_cache[cortar_no]
        url = f"https://new.land.naver.com/api/regions/list?cortarNo={cortar_no}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.region_cache[cortar_no] = data
            return data
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching regions for {cortar_no}: {e}")
            return None

    def get_cortar_no(self, address: str):
        """
        Translates a full address string into its corresponding cortarNo.
        This version includes a fix to handle unexpected API responses.
        """
        parts = address.split()
        current_cortar_no = "0000000000"
        for part in parts:
            regions = self._fetch_regions(current_cortar_no)
            if not regions:
                return None
            
            # === FIX: Start ===
            # This check prevents the "string indices must be integers" error
            # by ensuring the API response is a list before processing.
            if not isinstance(regions, list):
                print(f"API Error: Expected a list of regions, but got something else for code {current_cortar_no}.")
                print(f"Received: {regions}")
                return None
            # === FIX: End ===

            found = False
            for region in regions:
                # Using.get() for safer dictionary access
                if region.get('regionName') == part:
                    current_cortar_no = region.get('regionNo')
                    found = True
                    break
            if not found:
                print(f"Could not find region part: '{part}' in address '{address}'")
                return None
        return current_cortar_no

# ==============================================================================
# MODULE 2: PROPERTY SCRAPER (From Technical Guide, adapted for GUI)
# ==============================================================================
class NaverLandScraper:
    def __init__(self, region_finder: RegionFinder):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'Referer': 'https://new.land.naver.com/',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9,ko;q=0.8',
        })
        self.region_finder = region_finder

    def _fetch_page(self, criteria, page=1):
        base_url = "https://new.land.naver.com/api/articles"
        params = {
            'cortarNo': criteria['cortar_no'],
            'order': 'rank',
            'realEstateType': criteria['building_code'],
            'tradeType': criteria['deal_type_code'],
            'page': page,
            'priceMin': criteria.get('price_min', 0),
            'priceMax': criteria.get('price_max', 0),
            'areaMin': criteria.get('min_area', 0),
            'areaMax': criteria.get('max_area', 0),
        }
        try:
            response = self.session.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching page {page} for {criteria['cortar_no']}: {e}")
            return None

    def scrape_region(self, criteria, max_pages=10):
        cortar_no = self.region_finder.get_cortar_no(criteria['address'])
        if not cortar_no:
            print(f"Could not find cortarNo for address: {criteria['address']}")
            return []
        
        criteria['cortar_no'] = cortar_no
        print(f"Starting scrape for {criteria['address']} (cortarNo: {cortar_no})")
        all_articles = []  # Fixed: Added empty list initialization
        for page in range(1, max_pages + 1):
            print(f"Fetching page {page}...")
            data = self._fetch_page(criteria, page)
            if not data or 'articleList' not in data or not data['articleList']:
                print("No more articles found. Ending scrape.")
                break
            all_articles.extend(data['articleList'])
            time.sleep(random.uniform(1.0, 2.5))
        return all_articles

# ==============================================================================
# MODULE 3: DATA PARSER (From Technical Guide)
# ==============================================================================
def parse_articles(articles: list):
    parsed_list = []  # Fixed: Added empty list initialization
    if not articles:
        return parsed_list
    for article in articles:
        parsed_item = {
            'listing_id': article.get('articleNo'),
            'complex_name': article.get('articleName'),
            'property_type': article.get('realEstateTypeName'),
            'transaction_type': article.get('tradeTypeName'),
            'price': article.get('dealOrWarrantPrc'),
            'supply_area_m2': article.get('areaName'),
            'dedicated_area_m2': article.get('area2'),
            'floor': article.get('floorInfo'),
            'description': article.get('articleFeatureDesc'),
            'agent_name': article.get('realtorName')
        }
        parsed_list.append(parsed_item)
    return parsed_list

# ==============================================================================
# MODULE 4: THE GRAPHICAL USER INTERFACE (Original GUI, adapted)
# ==============================================================================
class NaverRealEstateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Naver Real Estate Scraper - Integrated Version")
        self.root.geometry("1200x800")
        
        self.region_finder = RegionFinder()
        self.scraper = NaverLandScraper(self.region_finder)
        
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
        
        self.results_data = []  # Fixed: Added empty list initialization
        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        search_frame = ttk.Frame(notebook)
        notebook.add(search_frame, text="Search Criteria")
        
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        
        self.create_search_tab(search_frame)
        self.create_results_tab(results_frame)

    def create_search_tab(self, parent):
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
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
        ttk.Button(button_frame, text="Search Properties", command=self.search_properties).pack(side='left', padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_results_tab(self, parent):
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.results_info = ttk.Label(results_frame, text="No search performed yet")
        self.results_info.pack(anchor='w', pady=5)
        
        columns = ('ID', 'Type', 'Complex', 'Deal', 'Price', 'Size (m²)', 'Floor', 'Agent')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=20)
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120, anchor='w')
        self.results_tree.column('Complex', width=200)

        tree_scroll_y = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        tree_scroll_x = ttk.Scrollbar(results_frame, orient='horizontal', command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.results_tree.bind('<Double-1>', self.on_property_double_click)
        
        self.results_tree.pack(side='left', fill='both', expand=True)
        tree_scroll_y.pack(side='right', fill='y')
        tree_scroll_x.pack(side='bottom', fill='x')
        
        export_frame = ttk.Frame(parent)
        export_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(export_frame, text="Export to CSV", command=self.export_to_csv).pack(side='left')
        ttk.Label(export_frame, text="💡 Double-click a property to view on Naver", foreground='blue').pack(side='right')

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

    def search_properties(self):
        if not self.validate_inputs():
            return
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Searching...")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="Scraping live data from Naver...").pack(pady=20)
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, fill='x', expand=True, padx=10)
        progress_bar.start()
        
        search_thread = threading.Thread(target=self._perform_search, args=(progress_window,))
        search_thread.daemon = True
        search_thread.start()

    def _build_criteria(self):
        category = self.selected_category.get()
        subcategory = self.selected_subcategory.get()
        deal_type = self.selected_deal_type.get()

        criteria = {
            'address': f"{self.selected_city.get()} {self.selected_district.get()}",
            'building_code': BUILDING_TYPES[category]['subcategories'][subcategory]['code'],
            'deal_type_code': DEAL_TYPES[deal_type]['code'],
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

    def _perform_search(self, progress_window):
        try:
            criteria = self._build_criteria()
            raw_articles = self.scraper.scrape_region(criteria, max_pages=5)
            parsed_results = parse_articles(raw_articles)
            self.root.after(0, self._update_results, parsed_results, progress_window)
        except Exception as e:
            self.root.after(0, self._handle_search_error, str(e), progress_window)

    def _update_results(self, results, progress_window):
        progress_window.destroy()
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        self.results_data = results
        if not results:
            self.results_info.config(text="No properties found matching your criteria.")
            return
        
        self.results_info.config(text=f"Found {len(results)} properties.")
        for item in results:
            self.results_tree.insert('', 'end', values=(
                item.get('listing_id', 'N/A'),
                item.get('property_type', 'N/A'),
                item.get('complex_name', 'N/A'),
                item.get('transaction_type', 'N/A'),
                item.get('price', 'N/A'),
                item.get('supply_area_m2', 'N/A'),
                item.get('floor', 'N/A'),
                item.get('agent_name', 'N/A')
            ))

    def _handle_search_error(self, error_message, progress_window):
        progress_window.destroy()
        messagebox.showerror("Search Error", f"An error occurred during scraping:\n{error_message}")

    def on_property_double_click(self, event):
        selection = self.results_tree.selection()
        if not selection:
            return
        
        # Fixed: Correctly get the selected item ID and its index
        selected_item_id = selection[0]  # Fixed: Added [0] to get first selected item
        item_index = self.results_tree.index(selected_item_id)

        if item_index < len(self.results_data):
            property_data = self.results_data[item_index]
            listing_id = property_data.get('listing_id')
            if listing_id:
                property_url = f"https://new.land.naver.com/articles/{listing_id}"
                if messagebox.askyesno("Open in Browser", f"Open details for listing {listing_id} in your web browser?"):
                    webbrowser.open(property_url)
            else:
                messagebox.showwarning("No URL", "No listing ID available for this property.")

    def export_to_csv(self):
        if not self.results_data:
            messagebox.showwarning("No Data", "No results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],  # Fixed: Added proper filetypes list
            initialfile=f"naver_listings_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if not filename:
            return
            
        try:
            df = pd.DataFrame(self.results_data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            messagebox.showinfo("Export Successful", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NaverRealEstateGUI(root)
    root.mainloop()