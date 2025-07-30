import csv
import datetime
import random

# --- Predefined Data ---
# Comprehensive building types based on Naver real estate classification
BUILDING_TYPES = {
    "residential": {
        "display": "주거용 부동산",
        "subcategories": {
            "apartments": {"display": "아파트", "code": "APT"},
            "apartment_presale": {"display": "아파트 분양권", "code": "ABYG"},
            "officetels": {"display": "오피스텔", "code": "OPST"},
            "officetel_presale": {"display": "오피스텔 분양권", "code": "OBYG"},
            "villas": {"display": "빌라", "code": "VL"},
            "detached_houses": {"display": "단독/다가구주택", "code": "DDDGG"},
            "onerooms": {"display": "원룸", "code": "OR"},
            "studios": {"display": "고시원", "code": "GSW"},
            "city_housing": {"display": "도시형생활주택", "code": "DSH"},
            "multi_residential": {"display": "연립주택", "code": "YR"},
            "multi_household": {"display": "다세대주택", "code": "DSD"},
            "traditional_houses": {"display": "한옥주택", "code": "HOJT"},
            "redevelopment": {"display": "재개발", "code": "JGB"},
            "reconstruction": {"display": "재건축", "code": "JGC"}
        }
    },
    "commercial": {
        "display": "상업용 부동산",
        "subcategories": {
            "retail_spaces": {"display": "상가", "code": "SG"},
            "offices": {"display": "사무실", "code": "SMS"},
            "commercial_housing": {"display": "상가주택", "code": "SGJT"},
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
    'short_rent': {'display': '단기임대', 'code': 'B3'}
}

REGIONS = {
    "서울특별시": [
        "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
        "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구",
        "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구",
        "종로구", "중구", "중랑구"
    ],
    # Add other regions as needed
}

APT_NAMES = ["자이", "래미안", "힐스테이트", "아이파크", "더샵", "롯데캐슬", "푸르지오"]

# --- Helper Functions ---
def get_choice_from_list(items, title, default_index=None):
    print(f"\n--- Select {title} ---")
    item_list = list(items)
    for i, item in enumerate(item_list):
        print(f"[{i + 1}] {item}")
    
    prompt = f"Enter number (1-{len(item_list)}): "
    if default_index != None:
        prompt = f"Enter number (default: {default_index+1}): "

    while True:
        try:
            choice_str = input(prompt)
            if choice_str == "" and default_index != None:
                return item_list[default_index]
            choice = int(choice_str)
            if 1 <= choice <= len(item_list):
                return item_list[choice - 1]
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_numeric_input(prompt, default):
    val = input(f"{prompt} (default: {default}): ")
    if val == "":
        return default
    try:
        return int(val)
    except ValueError:
        print(f"Invalid number. Using default: {default}.")
        return default

# --- Core Functions ---
def get_user_input():
    print("--- Naver Real Estate Scraper ---")
    criteria = {}

    # Three-level building type selection
    print("\n--- Select Property Category ---")
    category_keys = list(BUILDING_TYPES.keys())
    for i, key in enumerate(category_keys):
        print(f"[{i + 1}] {BUILDING_TYPES[key]['display']} ({key})")
    
    while True:
        try:
            choice = int(input(f"Enter number (1-{len(category_keys)}): ")) - 1
            if 0 <= choice < len(category_keys):
                selected_category = category_keys[choice]
                break
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Select subcategory
    subcategories = BUILDING_TYPES[selected_category]['subcategories']
    subcategory_name = get_choice_from_list(subcategories.keys(), f"{BUILDING_TYPES[selected_category]['display']} 세부 유형")
    
    criteria['building_category'] = selected_category
    criteria['building_type_display'] = subcategories[subcategory_name]['display']
    criteria['building_code'] = subcategories[subcategory_name]['code']

    criteria['city'] = get_choice_from_list(REGIONS.keys(), "City")
    criteria['district'] = get_choice_from_list(REGIONS[criteria['city']], f"District in {criteria['city']}")
    
    # Deal type selection with display names
    deal_type_key = get_choice_from_list(DEAL_TYPES.keys(), "Deal Type")
    criteria['deal_type'] = deal_type_key
    criteria['deal_type_display'] = DEAL_TYPES[deal_type_key]['display']
    criteria['deal_type_code'] = DEAL_TYPES[deal_type_key]['code']

    print("\n--- Enter Additional Filters (press Enter to use defaults) ---")
    if criteria['deal_type'] in ['rent', 'short_rent']:
        criteria['max_deposit'] = get_numeric_input("Max Deposit (만원)", 10000)
        criteria['max_monthly'] = get_numeric_input("Max Monthly Rent (만원)", 500)
    elif criteria['deal_type'] == 'jeonse':
        criteria['max_jeonse'] = get_numeric_input("Max Jeonse (억원)", 30)
    else:  # sale
        criteria['max_price'] = get_numeric_input(f"Max Price (억원) for {criteria['deal_type_display']}", 50)

    # Area filters with different defaults based on property type
    if criteria['building_code'] in ['OR', 'GSW']:  # One-rooms and studios
        default_min_area, default_max_area = 10, 50
    elif criteria['building_code'] in ['OPST']:  # Officetels
        default_min_area, default_max_area = 20, 100
    elif criteria['building_code'] in ['APT', 'ABYG']:  # Apartments
        default_min_area, default_max_area = 30, 200
    else:  # Other types
        default_min_area, default_max_area = 0, 300
        
    criteria['min_area'] = get_numeric_input("Min Area (m²)", default_min_area)
    criteria['max_area'] = get_numeric_input("Max Area (m²)", default_max_area)
    
    # Floor and age filters
    criteria['min_floor'] = get_numeric_input("Min Floor", 1)
    criteria['max_floor'] = get_numeric_input("Max Floor (0 for no limit)", 0)
    criteria['max_age'] = get_numeric_input("Max Building Age (years)", 20)
    
    # Additional filters
    criteria['parking'] = get_choice_from_list(["Yes", "No", "Any"], "Parking Available?", default_index=2)
    criteria['elevator'] = get_choice_from_list(["Yes", "No", "Any"], "Elevator Available?", default_index=2)
    
    # Commercial property specific filters
    if criteria['building_category'] == 'commercial':
        criteria['business_type'] = get_choice_from_list(
            ["Restaurant", "Retail", "Office", "Service", "Any"], 
            "Business Type Preference", 
            default_index=4
        )

    return criteria

def construct_search_url(criteria):
    base_url = "https://new.land.naver.com/search"
    
    # Build query parameters based on criteria
    params = []
    params.append(f"sk={criteria.get('district', '')}")  # Search keyword (district)
    
    # Building type codes - can be multiple, separated by colons
    if criteria.get('building_code'):
        params.append(f"a={criteria['building_code']}")
    
    # Deal type
    if criteria.get('deal_type_code'):
        params.append(f"b={criteria['deal_type_code']}")
    
    # Price range (simplified)
    if criteria.get('max_price'):
        params.append(f"p=0~{criteria['max_price']}억")
    elif criteria.get('max_jeonse'):
        params.append(f"p=0~{criteria['max_jeonse']}억")
    elif criteria.get('max_deposit') and criteria.get('max_monthly'):
        params.append(f"p={criteria['max_deposit']}만~{criteria['max_monthly']}만")
    
    # Area range
    min_area = criteria.get('min_area', 0)
    max_area = criteria.get('max_area', 999)
    if min_area > 0 or max_area < 999:
        params.append(f"s={min_area}~{max_area}m²")
    
    # Construct final URL
    if params:
        url = f"{base_url}?" + "&".join(params)
    else:
        url = base_url
    
    print(f"\n[INFO] Constructed URL: {url}")
    return url

def scrape_data_from_naver(criteria):
    print(f"\n[SIMULATION] Generating data for {criteria['building_type_display']} in {criteria['district']}...")
    
    pool_size = 300
    generated_pool = []
    building_code = criteria['building_code']
    building_display = criteria['building_type_display']
    
    for i in range(pool_size):
        district = criteria['district']
        
        # Generate appropriate titles based on building type
        if building_code == 'APT':
            title = f'{district} {random.choice(APT_NAMES)} {i+1}동'
        elif building_code == 'OPST':
            title = f'{district} {building_display} {i+1}호'
        elif building_code in ['SG', 'SMS']:
            title = f'{district} {building_display} {i+1}호'
        elif building_code == 'VL':
            title = f'{district} {building_display} {i+1}호'
        else:
            title = f'{district} {building_display} #{i+1}'

        # Generate realistic floor ranges based on building type
        if building_code in ['APT', 'ABYG']:  # Apartments - high rise
            floor = random.randint(1, 30)
            total_floors = floor + random.randint(5, 15)
        elif building_code == 'OPST':  # Officetels - medium rise
            floor = random.randint(1, 20)
            total_floors = floor + random.randint(3, 10)
        elif building_code == 'VL':  # Villas - low rise
            floor = random.randint(1, 5)
            total_floors = floor + random.randint(0, 2)
        else:  # Others
            floor = random.randint(1, 15)
            total_floors = floor + random.randint(2, 8)

        # Generate area based on property type
        if building_code in ['OR', 'GSW']:  # Studios and one-rooms
            area = round(random.uniform(12.0, 40.0), 1)
        elif building_code == 'OPST':  # Officetels
            area = round(random.uniform(20.0, 80.0), 1)
        elif building_code in ['APT', 'ABYG']:  # Apartments
            area = round(random.uniform(40.0, 150.0), 1)
        else:
            area = round(random.uniform(15.0, 200.0), 1)

        listing = {
            'listing_id': str(random.randint(2500000000, 2600000000)),
            'building_category': criteria['building_category'],
            'building_type': building_display,
            'building_code': building_code,
            'title': title,
            'deal_type': criteria['deal_type'],
            'deal_type_display': criteria['deal_type_display'],
            'size_m2': area,
            'floor': f"{floor}/{total_floors}",
            '_floor': floor,
            '_total_floors': total_floors,
            '_age': random.randint(0, 30),
            '_parking': random.choice([True, False]),
            '_elevator': random.choice([True, False]) if total_floors > 4 else False,
            'deposit': str(random.randint(1, 50) * 100000000),
            'monthly_rent': str(random.randint(50, 1000) * 10000) if criteria['deal_type'] in ['rent', 'short_rent'] else '0',
            'jeonse_price': str(random.randint(5, 30) * 100000000) if criteria['deal_type'] == 'jeonse' else '0',
            'sale_price': str(random.randint(10, 100) * 100000000) if criteria['deal_type'] == 'sale' else '0',
        }
        
        # Add commercial-specific fields
        if criteria['building_category'] == 'commercial':
            listing['business_types'] = random.choice(['Restaurant', 'Retail', 'Office', 'Service'])
            
        generated_pool.append(listing)

    filtered_listings = []
    for listing in generated_pool:
        # Area filter
        if not (criteria['min_area'] <= listing['size_m2'] <= criteria['max_area']):
            continue
            
        # Floor filters
        if listing['_floor'] < criteria['min_floor']:
            continue
        if criteria.get('max_floor', 0) > 0 and listing['_floor'] > criteria['max_floor']:
            continue
            
        # Age filter
        if listing['_age'] > criteria['max_age']:
            continue
            
        # Parking filter
        if criteria['parking'] != 'Any':
            if (criteria['parking'] == 'Yes' and not listing['_parking']) or \
               (criteria['parking'] == 'No' and listing['_parking']):
                continue
                
        # Elevator filter
        if criteria['elevator'] != 'Any':
            if (criteria['elevator'] == 'Yes' and not listing['_elevator']) or \
               (criteria['elevator'] == 'No' and listing['_elevator']):
                continue
        
        # Price filters based on deal type
        if criteria['deal_type'] in ['rent', 'short_rent']:
            if int(listing['deposit']) > criteria['max_deposit'] * 10000000 or \
               int(listing['monthly_rent']) > criteria['max_monthly'] * 10000:
                continue
        elif criteria['deal_type'] == 'jeonse':
            if int(listing['jeonse_price']) > criteria['max_jeonse'] * 100000000:
                continue
        elif criteria['deal_type'] == 'sale':
            if int(listing['sale_price']) > criteria['max_price'] * 100000000:
                continue
        
        # Commercial property business type filter
        if criteria['building_category'] == 'commercial' and criteria.get('business_type', 'Any') != 'Any':
            if listing.get('business_types') != criteria['business_type']:
                continue

        # Clean up temporary fields
        del listing['_floor']
        del listing['_total_floors']
        del listing['_age']
        del listing['_parking']
        del listing['_elevator']
        filtered_listings.append(listing)

    print(f"[SIMULATION] Found {len(filtered_listings)} listings matching your criteria (out of {pool_size} generated).")
    return filtered_listings

def display_results(data):
    if not data:
        print("\n[INFO] No results to display. Try broadening your filter criteria.")
        return

    print("\n--- Scraped Results ---")
    
    # Dynamic headers based on deal type
    first_item = data[0]
    deal_type = first_item.get('deal_type', 'sale')
    
    if deal_type in ['rent', 'short_rent']:
        headers = ['ID', 'Type', 'Title', 'Deal', 'Deposit(만원)', 'Monthly(만원)', 'Size(m²)', 'Floor']
        price_fields = ['deposit', 'monthly_rent']
        price_divisors = [10000, 10000]
    elif deal_type == 'jeonse':
        headers = ['ID', 'Type', 'Title', 'Deal', 'Jeonse(억원)', 'Size(m²)', 'Floor', 'Business']
        price_fields = ['jeonse_price']
        price_divisors = [100000000]
    else:  # sale
        headers = ['ID', 'Type', 'Title', 'Deal', 'Price(억원)', 'Size(m²)', 'Floor', 'Business']
        price_fields = ['sale_price']
        price_divisors = [100000000]
    
    # Adjust header format based on number of columns
    col_widths = [8, 10, 20, 8, 12, 10, 8, 10]
    header_format = " | ".join([f"{{:<{w}}}" for w in col_widths[:len(headers)]])
    row_format = header_format
    
    print(header_format.format(*headers))
    print("-" * (sum(col_widths[:len(headers)]) + 3 * (len(headers) - 1)))

    for item in data:
        row_data = [
            str(item.get('listing_id', 'N/A'))[-8:],  # Last 8 chars of ID
            item.get('building_type', 'N/A')[:10],
            item.get('title', 'N/A')[:20],
            item.get('deal_type_display', item.get('deal_type', 'N/A'))[:8],
        ]
        
        # Add price fields
        for field, divisor in zip(price_fields, price_divisors):
            value = int(item.get(field, 0))
            if divisor == 10000:  # 만원 units
                formatted_value = f"{value//10000:,}만" if value > 0 else "0"
            else:  # 억원 units
                formatted_value = f"{value//100000000:,}억" if value > 0 else "0"
            row_data.append(formatted_value)
        
        # Add remaining fields
        row_data.extend([
            str(item.get('size_m2', 'N/A')),
            item.get('floor', 'N/A'),
        ])
        
        # Add business type for commercial properties
        if len(headers) > 7:
            business = item.get('business_types', 'N/A')[:10]
            row_data.append(business)
        
        print(row_format.format(*row_data[:len(headers)]))
    
    print("-" * (sum(col_widths[:len(headers)]) + 3 * (len(headers) - 1)))

def save_to_csv(data, criteria):
    if not data:
        return

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    building_type_clean = criteria['building_type_display'].replace('/', '_')
    filename = f"naver_{building_type_clean}_{criteria['district']}_{criteria['deal_type']}_{now}.csv"
    
    # Dynamic headers based on available data
    sample_item = data[0]
    headers = [
        'listing_id', 'building_category', 'building_type', 'building_code', 
        'title', 'deal_type', 'deal_type_display', 'size_m2', 'floor'
    ]
    
    # Add price fields based on deal type
    if criteria['deal_type'] in ['rent', 'short_rent']:
        headers.extend(['deposit', 'monthly_rent'])
    elif criteria['deal_type'] == 'jeonse':
        headers.append('jeonse_price')
    elif criteria['deal_type'] == 'sale':
        headers.append('sale_price')
    
    # Add commercial fields if present
    if 'business_types' in sample_item:
        headers.append('business_types')

    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)
        print(f"\n[SUCCESS] Data successfully saved to '{filename}'")
        print(f"[INFO] File contains {len(data)} listings with {len(headers)} fields")
    except IOError as e:
        print(f"\n[ERROR] Could not save file: {e}")

if __name__ == "__main__":
    search_criteria = get_user_input()
    search_url = construct_search_url(search_criteria)
    scraped_data = scrape_data_from_naver(search_criteria)
    display_results(scraped_data)

    if scraped_data:
        save_choice = input("\nSave these results to a CSV file? (y/n): ").lower()
        if save_choice == 'y':
            save_to_csv(scraped_data, search_criteria)
        else:
            print("\nExiting without saving.")