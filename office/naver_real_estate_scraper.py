import csv
import datetime
import random

# --- Predefined Data ---
# New two-level structure for building types.
# Note: Codes are simplified for this simulation.
BUILDING_TYPES = {
    "complexes": {"display": "아파트/주상복합", "code": "APTHGJ"},
    "houses": {"display": "빌라/주택", "code": "VL"},
    "rooms": {"display": "원룸/투룸", "code": "OR"},
    "offices": {
        "상가 (Retail)": {"display": "상가", "code": "SG"},
        "사무실 (Office)": {"display": "사무실", "code": "SMS"},
        "오피스텔 (Officetel)": {"display": "오피스텔", "code": "OPST"}
    }
}

DEAL_TYPES = {
    'sale': 'A1',
    'jeonse': 'B1',
    'rent': 'B2',
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

    # Two-level building type selection
    level1_type_name = get_choice_from_list(BUILDING_TYPES.keys(), "Building Type (Level 1)")
    level1_type_data = BUILDING_TYPES[level1_type_name]

    if isinstance(level1_type_data, dict) and "display" not in level1_type_data:
        # This is a Level 2 category (like 'offices')
        level2_type_name = get_choice_from_list(level1_type_data.keys(), f"{level1_type_name} Type (Level 2)")
        final_type_data = level1_type_data[level2_type_name]
        criteria['building_type_display'] = final_type_data['display']
        criteria['building_code'] = final_type_data['code']
    else:
        # This is a Level 1 category with no sub-levels
        criteria['building_type_display'] = level1_type_data['display']
        criteria['building_code'] = level1_type_data['code']

    criteria['city'] = get_choice_from_list(REGIONS.keys(), "City")
    criteria['district'] = get_choice_from_list(REGIONS[criteria['city']], f"District in {criteria['city']}")
    criteria['deal_type'] = get_choice_from_list(DEAL_TYPES.keys(), "Deal Type")

    print("\n--- Enter Additional Filters (press Enter to use defaults) ---")
    if criteria['deal_type'] == 'rent':
        criteria['max_deposit'] = get_numeric_input("Max Deposit (만원)", 10000)
        criteria['max_monthly'] = get_numeric_input("Max Monthly Rent (만원)", 500)
    else:
        criteria['max_price'] = get_numeric_input(f"Max Price (억원) for {criteria['deal_type']}", 20)

    criteria['min_area'] = get_numeric_input("Min Area (m²)", 0)
    criteria['max_area'] = get_numeric_input("Max Area (m²)", 300)
    criteria['min_floor'] = get_numeric_input("Min Floor", 1)
    criteria['max_age'] = get_numeric_input("Max Building Age (years)", 20)
    criteria['parking'] = get_choice_from_list(["Yes", "No", "Any"], "Parking Available?", default_index=2)

    return criteria

def construct_search_url(criteria):
    base_url = "https://new.land.naver.com/search?sk="
    # The real URL is complex; this is a simplified representation
    url = f"{base_url}{criteria['building_code']}"
    print("\n[INFO] The constructed URL would look something like this (simplified):")
    print(url)
    return url

def scrape_data_from_naver(criteria):
    print("\n[SIMULATION] Generating and filtering dynamic data based on your selections...")
    
    pool_size = 300
    generated_pool = []
    for i in range(pool_size):
        district = criteria['district']
        building_display = criteria['building_type_display']
        
        title = f'{district} {building_display} #{i+1}'
        if building_display == "아파트/주상복합":
             title = f'{district} {random.choice(APT_NAMES)} #{i+1}'

        floor = random.randint(1, 40)
        listing = {
            'listing_id': str(random.randint(2500000000, 2600000000)),
            'building_type': building_display,
            'title': title,
            'deal_type': criteria['deal_type'],
            'size_m2': round(random.uniform(10.0, 500.0), 1),
            'floor': f"{floor}/{floor + random.randint(0, 10)}",
            '_floor': floor,
            '_age': random.randint(0, 30),
            '_parking': random.choice([True, False]),
            'deposit': str(random.randint(1, 50) * 100000000),
            'monthly_rent': str(random.randint(50, 1000) * 10000),
        }
        generated_pool.append(listing)

    filtered_listings = []
    for listing in generated_pool:
        if not (criteria['min_area'] <= listing['size_m2'] <= criteria['max_area']):
            continue
        if listing['_floor'] < criteria['min_floor']:
            continue
        if listing['_age'] > criteria['max_age']:
            continue
        if criteria['parking'] != 'Any':
            if (criteria['parking'] == 'Yes' and not listing['_parking']) or \
               (criteria['parking'] == 'No' and listing['_parking']):
                continue
        
        if criteria['deal_type'] == 'rent':
            if int(listing['deposit']) > criteria['max_deposit'] * 10000 or \
               int(listing['monthly_rent']) > criteria['max_monthly'] * 10000:
                continue
        else:
            if int(listing['deposit']) > criteria['max_price'] * 100000000:
                continue
            listing['monthly_rent'] = '0'

        del listing['_floor']
        del listing['_age']
        del listing['_parking']
        filtered_listings.append(listing)

    print(f"[SIMULATION] Found {len(filtered_listings)} listings matching your criteria (out of {pool_size} generated).")
    return filtered_listings

def display_results(data):
    if not data:
        print("\n[INFO] No results to display. Try broadening your filter criteria.")
        return

    print("\n--- Scraped Results ---")
    headers = ['Listing ID', 'Building Type', 'Title', 'Deal Type', 'Deposit (KRW)', 'Monthly (KRW)', 'Size (m²)', 'Floor']
    header_format = "{:<12} | {:<15} | {:<25} | {:<10} | {:<15} | {:<15} | {:<10} | {:<8}"
    row_format = "{:<12} | {:<15} | {:<25} | {:<10} | {:<15} | {:<15} | {:<10} | {:<8}"

    print(header_format.format(*headers))
    print("-" * 135)

    for item in data:
        deposit = f"{int(item.get('deposit', 0)):,}"
        monthly_rent = f"{int(item.get('monthly_rent', 0)):,}"
        print(row_format.format(
            item.get('listing_id', 'N/A'),
            item.get('building_type', 'N/A'),
            item.get('title', 'N/A')[:23],
            item.get('deal_type', 'N/A'),
            deposit,
            monthly_rent,
            item.get('size_m2', 'N/A'),
            item.get('floor', 'N/A')
        ))
    print("-" * 135)

def save_to_csv(data, criteria):
    if not data:
        return

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"naver_{criteria['building_type_display']}_{criteria['district']}_{criteria['deal_type']}_{now}.csv"
    headers = ['listing_id', 'building_type', 'title', 'deal_type', 'deposit', 'monthly_rent', 'size_m2', 'floor']

    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        print(f"\n[SUCCESS] Data successfully saved to '{filename}'")
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