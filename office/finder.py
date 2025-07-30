import requests
import time
import random
import pandas as pd
import json

# ==============================================================================
# MODULE 1: REGION FINDER
# ==============================================================================
class RegionFinder:
    """
    Handles the conversion of human-readable addresses to Naver's internal
    region code (cortarNo). This is a necessary first step before scraping listings.
    """
    def __init__(self):
        """Initializes the session with appropriate headers."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'Referer': 'https://new.land.naver.com/'
        })
        self.region_cache = {}

    def _fetch_regions(self, cortar_no="0000000000"):
        """
        Fetches sub-regions for a given cortar_no. It uses a cache to avoid
        redundant network calls for the same region code.
        """
        if cortar_no in self.region_cache:
            return self.region_cache[cortar_no]

        url = f"https://new.land.naver.com/api/regions/list?cortarNo={cortar_no}"
        try:
            # Set a timeout for the request
            response = self.session.get(url, timeout=10)
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            data = response.json()
            self.region_cache[cortar_no] = data
            return data
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching regions for {cortar_no}: {e}")
            return None

    def get_cortar_no(self, address: str):
        """
        Translates a full address string (e.g., '경기도 성남시 분당구')
        into its corresponding cortarNo by traversing the region hierarchy.
        """
        parts = address.split()
        current_cortar_no = "0000000000"  # Root code for all of Korea
        
        for part in parts:
            regions = self._fetch_regions(current_cortar_no)
            if not regions:
                print(f"Could not fetch sub-regions for {part} with code {current_cortar_no}")
                return None
            
            # Debug: print the structure of the API response
            print(f"Debug - Regions response type: {type(regions)}")
            if regions and len(regions) > 0:
                print(f"Debug - First region type: {type(regions[0])}")
                print(f"Debug - First region data: {regions[0]}")
            
            found = False
            for region in regions:
                # Handle both dict and string responses from API
                if isinstance(region, dict):
                    if region.get('regionName') == part:
                        current_cortar_no = region.get('regionNo')
                        found = True
                        break
                elif isinstance(region, str):
                    # If regions is a list of strings, we need a different approach
                    print(f"Unexpected API response format. Region data: {region}")
                    return None
            
            if not found:
                print(f"Could not find region part: '{part}' in address '{address}'")
                return None
        
        return current_cortar_no

# ==============================================================================
# MODULE 2: PROPERTY SCRAPER
# ==============================================================================
class NaverLandScraper:
    """
    Scrapes property listings from Naver Real Estate for a given region.
    It uses the internal API to fetch structured JSON data directly.
    """
    def __init__(self, region_finder: RegionFinder):
        """Initializes the session with headers that mimic a real browser."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'Referer': 'https://new.land.naver.com/',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9,ko;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        })
        self.region_finder = region_finder

    def _fetch_page(self, cortar_no, page=1):
        """Fetches a single page of property listings from the internal API."""
        base_url = "https://new.land.naver.com/api/articles"
        # These parameters correspond to the filters on the website
        params = {
            'cortarNo': cortar_no,
            'order': 'rank',  # Sort by rank, can also be 'prc' (price) or 'date'
            'realEstateType': 'A01:A03', # A01: Apartment, A03: Residential/Commercial Complex
            'tradeType': 'A1', # A1: Sale
            'page': page
        }
        
        try:
            response = self.session.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching page {page} for {cortar_no}: {e}")
            return None

    def scrape_region(self, address: str, max_pages=10):
        """
        Scrapes all listings for a given address up to a maximum number of pages.
        """
        cortar_no = self.region_finder.get_cortar_no(address)
        if not cortar_no:
            print(f"Could not find cortarNo for address: {address}. Aborting scrape.")
            return

        print(f"Starting scrape for {address} (cortarNo: {cortar_no})")
        all_articles = []
        for page in range(1, max_pages + 1):
            print(f"Fetching page {page}...")
            data = self._fetch_page(cortar_no, page)
            
            # Stop if no data is returned or the article list is empty
            if not data or 'articleList' not in data or not data['articleList']:
                print("No more articles found. Ending scrape.")
                break
            
            all_articles.extend(data['articleList'])
            
            # Ethical scraping practice: wait between requests to avoid overwhelming the server
            time.sleep(random.uniform(1.5, 3.0))
            
        return all_articles

# ==============================================================================
# MODULE 3: DATA PARSER
# ==============================================================================
def parse_articles(articles: list):
    """
    Parses the raw list of article dictionaries from the API into a clean,
    flat list of dictionaries with more readable keys.
    """
    parsed_list = []
    for article in articles:
        # Extracts relevant fields, using.get() to avoid errors if a key is missing
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
# MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """
    Main function to orchestrate the scraper's workflow.
    """
    # 1. DEFINE YOUR TARGET AND OUTPUT FILENAME
    # You can change this to any address in Korea (e.g., '서울특별시 강남구', '부산광역시 해운대구')
    target_address = "경기도 성남시 분당구"
    output_filename = "naver_real_estate_listings.csv"
    
    # 2. INSTANTIATE THE HELPER CLASSES
    finder = RegionFinder()
    scraper = NaverLandScraper(finder)
    
    # 3. RUN THE SCRAPER to get the raw article data from the API
    # The 'max_pages' parameter prevents excessively long scrapes. Adjust as needed.
    raw_articles = scraper.scrape_region(target_address, max_pages=5)
    
    if not raw_articles:
        print("Scraping finished with no data. Exiting.")
        return
        
    # 4. PARSE THE RAW DATA into a clean format
    parsed_data = parse_articles(raw_articles)
    
    # 5. CONVERT TO A PANDAS DATAFRAME for easy manipulation and saving
    df = pd.DataFrame(parsed_data)
    
    # 6. DISPLAY A SAMPLE of the results
    print("\n--- Scraped Data Sample ---")
    print(df.head())
    print(f"\nTotal listings scraped: {len(df)}")
    
    # 7. SAVE THE DATAFRAME TO A CSV FILE
    try:
        # Use 'utf-8-sig' encoding to ensure Korean characters are displayed correctly in Excel
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\nData successfully saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving data to CSV: {e}")

if __name__ == "__main__":
    # This ensures the main() function is called only when the script is executed directly
    main()