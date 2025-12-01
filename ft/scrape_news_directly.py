
import json
import requests
from bs4 import BeautifulSoup

def scrape_and_save(urls, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # This is a generic approach. For better results, selectors should be
                # tailored to the specific structure of each news site.
                paragraphs = soup.find_all('p')
                article_text = "\n".join([p.get_text() for p in paragraphs])
                
                if article_text:
                    news_item = {"text": article_text.strip()}
                    f.write(json.dumps(news_item, ensure_ascii=False) + '\n')
                    print(f"Successfully scraped and saved content from {url}")
                else:
                    print(f"No <p> tags found or they were empty in {url}")

            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch {url}: {e}")

if __name__ == '__main__':
    # URLs from the previous Google search
    news_urls = [
        "https://www.etnews.com/20251125000180",
        "https://www.itdaily.kr/news/articleView.html?idxno=219985",
        "https://www.zdnet.co.kr/view/?no=20240116141311",
        "https://www.etnews.com/20240116000219",
        "https://altibase.com/kr/company/news_20230913/",
        "https://www.kdpress.co.kr/news/articleView.html?idxno=124048",
        "https://altibase.com/kr/company/news_20240116/",
        "https://altibase.com/kr/company/news_20251125/"
    ]
    output_filename = "altibase_news_corpus.jsonl"
    scrape_and_save(news_urls, output_filename)
