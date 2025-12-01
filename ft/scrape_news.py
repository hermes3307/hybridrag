import json
from tool_code import web_fetch, google_web_search
from bs4 import BeautifulSoup

def scrape_and_save_news(query, output_file):
    # Use the google_web_search tool to find news articles
    search_results = google_web_search(query=query)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Assuming search_results is a list of dictionaries with a 'url' key
        for result in search_results:
            url = result.get('link')
            if not url:
                continue

            # Use the web_fetch tool to get the HTML content of the URL
            fetched_content = web_fetch(prompt=f"Fetch the content of the URL: {url}")
            
            # The output of web_fetch is a string, we need to parse it to get the HTML
            # This is a simplified assumption, the actual output might need more complex parsing
            if fetched_content:
                # Find the part of the string that contains the content
                content_part = fetched_content.split('Content: ', 1)
                if len(content_part) > 1:
                    html_content = content_part[1]
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # A simple way to get text, might need to be adjusted for specific sites
                    paragraphs = soup.find_all('p')
                    article_text = "\n".join([p.get_text() for p in paragraphs])
                    
                    if article_text:
                        # Create a JSON object with the text
                        news_item = {"text": article_text}
                        # Write the JSON object to the file as a new line
                        f.write(json.dumps(news_item, ensure_ascii=False) + '\n')
                        print(f"Successfully scraped and saved content from {url}")
                    else:
                        print(f"Could not extract text from {url}")
                else:
                    print(f"Could not find 'Content: ' in the fetched output from {url}")
            else:
                print(f"Failed to fetch content from {url}")

if __name__ == '__main__':
    search_query = "ALTIBASE 뉴스"
    output_filename = "altibase_news_corpus.jsonl"
    scrape_and_save_news(search_query, output_filename)