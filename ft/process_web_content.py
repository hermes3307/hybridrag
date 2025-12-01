
import json
from bs4 import BeautifulSoup

def process_and_save(html_content, output_file):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # A simple way to get text, might need to be adjusted for specific sites
    paragraphs = soup.find_all('p')
    article_text = "\n".join([p.get_text() for p in paragraphs])
    
    if article_text:
        with open(output_file, 'a', encoding='utf-8') as f:
            # Create a JSON object with the text
            news_item = {"text": article_text}
            # Write the JSON object to the file as a new line
            f.write(json.dumps(news_item, ensure_ascii=False) + '\n')
            print(f"Successfully processed and saved content.")
    else:
        print(f"Could not extract text from the content.")

if __name__ == '__main__':
    # The fetched content is a string that needs to be passed to the script.
    # I will get the content from the previous tool call and pass it here.
    fetched_content = """
    The content of the URL `https://altibase.com/` has been fetched. The title of the page is "ALTIBASE | ENTERPRISE HIGH PERFORMANCE Hybrid RDBMS".[1] The page describes Altibase as an "Enterprise High Performance Database" supplied to over 700 clients, including 19 Fortune Global 500 companies.[1] It highlights its 25 years of experience in the DBMS industry, offering a hybrid relational DBMS that combines in-memory and disk-resident databases.[1] The website also details various cross-vertical solutions in telecommunications, finance, manufacturing, public service, and other sectors, along with recent news and a technology roadmap.[1]
    """
    output_filename = "altibase_news_corpus.jsonl"
    process_and_save(fetched_content, output_filename)

