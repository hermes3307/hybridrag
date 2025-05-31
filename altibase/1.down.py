import requests
import os
import time
import sys
import argparse
import re
from urllib.parse import urlparse, unquote
import json
import logging
import traceback
from typing import Dict, List, Optional, Tuple

# Import PyMuPDF
try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Please install it: pip install PyMuPDF")
    sys.exit(1)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GitHub Interaction Functions (largely from your original script) ---

def get_github_token() -> Optional[str]:
    """Gets the GitHub API token from environment variables."""
    token = os.environ.get('GITHUB_API_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token:
        logger.info("Found GitHub API token in environment variables.")
    else:
        logger.info("GitHub API token not found in environment variables. API requests may be rate-limited.")
    return token

def extract_github_links_via_api(
    repo_or_folder_url: str,
    pattern: Optional[str] = None,
    github_token: Optional[str] = None,
    max_depth: int = 5, # Max recursion depth for subdirectories
    current_depth: int = 0
) -> List[str]:
    """
    Extracts links to files matching a pattern from a GitHub repository or folder URL using the GitHub API.
    Returns HTML URLs for files.
    """
    if current_depth >= max_depth:
        logger.warning(f"Max depth ({max_depth}) reached for URL: {repo_or_folder_url}")
        return []

    try:
        parsed_url = urlparse(repo_or_folder_url)
        path_parts = parsed_url.path.strip('/').split('/')

        if len(path_parts) < 2:
            logger.error(f"Invalid GitHub repository URL: {repo_or_folder_url}. Expected format like 'https://github.com/owner/repo/tree/branch/path'")
            return []

        owner = path_parts[0]
        repo_name = path_parts[1]
        
        # Determine the path within the repo
        # Example: /owner/repo/tree/branch/path/to/folder
        # Example: /owner/repo/blob/branch/path/to/file.pdf (less common for initial call)
        api_path_parts = []
        if 'tree' in path_parts or 'blob' in path_parts:
            # Find branch and subsequent path
            # e.g., .../tree/master/Manuals/Altibase_7.3/eng/PDF
            # e.g., .../blob/master/somefile.md
            idx = path_parts.index('tree') if 'tree' in path_parts else path_parts.index('blob')
            if len(path_parts) > idx + 2: # Ensure there's a path after branch
                 api_path_parts = path_parts[idx+2:] # Skip branch name
        
        api_path = '/'.join(api_path_parts)
        api_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{api_path}"
        
        logger.info(f"Accessing GitHub API: {api_url} (Depth: {current_depth})")

        headers = {
            'User-Agent': 'Python-GitHub-Downloader/1.0',
            'Accept': 'application/vnd.github.v3+json'
        }
        if github_token:
            headers['Authorization'] = f"token {github_token}"

        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status() # Raise an exception for HTTP errors
        
        items = response.json()
        links = []

        # If items is a dict, it's a single file response (less likely for folder Browse)
        if isinstance(items, dict) and items.get('type') == 'file':
            if not pattern or re.search(pattern, items.get('name', '')):
                links.append(items.get('html_url')) # Return html_url, will convert to raw later
            return links
        
        if not isinstance(items, list):
             logger.warning(f"Unexpected API response format from {api_url}. Expected a list of items.")
             return []

        for item in items:
            item_type = item.get('type')
            item_name = item.get('name', '')
            item_html_url = item.get('html_url') # This is the browser URL

            if item_type == 'file':
                if not pattern or re.search(pattern, item_name):
                    links.append(item_html_url)
            elif item_type == 'dir':
                # Recursive call for subdirectories
                time.sleep(0.1) # Be polite to the API
                dir_links = extract_github_links_via_api(
                    item_html_url, # Pass the HTML URL of the directory
                    pattern,
                    github_token,
                    max_depth,
                    current_depth + 1
                )
                links.extend(dir_links)
        return links

    except requests.exceptions.RequestException as e:
        logger.error(f"GitHub API request error for {repo_or_folder_url}: {e}")
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 403:
            logger.error("API rate limit likely exceeded. Consider using a GitHub token or waiting.")
        return []
    except Exception as e:
        logger.error(f"Error extracting links from {repo_or_folder_url}: {e}")
        logger.error(traceback.format_exc())
        return []

def convert_github_html_to_raw_url(html_url: str) -> str:
    """Converts a GitHub HTML file URL to its raw content URL."""
    parsed_url = urlparse(html_url)
    if 'blob' in parsed_url.path:
        raw_path = parsed_url.path.replace('/blob/', '/', 1)
        return f"https://raw.githubusercontent.com{raw_path}"
    else:
        # This might happen if the URL is already a raw URL or a different kind
        logger.warning(f"URL {html_url} does not contain '/blob/'. Assuming it might be a raw URL or needs different handling.")
        # For non-text files, this direct conversion might still work if it points to a file in repo.
        # However, for release assets or other types of links, this would be incorrect.
        # The extract_github_links_via_api gives html_url which points to a file in the repo.
        # So, this conversion is generally appropriate for files in the repository.
        # Let's assume it's a file in the repo and try to construct raw URL if possible.
        # Example: https://github.com/owner/repo/path/to/file.pdf -> needs to be raw.
        # This case is less common as API usually gives blob URLs for files in repo.
        # Fallback: try to convert, but it might be error-prone if not a blob URL.
        path_parts = parsed_url.path.split('/')
        # Example: /owner/repo/tree/master/file.pdf or /owner/repo/raw/master/file.pdf
        # This conversion is tricky if not 'blob'. Best if API gives download_url for binaries,
        # but contents API gives html_url, which we convert.
        if len(path_parts) > 3 and path_parts[3] in ['tree', 'raw']: # Heuristic
            raw_path_parts = [part for i, part in enumerate(path_parts) if not (i==3 and part=='tree')]
            raw_path = "/".join(raw_path_parts)
            return f"https://raw.githubusercontent.com{raw_path}"
        logger.info(f"Trying direct URL as raw (may not work for all cases): {html_url}")
        return html_url # As a fallback, but less reliable for non-blob text files

# --- PDF Processing Functions ---

def download_pdf_from_github(
    pdf_html_url: str,
    output_folder: str,
    file_name: Optional[str] = None
) -> Optional[str]:
    """
    Downloads a PDF file from a GitHub HTML URL to the specified output folder.
    Converts HTML URL to raw content URL for downloading.
    """
    try:
        raw_url = convert_github_html_to_raw_url(pdf_html_url)
        logger.info(f"Attempting to download PDF from raw URL: {raw_url}")

        headers = {'User-Agent': 'Python-PDF-Downloader/1.0'}
        response = requests.get(raw_url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        if not file_name:
            # Try to derive a filename from the URL
            parsed_raw_url = urlparse(raw_url)
            file_name = os.path.basename(unquote(parsed_raw_url.path))
            if not file_name.lower().endswith(".pdf"):
                file_name = f"{os.path.splitext(file_name)[0]}.pdf" # Ensure .pdf extension

        # Sanitize filename
        file_name = re.sub(r'[^\w\s.-]', '_', file_name)
        file_name = re.sub(r'\s+', '_', file_name)
        
        output_file_path = os.path.join(output_folder, file_name)
        os.makedirs(output_folder, exist_ok=True)

        with open(output_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"PDF downloaded successfully: {output_file_path}")
        return output_file_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF from {pdf_html_url} (raw: {raw_url}): {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while downloading {pdf_html_url}: {e}")
        logger.error(traceback.format_exc())
        return None


def _recursive_text_split(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Recursively splits text trying to respect separators and chunk size.
    """
    final_chunks = []
    if not text:
        return []

    if not separators: # Base case: no more separators to try
        # Split by fixed size if text is still too large
        for i in range(0, len(text), chunk_size - chunk_overlap):
            final_chunks.append(text[i:i + chunk_size])
        return final_chunks

    separator = separators[0]
    remaining_separators = separators[1:]
    
    # Split by the current separator
    # Keep the separator for more natural text flow if it's not just whitespace
    if separator == " ": # Special case for space, don't try to keep it if splitting by it
         splits = text.split(separator)
    else:
        # Use regex to split and keep the separator at the end of the previous chunk
        # This helps if separators are like ". "
        # For "\n\n" or "\n", we just split.
        if separator.strip() == "": # e.g. "\n\n", "\n"
             splits = text.split(separator)
        else: # e.g. ". ", "? "
            # Simpler split, then add separator back if needed, or just split
            splits = []
            current_pos = 0
            for match in re.finditer(re.escape(separator), text):
                splits.append(text[current_pos:match.end()])
                current_pos = match.end()
            if current_pos < len(text):
                splits.append(text[current_pos:])


    current_chunk = ""
    for i, part in enumerate(splits):
        # Potential new chunk if we add this part
        potential_chunk = current_chunk + (separator if current_chunk and separator.strip() != "" and not current_chunk.endswith(separator) else "") + part
        
        if len(potential_chunk) <= chunk_size:
            current_chunk = potential_chunk
        else:
            # If current_chunk is not empty, it's a complete chunk (or needs further splitting)
            if current_chunk:
                if len(current_chunk) > chunk_size: # It was already too big, or became too big
                    # This happens if a single 'part' after splitting by current_separator is > chunk_size
                    # So, we take what we had (current_chunk without the 'part' that made it too big)
                    # and then recursively split that 'part'
                    if current_chunk != potential_chunk : # if current_chunk has content before adding 'part'
                        final_chunks.extend(_recursive_text_split(current_chunk, remaining_separators, chunk_size, chunk_overlap))
                    
                    # Recursively split the 'part' itself that was too large
                    final_chunks.extend(_recursive_text_split(part, remaining_separators, chunk_size, chunk_overlap))
                else:
                     final_chunks.append(current_chunk)
                current_chunk = part # Start new chunk with the current part (with overlap logic below)
            else: # current_chunk was empty, meaning 'part' itself is > chunk_size
                final_chunks.extend(_recursive_text_split(part, remaining_separators, chunk_size, chunk_overlap))
                current_chunk = "" # Reset, as 'part' has been processed

    # Add the last remaining chunk
    if current_chunk:
        if len(current_chunk) > chunk_size:
            final_chunks.extend(_recursive_text_split(current_chunk, remaining_separators, chunk_size, chunk_overlap))
        else:
            final_chunks.append(current_chunk)

    # Apply overlap (simplified version for now, proper overlap needs more sophisticated joining)
    # A more robust overlap would combine small trailing chunks or prepend/append context.
    # For now, this recursive splitter focuses on breaking down large pieces.
    # True overlap might be better applied after this initial splitting.
    
    # Let's refine chunks by ensuring no chunk is excessively small due to splitting, merge if necessary
    merged_chunks = []
    buffer_chunk = ""
    for chunk_text in final_chunks:
        if not chunk_text.strip(): continue # Skip empty chunks
        if len(buffer_chunk) + len(chunk_text) < chunk_size / 2 and buffer_chunk: # Arbitrary merge threshold
            buffer_chunk += (separator if separator.strip() else " ") + chunk_text
        else:
            if buffer_chunk: merged_chunks.append(buffer_chunk)
            buffer_chunk = chunk_text
    if buffer_chunk: merged_chunks.append(buffer_chunk)
    
    return [c for c in merged_chunks if c.strip()] if merged_chunks else [c for c in final_chunks if c.strip()]


def chunk_pdf_text(
    pdf_path: str,
    chunk_size: int = 1000,  # Target characters per chunk
    chunk_overlap: int = 150 # Characters of overlap
) -> List[Dict]:
    """
    Extracts text from a PDF and chunks it.
    Returns a list of dictionaries, each containing the chunk text and metadata.
    """
    chunks_with_metadata = []
    try:
        doc = fitz.open(pdf_path)
        pdf_filename = os.path.basename(pdf_path)
        full_text = ""
        page_map = [] # To map character position in full_text to page number

        logger.info(f"Extracting text from {pdf_filename}...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text:
                start_char_idx = len(full_text)
                full_text += text + "\n" # Add newline between pages
                end_char_idx = len(full_text)
                page_map.append({"page": page_num + 1, "start": start_char_idx, "end": end_char_idx})
        
        logger.info(f"Total characters extracted: {len(full_text)}")
        if not full_text.strip():
            logger.warning(f"No text extracted from {pdf_filename}.")
            return []

        # Define separators for recursive splitting, from broader to finer
        separators = [
            "\n\n\n", # Multiple paragraphs
            "\n\n",   # Paragraphs
            "\n",     # Line breaks
            ". ",     # Sentences
            "? ",
            "! ",
            ", ",
            " ",      # Words
        ]
        
        # Use a simpler splitting logic if _recursive_text_split is too complex or buggy initially
        # For simplicity, let's try a direct split then manage size.
        # The _recursive_text_split is more robust if implemented carefully.
        # For now, let's use a character-based splitter as a fallback to ensure functionality.
        
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # raw_chunks = text_splitter.split_text(full_text)
        # For a self-contained example without Langchain:
        
        raw_chunks = []
        if len(full_text) <= chunk_size:
            raw_chunks.append(full_text)
        else:
            # Simplified iterative splitting:
            current_pos = 0
            while current_pos < len(full_text):
                end_pos = min(current_pos + chunk_size, len(full_text))
                chunk = full_text[current_pos:end_pos]
                
                # Try to find a natural break point if not at the end of text
                if end_pos < len(full_text):
                    best_break = -1
                    # Prefer splitting at paragraph, then sentence, then line break near the end of chunk
                    for sep in ["\n\n", ". ", "\n"]:
                        found_break = chunk.rfind(sep, max(0, chunk_size - chunk_overlap - len(sep))) # search in the overlap window
                        if found_break != -1:
                            best_break = found_break + len(sep) # include separator
                            break
                    if best_break != -1:
                        chunk = chunk[:best_break]
                        end_pos = current_pos + best_break
                
                raw_chunks.append(chunk)
                current_pos = end_pos - chunk_overlap if end_pos < len(full_text) else end_pos # Move with overlap
                if current_pos >= len(full_text) and end_pos < len(full_text): # ensure progress if overlap is large
                     current_pos = end_pos


        for i, text_chunk in enumerate(raw_chunks):
            if not text_chunk.strip():
                continue
            # Determine page numbers for the chunk (simplified)
            # A more accurate mapping would track character indices of each chunk
            # For now, we can list pages that have *any* overlap with the chunk's assumed position
            # This is a simplification; precise chunk-to-page mapping is complex.
            # We'll just list all pages as a placeholder metadata for now.
            # A better way: determine start/end char of chunk in full_text, then map to pages.
            chunk_pages = set()
            # This rough page association is not ideal. For better mapping,
            # one would need to track character offsets of chunks in the original `full_text`.
            # For this example, let's just put a general source.
            # For a more accurate page attribution, one would need to find the start and end
            # character index of `text_chunk` within `full_text` and then consult `page_map`.

            chunks_with_metadata.append({
                "text": text_chunk.strip(),
                "metadata": {
                    "source_pdf": pdf_filename,
                    "chunk_index": i,
                    "original_text_length": len(full_text)
                    # "pages": list(chunk_pages) # Add more accurate page mapping if needed
                }
            })
        
        logger.info(f"Created {len(chunks_with_metadata)} chunks for {pdf_filename}.")
        doc.close()
        return chunks_with_metadata

    except Exception as e:
        logger.error(f"Error chunking PDF {pdf_path}: {e}")
        logger.error(traceback.format_exc())
        return []


# --- Batch Processing ---

def download_and_chunk_pdfs_batch(
    repo_or_folder_url: str,
    pdf_pattern: str = r".+\.pdf$", # Regex for PDF files
    output_dir_pdfs: str = "downloaded_pdfs",
    output_dir_chunks: str = "chunked_data",
    github_token: Optional[str] = None,
    max_files: int = 100,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
):
    """
    Downloads PDFs from GitHub, then extracts text and chunks them.
    """
    os.makedirs(output_dir_pdfs, exist_ok=True)
    os.makedirs(output_dir_chunks, exist_ok=True)
    logger.info(f"PDFs will be saved to: {output_dir_pdfs}")
    logger.info(f"Chunked JSON data will be saved to: {output_dir_chunks}")

    logger.info(f"Fetching PDF links from: {repo_or_folder_url} with pattern: {pdf_pattern}")
    pdf_html_urls = extract_github_links_via_api(repo_or_folder_url, pdf_pattern, github_token)

    if not pdf_html_urls:
        logger.warning("No PDF links found matching the criteria.")
        return 0

    if len(pdf_html_urls) > max_files:
        logger.info(f"Found {len(pdf_html_urls)} PDFs. Processing the first {max_files}.")
        pdf_html_urls = pdf_html_urls[:max_files]
    else:
        logger.info(f"Found {len(pdf_html_urls)} PDFs to process.")

    successful_chunks = 0
    for i, pdf_html_url in enumerate(pdf_html_urls):
        logger.info(f"--- Processing PDF {i+1}/{len(pdf_html_urls)}: {pdf_html_url} ---")
        
        # 1. Download PDF
        # Derive a unique name for the PDF based on its URL path
        parsed_html_url = urlparse(pdf_html_url)
        pdf_original_name = os.path.basename(unquote(parsed_html_url.path))
        
        downloaded_pdf_path = download_pdf_from_github(pdf_html_url, output_dir_pdfs, file_name=pdf_original_name)

        if downloaded_pdf_path:
            # 2. Chunk PDF
            logger.info(f"Chunking PDF: {downloaded_pdf_path}")
            chunks = chunk_pdf_text(downloaded_pdf_path, chunk_size, chunk_overlap)

            if chunks:
                # 3. Save chunks to JSON
                base_name = os.path.splitext(os.path.basename(downloaded_pdf_path))[0]
                chunk_output_file = os.path.join(output_dir_chunks, f"{base_name}_chunks.json")
                try:
                    with open(chunk_output_file, 'w', encoding='utf-8') as f:
                        json.dump(chunks, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved {len(chunks)} chunks to {chunk_output_file}")
                    successful_chunks += 1

                    # Log first chunk of the first processed PDF as per request
                    if i == 0 and chunks:
                         logger.info(f"--- Sample Chunk from first PDF ({os.path.basename(downloaded_pdf_path)}) ---")
                         logger.info(json.dumps(chunks[0], indent=2, ensure_ascii=False))
                         logger.info("--- End Sample Chunk ---")

                except IOError as e:
                    logger.error(f"Could not write chunks to file {chunk_output_file}: {e}")
                except Exception as e:
                    logger.error(f"Error saving chunks for {downloaded_pdf_path}: {e}")
            else:
                logger.warning(f"No chunks generated for {downloaded_pdf_path}.")
        else:
            logger.error(f"Failed to download PDF: {pdf_html_url}")
        
        time.sleep(1) # Pause between processing files

    logger.info(f"Batch processing complete. Successfully chunked {successful_chunks} PDF(s).")
    return successful_chunks

# --- Main and Interactive ---

def interactive_mode_pdf():
    """Interactive mode for PDF downloading and chunking."""
    print("=" * 60)
    print("   PDF Downloader and Chunker - Interactive Mode")
    print("=" * 60)

    github_token = get_github_token()
    if not github_token:
        github_token_input = input("Enter GitHub Personal Access Token (optional, press Enter to skip): ").strip()
        if github_token_input:
            github_token = github_token_input
            os.environ['GITHUB_TOKEN'] = github_token # Set for current session if needed by other parts

    default_repo_url = "https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/eng/PDF"
    repo_url = input(f"Enter GitHub repository/folder URL containing PDFs [{default_repo_url}]: ").strip() or default_repo_url

    default_pattern = r".+\.pdf$"
    pattern = input(f"Enter PDF file pattern (regex) [{default_pattern}]: ").strip() or default_pattern
    
    output_pdfs = input("Enter directory to save downloaded PDFs [default: downloaded_pdfs]: ").strip() or "downloaded_pdfs"
    output_chunks = input("Enter directory to save chunked JSON data [default: chunked_data]: ").strip() or "chunked_data"
    
    try:
        max_f = int(input("Max number of PDF files to process [default: 100]: ").strip() or "100")
        c_size = int(input("Target chunk size (characters) [default: 1000]: ").strip() or "1000")
        c_overlap = int(input("Chunk overlap (characters) [default: 150]: ").strip() or "150")
    except ValueError:
        logger.error("Invalid number for max_files, chunk_size, or chunk_overlap. Using defaults.")
        max_f, c_size, c_overlap = 100, 1000, 150

    print("\nStarting PDF download and chunking process...")
    download_and_chunk_pdfs_batch(
        repo_url,
        pattern,
        output_pdfs,
        output_chunks,
        github_token,
        max_f,
        c_size,
        c_overlap
    )

def main():
    parser = argparse.ArgumentParser(description='GitHub PDF Downloader and Chunker.')
    parser.add_argument('--repo_url', type=str, help='GitHub repository or folder URL containing PDFs.')
    parser.add_argument('--pattern', type=str, default=r".+\.pdf$", help='Regex pattern for PDF files.')
    parser.add_argument('--output_pdfs_dir', type=str, default='downloaded_pdfs', help='Directory to save downloaded PDFs.')
    parser.add_argument('--output_chunks_dir', type=str, default='chunked_data', help='Directory to save chunked JSON data.')
    parser.add_argument('--token', type=str, help='GitHub API Token (can also be set via GITHUB_TOKEN env var).')
    parser.add_argument('--max_files', type=int, default=10, help='Maximum number of PDF files to process.')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Target chunk size in characters.')
    parser.add_argument('--chunk_overlap', type=int, default=150, help='Character overlap between chunks.')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode.')

    args = parser.parse_args()

    # Set token from arg if provided, else try env var
    github_token = args.token or get_github_token()
    if github_token and not args.token: # If found in env and not passed as arg
        logger.info("Using GitHub token from environment variable.")
    elif args.token:
         logger.info("Using GitHub token from command line argument.")
         os.environ['GITHUB_TOKEN'] = args.token # Make it available if other parts expect env var

    if args.interactive:
        interactive_mode_pdf()
    elif args.repo_url:
        download_and_chunk_pdfs_batch(
            args.repo_url,
            args.pattern,
            args.output_pdfs_dir,
            args.output_chunks_dir,
            github_token,
            args.max_files,
            args.chunk_size,
            args.chunk_overlap
        )
    else:
        print("No action specified. Use --repo_url <URL> to process PDFs or --interactive for interactive mode.")
        print("Example: python your_script_name.py --repo_url https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/eng/PDF")
        parser.print_help()

if __name__ == "__main__":
    main()