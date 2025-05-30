import requests
import os
from pathlib import Path

def simple_download_test():
    """Simple script to test PDF downloads with working URLs."""
    
    # Working PDF URLs for testing
    test_urls = [
        {
            "url": "https://sample-files.com/downloads/documents/pdf/sample-report.pdf",
            "filename": "sample_report.pdf",
            "description": "Multi-page report (2.39 MB)"
        },
        {
            "url": "https://sample-files.com/downloads/documents/pdf/dev-example.pdf", 
            "filename": "dev_example.pdf",
            "description": "Developer example (690 KB)"
        },
        {
            "url": "https://sample-files.com/downloads/documents/pdf/sample-pdf-a4-size.pdf",
            "filename": "sample_a4.pdf", 
            "description": "A4 size sample PDF"
        },
        {
            "url": "https://www.gutenberg.org/files/74/74-pdf.pdf",
            "filename": "tom_sawyer.pdf",
            "description": "Tom Sawyer from Project Gutenberg"
        },
        {
            "url": "https://www.gutenberg.org/files/1342/1342-pdf.pdf",
            "filename": "pride_and_prejudice.pdf",
            "description": "Pride and Prejudice from Project Gutenberg"
        }
    ]
    
    # Download directory
    download_dir = Path.home() / "Documents" / "test_downloads"
    download_dir.mkdir(exist_ok=True)
    
    print(f"Download directory: {download_dir}")
    print("="*70)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/pdf,application/octet-stream,*/*',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    successful_downloads = []
    
    for i, pdf_info in enumerate(test_urls, 1):
        url = pdf_info["url"]
        filename = pdf_info["filename"] 
        description = pdf_info["description"]
        output_path = download_dir / filename
        
        print(f"\n{i}. Testing: {description}")
        print(f"   URL: {url}")
        print(f"   File: {filename}")
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                file_size = len(response.content)
                print(f"   ✅ SUCCESS - Downloaded {file_size:,} bytes")
                successful_downloads.append({
                    'filename': filename,
                    'size': file_size,
                    'path': str(output_path)
                })
            else:
                print(f"   ❌ FAILED - HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ ERROR - {str(e)}")
    
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY:")
    print("="*70)
    
    if successful_downloads:
        print(f"✅ {len(successful_downloads)} files downloaded successfully:")
        for download in successful_downloads:
            print(f"   • {download['filename']} ({download['size']:,} bytes)")
            print(f"     Path: {download['path']}")
        
        # Return the first successful download info for your script
        first_success = successful_downloads[0]
        print(f"\n🎯 For your script, use:")
        print(f"   URL: {test_urls[0]['url']}")
        print(f"   Filename: {first_success['filename']}")
        
    else:
        print("❌ No files downloaded successfully")
    
    return successful_downloads

if __name__ == "__main__":
    simple_download_test()