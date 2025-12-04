"""
Test script for batch receipt processing endpoint.
This script demonstrates how to send multiple receipt images to the batch processing endpoint.
"""

import requests
import json
import sys
from pathlib import Path


def test_batch_processing(receipt_files, api_url="http://localhost:8000"):
    """
    Test the batch receipt processing endpoint.

    Args:
        receipt_files: List of file paths to receipt images
        api_url: Base URL of the API
    """
    endpoint = f"{api_url}/api/v1/process-receipts-batch"

    # Prepare files for multipart upload
    files = []
    for file_path in receipt_files:
        if not Path(file_path).exists():
            print(f"Warning: File not found: {file_path}")
            continue
        files.append(('files', (Path(file_path).name, open(file_path, 'rb'), 'image/jpeg')))

    if not files:
        print("Error: No valid files to process")
        return None

    # Prepare form data
    data = {
        'ocr_engine': 'paddleocr',
        'preprocess': 'true',
        'use_llm': 'true'
    }

    print(f"\nSending {len(files)} receipt(s) to batch processing endpoint...")
    print(f"Endpoint: {endpoint}\n")

    try:
        response = requests.post(endpoint, files=files, data=data)

        # Close file handles
        for _, file_tuple in files:
            file_tuple[1].close()

        if response.status_code == 200:
            result = response.json()
            print("="*80)
            print("BATCH PROCESSING RESULTS")
            print("="*80)
            print(f"Total Processing Time: {result['total_processing_time']:.2f} seconds")
            print(f"OCR Engine: {result['ocr_engine_used']}")
            print()

            # Print aggregated totals
            agg = result['aggregated_totals']
            print("-"*80)
            print("AGGREGATED TOTALS")
            print("-"*80)
            print(f"Total Receipts: {agg['total_receipts']}")
            print(f"Successful: {agg['successful_receipts']}")
            print(f"Failed: {agg['failed_receipts']}")
            print(f"\nGRAND TOTAL: ${agg['grand_total']:.2f}")
            print(f"Total Tax: ${agg['total_tax']:.2f}")
            print(f"Total Tip: ${agg['total_tip']:.2f}")
            print(f"Total Discount: ${agg['total_discount']:.2f}")
            print(f"Total Items Count: {agg['total_items_count']}")
            print()

            # Print top items
            if agg['items_by_name']:
                print("-"*80)
                print("TOP ITEMS BY TOTAL AMOUNT")
                print("-"*80)
                for i, item in enumerate(agg['items_by_name'][:10], 1):
                    print(f"{i}. {item['name']}")
                    print(f"   Total Amount: ${item['total_amount']:.2f}")
                    print(f"   Quantity: {item['total_quantity']}")
                    print(f"   Occurrences: {item['occurrences']} receipt(s)")
                    if item['category']:
                        print(f"   Category: {item['category']}")
                    print()

            # Print merchants
            if agg['merchants']:
                print("-"*80)
                print("MERCHANTS")
                print("-"*80)
                for merchant in agg['merchants']:
                    print(f"  - {merchant}")
                print()

            # Print individual results
            print("-"*80)
            print("INDIVIDUAL RECEIPT RESULTS")
            print("-"*80)
            for i, receipt in enumerate(result['individual_results'], 1):
                print(f"\n{i}. {receipt['filename']}")
                print(f"   Success: {receipt['success']}")
                if receipt['success'] and receipt['receipt_summary']:
                    summary = receipt['receipt_summary']
                    if summary['merchant_name']:
                        print(f"   Merchant: {summary['merchant_name']}")
                    print(f"   Total: ${summary['total']:.2f}")
                    print(f"   Items: {len(summary['items'])}")
                elif receipt['error']:
                    print(f"   Error: {receipt['error']}")

            print("\n" + "="*80)

            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python test_batch.py <receipt_file1> <receipt_file2> ...")
        print("\nExample:")
        print("  python test_batch.py receipts/receipt1.jpg receipts/receipt2.jpg receipts/receipt3.jpg")
        sys.exit(1)

    receipt_files = sys.argv[1:]
    result = test_batch_processing(receipt_files)

    if result:
        # Optionally save to JSON file
        output_file = "batch_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
