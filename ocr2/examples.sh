#!/bin/bash
# Example API calls for OCR Receipt Processing

API_URL="http://localhost:8000"

echo "OCR Receipt Processing - API Examples"
echo "======================================"
echo ""

# Check if API is running
echo "1. Health Check"
echo "---------------"
curl -s "${API_URL}/health" | python -m json.tool
echo -e "\n"

# Example 1: Process receipt with PaddleOCR and LLM
echo "2. Process Receipt (Full Pipeline - PaddleOCR + LLM)"
echo "---------------------------------------------------"
echo "curl -X POST '${API_URL}/api/v1/process-receipt' \\"
echo "  -F 'file=@receipt.jpg' \\"
echo "  -F 'ocr_engine=paddleocr' \\"
echo "  -F 'preprocess=true' \\"
echo "  -F 'use_llm=true'"
echo ""
# Uncomment to run:
# curl -X POST "${API_URL}/api/v1/process-receipt" \
#   -F "file=@receipt.jpg" \
#   -F "ocr_engine=paddleocr" \
#   -F "preprocess=true" \
#   -F "use_llm=true" | python -m json.tool

# Example 2: Process PDF receipt
echo "3. Process PDF Receipt"
echo "---------------------"
echo "curl -X POST '${API_URL}/api/v1/process-receipt' \\"
echo "  -F 'file=@receipt.pdf' \\"
echo "  -F 'ocr_engine=paddleocr' \\"
echo "  -F 'use_llm=true'"
echo ""

# Example 3: OCR only (no LLM, faster)
echo "4. OCR Only (No LLM - Faster)"
echo "----------------------------"
echo "curl -X POST '${API_URL}/api/v1/process-receipt-simple' \\"
echo "  -F 'file=@receipt.jpg' \\"
echo "  -F 'ocr_engine=easyocr'"
echo ""

# Example 4: Different OCR engines
echo "5. Using Different OCR Engines"
echo "-----------------------------"
echo ""
echo "Tesseract:"
echo "curl -X POST '${API_URL}/api/v1/process-receipt-simple' \\"
echo "  -F 'file=@receipt.jpg' \\"
echo "  -F 'ocr_engine=tesseract'"
echo ""
echo "EasyOCR:"
echo "curl -X POST '${API_URL}/api/v1/process-receipt-simple' \\"
echo "  -F 'file=@receipt.jpg' \\"
echo "  -F 'ocr_engine=easyocr'"
echo ""
echo "PaddleOCR (recommended):"
echo "curl -X POST '${API_URL}/api/v1/process-receipt-simple' \\"
echo "  -F 'file=@receipt.jpg' \\"
echo "  -F 'ocr_engine=paddleocr'"
echo ""

# Example 5: No preprocessing (faster but less accurate)
echo "6. Without Preprocessing (Faster)"
echo "---------------------------------"
echo "curl -X POST '${API_URL}/api/v1/process-receipt' \\"
echo "  -F 'file=@receipt.jpg' \\"
echo "  -F 'ocr_engine=paddleocr' \\"
echo "  -F 'preprocess=false' \\"
echo "  -F 'use_llm=false'"
echo ""

# Python example
echo "7. Python Example"
echo "----------------"
cat << 'EOF'
import requests

# Process receipt
with open('receipt.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'ocr_engine': 'paddleocr',
        'preprocess': 'true',
        'use_llm': 'true'
    }
    response = requests.post(
        'http://localhost:8000/api/v1/process-receipt',
        files=files,
        data=data
    )
    result = response.json()

    if result['success']:
        print(f"Total: ${result['receipt_summary']['total']}")
        print(f"Items: {len(result['receipt_summary']['items'])}")
        for item in result['receipt_summary']['items']:
            print(f"  - {item['name']}: ${item['total_price']}")
    else:
        print(f"Error: {result['error']}")
EOF

echo ""
echo "======================================"
echo "For more information, see README.md"
