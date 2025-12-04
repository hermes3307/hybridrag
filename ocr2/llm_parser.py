import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
import logging
from typing import Dict, Any, Optional
from models import ReceiptSummary, ReceiptItem

logger = logging.getLogger(__name__)


class Qwen3ReceiptParser:
    """Parser that uses Qwen3 LLM to extract structured data from receipt text."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", use_4bit: bool = True):
        """
        Initialize Qwen3 parser.

        Args:
            model_name: HuggingFace model name (Qwen2.5-7B-Instruct or smaller variants)
            use_4bit: Whether to use 4-bit quantization for lower memory usage
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Load the Qwen3 model with optimizations."""
        if self.model is not None:
            return

        logger.info(f"Loading model: {self.model_name}")

        # Configure quantization for lightweight deployment
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if not self.use_4bit else None
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def parse_receipt(self, ocr_text: str) -> Optional[ReceiptSummary]:
        """
        Parse OCR text into structured receipt data using Qwen3.

        Args:
            ocr_text: Raw text from OCR

        Returns:
            ReceiptSummary object or None if parsing fails
        """
        self._load_model()

        # Create prompt for structured extraction
        prompt = self._create_parsing_prompt(ocr_text)

        try:
            # Generate response
            response = self._generate(prompt)

            # Parse JSON response
            receipt_data = self._extract_json(response)

            if receipt_data:
                return self._create_receipt_summary(receipt_data)

        except Exception as e:
            logger.error(f"Failed to parse receipt: {e}")

        return None

    def _create_parsing_prompt(self, ocr_text: str) -> str:
        """Create a prompt for receipt parsing."""
        return f"""You are a receipt parser. Extract structured information from the following receipt text and return ONLY a JSON object.

Receipt Text:
{ocr_text}

Extract the following information and return ONLY valid JSON (no markdown, no explanation):
{{
  "merchant_name": "store name",
  "merchant_address": "store address if available",
  "date": "date in YYYY-MM-DD format if available",
  "time": "time if available",
  "items": [
    {{
      "name": "item name",
      "quantity": 1.0,
      "unit_price": 10.50,
      "total_price": 10.50,
      "category": "category if identifiable"
    }}
  ],
  "subtotal": 100.50,
  "tax": 8.50,
  "tip": 0.00,
  "discount": 0.00,
  "total": 109.00,
  "currency": "USD",
  "payment_method": "payment method if available"
}}

Rules:
1. Return ONLY the JSON object, no other text
2. If a field is not available, use null
3. All prices should be numbers, not strings
4. Parse all line items from the receipt
5. Ensure subtotal + tax + tip - discount = total

JSON:"""

    def _generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """Generate response from the model."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts structured data from receipts."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                top_p=0.95
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response text."""
        # Try to find JSON in the response
        try:
            # First, try direct parsing
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON between curly braces
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to extract from markdown code block
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        logger.error(f"Could not extract JSON from response: {text[:200]}")
        return None

    def _create_receipt_summary(self, data: Dict[str, Any]) -> ReceiptSummary:
        """Create ReceiptSummary from parsed data."""
        items = []
        for item_data in data.get('items', []):
            items.append(ReceiptItem(
                name=item_data.get('name', 'Unknown'),
                quantity=item_data.get('quantity', 1.0),
                unit_price=item_data.get('unit_price'),
                total_price=item_data.get('total_price', 0.0),
                category=item_data.get('category')
            ))

        return ReceiptSummary(
            merchant_name=data.get('merchant_name'),
            merchant_address=data.get('merchant_address'),
            date=data.get('date'),
            time=data.get('time'),
            items=items,
            subtotal=data.get('subtotal'),
            tax=data.get('tax'),
            tip=data.get('tip'),
            discount=data.get('discount'),
            total=data.get('total', 0.0),
            currency=data.get('currency', 'USD'),
            payment_method=data.get('payment_method')
        )
