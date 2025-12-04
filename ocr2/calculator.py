from typing import Optional
import logging
from models import ReceiptSummary

logger = logging.getLogger(__name__)


class ReceiptCalculator:
    """Calculator for validating and computing receipt totals."""

    @staticmethod
    def validate_and_calculate(receipt: ReceiptSummary) -> ReceiptSummary:
        """
        Validate receipt calculations and compute totals.

        Args:
            receipt: ReceiptSummary object

        Returns:
            Updated ReceiptSummary with validation results
        """
        # Calculate subtotal from items
        calculated_subtotal = sum(item.total_price for item in receipt.items)
        receipt.calculated_subtotal = round(calculated_subtotal, 2)

        # Calculate total
        subtotal = receipt.subtotal if receipt.subtotal is not None else calculated_subtotal
        tax = receipt.tax if receipt.tax is not None else 0.0
        tip = receipt.tip if receipt.tip is not None else 0.0
        discount = receipt.discount if receipt.discount is not None else 0.0

        calculated_total = subtotal + tax + tip - discount
        receipt.calculated_total = round(calculated_total, 2)

        # Check if calculations match
        tolerance = 0.02  # Allow 2 cent difference due to rounding

        if receipt.total is not None:
            difference = abs(receipt.total - receipt.calculated_total)
            receipt.calculation_difference = round(difference, 2)
            receipt.calculation_matches = difference <= tolerance

            if not receipt.calculation_matches:
                logger.warning(
                    f"Receipt total mismatch: reported={receipt.total}, "
                    f"calculated={receipt.calculated_total}, "
                    f"difference={receipt.calculation_difference}"
                )
        else:
            receipt.total = receipt.calculated_total
            receipt.calculation_matches = True
            receipt.calculation_difference = 0.0

        # Validate subtotal if provided
        if receipt.subtotal is not None:
            subtotal_diff = abs(receipt.subtotal - receipt.calculated_subtotal)
            if subtotal_diff > tolerance:
                logger.warning(
                    f"Subtotal mismatch: reported={receipt.subtotal}, "
                    f"calculated={receipt.calculated_subtotal}"
                )

        return receipt

    @staticmethod
    def generate_summary_text(receipt: ReceiptSummary) -> str:
        """
        Generate a human-readable summary of the receipt.

        Args:
            receipt: ReceiptSummary object

        Returns:
            Formatted summary text
        """
        lines = []
        lines.append("=" * 50)
        lines.append("RECEIPT SUMMARY")
        lines.append("=" * 50)

        # Merchant info
        if receipt.merchant_name:
            lines.append(f"Merchant: {receipt.merchant_name}")
        if receipt.merchant_address:
            lines.append(f"Address: {receipt.merchant_address}")
        if receipt.date:
            lines.append(f"Date: {receipt.date}")
        if receipt.time:
            lines.append(f"Time: {receipt.time}")

        lines.append("-" * 50)
        lines.append("ITEMS:")
        lines.append("-" * 50)

        # Items
        for i, item in enumerate(receipt.items, 1):
            qty_price = ""
            if item.quantity and item.unit_price:
                qty_price = f" ({item.quantity} x {item.unit_price:.2f})"
            elif item.quantity and item.quantity != 1.0:
                qty_price = f" (qty: {item.quantity})"

            lines.append(f"{i}. {item.name}{qty_price}")
            lines.append(f"   ${item.total_price:.2f}")

        lines.append("-" * 50)

        # Totals
        if receipt.subtotal is not None:
            lines.append(f"Subtotal:        ${receipt.subtotal:.2f}")

        if receipt.discount and receipt.discount > 0:
            lines.append(f"Discount:       -${receipt.discount:.2f}")

        if receipt.tax is not None:
            lines.append(f"Tax:            +${receipt.tax:.2f}")

        if receipt.tip and receipt.tip > 0:
            lines.append(f"Tip:            +${receipt.tip:.2f}")

        lines.append("-" * 50)
        lines.append(f"TOTAL:           ${receipt.total:.2f}")

        # Validation info
        if receipt.calculated_total is not None:
            lines.append("")
            lines.append("CALCULATION VALIDATION:")
            lines.append(f"Calculated Total: ${receipt.calculated_total:.2f}")
            lines.append(f"Match: {'✓ YES' if receipt.calculation_matches else '✗ NO'}")

            if not receipt.calculation_matches and receipt.calculation_difference:
                lines.append(f"Difference: ${receipt.calculation_difference:.2f}")

        # Payment info
        if receipt.payment_method:
            lines.append("")
            lines.append(f"Payment Method: {receipt.payment_method}")

        lines.append("=" * 50)

        return "\n".join(lines)
