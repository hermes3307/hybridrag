"""
Receipt Aggregation Service
Aggregates multiple receipt summaries into totals and statistics.
"""

from typing import List, Dict
from collections import defaultdict
import logging

from models import ReceiptSummary, AggregatedTotals, ItemAggregate, BatchReceiptResult

logger = logging.getLogger(__name__)


class ReceiptAggregator:
    """Aggregates multiple receipt summaries into combined totals."""

    def aggregate_receipts(self, results: List[BatchReceiptResult]) -> AggregatedTotals:
        """
        Aggregate multiple receipt processing results.

        Args:
            results: List of BatchReceiptResult objects

        Returns:
            AggregatedTotals with combined statistics
        """
        total_receipts = len(results)
        successful_results = [r for r in results if r.success and r.receipt_summary]
        failed_receipts = total_receipts - len(successful_results)

        if not successful_results:
            logger.warning("No successful receipt parsing results to aggregate")
            return AggregatedTotals(
                total_receipts=total_receipts,
                successful_receipts=0,
                failed_receipts=failed_receipts,
                grand_total=0.0,
                total_tax=0.0,
                total_tip=0.0,
                total_discount=0.0,
                total_items_count=0,
                items_by_name=[],
                merchants=[]
            )

        # Initialize accumulators
        grand_total = 0.0
        total_tax = 0.0
        total_tip = 0.0
        total_discount = 0.0
        total_items_count = 0

        # Track items by name for aggregation
        items_dict: Dict[str, Dict] = defaultdict(lambda: {
            'total_quantity': 0.0,
            'total_amount': 0.0,
            'occurrences': 0,
            'category': None
        })

        # Track unique merchants
        merchants = set()

        # Aggregate data from each receipt
        for result in successful_results:
            summary = result.receipt_summary
            if not summary:
                continue

            # Add to totals
            grand_total += summary.total or 0.0
            total_tax += summary.tax or 0.0
            total_tip += summary.tip or 0.0
            total_discount += summary.discount or 0.0
            total_items_count += len(summary.items)

            # Track merchant
            if summary.merchant_name:
                merchants.add(summary.merchant_name)

            # Aggregate items
            for item in summary.items:
                # Normalize item name (case-insensitive)
                item_key = item.name.lower().strip()

                items_dict[item_key]['total_quantity'] += item.quantity or 1.0
                items_dict[item_key]['total_amount'] += item.total_price
                items_dict[item_key]['occurrences'] += 1

                # Keep original name (first occurrence) and category
                if 'original_name' not in items_dict[item_key]:
                    items_dict[item_key]['original_name'] = item.name
                if item.category and not items_dict[item_key]['category']:
                    items_dict[item_key]['category'] = item.category

        # Convert items dict to list of ItemAggregate
        items_by_name = [
            ItemAggregate(
                name=data['original_name'],
                total_quantity=data['total_quantity'],
                total_amount=data['total_amount'],
                occurrences=data['occurrences'],
                category=data['category']
            )
            for data in items_dict.values()
        ]

        # Sort items by total amount (descending)
        items_by_name.sort(key=lambda x: x.total_amount, reverse=True)

        logger.info(
            f"Aggregated {len(successful_results)} receipts: "
            f"${grand_total:.2f} total, {total_items_count} items, "
            f"{len(items_by_name)} unique items"
        )

        return AggregatedTotals(
            total_receipts=total_receipts,
            successful_receipts=len(successful_results),
            failed_receipts=failed_receipts,
            grand_total=grand_total,
            total_tax=total_tax,
            total_tip=total_tip,
            total_discount=total_discount,
            total_items_count=total_items_count,
            items_by_name=items_by_name,
            merchants=sorted(list(merchants))
        )

    def get_items_by_category(self, aggregated: AggregatedTotals) -> Dict[str, float]:
        """
        Get total spending by category.

        Args:
            aggregated: AggregatedTotals object

        Returns:
            Dictionary mapping category names to total amounts
        """
        category_totals = defaultdict(float)

        for item in aggregated.items_by_name:
            category = item.category or "Uncategorized"
            category_totals[category] += item.total_amount

        return dict(category_totals)
