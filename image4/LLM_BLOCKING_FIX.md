# LLM Assistant Non-Blocking Fix

## Problem

When downloading images and then asking the LLM to do embedding, the chat interface would stop responding and freeze. Users couldn't send new messages while operations were running.

## Root Cause

The issue was in the `_handle_llm_response()` method:
1. It was executing actions synchronously
2. The input reset (`_reset_llm_input()`) was in a `finally` block
3. This meant the chat was locked until the action completed
4. Long-running operations (downloads, embeddings) blocked all user input

## Solution

### 1. Immediate Input Reset

**Before:**
```python
def _handle_llm_response(self, response: dict):
    try:
        # Display response
        # Execute action (BLOCKS HERE)
    finally:
        self._reset_llm_input()  # Only resets after action completes
```

**After:**
```python
def _handle_llm_response(self, response: dict):
    # Display response
    self._reset_llm_input()  # Reset IMMEDIATELY
    # Execute action (doesn't block chat anymore)
```

### 2. Background Operations Status Indicator

Added a new status label to show active background operations:

```
Operations: Downloading [5/20]...
```

This label updates in real-time showing:
- "Idle" (gray) - No operations running
- "Downloading [N/total]..." (blue) - Download in progress
- "Processing new images..." (blue) - Embedding in progress

### 3. Status Update Function

Added helper function to update operation status:

```python
def _update_operations_status(self, status_text: str, color: str = "gray"):
    """Update the operations status label"""
    if hasattr(self, 'llm_operations_label'):
        self.llm_operations_label.config(text=status_text, foreground=color)
```

### 4. Download Progress Enhancement

Updated `_llm_download_images_with_progress()` to:
- Set status at start: "Downloading N images..."
- Update status per image: "Downloading [N/total]..."
- Reset status at end: "Idle"

## Changes Made

### Modified Files

**image.py:**
1. Line 697-708: Added "Operations" status indicator
2. Line 870-900: Refactored `_handle_llm_response()` for non-blocking
3. Line 951-959: Added `_update_operations_status()` helper
4. Line 997-1059: Updated download progress with status updates
5. Line 1067-1097: Updated embedding commands with status updates

### New Features

1. **Non-Blocking Chat**
   - Chat input is available immediately after sending a message
   - Can send multiple commands while operations run
   - Operations execute in background threads

2. **Operation Status Indicator**
   - Shows current background operation
   - Updates in real-time with progress
   - Color-coded: gray (idle), blue (active)

3. **Concurrent Operations**
   - Start download, then immediately ask for embedding
   - Both run concurrently
   - Progress reported for both

## Usage Examples

### Example 1: Concurrent Download and Status Check

```
You: Download 20 images
Assistant: Starting download of 20 images...
[Operations status shows: "Downloading [1/20]..."]

System: 📥 Download started...
System: ✓ [1/20] Downloaded: image_001.jpg

You: Show system status
[Chat is NOT blocked - you can ask immediately!]Assistant: [Responds with status - chat was available!]
```

### Example 2: Start Download Then Embed

```
You: Download 10 images
Assistant: Starting download of 10 images...
[Operations: "Downloading [1/10]..."]

[Download is running in background]

You: Now process all new images
[Chat accepts your message immediately!]
