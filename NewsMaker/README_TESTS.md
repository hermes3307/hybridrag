# Test Suite for main.py

This directory contains comprehensive unit tests for the `main.py` module of the NewsMaker application.

## Test Files

### `test_main_fixed.py` - Main Test Suite ✅
The primary test file that provides comprehensive coverage without external dependencies.

**What it tests:**
- ✅ **Data Structures**: `NewsArticle`, `NewsMetadata`, `NewsChunk` dataclasses
- ✅ **EnhancedNaverNewsAPI**: Initialization, HTML cleaning, dummy news generation
- ✅ **EnhancedPromptManager**: All prompt generation methods
- ✅ **Edge Cases**: Empty inputs, special characters, large data
- ✅ **Test Mode Functionality**: API behavior without real credentials

**Features:**
- Mocks external dependencies (chromadb, anthropic, requests, bs4)
- Suppresses logging output for clean test results
- Comprehensive edge case testing
- 30 individual test cases covering core functionality

### `test_main.py` - Full Integration Tests (Requires Dependencies)
A more comprehensive test suite that requires actual dependencies to be installed.

### `test_simple.py` - Basic Tests (Deprecated)
Simpler version used during development. Use `test_main_fixed.py` instead.

## Running the Tests

### Quick Test (Recommended)
```bash
python3 test_main_fixed.py
```

### With pytest (if available)
```bash
python3 -m pytest test_main_fixed.py -v
```

### Full Integration Tests (requires dependencies)
```bash
# Install dependencies first
pip install -r requirements.txt
python3 test_main.py
```

## Test Results

When you run `test_main_fixed.py`, you should see:

```
============================================================
Running Comprehensive Unit Tests for main.py
============================================================

Tests run: 30
Failures: 0
Errors: 0

============================================================
✅ All tests passed!

What was tested:
• Data structure creation and validation
• HTML cleaning functionality
• Dummy news generation
• Prompt generation for various scenarios
• Edge cases and error conditions
• Test mode functionality
============================================================
```

## Test Coverage

The test suite covers:

### Data Structures (4 tests)
- `NewsArticle` creation with and without content
- `NewsMetadata` creation with all fields
- `NewsChunk` creation and validation

### EnhancedNaverNewsAPI (9 tests)
- Initialization with various credential types
- HTML tag cleaning functionality
- Dummy news generation for different companies
- Test mode behavior
- Search functionality in test mode

### EnhancedPromptManager (6 tests)
- News analysis prompt generation
- News chunking prompt generation
- Enhanced news generation prompts
- Length specification handling (lines/words)
- Quality check prompt generation

### Utility Functions (5 tests)
- String representation of data structures
- Equality testing
- Empty list handling
- Special character support
- Score range validation

### Edge Cases (6 tests)
- Empty company names
- Zero article requests
- Large number requests
- Special characters in company names
- Very long content handling

## Mocked Dependencies

The tests mock these external libraries:
- `chromadb` - Vector database
- `anthropic` - Claude AI API
- `requests` - HTTP requests
- `bs4` (BeautifulSoup) - HTML parsing
- `dotenv` - Environment variable loading

This allows the tests to run without installing these dependencies and without making actual API calls.

## Adding New Tests

To add new tests:

1. Add test methods to existing test classes or create new test classes
2. Follow the naming convention: `test_<functionality>_<scenario>`
3. Use descriptive docstrings
4. Test both success and failure cases
5. Test edge cases and invalid inputs

Example:
```python
def test_new_functionality(self):
    """Test description of what this tests"""
    # Arrange
    test_input = "test data"

    # Act
    result = function_under_test(test_input)

    # Assert
    self.assertEqual(result, expected_value)
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running the tests from the same directory as `main.py`:
```bash
cd /path/to/NewsMaker
python3 test_main_fixed.py
```

### Missing Dependencies for Full Tests
For `test_main.py`, install dependencies:
```bash
pip install chromadb anthropic requests beautifulsoup4 python-dotenv
```

### Permission Errors
If you get permission errors, try:
```bash
chmod +x test_main_fixed.py
python3 test_main_fixed.py
```

## Test Philosophy

These tests follow the principle of testing the **public interface** and **business logic** without relying on external services. They ensure that:

1. **Data structures work correctly** - All dataclasses can be created and used
2. **Core logic is sound** - HTML cleaning, dummy data generation works
3. **Prompt generation is reliable** - All prompt templates contain expected content
4. **Edge cases are handled** - Empty inputs, special characters, large data
5. **Test mode works** - Application can run without real API keys

The tests prioritize **reliability** and **speed** over testing integration with external services.