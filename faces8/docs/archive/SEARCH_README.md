# 🎯 Advanced Search - Quick Start

## What's New?

I've added **powerful search improvements** to your face database system! You can now:

✅ **Search by demographics** - sex, age, skin tone, hair color
✅ **Search by multiple values** - OR logic (e.g., "blonde OR brown hair")
✅ **Text-based search** - natural language queries
✅ **Range queries** - brightness, age ranges
✅ **Hybrid search** - combine image similarity with filters
✅ **Export results** - JSON, CSV, or TXT
✅ **Save queries** - reuse common searches
✅ **Database statistics** - understand your dataset

---

## 📊 What Data is Saved?

**YES!** All demographic data IS automatically saved when you process faces:

### Demographics (AI-estimated):
- **Sex**: male, female, unknown
- **Age Group**: child, young_adult, adult, middle_aged, senior
- **Skin Tone**: very_light, light, medium, tan, brown, dark
- **Hair Color**: black, dark_brown, brown, blonde, red, gray, other

### Image Properties:
- Brightness, contrast, quality
- Dimensions, format, file size
- Face detection (yes/no, count)

All this metadata is **searchable** via the CLI or Python API!

---

## 🚀 Quick Start (3 Steps)

### Step 1: Download & Process Faces
```bash
# Option A: Use GUI
python faces.py
# Then: Download Faces → Start Download (wait ~1 min)
# Then: Process & Embed → Process All Faces

# Option B: Check if you already have data
python inspect_database.py
```

### Step 2: View What's Available
```bash
# Show database statistics
python search_cli.py --stats

# List all searchable values
python search_cli.py --list-values
```

### Step 3: Try Searches!
```bash
# Simple text search
python search_cli.py --text "blonde female"

# Demographic filters
python search_cli.py --sex female --hair blonde --age young_adult

# Complex query
python search_cli.py --sex female --age young_adult adult --hair blonde brown
```

---

## 📁 New Files Added

```
advanced_search.py      - Core search engine with advanced filtering
search_cli.py          - Command-line interface for searches
search_examples.py     - Interactive Python examples
inspect_database.py    - Database inspection tool
SEARCH_GUIDE.md        - Comprehensive documentation
SEARCH_README.md       - This quick start guide (you're here!)
```

---

## 💡 Common Use Cases

### 1. Find specific demographics
```bash
python search_cli.py --sex female --hair blonde --age young_adult
```

### 2. Natural language search
```bash
python search_cli.py --text "seniors with gray hair"
```

### 3. Export dataset by demographics
```bash
python search_cli.py --sex male --age adult --export males_adults.csv --format csv
```

### 4. View database statistics
```bash
python search_cli.py --stats
```

### 5. Find similar faces with filters
```bash
python search_cli.py --image query.jpg --sex female --mode hybrid
```

---

## 🐍 Python API Examples

### Basic Search
```python
from core import IntegratedFaceSystem
from advanced_search import AdvancedSearchEngine, SearchQuery

# Initialize
system = IntegratedFaceSystem()
system.initialize()
search = AdvancedSearchEngine(system)

# Search for blonde females
query = SearchQuery(sex=['female'], hair_colors=['blonde'])
results = search.search(query)
```

### Text Search
```python
results = search.text_search("young blonde females", n_results=10)
```

### Export Results
```python
search.export_results(results, "output.json", format='json')
```

---

## 📖 Full Documentation

- **SEARCH_GUIDE.md** - Comprehensive guide with all features
- **search_examples.py** - Run interactive examples
- **search_cli.py --help** - CLI command reference

---

## 🧪 Test It Now!

### Option 1: Interactive Examples
```bash
python search_examples.py
```

### Option 2: Command Line
```bash
# Show what's in database
python search_cli.py --stats

# Try a search
python search_cli.py --text "blonde female"
```

### Option 3: Python API
```python
python3
>>> from advanced_search import *
>>> from core import *
>>> system = IntegratedFaceSystem()
>>> system.initialize()
>>> search = AdvancedSearchEngine(system)
>>> results = search.text_search("blonde female")
>>> print(f"Found {len(results)} results")
```

---

## ❓ Troubleshooting

### "Database is empty"
**Solution**: Download and process faces first
```bash
python faces.py  # Use GUI to download and process
```

### "No results found"
**Possible reasons**:
1. No faces match your filters (try broader search)
2. Database has limited data (download more faces)
3. Check available values: `python search_cli.py --stats`

### "Module not found"
**Solution**: Make sure all dependencies are installed
```bash
pip install chromadb numpy Pillow requests opencv-python
```

---

## 🎓 Learn More

1. **Read the full guide**: `cat SEARCH_GUIDE.md`
2. **Run examples**: `python search_examples.py`
3. **Try CLI**: `python search_cli.py --help`
4. **Check your data**: `python inspect_database.py`

---

## 🎯 Next Steps

Now you can:

1. ✅ **Search by demographics** - Find specific types of faces
2. ✅ **Build datasets** - Export faces by criteria
3. ✅ **Analyze diversity** - Check demographic distribution
4. ✅ **Filter searches** - Combine multiple criteria
5. ✅ **Find duplicates** - Use similarity search

**Need help?** Check `SEARCH_GUIDE.md` or run:
```bash
python search_cli.py --help
python search_examples.py
```

---

## 🚀 What's Possible Now

### Before (Basic Search):
- ❌ Search by image similarity only
- ❌ Limited filters in GUI
- ❌ No text-based queries
- ❌ No export capabilities
- ❌ No saved searches

### After (Advanced Search):
- ✅ Search by demographics (sex, age, hair, skin)
- ✅ Multiple value filters (OR logic)
- ✅ Text-based natural language queries
- ✅ Export to JSON/CSV/TXT
- ✅ Save and reuse queries
- ✅ Range queries (brightness, age)
- ✅ Database statistics
- ✅ Hybrid search (image + filters)
- ✅ Python API + CLI

---

**Happy Searching! 🔍**
