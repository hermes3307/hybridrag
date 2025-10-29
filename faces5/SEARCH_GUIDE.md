# Advanced Search Guide

## 🎯 Overview

Your face database now has **advanced search capabilities** with rich metadata filtering!

## 📊 What Metadata is Saved?

When you download and process faces, the following metadata is **automatically extracted and saved**:

### 👤 **Demographic Features** (AI-estimated)
```
estimated_sex     → 'male', 'female', 'unknown'
age_group         → 'child', 'young_adult', 'adult', 'middle_aged', 'senior'
estimated_age     → '0-12', '18-25', '25-40', '40-60', '60+'
skin_tone         → 'very_light', 'light', 'medium', 'tan', 'brown', 'dark'
skin_color        → 'light', 'medium', 'dark'
hair_color        → 'black', 'dark_brown', 'brown', 'blonde', 'red', 'gray', 'other'
```

### 🖼️ **Image Properties**
```
width, height     → Image dimensions
brightness        → 0-255 (higher = brighter)
contrast          → 0-255 (higher = more contrast)
saturation_mean   → Color saturation
faces_detected    → Number of faces detected
brightness_level  → 'bright', 'dark'
image_quality     → 'high', 'medium'
has_face          → True/False
```

### ⚙️ **System Fields**
```
face_id           → Unique identifier
file_path         → File location
timestamp         → When processed
image_hash        → MD5 hash (for duplicate detection)
embedding_model   → Which AI model created the embedding
```

---

## 🚀 How to Use Advanced Search

### **Method 1: GUI (Existing)**
Run the GUI and use the "Search Faces" tab with demographic filters.
```bash
python faces.py
```

### **Method 2: Command Line (NEW!)**

#### Show all possible search values:
```bash
python search_cli.py --list-values
```

#### Show database statistics:
```bash
python search_cli.py --stats
```

#### Simple text search:
```bash
# Find blonde females
python search_cli.py --text "blonde female"

# Find young adult males with dark hair
python search_cli.py --text "young adult males with dark hair"

# Find seniors with gray hair
python search_cli.py --text "seniors with gray hair"
```

#### Structured search with filters:
```bash
# Find females with blonde hair
python search_cli.py --sex female --hair blonde

# Find young adults or adults (OR logic)
python search_cli.py --age young_adult adult

# Multiple filters (AND logic between categories, OR within)
python search_cli.py --sex female --age young_adult adult --hair blonde brown
# → Finds: females who are (young_adult OR adult) with (blonde OR brown) hair

# Find bright, high quality images
python search_cli.py --quality high --brightness-min 150 --brightness-max 255

# Find males with dark skin
python search_cli.py --sex male --skin-color dark
```

#### Image-based similarity search with filters:
```bash
# Find similar faces to query image, but only females
python search_cli.py --image path/to/face.jpg --sex female --mode hybrid

# Find similar faces with specific hair color
python search_cli.py --image query.jpg --hair blonde --mode hybrid
```

#### Export results:
```bash
# Export to JSON
python search_cli.py --sex female --hair blonde --export results.json

# Export to CSV
python search_cli.py --sex male --age adult --export results.csv --format csv

# Export to text
python search_cli.py --text "blonde female" --export results.txt --format txt
```

---

## 💻 Programmatic Usage (Python API)

### Example 1: Simple metadata search
```python
from core import IntegratedFaceSystem
from advanced_search import AdvancedSearchEngine, SearchQuery

# Initialize
system = IntegratedFaceSystem()
system.initialize()
search_engine = AdvancedSearchEngine(system)

# Search for blonde females
query = SearchQuery(
    sex=['female'],
    hair_colors=['blonde'],
    n_results=10,
    search_mode='metadata'
)
results = search_engine.search(query)

for result in results:
    print(result['metadata']['file_path'])
```

### Example 2: Complex query with multiple filters
```python
# Find young adults OR adults with brown OR black hair
query = SearchQuery(
    sex=['male', 'female'],  # Both sexes
    age_groups=['young_adult', 'adult'],  # OR logic
    hair_colors=['brown', 'black'],  # OR logic
    brightness_range=(100, 200),  # Medium brightness
    has_face=True,
    n_results=20,
    search_mode='metadata'
)
results = search_engine.search(query)
```

### Example 3: Text-based natural language search
```python
# Natural language query
results = search_engine.text_search("young blonde females", n_results=10)
```

### Example 4: Hybrid search (image + filters)
```python
# Find faces similar to query image, but only males
query = SearchQuery(
    query_image='path/to/query.jpg',
    sex=['male'],
    n_results=10,
    search_mode='hybrid'
)
results = search_engine.search(query)
```

### Example 5: Save and reuse queries
```python
# Save a query
query = SearchQuery(sex=['female'], hair_colors=['blonde'])
search_engine.save_query("blonde_females", query)

# Load and reuse later
saved_query = search_engine.load_query("blonde_females")
results = search_engine.search(saved_query)
```

### Example 6: Export results
```python
# Export to different formats
search_engine.export_results(results, "output.json", format='json')
search_engine.export_results(results, "output.csv", format='csv')
search_engine.export_results(results, "output.txt", format='txt')
```

### Example 7: Get database statistics
```python
stats = search_engine.get_statistics()
print(f"Total faces: {stats['total_count']}")
print(f"Sex distribution: {stats['sex_distribution']}")
print(f"Age distribution: {stats['age_distribution']}")
```

---

## 🔍 Search Query Logic

### OR Logic (within same category):
When you provide multiple values for the same filter, they use **OR logic**:
```python
# Find males OR females
sex=['male', 'female']

# Find blonde OR brown OR black hair
hair_colors=['blonde', 'brown', 'black']

# Find young_adult OR adult OR middle_aged
age_groups=['young_adult', 'adult', 'middle_aged']
```

### AND Logic (between categories):
Different filter categories are combined with **AND logic**:
```python
query = SearchQuery(
    sex=['female'],           # Must be female
    age_groups=['adult'],     # AND must be adult
    hair_colors=['blonde']    # AND must have blonde hair
)
# Result: female AND adult AND blonde
```

### Combined Example:
```python
query = SearchQuery(
    sex=['male', 'female'],           # (male OR female)
    age_groups=['young_adult', 'adult'],  # AND (young_adult OR adult)
    hair_colors=['brown', 'black']    # AND (brown OR black hair)
)
# Result: (male OR female) AND (young_adult OR adult) AND (brown OR black hair)
```

---

## 🎨 Search Modes

### 1. **Metadata Search** (`mode='metadata'`)
Search by demographics/properties only, no image needed.
- **Use case**: "Find all blonde females", "Find all seniors"
- **Fast**: No vector similarity computation
- **No query image needed**

### 2. **Vector Search** (`mode='vector'`)
Search by image similarity using embeddings.
- **Use case**: "Find faces similar to this image"
- **Requires**: Query image
- **Accurate**: Uses deep learning embeddings

### 3. **Hybrid Search** (`mode='hybrid'`)
Combines vector similarity with metadata filters.
- **Use case**: "Find similar faces to this image, but only females with blonde hair"
- **Requires**: Query image + metadata filters
- **Most powerful**: Best of both worlds

---

## 📈 Understanding Results

Each result contains:
```python
{
    'id': 'face_1234567890_abc12345',    # Unique ID
    'distance': 0.234,                    # Similarity distance (lower = more similar)
    'metadata': {                         # All metadata
        'file_path': '/path/to/face.jpg',
        'estimated_sex': 'female',
        'age_group': 'young_adult',
        'estimated_age': '18-25',
        'skin_tone': 'light',
        'hair_color': 'blonde',
        'brightness': 180.5,
        'width': 1024,
        'height': 1024,
        # ... more fields
    }
}
```

**Distance interpretation**:
- `0.0 - 0.3`: Very similar (likely same person or very similar features)
- `0.3 - 0.6`: Similar (similar demographics/features)
- `0.6 - 1.0`: Different (different people)
- `> 1.0`: Very different

---

## 🧪 Testing the Search

### Step 1: Download and process some faces
```bash
# Start GUI and download ~50 faces
python faces.py
# In GUI: Go to "Download Faces" → Start Download (let it run for a minute)
# Then: Go to "Process & Embed" → Process All Faces
```

### Step 2: Inspect what was saved
```bash
python inspect_database.py
```

### Step 3: View statistics
```bash
python search_cli.py --stats
```

### Step 4: Try searches
```bash
# Simple searches
python search_cli.py --text "blonde female"
python search_cli.py --sex male --age adult
python search_cli.py --hair red

# Complex searches
python search_cli.py --sex female --age young_adult adult --hair blonde brown
python search_cli.py --skin-color light --hair blonde --limit 20
```

---

## 🎯 Real-World Use Cases

### Use Case 1: Build a dataset by demographics
```bash
# Export all blonde females for training
python search_cli.py --sex female --hair blonde --limit 1000 --export blonde_females.csv
```

### Use Case 2: Find duplicates or similar faces
```python
# Find faces similar to a specific image
python search_cli.py --image target_face.jpg --limit 50 --export similar_faces.json
```

### Use Case 3: Dataset quality analysis
```bash
# Check demographic distribution
python search_cli.py --stats

# Find low quality images
python search_cli.py --brightness-min 0 --brightness-max 100 --limit 100
```

### Use Case 4: Create demographic-specific datasets
```python
from advanced_search import AdvancedSearchEngine, SearchQuery

# Get diverse age groups
for age in ['child', 'young_adult', 'adult', 'middle_aged', 'senior']:
    query = SearchQuery(age_groups=[age], n_results=100)
    results = search_engine.search(query)
    search_engine.export_results(results, f"dataset_{age}.json")
```

---

## 🔧 Advanced Features

### Feature 1: Range Queries
```python
# Find faces with specific brightness range
query = SearchQuery(
    brightness_range=(150, 200),  # Medium-bright images
    n_results=50
)
```

### Feature 2: Age Range Queries
```python
# Find specific age range (parsed from text)
results = search_engine.text_search("people aged 25 to 35")
```

### Feature 3: Multiple Demographics (OR logic)
```python
# Find people with multiple hair colors
query = SearchQuery(
    hair_colors=['blonde', 'brown', 'red'],  # OR logic
    sex=['female'],
    n_results=100
)
```

### Feature 4: Saved Queries
```python
# Save frequently used queries
search_engine.save_query("diverse_adults", SearchQuery(
    age_groups=['young_adult', 'adult'],
    sex=['male', 'female'],
    n_results=100
))

# Reuse later
results = search_engine.search(search_engine.load_query("diverse_adults"))
```

---

## 📋 Quick Reference

### All Searchable Fields:
- **Sex**: `male`, `female`, `unknown`
- **Age**: `child`, `young_adult`, `adult`, `middle_aged`, `senior`
- **Skin Tone**: `very_light`, `light`, `medium`, `tan`, `brown`, `dark`
- **Skin Color**: `light`, `medium`, `dark`
- **Hair**: `black`, `dark_brown`, `brown`, `blonde`, `red`, `gray`, `light_gray`, `other`
- **Quality**: `high`, `medium`
- **Brightness**: `bright`, `dark` (or custom range 0-255)
- **Has Face**: `True`, `False`

### CLI Commands Cheat Sheet:
```bash
# Statistics
python search_cli.py --stats
python search_cli.py --list-values

# Simple searches
python search_cli.py --text "query"
python search_cli.py --sex VALUE --age VALUE --hair VALUE

# Export
python search_cli.py --sex female --export output.json
python search_cli.py --text "query" --export output.csv --format csv

# Image search
python search_cli.py --image path.jpg --mode hybrid --sex female
```

---

## 🚀 Next Steps

Now that you have advanced search, you can:

1. **Build demographic datasets**: Export specific demographics for training
2. **Find duplicates**: Use vector similarity to find duplicate faces
3. **Quality control**: Filter by image quality and brightness
4. **Diversity analysis**: Analyze demographic distribution
5. **Create training sets**: Build balanced datasets by demographics

Need help? Check the examples or run:
```bash
python search_cli.py --help
```
