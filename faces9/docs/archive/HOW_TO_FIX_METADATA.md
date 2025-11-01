# How to Fix Incorrect Metadata (Sex, Age, etc.)

## The Problem

The current demographic metadata (estimated_sex, age_group, hair_color, etc.) is being generated using **simple heuristics** in the `_estimate_sex()` function in `core.py`. These heuristics are not accurate because they rely on basic measurements like:
- Face aspect ratio
- Jaw sharpness
- Skin smoothness

This causes incorrect classifications where women are labeled as men and vice versa.

## The Solution

Use **DeepFace**, a much more accurate deep learning library for demographic analysis. DeepFace uses state-of-the-art neural networks to predict age, gender, and race with high accuracy.

## How to Fix It

### Option 1: Run the Fix Script (Recommended)

I've created `fix_demographics.py` which will:
1. Re-analyze all your face images using DeepFace
2. Update the database with correct demographics
3. Show progress with a progress bar

**First time setup (models will download ~1.5GB):**
```bash
# Install DeepFace if not already installed
pip install deepface

# Test on a small sample first (5 images)
python3 fix_demographics.py --limit 5
```

**Once verified, process all images:**
```bash
# Process all images
python3 fix_demographics.py

# Or process a random sample
python3 fix_demographics.py --limit 100 --sample
```

**This will take time!** DeepFace needs to:
- Download models (~1.5GB) on first run
- Analyze each image (1-3 seconds per image)
- Update database

For 10,000 images, expect 3-8 hours depending on your CPU.

### Option 2: Use the Pre-configured Script

The `fix_demographics.py` script automatically:
- Uses DeepFace's accurate age, gender, and race detection
- Maps results to your existing schema
- Updates the database progressively
- Shows a progress bar

### Option 3: Manually Improve Future Downloads

To fix the issue for **new downloads**, modify the analyzer in `core.py` to use DeepFace instead of heuristics.

## What DeepFace Provides

DeepFace gives you:
- **Age**: Exact age (e.g., 28, 45, 62)
- **Gender**: Male/Female with confidence score
- **Race**: asian, white, middle eastern, indian, latino hispanic, black
- **Emotion**: happy, sad, angry, surprise, fear, disgust, neutral

These are mapped to your existing fields:
- `estimated_sex` → 'male' or 'female'
- `age_group` → 'child', 'young_adult', 'adult', 'middle_aged', 'senior'
- `skin_color` / `skin_tone` → Derived from race
- `dominant_race` → New field with race information

## Performance Expectations

| # Images | Time (CPU) | Time (GPU) |
|----------|------------|------------|
| 100      | 5-10 min   | 1-2 min    |
| 1,000    | 1-2 hours  | 10-20 min  |
| 10,000   | 5-10 hours | 1-2 hours  |
| 62,000   | 2-3 days   | 6-12 hours |

## After Fixing

Once you run the fix script:
1. ✅ Metadata search will work correctly
2. ✅ Male/female filters will return accurate results
3. ✅ Age groups will be more accurate
4. ✅ You'll have race/ethnicity information

## Testing the Fix

Test if it worked:
```bash
python3 test_metadata_search.py
```

This will search for:
- Male faces
- Female faces
- Male + blonde hair
- And show you the results

## Technical Details

### What Changed in the Database

The `metadata` JSONB column in the `faces` table gets updated with:
```json
{
  "estimated_sex": "female",           // Was: "male" (wrong)
  "estimated_age": 32,                 // Was: inaccurate
  "age_group": "adult",                // Updated based on exact age
  "dominant_race": "white",            // New field
  "gender_confidence": 98.7,           // New field
  "skin_color": "light",               // Updated based on race
  "skin_tone": "light"                 // Updated based on race
}
```

### Why It Was Wrong Before

The old method in `core.py:_estimate_sex()` used:
```python
# Simple heuristics - NOT ACCURATE
if aspect_ratio > 0.85:
    male_score += 1
if jaw_variance > 500:
    male_score += 1
if skin_variance < 600:
    female_score += 1
```

These thresholds are arbitrary and don't work well for AI-generated faces.

### Why DeepFace Is Better

DeepFace uses:
- **VGG-Face** model trained on millions of real faces
- **Deep Convolutional Neural Networks** that learn features
- **Transfer learning** from large-scale face recognition
- **Confidence scores** for each prediction

## Troubleshooting

### "DeepFace not installed"
```bash
pip install deepface
```

### "Models downloading slowly"
This is normal the first time. Models are ~1.5GB and come from GitHub.

### "Segmentation fault" or crashes
Try:
```bash
# Use simpler detector
python3 fix_demographics_simple.py
```

### "Out of memory"
Process in smaller batches:
```bash
python3 fix_demographics.py --limit 1000
# Wait for it to finish, then run again for next batch
```

## Need Help?

If the scripts don't work, you can:
1. Check the log output for errors
2. Try processing just 5-10 images first with `--limit 5`
3. Check if DeepFace is properly installed: `python3 -c "from deepface import DeepFace; print('OK')"`

## Summary

**The metadata search now works**, but the **data itself is inaccurate** because of simple heuristics.

**To fix permanently:** Run `python3 fix_demographics.py` to re-analyze all images with DeepFace.

**Time required:** 3-8 hours for 10,000 images, but you can start with smaller batches.
