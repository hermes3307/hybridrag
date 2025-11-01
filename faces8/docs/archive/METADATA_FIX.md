# Metadata Display Fix

## Issue
Search results were showing "unknown" for demographic information (sex, age) even though the data existed in the database.

## Root Cause
The database has **two storage locations** for demographic data:

1. **Dedicated columns** (from schema.sql):
   - `gender` VARCHAR(20)
   - `age_estimate` INTEGER
   - `brightness` FLOAT
   - `contrast` FLOAT
   - `sharpness` FLOAT

2. **JSONB metadata field** (flexible storage):
   - `estimated_sex`
   - `age_group`
   - `hair_color`
   - `skin_tone`
   - etc.

**Data inconsistency:**
- Some records had data in dedicated columns but NOT in JSONB metadata
- Search/display code only checked JSONB metadata
- Result: Records showed "unknown" even though data existed

### Example
```
Database record #3:
- gender (column): "female"  ✓
- age_estimate (column): 48  ✓
- metadata->>'estimated_sex': NULL  ✗

Display showed: Sex: unknown  ❌
Should show: Sex: female  ✅
```

## Solution

### 1. Updated `pgvector_db.py` (lines 414-450 and 533-569)

**Added metadata merging logic** in both `search_faces()` and `search_by_metadata()`:

```python
# Merge dedicated columns with JSONB metadata
jsonb_metadata = row[10] if row[10] else {}

metadata = {
    'face_id': row[0],
    'file_path': row[1],
    'age_estimate': row[5],
    'gender': row[6],
    'brightness': row[7],
    # ... other columns
}

# Merge JSONB metadata (takes precedence)
metadata.update(jsonb_metadata)

# Fallback: If JSONB is missing, use dedicated columns
if not metadata.get('estimated_sex') and metadata.get('gender'):
    metadata['estimated_sex'] = metadata['gender']

if not metadata.get('estimated_age') and metadata.get('age_estimate'):
    metadata['estimated_age'] = str(metadata['age_estimate'])
```

**Result:** Search results now include data from both sources, with JSONB taking precedence when available.

### 2. Updated `search_cli.py` (lines 36-41)

**Added fallback logic** in display code:

```python
# Check both JSONB metadata and dedicated columns
sex = metadata.get('estimated_sex') or metadata.get('gender', 'unknown')
age_range = metadata.get('estimated_age') or metadata.get('age_estimate', 'unknown')
```

**Result:** Display shows data from whichever source has it.

## Testing

### Before Fix
```
#3 - Demographics:
   Sex: unknown        ❌
   Age: unknown        ❌
```

### After Fix
```
#3 - Demographics:
   Sex: female         ✅
   Age: unknown (48)   ✅
```

The age group shows "unknown" because it was never populated, but the numeric age (48) is now correctly displayed.

## Data Migration Recommendation

For complete consistency, consider running a migration to populate JSONB metadata from dedicated columns:

```sql
UPDATE faces
SET metadata = jsonb_set(
    COALESCE(metadata, '{}'::jsonb),
    '{estimated_sex}',
    to_jsonb(gender)
)
WHERE gender IS NOT NULL
  AND (metadata->>'estimated_sex' IS NULL);

UPDATE faces
SET metadata = jsonb_set(
    COALESCE(metadata, '{}'::jsonb),
    '{estimated_age}',
    to_jsonb(age_estimate::text)
)
WHERE age_estimate IS NOT NULL
  AND (metadata->>'estimated_age' IS NULL);
```

This would ensure metadata filters work correctly on all records.

## Impact

### Files Modified
1. `pgvector_db.py` - Added metadata merging in result formatting
2. `search_cli.py` - Added fallback logic in display

### Benefits
- ✅ Search results now show correct demographic data
- ✅ Works with records from different data sources
- ✅ Backwards compatible with existing data
- ✅ JSONB metadata takes precedence when available
- ✅ Falls back to dedicated columns when JSONB is empty

## Future Recommendations

1. **Standardize data insertion**: Ensure all new records populate both dedicated columns AND JSONB metadata

2. **Run migration**: Update existing records to have consistent metadata

3. **Add validation**: Check that demographic data is populated in at least one location

4. **Consider schema cleanup**: Decide whether to use dedicated columns OR JSONB, not both

## Date
2025-10-30
