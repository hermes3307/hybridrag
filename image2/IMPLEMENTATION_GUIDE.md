# IMPLEMENTATION GUIDE - WHAT TO MODIFY FOR MULTI-MODEL SUPPORT

## EXECUTIVE SUMMARY

This image search program currently supports multiple embedding models (Statistical, CLIP, YOLO, Action) but has a critical architectural limitation: **the database can only store ONE embedding per image at a time**.

To enable true multi-model support where each image has embeddings from multiple models simultaneously, you need to modify these three files:

1. **schema.sql** - Add separate columns for each model's embedding
2. **pgvector_db.py** - Update storage and search methods
3. **core.py** - Implement missing processing method and multi-model support

---

## FILE MODIFICATION CHECKLIST

### PRIORITY 1: CRITICAL (REQUIRED FOR FUNCTIONALITY)

#### 1. schema.sql (File: /home/pi/hybridrag/image/schema.sql)

**Current State**: Single `embedding vector(512)` column
**Problem**: Can only store one embedding type per image

**Required Changes**:
- Replace single embedding column with per-model columns
- Add HNSW indexes for each embedding
- Update search function to support model selection

**Modification Template**:
```sql
-- BEFORE (lines 10-22):
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    image_hash VARCHAR(64) NOT NULL,
    embedding_model VARCHAR(50) NOT NULL,
    
    embedding vector(512),  -- SINGLE COLUMN LIMITATION
    
    brightness FLOAT,
    contrast FLOAT,
    sharpness FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- AFTER (what you need):
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    image_hash VARCHAR(64) NOT NULL,
    
    -- Per-model embeddings with different dimensions
    embedding_statistical vector(512),      -- Always present
    embedding_clip vector(512),             -- Vision-language
    embedding_yolo vector(80),              -- Object detection
    embedding_action vector(15),            -- Action recognition
    
    -- Track which models have been computed
    has_embedding_statistical BOOLEAN DEFAULT FALSE,
    has_embedding_clip BOOLEAN DEFAULT FALSE,
    has_embedding_yolo BOOLEAN DEFAULT FALSE,
    has_embedding_action BOOLEAN DEFAULT FALSE,
    
    brightness FLOAT,
    contrast FLOAT,
    sharpness FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Additional Changes Needed**:
```sql
-- Update indexes
CREATE INDEX idx_embedding_statistical_hnsw ON images
USING hnsw (embedding_statistical vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_embedding_clip_hnsw ON images
USING hnsw (embedding_clip vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_embedding_yolo_hnsw ON images
USING hnsw (embedding_yolo vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_embedding_action_hnsw ON images
USING hnsw (embedding_action vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Update search function to accept model parameter
CREATE OR REPLACE FUNCTION search_similar_images(
    query_embedding vector,
    embedding_model VARCHAR(50),
    limit_count INTEGER DEFAULT 10,
    distance_threshold FLOAT DEFAULT 1.0
)
RETURNS TABLE (
    image_id VARCHAR(255),
    file_path TEXT,
    distance FLOAT,
    metadata JSONB
) AS $$
BEGIN
    IF embedding_model = 'clip' THEN
        RETURN QUERY SELECT ... FROM images WHERE has_embedding_clip = TRUE ...
    ELSIF embedding_model = 'yolo' THEN
        RETURN QUERY SELECT ... FROM images WHERE has_embedding_yolo = TRUE ...
    -- ... etc for other models
END;
$$ LANGUAGE plpgsql;
```

**Impact**: Breaking change - requires database migration
**Effort**: ~1 hour

---

#### 2. pgvector_db.py (File: /home/pi/hybridrag/image/pgvector_db.py)

**Current State** (lines 66-87):
```python
def add_image(self, image_data, embedding_model):
    cursor.execute(
        """INSERT INTO images (image_id, file_path, timestamp, image_hash, 
           embedding_model, embedding, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (image_data.image_id, image_data.file_path, image_data.timestamp,
         image_data.image_hash, embedding_model, image_data.embedding, ...)
```

**Problem**: Stores single embedding, overwrites `embedding_model` field on re-processing

**Required Changes**:
```python
def add_image(self, image_data, embedding_model, embeddings_dict=None):
    """
    Args:
        image_data: ImageData with image info
        embedding_model: "statistical", "clip", "yolo", "action"
        embeddings_dict: Optional dict with keys: 
                        {'statistical': [...], 'clip': [...], 'yolo': [...], 'action': [...]}
    """
    try:
        if embeddings_dict is None:
            # Legacy: single embedding
            embeddings_dict = {embedding_model: image_data.embedding}
        
        with self.conn.cursor() as cursor:
            # Build dynamic INSERT based on available embeddings
            columns = [
                'image_id', 'file_path', 'timestamp', 'image_hash',
                'metadata', 'brightness', 'contrast', 'sharpness'
            ]
            values_placeholders = [f'%s'] * len(columns)
            values = [
                image_data.image_id, image_data.file_path, image_data.timestamp,
                image_data.image_hash, psycopg2.extras.Json(image_data.features),
                image_data.features.get('brightness'), 
                image_data.features.get('contrast'),
                image_data.features.get('sharpness')
            ]
            
            # Add embeddings if present
            if embeddings_dict.get('statistical'):
                columns.append('embedding_statistical')
                columns.append('has_embedding_statistical')
                values_placeholders.extend(['%s', 'TRUE'])
                values.extend([embeddings_dict['statistical'], True])
            
            if embeddings_dict.get('clip'):
                columns.append('embedding_clip')
                columns.append('has_embedding_clip')
                values_placeholders.extend(['%s', 'TRUE'])
                values.extend([embeddings_dict['clip'], True])
            
            # ... repeat for yolo and action ...
            
            query = f"""
                INSERT INTO images ({', '.join(columns)})
                VALUES ({', '.join(values_placeholders)})
                ON CONFLICT (image_hash) DO UPDATE SET
                    {', '.join([f'{col}=EXCLUDED.{col}' for col in columns[4:]])}
            """
            
            cursor.execute(query, values)
        
        self.conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error adding image: {e}")
        self.conn.rollback()
        return False
```

**Update search_images() method**:
```python
def search_images(self, embedding, limit, embedding_model="statistical"):
    """
    Args:
        embedding: List[float] - vector embedding
        limit: int - max results
        embedding_model: str - which model's embedding to use for search
    """
    try:
        column_name = f"embedding_{embedding_model}"
        has_column_name = f"has_embedding_{embedding_model}"
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = f"""
                SELECT * FROM search_similar_images(%s, %s, %s)
                WHERE {has_column_name} = TRUE
                ORDER BY {column_name} <=> %s
                LIMIT %s
            """
            cursor.execute(query, (embedding, embedding_model, limit))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error searching images: {e}")
        return []
```

**Add new method for true multi-model search**:
```python
def multi_model_search(self, embeddings_dict, weights=None, limit=10):
    """
    Search using multiple models simultaneously
    
    Args:
        embeddings_dict: {'clip': [...], 'yolo': [...], ...}
        weights: {'clip': 0.5, 'yolo': 0.3, 'action': 0.2}
        limit: max results
    
    Returns: Combined results with weighted distances
    """
    if weights is None:
        # Equal weights
        weights = {model: 1.0/len(embeddings_dict) for model in embeddings_dict}
    
    try:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # For each model, get distances
            queries = []
            for model, embedding in embeddings_dict.items():
                if embedding:
                    weight = weights.get(model, 0)
                    col = f"embedding_{model}"
                    queries.append(f"""
                        ({col} <=> %s) * {weight} as {model}_distance
                    """)
            
            if not queries:
                return []
            
            # Combined query with weighted distances
            full_query = f"""
                SELECT 
                    image_id, file_path, metadata,
                    {' + '.join([f"({col} <=> %s) * {weights.get(col.replace('embedding_', ''), 0)}" 
                                for col in [f"embedding_{m}" for m in embeddings_dict]])} 
                    as combined_distance
                FROM images
                WHERE {' OR '.join([f"has_embedding_{m} = TRUE" for m in embeddings_dict])}
                ORDER BY combined_distance
                LIMIT %s
            """
            
            all_embeddings = [e for e in embeddings_dict.values() if e]
            cursor.execute(full_query, (*all_embeddings, limit))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error in multi-model search: {e}")
        return []
```

**Impact**: Significant changes to database interface
**Effort**: ~2-3 hours

---

#### 3. core.py (File: /home/pi/hybridrag/image/core.py)

**Issue 1: Missing process_image_file() method**

**Location**: Called at lines 1285 and 1316 but not defined
**Impact**: Critical - image processing completely broken

**Implementation Required**:
```python
class ImageProcessor:
    # ... existing code ...
    
    def process_image_file(self, file_path: str, callback=None):
        """
        Process single image file: analyze, embed, and store
        
        Args:
            file_path: Path to image file
            callback: Called with ImageData after processing
        
        Returns: True if successful, False otherwise
        """
        try:
            self.stats.increment_embed_processed()
            
            # 1. Analyze image
            features = self.analyzer.analyze_image(file_path)
            if not features:
                self.stats.increment_embed_errors()
                return False
            
            # 2. Calculate image hash
            image_hash = self._get_file_hash(file_path)
            if not image_hash:
                self.stats.increment_embed_errors()
                return False
            
            # 3. Create embedding with configured model
            embed_start = time.time()
            embedding = self.embedder.create_embedding(file_path, features)
            embed_time = time.time() - embed_start
            
            # 4. Build ImageData
            image_data = ImageData(
                image_id=datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                file_path=file_path,
                features=features,
                embedding=embedding,
                timestamp=datetime.now().isoformat(),
                image_hash=image_hash
            )
            
            # 5. Store in database
            if self.db_manager.add_image(image_data, self.config.embedding_model):
                self.processed_files.add(file_path)
                self.stats.increment_embed_success(embed_time)
                
                if callback:
                    callback(image_data)
                
                return True
            else:
                self.stats.increment_embed_errors()
                return False
                
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            self.stats.increment_embed_errors()
            return False
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash: {e}")
            return ""
```

**Issue 2: Add multi-model embedding support to ImageEmbedder**

**Current limitation**: Creates one embedding at a time
**Enhancement**: Add batch/multi-model methods

```python
class ImageEmbedder:
    # ... existing code ...
    
    def create_embeddings_all_models(self, image_path: str, features: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Create embeddings using all available models
        
        Returns: {'statistical': [...], 'clip': [...], 'yolo': [...], 'action': [...]}
        """
        embeddings = {}
        
        # Statistical (always works)
        try:
            image = Image.open(image_path)
            img_array = np.array(image)
            stat_emb = self._create_statistical_embedding(img_array, features)
            embeddings['statistical'] = stat_emb.tolist()
        except Exception as e:
            logger.warning(f"Statistical embedding failed: {e}")
        
        # CLIP
        if AVAILABLE_MODELS.get('clip'):
            try:
                clip_embedder = ImageEmbedder(model_name="clip")
                embeddings['clip'] = clip_embedder.create_embedding(image_path, features)
            except Exception as e:
                logger.warning(f"CLIP embedding failed: {e}")
        
        # YOLO
        if AVAILABLE_MODELS.get('yolo'):
            try:
                yolo_embedder = ImageEmbedder(model_name="yolo")
                embeddings['yolo'] = yolo_embedder.create_embedding(image_path, features)
            except Exception as e:
                logger.warning(f"YOLO embedding failed: {e}")
        
        # Action
        if AVAILABLE_MODELS.get('action'):
            try:
                action_embedder = ImageEmbedder(model_name="action")
                embeddings['action'] = action_embedder.create_embedding(image_path, features)
            except Exception as e:
                logger.warning(f"Action embedding failed: {e}")
        
        return embeddings
```

**Update ImageProcessor to support multi-model processing**:
```python
class ImageProcessor:
    
    def __init__(self, config, stats, db_manager, embedding_models=None):
        """
        Args:
            embedding_models: List of models to embed ['statistical', 'clip', ...]
                            or None for default single model
        """
        self.config = config
        self.stats = stats
        self.db_manager = db_manager
        self.analyzer = ImageAnalyzer()
        
        # Support multiple embeddings
        self.embedding_models = embedding_models or [config.embedding_model]
        self.embedders = {
            model: ImageEmbedder(model_name=model) 
            for model in self.embedding_models
        }
        self.processed_files = set()
    
    def process_image_file_multi_model(self, file_path: str, callback=None):
        """
        Process image with multiple embedding models
        
        Similar to process_image_file but creates all embeddings
        """
        try:
            features = self.analyzer.analyze_image(file_path)
            image_hash = self._get_file_hash(file_path)
            
            # Create embeddings for all configured models
            embeddings_dict = {}
            for model in self.embedding_models:
                embed_start = time.time()
                embedding = self.embedders[model].create_embedding(file_path, features)
                embeddings_dict[model] = embedding
                self.stats.increment_embed_success(time.time() - embed_start)
            
            # Store all embeddings
            image_data = ImageData(
                image_id=datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                file_path=file_path,
                features=features,
                embedding=embeddings_dict.get(self.config.embedding_model),
                timestamp=datetime.now().isoformat(),
                image_hash=image_hash
            )
            
            # Store in database with all embeddings
            if self.db_manager.add_image(
                image_data, 
                self.config.embedding_model,
                embeddings_dict=embeddings_dict
            ):
                self.processed_files.add(file_path)
                if callback:
                    callback(image_data)
                return True
            else:
                self.stats.increment_embed_errors()
                return False
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            self.stats.increment_embed_errors()
            return False
```

**Impact**: Adds substantial new functionality
**Effort**: ~3-4 hours

---

### PRIORITY 2: IMPORTANT (RECOMMENDED)

#### 4. image.py GUI Updates

**File**: /home/pi/hybridrag/image/image.py

**Modification 1: Add model selection in Process tab** (around line 385)
```python
def create_process_tab(self):
    # ... existing code ...
    
    # Add model selection
    ttk.Label(control_frame, text="Embedding Model(s):").grid(row=X, column=0, sticky="w")
    
    # Checkboxes for multi-model selection
    self.embedding_models_var = {
        'statistical': tk.BooleanVar(value=True),
        'clip': tk.BooleanVar(value=False),
        'yolo': tk.BooleanVar(value=False),
        'action': tk.BooleanVar(value=False)
    }
    
    models_frame = ttk.Frame(control_frame)
    for model in ['statistical', 'clip', 'yolo', 'action']:
        ttk.Checkbutton(
            models_frame, 
            text=model, 
            variable=self.embedding_models_var[model]
        ).pack(side="left", padx=5)
```

**Modification 2: Update search_images() to support model selection** (line 1980+)
```python
def search_images(self):
    # ... existing code ...
    
    # Add model selection for search
    search_model = self.search_model_var.get()  # New dropdown
    
    # Use specific model for search
    embedder = ImageEmbedder(model_name=search_model)
    embedding = embedder.create_embedding(image_path, features)
```

**Effort**: ~1-2 hours

---

## MIGRATION STRATEGY

### Step 1: Database Migration (1-2 hours)
1. Backup existing database
2. Modify schema.sql with new columns
3. Run migration on test database
4. Verify data integrity

### Step 2: Backend Updates (3-4 hours)
1. Update pgvector_db.py add_image() method
2. Update search methods to accept model parameter
3. Implement multi-model search
4. Add to ImageEmbedder for convenience methods
5. Complete ImageProcessor.process_image_file()

### Step 3: Frontend Updates (1-2 hours)
1. Add model selection UI
2. Update search UI for model choice
3. Add progress tracking for multi-model processing

### Step 4: Testing & Validation (2-3 hours)
1. Test single-model backward compatibility
2. Test multi-model processing
3. Test mixed-model search
4. Performance benchmarking

### Total Estimated Time: 7-13 hours

---

## QUICK START: MINIMAL VIABLE CHANGES

If you want to get multi-model support working ASAP with minimal changes:

1. **Only modify schema.sql** - Add embedding_clip, embedding_yolo, embedding_action columns
2. **Update pgvector_db.py add_image()** - Store all provided embeddings
3. **Fix core.py process_image_file()** - Implement the missing method
4. **Update ImageProcessor** - Create embeddings for all models before calling add_image

This creates parallel embedding storage without fully optimizing the search interface. Effort: ~4-5 hours

---

## TESTING CHECKLIST

After modifications:
- [ ] Database connection works
- [ ] Single embedding storage works
- [ ] Multi-embedding storage works
- [ ] Vector search works with each model
- [ ] Model mismatch check still works
- [ ] GUI doesn't crash on model selection
- [ ] Statistics tracking works correctly
- [ ] Performance acceptable for your use case

---

