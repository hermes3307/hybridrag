# Installing All Face Recognition Models

## Overview

This guide will help you install support for all 5 face recognition models:
- ✅ **FaceNet** - Already installed
- ✅ **ArcFace** - Already installed (via InsightFace)
- 🔄 **VGGFace2** - Needs DeepFace
- 🔄 **InsightFace** - Need to install
- ✅ **Statistical** - Built-in (no install needed)

---

## Quick Install (All Models)

```bash
cd /home/pi/hybridrag/faces10_mm
source venv/bin/activate

# Install in recommended order
pip install numpy<2.0.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow tf-keras
pip install -r requirements.txt
```

---

## Step-by-Step Installation

### Step 1: Activate Virtual Environment

```bash
cd /home/pi/hybridrag/faces10_mm
source venv/bin/activate
```

### Step 2: Install NumPy (Correct Version)

```bash
pip install 'numpy<2.0.0'
```

**Why?** Many face recognition models require NumPy 1.x

### Step 3: Install PyTorch (Already Done)

```bash
pip install torch torchvision
```

### Step 4: Install TensorFlow + tf-keras

```bash
pip install tensorflow tf-keras
```

This enables:
- VGGFace2
- DeepFace models
- Other Keras-based models

### Step 5: Install Face Recognition Libraries

```bash
pip install insightface deepface facenet-pytorch
```

### Step 6: Install Additional Dependencies

```bash
pip install -r requirements.txt
```

---

## Verify Installation

### Test Each Model:

```bash
# Test Statistical (always works, built-in)
python3 -c "print('✅ Statistical: Built-in, always available')"

# Test FaceNet
python3 -c "from facenet_pytorch import InceptionResnetV1; print('✅ FaceNet OK')"

# Test InsightFace/ArcFace
python3 -c "from insightface.app import FaceAnalysis; print('✅ InsightFace/ArcFace OK')"

# Test DeepFace/VGGFace2
python3 -c "from deepface import DeepFace; print('✅ DeepFace/VGGFace2 OK')"

# Test TensorFlow
python3 -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__} OK')"
```

### All-in-One Test Script:

```bash
python3 << 'EOF'
import sys

models = {
    'Statistical': ('built-in', None),
    'FaceNet': ('facenet-pytorch', 'facenet_pytorch'),
    'InsightFace/ArcFace': ('insightface', 'insightface.app'),
    'DeepFace/VGGFace2': ('deepface', 'deepface'),
    'TensorFlow': ('tensorflow', 'tensorflow'),
}

print("=" * 60)
print("MODEL INSTALLATION STATUS")
print("=" * 60)

for name, (package, import_path) in models.items():
    try:
        if import_path:
            if '.' in import_path:
                module, submodule = import_path.rsplit('.', 1)
                exec(f"from {module} import {submodule}")
            else:
                exec(f"import {import_path}")
        print(f"✅ {name:20} OK")
    except ImportError as e:
        print(f"❌ {name:20} MISSING - pip install {package}")
    except Exception as e:
        print(f"⚠️  {name:20} ERROR: {e}")

print("=" * 60)
EOF
```

---

## Model Requirements Summary

| Model | Library | Size | Additional Notes |
|-------|---------|------|-----------------|
| **Statistical** | Built-in | 0 MB | ✅ No install needed |
| **FaceNet** | facenet-pytorch | ~100 MB | ✅ Already installed |
| **ArcFace** | insightface | ~200 MB | Uses buffalo_l model |
| **VGGFace2** | deepface + TF | ~500 MB | Requires TensorFlow |
| **InsightFace** | insightface | ~200 MB | Same as ArcFace |

**Total Disk Space:** ~1 GB for all models + dependencies

---

## Troubleshooting

### Error: NumPy version conflict

```bash
pip uninstall numpy
pip install 'numpy<2.0.0'
```

### Error: No module named 'tf_keras'

```bash
pip install tf-keras
```

### Error: TensorFlow installation fails

For systems with limited resources:
```bash
# Skip TensorFlow (won't have VGGFace2)
pip install insightface deepface --no-deps
pip install opencv-python pillow gdown
```

### Error: Out of memory during install

Install one at a time:
```bash
pip install insightface
pip install deepface
pip install tensorflow
```

### Error: "externally-managed-environment"

Use virtual environment (already done) or:
```bash
pip install --break-system-packages -r requirements.txt
```

---

## After Installation

### Test All Models with Embedding Manager:

```bash
# Test each model
./run_embedding.sh
# Choose each model 1-5 to verify they work
```

### Check Available Models:

```bash
python3 << 'EOF'
from core import check_embedding_models
available = check_embedding_models()
print("Available models:", available)
EOF
```

---

## Recommended Installation Order

1. **Start Fresh** (if issues):
   ```bash
   cd /home/pi/hybridrag/faces10_mm
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Core Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install 'numpy<2.0.0'
   ```

3. **Install Deep Learning Frameworks**:
   ```bash
   pip install torch torchvision
   pip install tensorflow tf-keras
   ```

4. **Install Face Recognition Libraries**:
   ```bash
   pip install facenet-pytorch
   pip install insightface
   pip install deepface
   ```

5. **Install Remaining Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Verify Everything**:
   ```bash
   python3 -c "from core import check_embedding_models; print(check_embedding_models())"
   ```

---

## Usage After Installation

### Run Single Model:
```bash
./run_embedding.sh
# Choose 1-5 for specific model
```

### Run Multiple Models:
```bash
./run_embedding.sh
# Choose 7 (multi)
# Enter: arcface,vggface2,insightface
```

### Run All Models:
```bash
./run_embedding.sh
# Choose 6 (all)
```

---

## Model Selection Guide

| Use Case | Recommended Model |
|----------|-------------------|
| **General Purpose** | FaceNet (fast, accurate) |
| **Best Accuracy** | ArcFace or InsightFace |
| **Lightweight** | Statistical |
| **Research/Comparison** | All models |
| **Production (Single)** | FaceNet or ArcFace |
| **Production (Ensemble)** | FaceNet + ArcFace + VGGFace2 |

---

## File Size Reference

After full installation:
```
venv/
├── lib/python3.12/site-packages/
│   ├── torch/          ~500 MB
│   ├── tensorflow/     ~400 MB
│   ├── insightface/    ~100 MB
│   ├── deepface/       ~50 MB
│   ├── facenet_pytorch/ ~50 MB
│   └── other/          ~200 MB
└── Total: ~1.3 GB
```

---

## Support Matrix

| Model | CPU | GPU | ONNX | TensorFlow | PyTorch |
|-------|-----|-----|------|------------|---------|
| Statistical | ✅ | - | - | - | - |
| FaceNet | ✅ | ✅ | - | - | ✅ |
| ArcFace | ✅ | ✅ | ✅ | - | - |
| VGGFace2 | ✅ | ✅ | - | ✅ | - |
| InsightFace | ✅ | ✅ | ✅ | - | - |

---

## Quick Commands Reference

```bash
# Activate environment
source venv/bin/activate

# Check what's installed
pip list | grep -E "facenet|insightface|deepface|tensorflow"

# Install missing package
pip install [package-name]

# Test embedding
./run_embedding.sh

# Check database status
PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c "
    SELECT COUNT(*) as total,
           COUNT(embedding_facenet) as facenet,
           COUNT(embedding_arcface) as arcface,
           COUNT(embedding_vggface2) as vggface2,
           COUNT(embedding_insightface) as insightface
    FROM faces;
"
```

---

**Installation Complete!** 🎉

You should now have all 5 face recognition models ready to use!
