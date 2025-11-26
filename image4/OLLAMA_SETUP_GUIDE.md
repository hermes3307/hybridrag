# Ollama Setup Guide for LLM Assistant

## What is Ollama?

Ollama is a tool that lets you run Large Language Models (LLMs) **locally** on your computer. This means:
- ✅ **No API costs** - Completely free
- ✅ **Privacy** - Your data never leaves your machine
- ✅ **Offline** - Works without internet
- ✅ **Fast** - No network latency

## Installation

### Option 1: One-Line Install (Linux/Mac)

```bash
curl https://ollama.ai/install.sh | sh
```

### Option 2: Manual Install

1. **Visit**: https://ollama.ai/
2. **Download** the installer for your OS:
   - Linux: Download and run installer
   - Mac: Download .dmg file
   - Windows: Download .exe file
3. **Install** following the prompts

### Verify Installation

```bash
ollama --version
```

You should see: `ollama version x.x.x`

## Quick Start

### Step 1: Pull a Model

Download a model to use:

```bash
# Recommended for general use (small, fast)
ollama pull llama3.2

# Or choose from other models:
ollama pull mistral        # Good for coding
ollama pull codellama      # Specialized for code
ollama pull phi            # Very small, runs on anything
```

### Step 2: Test Ollama

```bash
ollama run llama3.2
```

Type a message and press Enter. If you see a response, Ollama is working!

Type `/bye` to exit.

### Step 3: Configure in GUI

1. Open the Image Processing application
2. Go to **Configuration** tab
3. Find **LLM Assistant Configuration**
4. Set:
   - Provider: "Ollama (Local)"
   - Model: "llama3.2" (or whichever you pulled)
5. Click **"Test LLM Connection"**
6. Should see: ✓ Connected to Ollama!
7. Click **"Apply & Reinitialize"**
8. Go to **LLM Assistant** tab and start chatting!

## Available Models

### Small & Fast (Recommended for Most Users)

| Model | Size | Best For |
|-------|------|----------|
| llama3.2 | 2GB | General chat, commands |
| phi | 1.6GB | Very fast, basic tasks |
| mistral | 4GB | Balanced performance |

### Medium (Better Quality)

| Model | Size | Best For |
|-------|------|----------|
| llama3.1:8b | 4.7GB | Better reasoning |
| mistral-nemo | 7GB | High quality responses |

### Large (Best Quality, Needs Good Hardware)

| Model | Size | Best For |
|-------|------|----------|
| llama3.1:70b | 40GB | Professional use |
| codellama:34b | 19GB | Advanced coding |

## Managing Models

### List installed models
```bash
ollama list
```

### Pull a specific model
```bash
ollama pull <model-name>
```

### Remove a model
```bash
ollama rm <model-name>
```

### Update a model
```bash
ollama pull <model-name>
```

## Troubleshooting

### Error: "Connection refused" or "Cannot connect to Ollama"

**Problem**: Ollama service is not running

**Solutions**:

1. **Start Ollama manually**:
   ```bash
   ollama serve
   ```

2. **Check if Ollama is running**:
   ```bash
   ps aux | grep ollama
   ```

3. **On Linux, enable autostart**:
   ```bash
   sudo systemctl enable ollama
   sudo systemctl start ollama
   ```

### Error: "Model not found"

**Problem**: You haven't downloaded the model yet

**Solution**:
```bash
ollama pull llama3.2
```

### Error: "Out of memory"

**Problem**: Model too large for your system

**Solutions**:
1. Use a smaller model (phi, llama3.2:1b)
2. Close other applications
3. Upgrade system RAM

### Slow Performance

**Problem**: Model runs but is very slow

**Solutions**:
1. Use smaller model (phi, llama3.2)
2. Close other applications
3. Enable GPU acceleration (if available)
4. Reduce context window size

## Advanced Configuration

### Change Ollama Port

Default port is 11434. To change:

```bash
# Set environment variable
export OLLAMA_HOST=0.0.0.0:11435
```

Then update `.env` file:
```
OLLAMA_HOST=http://localhost:11435
```

### Enable GPU Acceleration

Ollama automatically uses GPU if available (NVIDIA, AMD, Apple Silicon).

Check GPU usage:
```bash
# NVIDIA
nvidia-smi

# AMD (Linux)
rocm-smi
```

### Run on Different Machine

You can run Ollama on a server and connect from your GUI:

**On server**:
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

**In `.env` file**:
```
OLLAMA_HOST=http://192.168.1.100:11434
```

## Performance Tips

### 1. Choose Right Model Size

- **< 8GB RAM**: Use phi or llama3.2:1b
- **8-16GB RAM**: Use llama3.2 or mistral
- **16-32GB RAM**: Use llama3.1:8b
- **> 32GB RAM**: Use llama3.1:70b or larger

### 2. Use SSD Storage

Models load faster from SSD than HDD.

### 3. Close Background Apps

Free up RAM for better model performance.

### 4. Use GPU

If you have a GPU, Ollama will use it automatically for much faster performance.

## Comparison with Cloud APIs

| Feature | Ollama (Local) | Cloud APIs |
|---------|---------------|------------|
| Cost | Free | Pay per use |
| Privacy | Complete | Data sent to cloud |
| Internet | Not needed | Required |
| Speed | Depends on hardware | Consistent |
| Model Quality | Varies by size | Generally higher |
| Setup | Need to install | Just API key |

## Recommended Setup for This Project

For the Image Processing System LLM Assistant:

**Minimum**:
- Model: `phi` or `llama3.2:1b`
- RAM: 4GB available
- Storage: 5GB free

**Recommended**:
- Model: `llama3.2` or `mistral`
- RAM: 8GB available
- Storage: 10GB free

**Optimal**:
- Model: `llama3.1:8b` or `mistral-nemo`
- RAM: 16GB+ available
- Storage: 15GB free
- GPU: NVIDIA/AMD/Apple Silicon

## Example Commands for This Project

Once Ollama is set up with the GUI:

```
You: Download 10 images
Assistant: I'll download 10 images for you.
[Downloads start...]

You: Process new images
Assistant: I'll process and embed the images.
[Processing starts...]

You: Show system status
Assistant: [Shows current status]
```

## Getting Help

- **Ollama Documentation**: https://ollama.ai/
- **GitHub**: https://github.com/ollama/ollama
- **Discord**: https://discord.gg/ollama
- **Model Library**: https://ollama.ai/library

## Next Steps

1. ✅ Install Ollama
2. ✅ Pull a model (`ollama pull llama3.2`)
3. ✅ Test it (`ollama run llama3.2`)
4. ✅ Configure in GUI (Configuration tab)
5. ✅ Test connection
6. ✅ Use LLM Assistant!

---

**Note**: The first time you use a model, it may take a moment to load into memory. Subsequent uses will be faster.