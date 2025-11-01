# Free Deployment Alternatives for Flask + PyTorch App

## Top Recommendations (Best to Good)

### 1. 🥇 **Render** (Highly Recommended)
**Why it's great:**
- ✅ **Free tier available** (with limitations)
- ✅ Supports Flask/Python natively
- ✅ Can handle PyTorch (up to 500MB free build size)
- ✅ Auto-deploy from Git (GitHub, GitLab)
- ✅ Supports environment variables
- ✅ Automatic HTTPS
- ✅ Easy Procfile support

**Limitations:**
- Free tier spins down after 15 minutes of inactivity (takes ~30s to wake up)
- Build size limit: 500MB (PyTorch may push this)
- Bandwidth limits

**Setup:**
```yaml
# render.yaml (optional, or use web dashboard)
services:
  - type: web
    name: agricultural-cost-predictor
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
    envVars:
      - key: PORT
        value: 8000
```

**Files needed:**
- `Procfile` (already have)
- `requirements.txt` (already have)
- `runtime.txt` (already have)

**URL:** https://render.com

---

### 2. 🥈 **Railway** (Very Easy)
**Why it's great:**
- ✅ **Free tier with $5 credit/month**
- ✅ Extremely easy setup (just connect Git repo)
- ✅ Auto-detects Python/Flask
- ✅ Good for ML apps
- ✅ No Procfile needed (auto-detects)
- ✅ Supports large dependencies

**Limitations:**
- Free tier: $5 credit/month (~100 hours of runtime)
- After credit expires, may need to pay

**Setup:**
1. Connect GitHub repo
2. Railway auto-detects Flask app
3. Deploy!

**URL:** https://railway.app

---

### 3. 🥉 **Fly.io** (Docker-based)
**Why it's great:**
- ✅ **Free tier: 3 shared-cpu VMs**
- ✅ Good for containerized apps
- ✅ Global edge network
- ✅ No cold starts

**Limitations:**
- Need to create Dockerfile
- Free tier: Limited resources
- More complex setup

**Required file: `Dockerfile`**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "300"]
```

**URL:** https://fly.io

---

### 4. **PythonAnywhere**
**Why it's good:**
- ✅ **Free tier available**
- ✅ Python-focused (perfect for Flask)
- ✅ Easy manual deployment
- ✅ Web-based console

**Limitations:**
- Free tier: Limited CPU time
- Free tier: App must be reloaded daily (manually)
- Smaller community than others
- May have issues with PyTorch size

**URL:** https://www.pythonanywhere.com

---

### 5. **Google Cloud Run** (Free Tier)
**Why it's good:**
- ✅ **Free tier: 2 million requests/month**
- ✅ Pay only for what you use
- ✅ Auto-scaling
- ✅ Good for ML apps

**Limitations:**
- Requires Dockerfile
- Requires Google Cloud account setup
- More complex configuration

**URL:** https://cloud.google.com/run

---

## Comparison Table

| Platform | Free Tier | Ease of Setup | PyTorch Support | Best For |
|----------|-----------|---------------|-----------------|----------|
| **Render** | ✅ Yes (limits) | ⭐⭐⭐⭐⭐ Very Easy | ✅ Yes | Most users |
| **Railway** | ✅ $5/month credit | ⭐⭐⭐⭐⭐ Very Easy | ✅ Yes | Quick deploy |
| **Fly.io** | ✅ 3 VMs free | ⭐⭐⭐ Medium | ✅ Yes | Advanced users |
| **PythonAnywhere** | ✅ Yes (limited) | ⭐⭐⭐⭐ Easy | ⚠️ May be tight | Python-only |
| **Cloud Run** | ✅ Generous | ⭐⭐ Complex | ✅ Yes | Cloud users |

---

## Recommendation: **Render or Railway**

For this Flask + PyTorch app, I recommend:

### **Primary Choice: Render**
- Best balance of ease and features
- Good PyTorch support
- Similar to Crane Cloud workflow
- Free tier sufficient for testing/demos

### **Secondary Choice: Railway**
- Easiest deployment (just connect Git)
- $5/month credit covers most usage
- Very developer-friendly

---

## Quick Setup Guides

### Render Setup (5 minutes)
1. Sign up at https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub/GitLab repo
4. Configure:
   - **Name:** `agricultural-cost-predictor`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Click "Create Web Service"
6. Wait for build (5-10 minutes for PyTorch)
7. Done! Get URL like: `https://your-app.onrender.com`

### Railway Setup (2 minutes)
1. Sign up at https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Flask app
5. Wait for deployment
6. Done! Get URL automatically

---

## Important Notes for All Platforms

### PyTorch Installation
PyTorch is large (~500MB). All platforms can handle it, but:
- Build times will be longer (5-15 minutes)
- Use CPU-only PyTorch if possible (lighter):
  ```txt
  torch==1.9.0+cpu  # Instead of full torch
  ```

### Model Files Size
Your model files are relatively small:
- `best_normalized_model.pth`: ~600KB
- `normalized_preprocessing.pkl`: ~2KB
- Total: < 1MB ✅

These are fine for Git and all platforms.

### Memory Requirements
- App needs ~500MB-1GB RAM for PyTorch
- Most free tiers provide 512MB-1GB ✅

### Cold Starts
- Render: ~30 seconds after inactivity
- Railway: Minimal
- Fly.io: None (always running)
- Cloud Run: ~5-10 seconds

---

## Alternative: Use PyTorch CPU-Only (Lighter)

To reduce deployment size and speed, consider using CPU-only PyTorch:

```txt
# requirements.txt (lighter version)
torch==1.9.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
flask>=2.0.0
numpy>=1.21.0,<2.0.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.0.0
gunicorn>=20.0.0
```

This reduces PyTorch from ~500MB to ~150MB and speeds up builds.

---

## My Recommendation

**Start with Render** - it's the most balanced option:
- Free tier
- Easy setup
- Good PyTorch support
- Similar workflow to Crane Cloud
- Good documentation

If Render doesn't work or you want even easier setup, try **Railway**.

