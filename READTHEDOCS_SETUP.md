# ReadTheDocs Setup Instructions for PandaDock

This document provides step-by-step instructions to set up and update your ReadTheDocs documentation for PandaDock v3.0.

## What's Been Created

The following files have been created/updated for ReadTheDocs:

✅ **Configuration Files:**
- `mkdocs.yml` - MkDocs configuration with Material theme
- `.readthedocs.yaml` - ReadTheDocs build configuration
- `docs/requirements.txt` - Python dependencies for building docs

✅ **Documentation Files:**
- `docs/index.md` - Updated main page with all new algorithms
- All documentation structure in place

---

## Step 1: Commit and Push Documentation Files

First, add all the new documentation files to git and push them:

```bash
cd /mnt/cce9630f-8b3b-4312-932d-ff04311ba514/SSD/PandaDock

# Check what files will be added
git status

# Add documentation files
git add mkdocs.yml
git add .readthedocs.yaml
git add docs/requirements.txt
git add docs/index.md
git add docs/algorithms/
git add docs/scoring/
git add docs/cli/
git add docs/tutorials/

# Commit
git commit -m "Update documentation for v3.0 with all new algorithms

- Added mkdocs.yml configuration with Material theme
- Created .readthedocs.yaml for automated builds
- Updated docs/index.md with 10+ new algorithms
- Added comprehensive algorithm documentation
- Configured for ReadTheDocs automatic builds"

# Push to GitHub
git push origin latest-v3.0
```

---

## Step 2: Configure ReadTheDocs Website

### 2.1 Log in to ReadTheDocs

1. Go to [https://readthedocs.org/](https://readthedocs.org/)
2. Sign in with your GitHub account

### 2.2 Import Your Repository (if not already imported)

If PandaDock is already connected:
1. Go to [https://readthedocs.org/dashboard/](https://readthedocs.org/dashboard/)
2. Find "PandaDock" in your project list
3. Skip to Step 2.3

If you need to import it:
1. Click "Import a Project"
2. Select "pritampanda15/PandaDock" from your GitHub repositories
3. Click "Next"
4. Confirm the project details
5. Click "Finish"

### 2.3 Configure Version Settings

1. Go to your PandaDock project on ReadTheDocs
2. Click "Admin" → "Versions"
3. **Activate the `latest-v3.0` branch:**
   - Find `latest-v3.0` in the list
   - Click "Edit"
   - Check "Active"
   - Check "Public"
   - Click "Save"

4. **Set default version:**
   - Go to "Admin" → "Advanced Settings"
   - Find "Default version"
   - Select `latest-v3.0`
   - Scroll down and click "Save"

### 2.4 Configure Build Settings

1. Go to "Admin" → "Advanced Settings"
2. Verify these settings:
   - **Documentation type**: `mkdocs`
   - **Requirements file**: `docs/requirements.txt`
   - **Python interpreter**: `CPython 3.x`
3. Click "Save"

---

## Step 3: Trigger Build

### Manual Build:

1. Go to your project dashboard
2. Click "Builds"
3. Click "Build Version: latest-v3.0"
4. Wait for the build to complete (usually 2-5 minutes)

### Automatic Builds:

ReadTheDocs will automatically rebuild when you push to the `latest-v3.0` branch.

---

## Step 4: Verify Documentation

Once the build succeeds:

1. Visit: `https://pandadock.readthedocs.io/en/latest-v3.0/`
2. Or: `https://pandadock.readthedocs.io/en/latest/` (if set as default)

### Check These Pages:

- ✅ Home page shows all new algorithms
- ✅ Algorithms page lists all 10+ algorithms
- ✅ GPU algorithms are documented
- ✅ Specialized modes (flex, metal, ML, tethered) are listed
- ✅ All 6 scoring functions are documented
- ✅ Installation instructions are up to date

---

## Step 5: Set Up Webhooks (Optional - for automatic builds)

1. Go to "Admin" → "Integrations"
2. Click "Add integration"
3. Select "GitHub incoming webhook"
4. Copy the webhook URL
5. Go to your GitHub repository settings
6. Navigate to "Webhooks" → "Add webhook"
7. Paste the ReadTheDocs webhook URL
8. Set "Content type" to `application/json`
9. Select "Just the push event"
10. Click "Add webhook"

Now documentation will rebuild automatically on every push!

---

## Step 6: Update Old Documentation (Clean Up)

If you want to keep old documentation as an archive:

1. In ReadTheDocs, go to "Admin" → "Versions"
2. Find your old `main` branch
3. Either:
   - **Option A**: Deactivate it (hide it)
   - **Option B**: Keep it active as "v0.x - Legacy"

---

## Troubleshooting

### Build Fails

**Check build logs:**
1. Go to "Builds" → Click on the failed build
2. Check the error message

**Common issues:**

#### Missing Dependencies
**Error:** `ModuleNotFoundError: No module named 'mkdocs_material'`

**Fix:** Check that `docs/requirements.txt` includes all dependencies:
```txt
mkdocs>=1.5.0
mkdocs-material>=9.0.0
mkdocs-minify-plugin>=0.7.0
```

#### Configuration Error
**Error:** `Error reading config file`

**Fix:** Validate `mkdocs.yml` syntax:
```bash
# Locally test the build
cd /mnt/cce9630f-8b3b-4312-932d-ff04311ba514/SSD/PandaDock
pip install mkdocs mkdocs-material
mkdocs build
```

#### .readthedocs.yaml Issues
**Error:** `Invalid configuration file`

**Fix:** Ensure `.readthedocs.yaml` follows the schema. The file has been created correctly, but if issues occur, check:
```yaml
version: 2  # Must be exactly 2, not "2" or 2.0
```

### Documentation Not Updating

1. **Check if build succeeded:**
   - Go to "Builds" tab
   - Latest build should be green (✓)

2. **Clear browser cache:**
   - Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)

3. **Check version:**
   - Make sure you're viewing the correct version
   - URL should be `/en/latest-v3.0/` or `/en/latest/`

### 404 Errors on Some Pages

**Cause:** Missing markdown files referenced in `mkdocs.yml`

**Fix:** Create placeholder files for any missing pages:
```bash
cd docs
mkdir -p tutorials guide api development
touch tutorials/quickstart.md guide/best-practices.md
```

---

## Updating Documentation in the Future

### To Update Existing Docs:

```bash
cd /mnt/cce9630f-8b3b-4312-932d-ff04311ba514/SSD/PandaDock

# Edit documentation files
vim docs/algorithms/enhanced-hierarchical-cpu.md

# Commit and push
git add docs/
git commit -m "Update: improved algorithm documentation"
git push origin latest-v3.0

# ReadTheDocs will auto-rebuild (if webhook is set up)
```

### To Add New Pages:

1. Create the new markdown file:
```bash
touch docs/tutorials/advanced-gpu-docking.md
```

2. Add it to `mkdocs.yml` navigation:
```yaml
nav:
  - Tutorials:
      - Advanced GPU Docking: tutorials/advanced-gpu-docking.md
```

3. Commit and push:
```bash
git add docs/tutorials/advanced-gpu-docking.md mkdocs.yml
git commit -m "Add: advanced GPU docking tutorial"
git push origin latest-v3.0
```

---

## Testing Documentation Locally

Before pushing to ReadTheDocs, test locally:

```bash
cd /mnt/cce9630f-8b3b-4312-932d-ff04311ba514/SSD/PandaDock

# Install MkDocs
pip install -r docs/requirements.txt

# Serve locally
mkdocs serve

# Open browser to http://127.0.0.1:8000
```

This lets you preview changes before publishing.

---

## Documentation Structure Created

```
PandaDock/
├── mkdocs.yml                      # Main configuration
├── .readthedocs.yaml               # ReadTheDocs build config
├── docs/
│   ├── requirements.txt            # Build dependencies
│   ├── index.md                    # Main page (UPDATED ✓)
│   ├── getting-started.md          # Installation guide
│   ├── algorithms/
│   │   ├── index.md                # Algorithms overview (EXISTS)
│   │   ├── enhanced-hierarchical-cpu.md (EXISTS)
│   │   └── ...                     # Other algorithm pages
│   ├── scoring/
│   │   └── index.md                # Scoring functions
│   ├── cli/
│   │   └── index.md                # CLI reference
│   └── tutorials/
│       └── index.md                # Tutorials
```

---

## Important URLs

- **Live Documentation**: https://pandadock.readthedocs.io/
- **ReadTheDocs Dashboard**: https://readthedocs.org/projects/pandadock/
- **Build Logs**: https://readthedocs.org/projects/pandadock/builds/
- **Settings**: https://readthedocs.org/dashboard/pandadock/
- **GitHub Repository**: https://github.com/pritampanda15/PandaDock

---

## Next Steps

1. ✅ Commit and push documentation files
2. ✅ Log in to ReadTheDocs
3. ✅ Activate `latest-v3.0` branch
4. ✅ Set as default version
5. ✅ Trigger build
6. ✅ Verify documentation loads correctly
7. ✅ Set up webhook for automatic builds
8. 📝 Create additional documentation pages as needed

---

## Support

If you encounter issues:

1. **ReadTheDocs Community**: https://readthedocs.org/support/
2. **MkDocs Documentation**: https://www.mkdocs.org/
3. **Material Theme**: https://squidfunk.github.io/mkdocs-material/
4. **Check Build Logs**: Always check the build logs for specific errors

---
