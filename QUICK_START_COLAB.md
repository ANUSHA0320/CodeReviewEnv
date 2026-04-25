# ⚡ Quick Reference: Colab Training Run

**Goal**: Get real training evidence in 20 minutes for hackathon submission.

---

## 🚀 5-Step Process

### 1️⃣ Open Notebook (1 min)
- Option A: Upload `training_trl_colab.ipynb` to https://colab.research.google.com/
- Option B: Click Colab badge in README

### 2️⃣ Enable GPU (1 min)
- **Runtime → Change runtime type**
- Set: **Hardware accelerator = T4 GPU**
- Click **Save**

### 3️⃣ Run All Cells (15 min)
- **Runtime → Run all** (or Ctrl+F9)
- Wait for completion (watch progress output)

### 4️⃣ Download Results (2 min)
- Click **📁 Files** icon (left sidebar)
- Navigate to `results/` folder
- Right-click → Download each file:
  - `training_plot.png`
  - `baseline_comparison.png`  
  - `training_summary.txt`

### 5️⃣ Commit to Repo (2 min)
```bash
cp ~/Downloads/*.png results/
cp ~/Downloads/*.txt results/
git add results/
git commit -m "Add TRL training evidence"
git push
```

---

## ✅ Success Criteria

After completion, you should have:
- ✅ Reward curve showing improvement trend
- ✅ Baseline comparison (trained > random)
- ✅ Summary with metrics (avg reward, improvement %)
- ✅ Files committed to `results/` in your repo

---

## 🐛 Quick Fixes

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size=2` in Cell 6 |
| Module not found | Add `!pip install -e .` to Cell 2 |
| Downloads missing | Run `files.download('results/training_plot.png')` in new cell |

---

## 📊 Expected Results

- **Random Baseline**: ~0.05 avg reward
- **Trained Agent**: ~0.40 avg reward
- **Improvement**: 8-10x better than random
- **Training Time**: ~15 minutes on T4 GPU

---

## ⏰ Timeline Tracker

- [ ] **Minute 0-1**: Upload & enable GPU
- [ ] **Minute 1-4**: Install dependencies
- [ ] **Minute 4-9**: Download distilgpt2 model
- [ ] **Minute 9-19**: Train 30 episodes
- [ ] **Minute 19-20**: Generate plots
- [ ] **Minute 20-22**: Download files
- [ ] **Minute 22-25**: Commit to repo

**Total: 25 minutes**

---

## 🎯 Why This Matters

This gives you:
- **20% of hackathon score** (Showing Improvement criterion)
- Evidence that your environment enables learning
- Proof that agent performs better than random
- Reproducible training pipeline judges can verify

---

**See [COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md) for detailed guide.**
