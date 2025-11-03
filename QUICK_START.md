# Quick Start Checklist

Follow these steps to run your first evaluation in 5 minutes:

## ✅ Pre-Flight Checklist

- [ ] Python 3.7+ installed? → `python --version`
- [ ] Files downloaded? → Check `bedrock/` folder exists
- [ ] Dependencies installed? → `pip install -r requirements.txt`
- [ ] AWS credentials ready? → Access Key ID and Secret Key

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Add AWS Credentials

Open `config.py` and replace:
```python
AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID_HERE"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_ACCESS_KEY_HERE"
```
With your actual credentials.

### Step 3: Run Test

```bash
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --max-prompts 1
```

**Expected:** You'll see progress messages and a summary table.

## 📊 View Results

Check the `results/` folder for 3 CSV files:
- `summary_TIMESTAMP.csv` - High-level comparison
- `detailed_results_TIMESTAMP.csv` - Full details
- `comparison_TIMESTAMP.csv` - Side-by-side comparison

## ❓ Need Help?

- **Full guide:** See [MANUAL_GUIDE.md](MANUAL_GUIDE.md)
- **Troubleshooting:** Check the Troubleshooting section in MANUAL_GUIDE.md

## 🎯 Common Commands

```bash
# Test with 1 prompt (recommended first)
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --max-prompts 1

# Test with 5 prompts
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --max-prompts 5

# Run all prompts
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json

# Use your own file
python evaluate.py --prompts your_file.csv
```

That's it! You're evaluating LLM models! 🎉

