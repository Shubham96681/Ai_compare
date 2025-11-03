# Step-by-Step Manual Guide: Running LLM Model Evaluation

This guide will walk you through running the evaluation framework from start to finish.

## Prerequisites Check

Before starting, make sure you have:
- ✅ Python 3.7 or higher installed
- ✅ AWS Account with Bedrock access enabled
- ✅ AWS Access Key ID and Secret Access Key

**Check Python version:**
```bash
python --version
```
or
```bash
python3 --version
```

## Step 1: Download/Clone the Repository

If you haven't already, download or clone the repository:

```bash
# If using Git
git clone <repository-url>
cd bedrock

# OR download ZIP and extract, then navigate to folder
cd bedrock
```

## Step 2: Install Required Python Packages

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Collecting boto3>=1.28.0
Collecting botocore>=1.31.0
...
Successfully installed boto3-x.x.x botocore-x.x.x
```

**If you get errors:**
- Try: `pip3 install -r requirements.txt`
- Or: `python -m pip install -r requirements.txt`

## Step 3: Configure AWS Credentials

Open the `config.py` file in any text editor (Notepad, VS Code, etc.).

### 3.1 Find the AWS Credentials Section

Look for these lines (around line 14-16):

```python
AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID_HERE"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_ACCESS_KEY_HERE"
```

### 3.2 Replace with Your Credentials

```python
AWS_ACCESS_KEY_ID = "AKIAXXXXXXXXXXXXX"  # Your actual Access Key ID
AWS_SECRET_ACCESS_KEY = "your_secret_key_here"  # Your actual Secret Key
```

### 3.3 (Optional) Update AWS Region

If your AWS region is different, update this line (around line 10):

```python
AWS_REGION = "us-east-2"  # Change to your region (e.g., "us-west-2", "eu-west-1")
```

**Save the file** after making changes.

## Step 4: Verify Your Setup

### 4.1 Check Files Are Present

Make sure these files exist in your directory:
- ✅ `config.py`
- ✅ `evaluate.py`
- ✅ `model_evaluator.py`
- ✅ `prompt_loader.py`
- ✅ `results_aggregator.py`
- ✅ `requirements.txt`
- ✅ `20251001T000153731Z_e9c5e90710a8738a.json` (sample prompts file)

### 4.2 Test Python Can Import Modules

Run this quick test:

```bash
python -c "import boto3; print('✓ boto3 installed successfully')"
```

If you see the checkmark, you're good to go!

## Step 5: Run Your First Evaluation

### Option A: Test with 1 Prompt (Recommended for First Run)

This will test with just 1 prompt to make sure everything works:

```bash
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --max-prompts 1
```

**What this does:**
- Loads 1 prompt from the included JSON file
- Evaluates both Claude Sonnet and Llama 3.2 11B models
- Shows progress in the terminal
- Generates results in `results/` folder

**Expected output:**
```
Loading prompts...
Loaded 1 prompts

Evaluating Claude 3.7 Sonnet (claude-sonnet) on 1 prompts...
  [1] Processing prompt... ✓ (XXXXms)

Evaluating Llama 3.2 11B Instruct (llama-3-2-11b) on 1 prompts...
  [1] Processing prompt... ✓ (XXXXms)

Generating reports...
Detailed results saved to: results\detailed_results_TIMESTAMP.csv
Summary report saved to: results\summary_TIMESTAMP.csv
Comparison report saved to: results\comparison_TIMESTAMP.csv

====================================================================================================
EVALUATION SUMMARY
====================================================================================================
...
✓ Evaluation complete!
```

**Time:** This typically takes 30-60 seconds for 1 prompt.

### Option B: Run with More Prompts

Once the test works, you can run with more prompts:

```bash
# Test with 5 prompts
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --max-prompts 5

# Run all prompts (may take longer)
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json
```

## Step 6: View Your Results

After the evaluation completes, check the `results/` folder:

```
results/
├── detailed_results_TIMESTAMP.csv
├── summary_TIMESTAMP.csv
└── comparison_TIMESTAMP.csv
```

### View in Terminal (Linux/Mac):

```bash
# View summary
cat results/summary_*.csv

# View first few lines of detailed results
head -20 results/detailed_results_*.csv
```

### View in Excel/Spreadsheet:

1. Open Excel, Google Sheets, or any spreadsheet application
2. File → Open → Navigate to `results/` folder
3. Open any of the CSV files
4. You'll see columns like:
   - Model name
   - Latency (ms)
   - Cost (USD)
   - JSON validity
   - Token counts

## Step 7: Understanding the Output

### Summary Report (`summary_TIMESTAMP.csv`)

This shows aggregated statistics:
- **avg_latency_ms**: Average response time
- **p50_latency_ms**: Median response time (50th percentile)
- **p95_latency_ms**: 95th percentile (95% of requests faster than this)
- **avg_cost_usd**: Average cost per prompt
- **valid_json_rate**: Percentage of valid JSON responses (0.0 to 1.0)

### Detailed Results (`detailed_results_TIMESTAMP.csv`)

This shows results for each individual prompt:
- Each row = one prompt evaluated by one model
- Columns include: prompt text, latency, tokens, cost, JSON validity, model output

### Comparison Report (`comparison_TIMESTAMP.csv`)

Side-by-side comparison:
- Each row = one prompt
- Columns show metrics for each model side-by-side
- Easy to compare which model performed better for each prompt

## Common Command Examples

### Evaluate Specific Models Only

```bash
# Only Claude Sonnet
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --models claude-sonnet

# Only Llama
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --models llama-3-2-11b

# Both models (default)
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --models claude-sonnet llama-3-2-11b
```

### Use Your Own Prompt File

```bash
# If you have a CSV file
python evaluate.py --prompts your_prompts.csv --max-prompts 10

# If you have a JSON file
python evaluate.py --prompts your_prompts.json --max-prompts 5
```

### Custom Output Location

```bash
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --output-dir ./my_results
```

## Troubleshooting Common Issues

### Issue 1: "No module named 'boto3'"

**Solution:**
```bash
pip install boto3 botocore
```

### Issue 2: "Invalid credentials" or AWS errors

**Check:**
1. Did you replace `YOUR_ACCESS_KEY_ID_HERE` in `config.py`?
2. Are your credentials correct?
3. Does your AWS account have Bedrock access?
4. Is your region correct?

**Test credentials:**
```bash
python -c "import boto3; client = boto3.client('sts'); print(client.get_caller_identity())"
```

### Issue 3: "Model not found" or "Access denied"

**Solutions:**
- Verify model IDs in `config.py` are correct for your region
- Check AWS Console → Bedrock → Model access
- Ensure your IAM user has `bedrock:InvokeModel` permission

### Issue 4: "File not found" errors

**Check:**
- Are you in the correct directory? (should be `bedrock/`)
- Does the prompt file exist?
- Use full path if needed: `python evaluate.py --prompts /full/path/to/file.json`

### Issue 5: Slow evaluation

**This is normal:**
- Each prompt takes 5-35 seconds depending on the model
- 5 prompts × 2 models = ~2-6 minutes
- Use `--max-prompts` to limit for testing

## Step-by-Step Checklist

Use this checklist to ensure you've done everything:

- [ ] Python 3.7+ installed (`python --version`)
- [ ] Repository downloaded/cloned
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] AWS credentials added to `config.py`
- [ ] AWS region updated (if needed)
- [ ] Test run successful (`--max-prompts 1`)
- [ ] Results files generated in `results/` folder
- [ ] Results viewed and understood

## What's Next?

Once you've successfully run an evaluation:

1. **Experiment with different prompts**: Create your own CSV or JSON files
2. **Compare models**: Review the comparison CSV to see which model performs better
3. **Adjust settings**: Modify `config.py` to add more models or adjust pricing
4. **Analyze results**: Use the detailed CSV for deep analysis in Excel/Python

## Getting Help

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review the main `README.md` for more details
3. Verify your AWS credentials and permissions
4. Check that all required files are present

## Quick Reference: Complete Command List

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (1 prompt)
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --max-prompts 1

# Test with 5 prompts
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --max-prompts 5

# Full evaluation (all prompts)
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json

# Custom prompts file
python evaluate.py --prompts your_file.csv

# Specific models only
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --models claude-sonnet

# Custom output directory
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --output-dir ./custom_results

# View help
python evaluate.py --help
```

That's it! You're ready to evaluate LLM models. 🚀

