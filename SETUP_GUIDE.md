# Quick Setup Guide

This guide will help you get the LLM Model Evaluation Framework running on your system in minutes.

## Step-by-Step Setup

### Step 1: Clone or Download the Repository

```bash
git clone <your-repo-url>
cd bedrock
```

Or download and extract the ZIP file.

### Step 2: Install Python Dependencies

Make sure you have Python 3.7 or higher installed, then run:

```bash
pip install -r requirements.txt
```

### Step 3: Configure AWS Credentials (Only Step You Need to Customize!)

Open `config.py` in a text editor and find these lines:

```python
# AWS Credentials (if using direct credentials instead of profile)
# TODO: Replace with your AWS credentials
AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID_HERE"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_ACCESS_KEY_HERE"
```

Replace `YOUR_ACCESS_KEY_ID_HERE` and `YOUR_SECRET_ACCESS_KEY_HERE` with your actual AWS credentials.

**Optional**: Also update the region if needed:
```python
AWS_REGION = "us-east-2"  # Change to your AWS region
```

### Step 4: Test the Setup

Run a quick test with the included sample file:

```bash
python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --max-prompts 1
```

This will:
- Load 1 prompt from the included JSON file
- Test both Claude Sonnet and Llama 3.2 11B models
- Generate results in the `results/` directory

If you see a summary output, you're all set! 🎉

## Troubleshooting

### "No module named 'boto3'"
Run: `pip install -r requirements.txt`

### "Invalid credentials"
Double-check your AWS Access Key ID and Secret Access Key in `config.py`

### "Model not found" or "Access denied"
- Verify your AWS account has Bedrock access
- Check that the model IDs in `config.py` are available in your region
- Ensure your IAM user/role has `bedrock:InvokeModel` permission

## Next Steps

Once setup is complete, you can:
- Use your own prompts: `python evaluate.py --prompts your_prompts.csv`
- Evaluate more prompts: `python evaluate.py --prompts 20251001T000153731Z_e9c5e90710a8738a.json --max-prompts 10`
- See all options: `python evaluate.py --help`

For more details, see the main [README.md](README.md).

