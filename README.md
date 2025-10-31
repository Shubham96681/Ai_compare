# 🚀 Model Evaluation Framework for AWS Bedrock LLMs

Compare multiple Bedrock-hosted LLMs (Claude Sonnet, Llama 3, Titan, etc.) using production-like prompts from CSV. The framework measures latency, token usage, JSON validity, and cost, aggregates results, and visualizes comparisons in a Streamlit dashboard.

## 📋 Features

- Multi-model evaluation on the same prompts
- Per-request metrics: input/output tokens, latency, validity, cost
- Aggregations: p50/p95 latency, averages, validity%, cost/request
- Interactive Streamlit dashboard with visualizations
- Config-driven models and pricing (YAML)
- CSV export functionality

## 📁 Project Structure

```
AICostOptimizer/
  ├─ src/
  │  ├─ __init__.py
  │  ├─ model_registry.py
  │  ├─ prompt_loader.py
  │  ├─ tokenizers.py
  │  ├─ evaluator.py
  │  ├─ metrics_logger.py
  │  ├─ report_generator.py
  │  ├─ dashboard.py
  │  └─ utils/
  │     ├─ bedrock_client.py
  │     ├─ timing.py
  │     └─ json_utils.py
  ├─ configs/
  │  ├─ models.yaml
  │  └─ settings.yaml
  ├─ data/
  │  ├─ test_prompts.csv
  │  ├─ runs/
  │  │  ├─ raw_metrics.csv (generated)
  │  │  └─ model_comparison.csv (generated)
  │  └─ cache/
  ├─ scripts/
  │  ├─ run_evaluation.py
  │  └─ extract_prompts_from_json.py
  ├─ .env.example
  ├─ requirements.txt
  └─ README.md
```

---

## 🔧 Step-by-Step Setup & Usage Guide

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd AICostOptimizer
```

### Step 2: Set Up Python Virtual Environment

**Windows PowerShell:**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- `boto3` (AWS SDK)
- `pandas` (Data manipulation)
- `streamlit` (Dashboard)
- `plotly` (Visualizations)
- `tiktoken` (Token counting)
- `tqdm` (Progress bars)
- And other dependencies...

### Step 4: Configure AWS Credentials

You have two options:

**Option A: Using `.env` file (Recommended)**

1. Copy the example environment file:
   ```bash
   # Windows PowerShell
   Copy-Item .env.example .env
   
   # Linux/Mac
   cp .env.example .env
   ```

2. Edit `.env` file and add your AWS credentials:
   ```env
   AWS_ACCESS_KEY_ID=your_access_key_here
   AWS_SECRET_ACCESS_KEY=your_secret_key_here
   AWS_REGION=us-east-1
   ```

   **⚠️ Important:** Never commit the `.env` file to version control!

**Option B: Using AWS Credentials File**

Configure AWS credentials using AWS CLI:
```bash
aws configure
```

Or manually edit `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_access_key_here
aws_secret_access_key = your_secret_key_here
region = us-east-1
```

### Step 5: Convert JSON Logs to CSV (Extract Prompts)

If you have Bedrock CloudTrail JSON logs and want to extract prompts:

1. Place your JSON log file in the `data/` directory (e.g., `data/my_logs.json`)

2. Run the extraction script:
   ```bash
   python scripts/extract_prompts_from_json.py --input data/my_logs.json --output data/test_prompts.csv
   ```

   Or use the default file if it exists:
   ```bash
   python scripts/extract_prompts_from_json.py
   ```

3. The script will:
   - Extract all prompts from JSON log file
   - Combine multi-message conversations into complete prompts
   - Detect JSON-expected prompts automatically
   - Save to `data/test_prompts.csv`

4. Verify the output:
   ```bash
   # Check the CSV file
   # Windows PowerShell
   Get-Content data/test_prompts.csv -Head 5
   
   # Linux/Mac
   head -5 data/test_prompts.csv
   ```

**CSV Format:**
```csv
prompt_id,prompt,expected_json,category
1,"Your complete prompt text here...",True,converse
2,"Another prompt...",False,general
```

### Step 6: Configure Models and Pricing

1. Edit `configs/models.yaml` and update with your Bedrock model IDs:

```yaml
region_name: us-east-1  # Change if your models are in a different region

models:
  - name: Claude 3 Sonnet
    provider: anthropic
    bedrock_model_id: anthropic.claude-3-sonnet-20240229-v1:0  # Update with your model ID
    tokenizer: anthropic
    pricing:
      input_per_1k_tokens_usd: 0.003    # Update with current pricing
      output_per_1k_tokens_usd: 0.015   # Update with current pricing
    generation_params:
      max_tokens: 512
      temperature: 0.2
      top_p: 0.95

  # Add more models as needed...
```

**💡 Tips:**
- Find available Bedrock models in AWS Console → Amazon Bedrock → Foundation models
- Update pricing from AWS pricing pages (prices may vary by region)
- Use model IDs that match your AWS account's available models

2. (Optional) Edit `configs/settings.yaml` for advanced settings:
```yaml
region_name: us-east-1
output_dir: data/runs
use_sqlite: false
cache_enabled: true
concurrency: 4
logging_level: INFO
```

### Step 7: Run the Evaluation

**Test Run (First 3 Prompts - Recommended for First Time):**
```bash
python scripts/run_evaluation.py --models all --limit 3
```

**Full Evaluation (All Prompts):**
```bash
python scripts/run_evaluation.py --models all --prompts data/test_prompts.csv --out data/runs
```

**Evaluate Specific Models:**
```bash
python scripts/run_evaluation.py --models "Claude 3 Sonnet,Llama 3 70B Instruct"
```

**Other Useful Options:**
```bash
# Custom output directory
python scripts/run_evaluation.py --models all --out data/my_results

# Custom run ID
python scripts/run_evaluation.py --models all --run-id my_test_run

# Skip report generation (faster for testing)
python scripts/run_evaluation.py --models all --limit 5 --skip-report
```

**What Happens During Evaluation:**
1. Loads prompts from CSV
2. For each prompt and each model:
   - Sends request to Bedrock API
   - Measures latency (start to finish)
   - Counts input/output tokens
   - Calculates cost based on pricing
   - Validates JSON if expected
   - Records all metrics
3. Saves raw metrics to `data/runs/raw_metrics.csv`
4. Generates aggregated report to `data/runs/model_comparison.csv`

**Progress Indicators:**
- You'll see a progress bar showing evaluation status
- Success (✅) or Error (❌) indicators per evaluation
- Final summary with metrics

### Step 8: View Results in Dashboard

1. Launch the Streamlit dashboard:
   ```bash
   streamlit run src/dashboard.py
   ```

2. The dashboard will automatically open in your browser at:
   ```
   http://localhost:8501
   ```

3. If it doesn't open automatically, copy the URL from the terminal output.

**Dashboard Features:**

- **Summary Cards:** Total evaluations, success rate, total cost, models compared
- **Model Comparison Table:** Aggregated metrics per model
- **Best Performer Highlights:** Best latency, cost, and JSON validity
- **Visualizations:**
  - **Latency Tab:** Box plots showing latency distribution
  - **Tokens Tab:** Bar charts for average input/output tokens
  - **Cost Tab:** Cost distribution and average cost per request
  - **JSON Validity Tab:** Validity percentage by model
- **Filters:**
  - Select specific models to compare
  - Filter by prompt IDs
  - Filter by status (success/error)
- **Export:** Download filtered results as CSV

**Using the Dashboard:**
1. Use sidebar to select data file paths (if different from defaults)
2. Apply filters to focus on specific models or prompts
3. Explore visualizations in different tabs
4. Export results using download buttons

### Step 9: Interpret Results

**Key Metrics to Compare:**

1. **Latency (p50/p95/p99):**
   - Lower is better
   - p95 shows worst-case performance
   - Important for user-facing applications

2. **Cost per Request:**
   - Lower is better for cost optimization
   - Consider both input and output token costs
   - Multiply by expected request volume

3. **JSON Validity:**
   - Higher percentage is better
   - Critical if you require structured outputs

4. **Token Usage:**
   - Compare input vs output tokens
   - Models with lower token usage may be more cost-effective

5. **Success Rate:**
   - Check for errors in the raw metrics
   - Investigate any model-specific failures

**Files Generated:**

- `data/runs/raw_metrics.csv`: Per-request detailed metrics
- `data/runs/model_comparison.csv`: Aggregated comparison table

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'tqdm'"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Error: "NoCredentialsError" or "Unable to locate credentials"
**Solution:** Configure AWS credentials (see Step 4)

### Error: "An error occurred (ValidationException)"
**Solution:** Check that your model IDs in `configs/models.yaml` match available models in your AWS account

### Error: "AccessDeniedException"
**Solution:** Ensure your AWS credentials have permissions for Amazon Bedrock

### Dashboard shows "No data found"
**Solution:** Run evaluation first (Step 7), then check file paths in dashboard sidebar

### Evaluation is slow
**Solution:** 
- Use `--limit` to test with fewer prompts first
- Check your internet connection
- Verify Bedrock model availability in your region

---

## 📊 Example Output

After running evaluation, you'll see:

```
✅ Found 3 model(s): ['Claude 3 Sonnet', 'Llama 3 70B Instruct', 'Titan Text Premier']
✅ Loaded 16 prompt(s)
🏃 Run ID: run_20250101_120000
🚀 Starting evaluation...
   Models: 3
   Prompts: 16
   Total evaluations: 48

Evaluating: 100%|████████████| 48/48 [02:15<00:00,  2.81s/eval]

✅ Evaluation complete! Collected 48 metric records
💾 Saving metrics...
   Saved to: data/runs/raw_metrics.csv
📊 Generating aggregated report...
   Saved to: data/runs/model_comparison.csv

📈 Summary:
model_name              count  avg_input_tokens  p95_latency_ms  avg_cost_usd_per_request
Claude 3 Sonnet         16     1250.3           2150.5          0.008425
Llama 3 70B Instruct    16     1248.7           2890.2          0.005234
```

---

## 📝 CSV Schema

**Prompts CSV (`data/test_prompts.csv`):**
```csv
prompt_id,prompt,expected_json,category
1,"Your prompt text here...",true,converse
2,"Another prompt...",false,general
```

**Raw Metrics CSV (`data/runs/raw_metrics.csv`):**
- timestamp, run_id, model_name, model_id, prompt_id
- input_tokens, output_tokens, latency_ms
- json_valid, error, status
- cost_usd_input, cost_usd_output, cost_usd_total

**Comparison CSV (`data/runs/model_comparison.csv`):**
- model_name, count, success_count, error_count
- avg_input_tokens, avg_output_tokens
- p50_latency_ms, p95_latency_ms, p99_latency_ms
- json_valid_pct
- avg_cost_usd_per_request, total_cost_usd

---

## 🔐 Security Notes

- **Never commit** `.env` file or AWS credentials to version control
- Use AWS IAM roles with minimal required permissions
- Review AWS CloudTrail logs regularly
- Keep dependencies updated for security patches

---

## 📚 Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

[Add your license here]

---

## 💡 Tips

- Start with `--limit 3` to test before full evaluation
- Use different run IDs to compare different configurations
- Export results regularly for tracking over time
- Monitor AWS costs in CloudWatch

---

**Happy Evaluating! 🎉**
