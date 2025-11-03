# Code Overview: Understanding Each Python File

This document explains what each Python file does and how they work together.

## 📁 Project Structure

```
bedrock/
├── config.py                  # Configuration settings
├── evaluate.py                # Main entry point (CLI)
├── model_evaluator.py         # Model invocation and metrics
├── prompt_loader.py           # Loads prompts from files/S3
└── results_aggregator.py      # Generates CSV reports
```

## 🔍 Detailed File Explanations

### 1. `config.py` - Configuration Module

**Purpose:** Central configuration file containing all settings.

**What it contains:**
- **AWS Configuration**: Credentials, region, profile settings
- **Model Definitions**: Model IDs, providers, and friendly names
- **Pricing Information**: Cost per 1K tokens for each model
- **Evaluation Settings**: Max tokens, temperature, retry logic
- **Prompt Settings**: Default file paths and CSV column names

**Key Variables:**
```python
AWS_ACCESS_KEY_ID = "..."      # Your AWS credentials
AWS_REGION = "us-east-2"       # AWS region for Bedrock
MODELS = {...}                 # Dictionary of model configurations
PRICING = {...}                # Pricing per model
EVAL_SETTINGS = {...}          # Evaluation parameters
PROMPT_SETTINGS = {...}        # Default prompt file settings
```

**When to modify:**
- Adding AWS credentials (required)
- Adding new models
- Updating pricing information
- Changing evaluation parameters (temperature, max tokens, etc.)

**Example:** Add a new model here:
```python
MODELS["claude-haiku"] = {
    "model_id": "us.anthropic.claude-3-5-haiku-20241022-v2:0",
    "provider": "anthropic",
    "name": "Claude 3.5 Haiku",
}
```

---

### 2. `evaluate.py` - Main Entry Point

**Purpose:** Command-line interface that orchestrates the entire evaluation process.

**What it does:**
1. Parses command-line arguments (which prompts, which models, etc.)
2. Loads prompts using `PromptLoader`
3. Creates `ModelEvaluator` instances for each model
4. Runs evaluations and collects results
5. Uses `ResultsAggregator` to generate reports
6. Prints summary to console

**Key Functions:**
- `main()`: Main entry point that coordinates everything

**Command-line Arguments:**
- `--prompts`: Path to prompt file (CSV/JSON/TXT)
- `--models`: Which models to evaluate (default: all)
- `--max-prompts`: Limit number of prompts
- `--output-dir`: Where to save results
- `--skip-summary`: Don't print summary table

**Example Usage:**
```bash
python evaluate.py --prompts prompts.csv --max-prompts 5
```

**What happens when you run it:**
```
User runs command
    ↓
evaluate.py parses arguments
    ↓
prompt_loader.py loads prompts
    ↓
For each model:
    model_evaluator.py evaluates all prompts
    ↓
results_aggregator.py generates CSV files
    ↓
Summary printed to console
```

**When to modify:**
- Adding new CLI options
- Changing evaluation workflow
- Adding new output formats

---

### 3. `prompt_loader.py` - Prompt Loading Module

**Purpose:** Loads and parses prompts from various sources (local files or S3).

**What it does:**
1. Detects the source type (local file or S3)
2. Determines file format (CSV, JSON, TXT, NDJSON)
3. Parses the file and extracts prompts
4. Combines multiple messages (for conversation history)
5. Returns structured prompt data

**Key Classes:**
- `PromptLoader`: Main class for loading prompts

**Key Methods:**
- `load_prompts(max_prompts)`: Main method to load prompts
- `_load_from_local()`: Loads from local files
- `_load_from_s3()`: Loads from S3 bucket
- `_load_from_csv()`: Parses CSV files
- `_load_from_json()`: Parses JSON/NDJSON files
- `_load_from_txt()`: Parses plain text files
- `_extract_prompt_from_dict()`: Extracts prompts from Bedrock log format

**Supported Formats:**
- **CSV**: `prompt` column containing text
- **JSON Array**: `[{"prompt": "..."}]`
- **NDJSON**: One JSON object per line (Bedrock log format)
- **TXT**: One prompt per line

**Special Feature:** Handles Bedrock log format
- Extracts from: `input.inputBodyJson.messages[].content[].text`
- Combines multiple messages into one prompt
- Preserves metadata (requestId, timestamp, etc.)

**Example:**
```python
loader = PromptLoader()
prompts = loader.load_prompts(max_prompts=5)
# Returns: [{"index": 1, "prompt": "...", "metadata": {...}}, ...]
```

**When to modify:**
- Adding support for new file formats
- Changing how prompts are extracted
- Adding new data sources (database, API, etc.)

---

### 4. `model_evaluator.py` - Model Evaluation Engine

**Purpose:** Invokes Bedrock models, collects metrics, and validates responses.

**What it does:**
1. Initializes Bedrock client for a specific model
2. Invokes the model with prompts
3. Measures latency (response time)
4. Estimates token counts
5. Validates JSON responses
6. Extracts JSON from markdown/wrapped responses
7. Retries on invalid JSON
8. Calculates costs
9. Collects all metrics

**Key Classes:**
- `ModelEvaluator`: Evaluates one model across multiple prompts

**Key Methods:**
- `__init__(model_key)`: Initialize evaluator for a model
- `invoke_model(prompt)`: Send prompt to model, get response
- `evaluate_prompts(prompts)`: Evaluate model on list of prompts
- `get_summary_stats()`: Calculate aggregated statistics (P50, P95, etc.)
- `clean_json_output(raw)`: Extract JSON from text (4 strategies)
- `is_valid_json(text)`: Validate JSON with cleaning
- `calculate_cost(input_tokens, output_tokens)`: Compute USD cost
- `estimate_tokens(text)`: Estimate token count

**JSON Extraction Strategies:**
1. **Markdown code blocks**: ` ```json [...] ``` `
2. **Balanced brackets**: Finds complete JSON arrays/objects
3. **Prefix detection**: "Here is the JSON response: [...]"
4. **Regex fallback**: Pattern matching as last resort

**Metrics Collected:**
- Input tokens (estimated)
- Output tokens (estimated)
- Latency (milliseconds)
- Cost (USD)
- JSON validity (boolean)
- Retry count
- Whether JSON was cleaned

**Retry Logic:**
- Automatically retries up to 2 times if JSON is invalid
- Enhanced prompts for Llama models on retry
- Tracks retry attempts

**Example:**
```python
evaluator = ModelEvaluator("claude-sonnet")
results = evaluator.evaluate_prompts(prompts)
# Returns: List of result dictionaries with all metrics
```

**When to modify:**
- Adding new models/providers
- Changing metrics collection
- Improving JSON extraction
- Adding new retry strategies

---

### 5. `results_aggregator.py` - Results Processing & Reporting

**Purpose:** Processes evaluation results and generates CSV reports.

**What it does:**
1. Aggregates results from multiple models
2. Calculates summary statistics (averages, percentiles)
3. Generates three types of CSV reports:
   - Detailed results (every prompt/model combination)
   - Summary statistics (aggregated per model)
   - Comparison report (side-by-side model comparison)
4. Prints formatted summary table to console
5. Provides recommendations (best model for cost, latency, reliability)

**Key Classes:**
- `ResultsAggregator`: Processes and reports results

**Key Methods:**
- `save_detailed_results(results)`: Saves full metrics to CSV
- `save_summary_report(summaries)`: Saves aggregated stats to CSV
- `save_comparison_report(results, summaries)`: Saves side-by-side comparison
- `print_summary_table(summaries)`: Prints formatted table to console

**Statistics Calculated:**
- Average latency
- P50 (median) latency
- P95 (95th percentile) latency
- Min/Max latency
- Total/Average cost
- JSON validity rate
- Total/Average token counts

**Recommendations Provided:**
- Most reliable (highest JSON validity)
- Most cost-effective (lowest cost)
- Best latency (lowest P95)
- Best overall (weighted score)

**Output Files:**
1. `detailed_results_TIMESTAMP.csv`: Every prompt/model with full metrics
2. `summary_TIMESTAMP.csv`: Aggregated statistics per model
3. `comparison_TIMESTAMP.csv`: Side-by-side comparison

**Example:**
```python
aggregator = ResultsAggregator(output_dir="results")
aggregator.save_detailed_results(all_results)
aggregator.save_summary_report(summaries)
aggregator.print_summary_table(summaries)
```

**When to modify:**
- Adding new report formats
- Changing summary calculations
- Adding new metrics to reports
- Customizing recommendations

---

## 🔄 How Files Work Together

### Execution Flow:

```
1. User runs: python evaluate.py --prompts file.json
                    ↓
2. evaluate.py reads config.py for settings
                    ↓
3. evaluate.py uses prompt_loader.py to load prompts
                    ↓
4. For each model in config:
    evaluate.py creates ModelEvaluator(model)
    ModelEvaluator.invoke_model() calls Bedrock API
    ModelEvaluator collects metrics
                    ↓
5. evaluate.py collects all results
                    ↓
6. evaluate.py uses results_aggregator.py to:
    - Generate CSV files
    - Print summary
                    ↓
7. Done! Results in results/ folder
```

### Data Flow:

```
config.py
    ↓ (provides settings)
evaluate.py
    ↓ (orchestrates)
    ├── prompt_loader.py (loads prompts)
    │       ↓ (returns list of prompts)
    ├── model_evaluator.py (evaluates models)
    │       ↓ (returns metrics per prompt)
    └── results_aggregator.py (generates reports)
            ↓ (creates CSV files)
```

## 📊 Key Data Structures

### Prompt Dictionary:
```python
{
    "index": 1,
    "prompt": "Full prompt text...",
    "metadata": {"requestId": "...", "timestamp": "..."}
}
```

### Result Dictionary:
```python
{
    "prompt_index": 1,
    "model_key": "claude-sonnet",
    "input_tokens": 1220,
    "output_tokens": 495,
    "latency_ms": 25880.89,
    "cost_usd": 0.0232,
    "valid_json": True,
    "retries": 0,
    "json_was_cleaned": False,
    "output": "..."
}
```

### Summary Dictionary:
```python
{
    "model_key": "claude-sonnet",
    "avg_latency_ms": 25880.89,
    "p50_latency_ms": 25880.89,
    "p95_latency_ms": 25880.89,
    "total_cost_usd": 0.0232,
    "valid_json_rate": 1.0
}
```

## 🛠️ Extension Points

### Adding a New Model:
1. **config.py**: Add to `MODELS` and `PRICING`
2. **model_evaluator.py**: Update `invoke_model()` if new provider

### Adding a New Metric:
1. **model_evaluator.py**: Collect metric in `invoke_model()`
2. **model_evaluator.py**: Add to `get_summary_stats()`
3. **results_aggregator.py**: Add column to CSV outputs

### Adding a New Input Format:
1. **prompt_loader.py**: Add new `_load_from_*()` method
2. **prompt_loader.py**: Update `load_prompts()` to detect format

### Adding a New Output Format:
1. **results_aggregator.py**: Add new `save_*()` method
2. **evaluate.py**: Call new method in `main()`

## 🔑 Key Design Patterns

1. **Separation of Concerns**: Each file has one clear responsibility
2. **Modularity**: Files can be imported and used independently
3. **Configuration-Driven**: Settings in `config.py`, code is generic
4. **Error Handling**: Try-catch blocks with graceful degradation
5. **Retry Logic**: Automatic retries for transient failures

## 📝 Summary

- **config.py**: Settings and configuration
- **evaluate.py**: Main CLI orchestrator
- **prompt_loader.py**: Input handling (files/S3)
- **model_evaluator.py**: Core evaluation logic (API calls, metrics)
- **results_aggregator.py**: Output generation (CSV reports)

Each file is independent but works together to create a complete evaluation framework!

