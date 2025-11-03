"""
Main evaluation script - orchestrates the model evaluation framework.
"""

import argparse
import sys
from typing import List, Optional
from prompt_loader import PromptLoader
from model_evaluator import ModelEvaluator
from results_aggregator import ResultsAggregator
from config import MODELS, PROMPT_SETTINGS


def main():
    """Main entry point for evaluation framework."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM models on Bedrock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all configured models with local CSV file
  python evaluate.py --prompts prompts.csv --models claude-sonnet llama-3-2-11b
  
  # Evaluate from S3
  python evaluate.py --s3-bucket my-bucket --s3-key prompts.csv --models claude-sonnet
  
  # Limit to 10 prompts
  python evaluate.py --prompts prompts.csv --max-prompts 10
  
  # Custom output directory
  python evaluate.py --prompts prompts.csv --output-dir ./eval_results
        """
    )
    
    # Prompt source options
    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument("--prompts", type=str, help="Local path to prompts file (.csv, .json, or .txt)")
    prompt_group.add_argument("--s3-bucket", type=str, help="S3 bucket name for prompts")
    prompt_group.add_argument("--s3-key", type=str, help="S3 key/path for prompts file")
    
    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help=f"Models to evaluate (default: all). Available: {', '.join(MODELS.keys())}"
    )
    
    # Evaluation options
    parser.add_argument("--max-prompts", type=int, help="Maximum number of prompts to evaluate")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results (default: results)")
    parser.add_argument("--skip-summary", action="store_true", help="Skip printing summary table")
    
    args = parser.parse_args()
    
    # Update config based on CLI arguments
    if args.prompts:
        PROMPT_SETTINGS["local_path"] = args.prompts
    if args.s3_bucket:
        PROMPT_SETTINGS["s3_bucket"] = args.s3_bucket
    if args.s3_key:
        PROMPT_SETTINGS["s3_key"] = args.s3_key
    
    # Validate models
    invalid_models = [m for m in args.models if m not in MODELS]
    if invalid_models:
        print(f"Error: Invalid model keys: {invalid_models}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)
    
    # Load prompts
    print("Loading prompts...")
    try:
        loader = PromptLoader(source_type="auto")
        prompts = loader.load_prompts(max_prompts=args.max_prompts)
        
        if not prompts:
            print("Error: No prompts loaded")
            sys.exit(1)
        
        print(f"Loaded {len(prompts)} prompts")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        sys.exit(1)
    
    # Evaluate each model
    all_results = []
    summaries = []
    
    for model_key in args.models:
        try:
            evaluator = ModelEvaluator(model_key)
            results = evaluator.evaluate_prompts(prompts)
            summary = evaluator.get_summary_stats()
            
            all_results.extend(results)
            summaries.append(summary)
        except Exception as e:
            print(f"Error evaluating {model_key}: {e}")
            continue
    
    if not all_results:
        print("Error: No results collected")
        sys.exit(1)
    
    # Generate reports
    print("\nGenerating reports...")
    aggregator = ResultsAggregator(output_dir=args.output_dir)
    
    detailed_file = aggregator.save_detailed_results(all_results)
    summary_file = aggregator.save_summary_report(summaries)
    comparison_file = aggregator.save_comparison_report(all_results, summaries)
    
    # Print summary
    if not args.skip_summary:
        aggregator.print_summary_table(summaries)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Detailed results: {detailed_file}")
    print(f"  Summary report: {summary_file}")
    print(f"  Comparison report: {comparison_file}")


if __name__ == "__main__":
    main()

