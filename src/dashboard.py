"""Premium Streamlit Dashboard for LLM Model Comparison - Enterprise Edition"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import time
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Import evaluation components
from src.model_registry import ModelRegistry
from src.evaluator import BedrockEvaluator
from src.metrics_logger import MetricsLogger
from src.report_generator import ReportGenerator
from src.utils.json_utils import is_valid_json

# Page configuration
st.set_page_config(
    page_title="AI Cost Optimizer Pro - Enterprise LLM Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS Styling
st.markdown("""
<style>
    /* Main Theme */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .premium-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        white-space: normal;
        text-align: center;
        width: 100%;
        word-wrap: break-word;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .primary-button {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.1rem;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f46b45 0%, #eea849 100%);
        color: white;
    }
    
    .badge-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Status indicators */
    .status-success {
        color: #00b09b;
        font-weight: 600;
    }
    
    .status-error {
        color: #f46b45;
        font-weight: 600;
    }
    
    /* Custom animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Header with premium design
st.markdown("""
<div class="main-header fade-in">
    <h1 style="color: white; margin: 0; font-size: 3rem;">🚀 AI Cost Optimizer Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: 300;">
        Enterprise-Grade LLM Performance & Cost Analytics
    </p>
    <div style="margin-top: 1rem;">
        <span class="badge badge-success">Real-time Analytics</span>
        <span class="badge badge-warning">Cost Optimization</span>
        <span class="badge badge-info">Multi-Model Comparison</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'show_tour' not in st.session_state:
    st.session_state.show_tour = False
if 'uploaded_prompts' not in st.session_state:
    st.session_state.uploaded_prompts = []
if 'uploaded_file_type' not in st.session_state:
    st.session_state.uploaded_file_type = None
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'run_evaluation' not in st.session_state:
    st.session_state.run_evaluation = False
if 'prompts_to_evaluate' not in st.session_state:
    st.session_state.prompts_to_evaluate = []
if 'data_reload_key' not in st.session_state:
    st.session_state.data_reload_key = 0

# Set default paths (removed from sidebar)
config_path = "configs/models.yaml"
raw_path = "data/runs/raw_metrics.csv"
agg_path = "data/runs/model_comparison.csv"

# Premium Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #667eea; margin-bottom: 0.5rem;">⚙️ Control Panel</h2>
        <p style="color: #666; font-size: 0.9rem;">Test prompts against LLM models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================
    # TEST YOUR PROMPT SECTION IN SIDEBAR
    # ==========================================
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h2 style="color: #667eea; margin-bottom: 0.5rem;">🧪 Test Your Prompt</h2>
        <p style="color: #666; font-size: 0.9rem;">Test prompts against LLM models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prompt Input Section in Sidebar
    with st.expander("📝 Custom Prompt", expanded=True):
        user_prompt = st.text_area(
            "Enter Your Prompt",
            height=100,
            placeholder="Enter your prompt here...",
            help="💡 Tip: Be specific about your expected output format for better analysis.",
            key="custom_prompt_input_sidebar"
        )
    
    # File Upload Section in Sidebar
    with st.expander("📁 Upload File (JSON or CSV)", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['json', 'csv'],
            help="Upload a JSON or CSV file containing prompts. CSV should have a 'prompt' column. JSON can be an array of objects with 'prompt' field.",
            key="prompt_file_uploader_sidebar"
        )
        
        # Handle file upload
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            st.session_state.uploaded_file_type = file_extension
            
            try:
                if file_extension == 'csv':
                    df_uploaded = pd.read_csv(uploaded_file)
                    if 'prompt' in df_uploaded.columns:
                        st.session_state.uploaded_prompts = df_uploaded['prompt'].tolist()
                        st.success(f"✅ Loaded {len(st.session_state.uploaded_prompts)} prompts")
                        if st.checkbox("Preview prompts", key="preview_csv"):
                            st.dataframe(df_uploaded[['prompt']].head(5), use_container_width=True)
                    else:
                        st.error("❌ CSV must have 'prompt' column")
                        st.session_state.uploaded_prompts = []
                        
                elif file_extension == 'json':
                    # Read file content as string (handles both bytes and text)
                    file_content = uploaded_file.read()
                    if isinstance(file_content, bytes):
                        file_content = file_content.decode('utf-8')
                    
                    # Try to parse as regular JSON first
                    data = None
                    json_error = None
                    
                    try:
                        # Try parsing as single JSON object/array
                        is_valid, parsed = is_valid_json(file_content)
                        if is_valid:
                            data = parsed
                    except Exception as e:
                        json_error = str(e)
                    
                    # If JSON parsing failed or has "Extra data" error, try JSONL format
                    if data is None or (json_error and "Extra data" in json_error):
                        # Try parsing as JSONL (one JSON object per line)
                        try:
                            lines = file_content.strip().split('\n')
                            jsonl_objects = []
                            for line_num, line in enumerate(lines, 1):
                                line = line.strip()
                                if not line:
                                    continue
                                
                                is_valid, parsed = is_valid_json(line)
                                if is_valid:
                                    jsonl_objects.append(parsed)
                                else:
                                    # If one line fails, might not be JSONL
                                    break
                            
                            # If we successfully parsed multiple lines as JSON, it's JSONL
                            if len(jsonl_objects) > 0:
                                data = jsonl_objects
                            elif data is None:
                                # If JSONL also failed, try to parse just the first valid JSON object
                                # This handles cases where there's extra data after the main JSON
                                try:
                                    # Find the first complete JSON object
                                    first_obj_end = file_content.find('\n')
                                    if first_obj_end > 0:
                                        first_line = file_content[:first_obj_end].strip()
                                        is_valid, parsed = is_valid_json(first_line)
                                        if is_valid:
                                            data = [parsed]  # Wrap in list for consistency
                                except:
                                    pass
                        except Exception as e:
                            pass
                    
                    # If still no data, show error
                    if data is None:
                        st.error(f"❌ Error parsing JSON file: {json_error or 'Invalid JSON format. Please ensure the file is valid JSON or JSONL format.'}")
                        st.session_state.uploaded_prompts = []
                    else:
                        # Handle different JSON structures
                        prompts = []
                        
                        # If data is a list (from JSON array or JSONL)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    # Try to extract prompt from various field names
                                    if 'prompt' in item:
                                        prompts.append(item['prompt'])
                                    elif 'input' in item and isinstance(item['input'], str):
                                        prompts.append(item['input'])
                                    elif 'text' in item and isinstance(item['text'], str):
                                        prompts.append(item['text'])
                                    elif 'message' in item:
                                        msg = item['message']
                                        if isinstance(msg, str):
                                            prompts.append(msg)
                                        elif isinstance(msg, dict) and 'content' in msg:
                                            prompts.append(str(msg['content']))
                                    else:
                                        # Try to find any string value that looks like a prompt
                                        for key, value in item.items():
                                            if isinstance(value, str) and len(value) > 10:
                                                prompts.append(value)
                                                break
                                elif isinstance(item, str):
                                    prompts.append(item)
                        
                        # If data is a single dict
                        elif isinstance(data, dict):
                            if 'prompts' in data:
                                prompt_list = data['prompts']
                                if isinstance(prompt_list, list):
                                    prompts = prompt_list
                                else:
                                    prompts = [prompt_list]
                            elif 'prompt' in data:
                                prompts = [data['prompt']]
                            elif 'input' in data:
                                prompts = [data['input']]
                            else:
                                st.error("❌ JSON structure not recognized. Expected 'prompt', 'prompts', or 'input' field.")
                                st.session_state.uploaded_prompts = []
                                prompts = []
                        
                        st.session_state.uploaded_prompts = prompts
                        
                        if st.session_state.uploaded_prompts:
                            st.success(f"✅ Loaded {len(st.session_state.uploaded_prompts)} prompts from JSON/JSONL file")
                            if st.checkbox("Preview prompts", key="preview_json"):
                                preview_df = pd.DataFrame({'prompt': st.session_state.uploaded_prompts[:5]})
                                st.dataframe(preview_df, use_container_width=True)
                            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.uploaded_prompts = []
        else:
            st.session_state.uploaded_prompts = []
            st.session_state.uploaded_file_type = None
    
    # Settings Section in Sidebar
    with st.expander("⚙️ Settings", expanded=False):
        expect_json = st.checkbox(
            "Expect JSON Response",
            value=True,
            help="Validate response as JSON and track validity metrics",
            key="expect_json_sidebar"
        )
        
        format_as_json = st.checkbox(
            "Format Prompt as JSON",
            value=False,
            help="Convert the prompt to JSON format before sending to models",
            key="format_as_json_sidebar"
        )
    
    # Model Selection in Sidebar
    st.markdown("### 🤖 Select Models")
    
    # Load model registry here in sidebar
    try:
        config_file = Path(config_path)
        if config_file.exists():
            sidebar_registry = ModelRegistry(config_path)
            available_models = sidebar_registry.list_models()
            if available_models:
                model_options = {model['name']: model for model in available_models}
                selected_model_names = []
                
                for name, model in model_options.items():
                    pricing = sidebar_registry.get_model_pricing(model)
                    pricing_info = f"${pricing['input_per_1k_tokens_usd']:.4f}/1k in, ${pricing['output_per_1k_tokens_usd']:.4f}/1k out"
                    
                    if st.checkbox(f"{name}", key=f"model_sidebar_{name}", help=f"Pricing: {pricing_info}"):
                        selected_model_names.append(name)
                
                st.session_state.selected_models = selected_model_names
            else:
                st.warning("⚠️ No models configured")
                st.session_state.selected_models = []
        else:
            st.warning(f"⚠️ Config file not found: {config_path}")
            st.session_state.selected_models = []
    except Exception as e:
        st.warning(f"⚠️ Error loading models: {str(e)}")
        st.session_state.selected_models = []
    
    # Input Summary in Sidebar
    prompts_to_use = []
    if user_prompt.strip():
        prompts_to_use.append(user_prompt)
    if st.session_state.uploaded_prompts:
        prompts_to_use.extend(st.session_state.uploaded_prompts)
    
    if prompts_to_use:
        st.info(f"**📊 Total Prompts:** {len(prompts_to_use)}")
        if len(prompts_to_use) > 1:
            st.caption(f"• 1 custom" + (f" + {len(st.session_state.uploaded_prompts)} from file" if st.session_state.uploaded_prompts else ""))
    
    # Run Button in Sidebar
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True, key="run_eval_sidebar"):
        st.session_state.run_evaluation = True
        st.session_state.prompts_to_evaluate = prompts_to_use
        st.rerun()
    
    # Support Section
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <p style="color: #666; font-size: 0.8rem;">Need help?</p>
        <p style="color: #667eea; font-size: 0.9rem; font-weight: 600;">support@aicostoptimizer.com</p>
    </div>
    """, unsafe_allow_html=True)

# Load data functions with cache key for syncing
@st.cache_data
def load_data(raw_path: str, agg_path: str, cache_key: int = 0):
    """Load and cache data files with enhanced error handling.
    cache_key allows cache invalidation when new data is saved."""
    raw_df = pd.DataFrame()
    agg_df = pd.DataFrame()
    
    try:
        if Path(raw_path).exists():
            raw_df = pd.read_csv(raw_path)
        else:
            pass  # Don't show warning in main sidebar - let it show in sidebar
    except Exception as e:
        pass  # Errors handled in sidebar
    
    try:
        if Path(agg_path).exists():
            agg_df = pd.read_csv(agg_path)
        else:
            pass
    except Exception as e:
        pass
    
    return raw_df, agg_df

@st.cache_resource(ttl=0)  # Disable caching to ensure fresh config is always loaded
def load_model_registry(config_path: str):
    """Load model registry with enhanced error handling."""
    try:
        config_file = Path(config_path)
        if config_file.exists():
            # Read file modification time to bust cache
            mtime = config_file.stat().st_mtime
            registry = ModelRegistry(config_path)
            return registry, mtime
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading model registry: {e}")
        return None, None

# Load data and models with cache key for syncing
raw_df, agg_df = load_data(raw_path, agg_path, st.session_state.data_reload_key)
model_registry_result = load_model_registry(config_path)
if model_registry_result:
    model_registry, _ = model_registry_result
else:
    model_registry = None

# Premium Tabs with Icons
tab1, tab2 = st.tabs(["📊 **Overview & Analytics**", "📈 **Historical Results**"])

# ==========================================
# TAB 1: Premium Overview & Analytics
# ==========================================
with tab1:
    # Handle evaluation triggered from sidebar
    if st.session_state.run_evaluation:
        prompts_to_evaluate = st.session_state.prompts_to_evaluate
        selected_model_names = st.session_state.selected_models
        expect_json = st.session_state.get('expect_json_sidebar', True)
        format_as_json = st.session_state.get('format_as_json_sidebar', False)
        
        # Get model registry
        model_registry_result = load_model_registry(config_path)
        if model_registry_result:
            model_registry, _ = model_registry_result
        else:
            model_registry = None
        
        if not prompts_to_evaluate:
            st.error("""
            ❌ **Prompt Required**
            
            Please enter a custom prompt or upload a file with prompts in the sidebar.
            """)
            st.session_state.run_evaluation = False
        elif not selected_model_names:
            st.error("""
            ❌ **Models Required**
            
            Please select at least one model in the sidebar.
            """)
            st.session_state.run_evaluation = False
        elif model_registry is None:
            st.error("""
            ❌ **Configuration Required**
            
            Please check your model configuration file path: `{config_path}`
            """.format(config_path=config_path))
            st.session_state.run_evaluation = False
        else:
            # Premium evaluation execution
            with st.status("🚀 **Running Comprehensive Evaluation...**", expanded=True) as status:
                try:
                    evaluator = BedrockEvaluator(model_registry)
                    selected_models = [model_registry.get_model_by_name(name) for name in selected_model_names]
                    
                    status.update(label="🔄 Initializing evaluation engine...")
                    time.sleep(0.5)
                    
                    results = []
                    progress_bar = st.progress(0)
                    total_evaluations = len(prompts_to_evaluate) * len(selected_models)
                    current_evaluation = 0
                    
                    for prompt_idx, current_prompt in enumerate(prompts_to_evaluate):
                        for model_idx, model in enumerate(selected_models):
                            if model is None:
                                continue
                            
                            current_evaluation += 1
                            status.update(label=f"🧪 Testing {model['name']} with prompt {prompt_idx+1}/{len(prompts_to_evaluate)}... ({current_evaluation}/{total_evaluations})")
                            progress_bar.progress(current_evaluation / total_evaluations)
                            
                            try:
                                # Format prompt as JSON if requested
                                final_prompt = current_prompt
                                if format_as_json:
                                    try:
                                        json.loads(current_prompt)
                                        final_prompt = current_prompt
                                    except (json.JSONDecodeError, ValueError):
                                        prompt_json = {
                                            "prompt": current_prompt,
                                            "instruction": "Please respond to the following prompt. If JSON format is requested, return your answer as valid JSON."
                                        }
                                        final_prompt = json.dumps(prompt_json, indent=2)
                                else:
                                    if expect_json:
                                        final_prompt = f"{current_prompt}\n\nPlease respond in valid JSON format."
                                
                                prompt_id = f"prompt_{prompt_idx+1}" if len(prompts_to_evaluate) > 1 else None
                                
                                metrics = evaluator.evaluate_prompt(
                                    prompt=final_prompt,
                                    model=model,
                                    prompt_id=prompt_id,
                                    expected_json=expect_json,
                                    run_id=f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                )
                                results.append(metrics)
                            
                            except Exception as e:
                                error_metric = {
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "run_id": f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                    "model_name": model.get("name", "unknown"),
                                    "model_id": model.get("bedrock_model_id", "unknown"),
                                    "prompt_id": f"prompt_{prompt_idx+1}" if len(prompts_to_evaluate) > 1 else None,
                                    "input_prompt": final_prompt,  # Store input prompt even for errors
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "latency_ms": 0,
                                    "json_valid": False,
                                    "error": str(e),
                                    "status": "error",
                                    "cost_usd_input": 0.0,
                                    "cost_usd_output": 0.0,
                                    "cost_usd_total": 0.0,
                                    "response": ""
                                }
                                results.append(error_metric)
                    
                    progress_bar.progress(1.0)
                    status.update(label="✅ Evaluation complete! Generating insights...", state="complete")
                    
                except Exception as e:
                    status.update(label="❌ Evaluation failed", state="error")
                    st.error(f"Evaluation error: {e}")
            
            # Save results automatically and reload data
            if results:
                try:
                    metrics_logger = MetricsLogger(Path("data/runs"))
                    metrics_logger.log_metrics(results)
                    
                    # Regenerate aggregated report to ensure all graphs are synced
                    report_generator = ReportGenerator(Path("data/runs"))
                    report_generator.generate_report()
                    
                    st.session_state.evaluation_results = results
                    # Increment cache key to force data reload and sync all graphs
                    st.session_state.data_reload_key += 1
                    st.cache_data.clear()
                    # Reload data with new cache key to ensure all components see fresh data
                    raw_df, agg_df = load_data(raw_path, agg_path, st.session_state.data_reload_key)
                    st.success("✅ **Evaluation Complete!** Results saved and dashboard updated. All graphs synced.")
                except Exception as e:
                    st.error(f"❌ Error saving results: {e}")
                    st.session_state.evaluation_results = results
            
            # Reset evaluation flag after processing
            st.session_state.run_evaluation = False
    
    # Always reload data with current cache key to ensure sync
    raw_df, agg_df = load_data(raw_path, agg_path, st.session_state.data_reload_key)
    
    # Show evaluation results summary if available
    if st.session_state.get('evaluation_results'):
        results = st.session_state.evaluation_results
        success_count = len([r for r in results if r.get('status') == 'success'])
        
        with st.expander("📊 Latest Evaluation Results", expanded=True):
            st.success(f"🎉 **Evaluation Complete!** {success_count} successful responses.")
            
            # Quick evaluation summary
            if results:
                eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
                success_results = [r for r in results if r.get('status') == 'success']
                
                with eval_col1:
                    if success_results:
                        avg_latency = pd.DataFrame(success_results)['latency_ms'].mean()
                        st.metric("⚡ Avg Latency", f"{avg_latency:.0f} ms")
                
                with eval_col2:
                    total_cost = pd.DataFrame(results)['cost_usd_total'].sum()
                    st.metric("💰 Total Cost", f"${total_cost:.6f}")
                
                with eval_col3:
                    if success_results:
                        valid_count = len([r for r in success_results if r.get('json_valid', False)])
                        validity_pct = (valid_count / len(success_results) * 100) if success_results else 0
                        st.metric("✅ JSON Validity", f"{validity_pct:.1f}%")
                
                with eval_col4:
                    if success_results:
                        total_tokens = pd.DataFrame(success_results)[['input_tokens', 'output_tokens']].sum().sum()
                        st.metric("📝 Total Tokens", f"{total_tokens:,}")
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            results_df = pd.DataFrame(results)
            display_cols = ['model_name', 'latency_ms', 'input_tokens', 'output_tokens', 
                          'cost_usd_total', 'json_valid', 'status']
            if 'error' in results_df.columns:
                display_cols.append('error')
            
            available_cols = [col for col in display_cols if col in results_df.columns]
            display_df = results_df[available_cols].copy()
            
            # Format for display
            if 'latency_ms' in display_df.columns:
                display_df['latency_ms'] = display_df['latency_ms'].apply(lambda x: f"{x:.0f} ms" if pd.notna(x) and x > 0 else "N/A")
            if 'input_tokens' in display_df.columns:
                display_df['input_tokens'] = display_df['input_tokens'].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")
            if 'output_tokens' in display_df.columns:
                display_df['output_tokens'] = display_df['output_tokens'].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")
            if 'cost_usd_total' in display_df.columns:
                display_df['cost_usd_total'] = display_df['cost_usd_total'].apply(lambda x: f"${x:.6f}" if pd.notna(x) and x > 0 else "$0.000000")
            if 'json_valid' in display_df.columns:
                def format_json_valid(x):
                    if pd.isna(x) or x is None:
                        return "➖ N/A"
                    elif x is True:
                        return "✅ Yes"
                    else:
                        return "❌ No"
                display_df['json_valid'] = display_df['json_valid'].apply(format_json_valid)
            if 'status' in display_df.columns:
                display_df['status'] = display_df['status'].apply(lambda x: "✅ Success" if x == "success" else "❌ Error")
            
            st.dataframe(display_df, use_container_width=True, height=200)
            
            # Display JSON Output Responses
            st.subheader("📄 Output JSON Responses")
            for idx, result in enumerate(results):
                model_name = result.get('model_name', 'Unknown')
                prompt_id = result.get('prompt_id', 'N/A')
                status = result.get('status', 'unknown')
                response = result.get('response', '')
                error = result.get('error', None)
                
                # Create expander title
                if prompt_id and prompt_id != 'N/A':
                    expander_title = f"🔹 {model_name} - {prompt_id}"
                else:
                    expander_title = f"🔹 {model_name}"
                
                if status == 'error':
                    expander_title += " ❌"
                else:
                    expander_title += " ✅"
                
                with st.expander(expander_title, expanded=False):
                    # Display Input Prompt/JSON
                    input_prompt = result.get('input_prompt', '')
                    if input_prompt:
                        st.markdown("### 📥 Input Prompt/JSON")
                        # Try to format as JSON if valid JSON
                        try:
                            input_json_obj = json.loads(input_prompt)
                            st.json(input_json_obj)
                        except (json.JSONDecodeError, ValueError, TypeError):
                            # If not valid JSON, display as text
                            st.code(input_prompt, language='text')
                        
                        st.markdown("---")
                    
                    # Display Output Response/JSON
                    if status == 'success' and response:
                        st.markdown("### 📤 Output Response/JSON")
                        # Try to format as JSON if valid JSON
                        try:
                            json_obj = json.loads(response)
                            st.json(json_obj)
                        except (json.JSONDecodeError, ValueError, TypeError):
                            # If not valid JSON, display as code/text
                            st.markdown("**Response (Text Format):**")
                            st.code(response, language='text')
                        
                        # Show raw response option
                        with st.expander("📋 View Raw Output Response", expanded=False):
                            st.text_area(
                                "Full Output Response Text",
                                value=response,
                                height=200,
                                key=f"raw_response_{idx}",
                                disabled=True
                            )
                    elif status == 'error':
                        st.markdown("### 📤 Output Response/JSON")
                        st.error(f"**Error:** {error if error else 'Unknown error occurred'}")
                        if response:
                            st.markdown("**Partial Response:**")
                            st.code(response, language='text')
                    else:
                        st.markdown("### 📤 Output Response/JSON")
                        st.warning("No response available")
        
        st.markdown("---")
    
    # Show existing data or empty state - Always show dashboard sections if data exists
    if not (agg_df.empty and raw_df.empty):
        # Use all data without filters (removed Analysis Filters section)
        filtered_raw = raw_df.copy()
        filtered_agg = agg_df.copy()
        
        # Premium Summary Cards
        st.header("📈 Executive Summary")
        
        if not filtered_agg.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-highlight">
                    <h3 style="margin: 0; font-size: 2rem; color: white; font-weight: 700;">{:,}</h3>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; color: white;">Total Evaluations</p>
                </div>
                """.format(len(filtered_raw) if not filtered_raw.empty else 0), unsafe_allow_html=True)
            
            with col2:
                success_rate = 0
                if not filtered_raw.empty and "status" in filtered_raw.columns:
                    total = len(filtered_raw)
                    success = len(filtered_raw[filtered_raw["status"] == "success"])
                    success_rate = (success / total * 100) if total > 0 else 0
                
                success_color = "#00b09b" if success_rate >= 90 else "#eea849"
                st.markdown(f"""
                <div class="premium-card" style="text-align: center; background: linear-gradient(135deg, {success_color} 0%, #96c93d 100%); color: white;">
                    <h3 style="margin: 0; font-size: 2rem;">{success_rate:.1f}%</h3>
                    <p style="margin: 0; opacity: 0.9;">Success Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_cost = filtered_agg["total_cost_usd"].sum() if "total_cost_usd" in filtered_agg.columns else 0
                st.markdown(f"""
                <div class="premium-card" style="text-align: center;">
                    <h3 style="margin: 0; font-size: 2rem; color: #333; font-weight: 700;">${total_cost:.4f}</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #666; font-weight: 500;">Total Cost</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                num_models = len(filtered_agg)
                st.markdown(f"""
                <div class="premium-card" style="text-align: center;">
                    <h3 style="margin: 0; font-size: 2rem; color: #333; font-weight: 700;">{num_models}</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #666; font-weight: 500;">Models Compared</p>
                </div>
                """, unsafe_allow_html=True)

        # Enhanced Best Performers Section
        if not filtered_agg.empty and len(filtered_agg) > 0:
            st.header("🏆 Performance Leaders")
            
            leader_cols = st.columns(3)
            metrics_to_show = [
                ("⚡ Fastest Response", "p95_latency_ms", "min", "#00b09b"),
                ("💰 Most Cost-Effective", "avg_cost_usd_per_request", "min", "#667eea"),
                ("✅ Best Quality", "json_valid_pct", "max", "#96c93d")
            ]
            
            for idx, (title, metric, operation, color) in enumerate(metrics_to_show):
                with leader_cols[idx]:
                    if metric in filtered_agg.columns:
                        if operation == "min":
                            best_idx = filtered_agg[metric].idxmin()
                        else:
                            best_idx = filtered_agg[metric].idxmax()
                        
                        best_model = filtered_agg.loc[best_idx, "model_name"]
                        best_value = filtered_agg.loc[best_idx, metric]
                        
                        unit = " ms" if "latency" in metric else "%" if "pct" in metric else ""
                        st.markdown(f"""
                        <div class="premium-card" style="border-left: 4px solid {color}; text-align: center;">
                            <h4 style="margin: 0 0 1rem 0; color: {color};">{title}</h4>
                            <h3 style="margin: 0; color: #333;">{best_model}</h3>
                            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 1.1rem;">
                                {best_value:.2f}{unit}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

        # Enhanced Interactive Visualizations
        # Ensure we use the most up-to-date synced data
        if not filtered_raw.empty and len(filtered_raw) > 0:
            st.header("📊 Interactive Analytics")
            
            # Custom visualization selector
            viz_option = st.selectbox(
                "Choose Visualization Type",
                ["Performance Dashboard", "Cost Analysis", "Quality Metrics", "Token Usage"],
                help="Select which metrics to visualize"
            )
            
            # Use synced data - ensure fresh copy from loaded data
            success_df = filtered_raw[filtered_raw["status"] == "success"].copy() if "status" in filtered_raw.columns else filtered_raw.copy()
            
            if not success_df.empty:
                if viz_option == "Performance Dashboard":
                    col1, col2 = st.columns(2)
                    with col1:
                        if "latency_ms" in success_df.columns and "model_name" in success_df.columns:
                            fig = px.box(success_df, x="model_name", y="latency_ms", 
                                        title="🚀 Response Time Distribution",
                                        color="model_name",
                                        color_discrete_sequence=px.colors.qualitative.Set2)
                            fig.update_layout(showlegend=False, height=400, template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if "model_name" in success_df.columns:
                            requests_per_model = success_df['model_name'].value_counts().reset_index()
                            requests_per_model.columns = ['model_name', 'request_count']
                            fig = px.bar(requests_per_model, x='model_name', y='request_count',
                                        title="📊 Requests per Model",
                                        color='model_name',
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig.update_layout(showlegend=False, height=400, template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif viz_option == "Cost Analysis":
                    if "cost_usd_total" in success_df.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.box(success_df, x="model_name", y="cost_usd_total",
                                        title="💰 Cost Distribution",
                                        color="model_name",
                                        color_discrete_sequence=px.colors.qualitative.Set3)
                            fig.update_layout(showlegend=False, height=400, template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            if not filtered_agg.empty and "avg_cost_usd_per_request" in filtered_agg.columns:
                                fig = px.bar(filtered_agg.sort_values("avg_cost_usd_per_request"),
                                            x="model_name", y="avg_cost_usd_per_request",
                                            title="💰 Average Cost per Request",
                                            color="avg_cost_usd_per_request",
                                            color_continuous_scale="Greens")
                                fig.update_layout(height=400, template="plotly_white")
                                st.plotly_chart(fig, use_container_width=True)
                
                elif viz_option == "Quality Metrics":
                    if "json_valid" in success_df.columns:
                        json_stats = success_df.groupby("model_name")["json_valid"].agg(["sum", "count"])
                        json_stats["validity_pct"] = (json_stats["sum"] / json_stats["count"] * 100).round(2)
                        json_stats = json_stats.reset_index()
                        
                        fig = px.bar(json_stats.sort_values("validity_pct", ascending=False),
                                    x="model_name", y="validity_pct",
                                    title="✅ JSON Validity Percentage",
                                    color="validity_pct",
                                    color_continuous_scale="RdYlGn",
                                    text="validity_pct")
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig.update_layout(height=400, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_option == "Token Usage":
                    col1, col2 = st.columns(2)
                    with col1:
                        if "input_tokens" in success_df.columns:
                            token_input = success_df.groupby("model_name")["input_tokens"].mean().reset_index()
                            fig = px.bar(token_input, x="model_name", y="input_tokens",
                                        title="📥 Average Input Tokens",
                                        color="model_name",
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig.update_layout(showlegend=False, height=400, template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if "output_tokens" in success_df.columns:
                            token_output = success_df.groupby("model_name")["output_tokens"].mean().reset_index()
                            fig = px.bar(token_output, x="model_name", y="output_tokens",
                                        title="📤 Average Output Tokens",
                                        color="model_name",
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig.update_layout(showlegend=False, height=400, template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
    else:
        # Empty state with call to action
        st.markdown("""
        <div class="premium-card" style="text-align: center; padding: 4rem;">
            <h2 style="color: #667eea;">📊 No Data Yet</h2>
            <p style="color: #666; font-size: 1.1rem; margin-bottom: 2rem;">
                Start by testing your first prompt to see beautiful analytics!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Test Your First Prompt", type="primary", use_container_width=True):
            st.info("💡 Use the sidebar to run evaluations!")

# ==========================================
# TAB 2: Premium Historical Results
# ==========================================
with tab2:
    st.header("📈 Historical Analysis & Export")
    
    # Reload data with current cache key to ensure sync in Tab 2
    raw_df, agg_df = load_data(raw_path, agg_path, st.session_state.data_reload_key)
    
    if raw_df.empty:
        st.info("""
        📭 **No Historical Data Available**
        
        Use the sidebar to run evaluations and build your historical dataset.
        """)
    else:
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.metric("Total Records", f"{len(raw_df):,}")
        
        with overview_col2:
            unique_models = raw_df['model_name'].nunique() if 'model_name' in raw_df.columns else 0
            st.metric("Unique Models", unique_models)
        
        with overview_col3:
            date_range = "N/A"
            if 'timestamp' in raw_df.columns:
                try:
                    dates = pd.to_datetime(raw_df['timestamp'])
                    date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                except:
                    pass
            st.metric("Date Range", date_range)
        
        with overview_col4:
            if 'status' in raw_df.columns:
                success_rate = (raw_df['status'] == 'success').mean() * 100
            else:
                success_rate = 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Interactive data explorer
        st.subheader("🔍 Data Explorer")
        
        explore_col1, explore_col2 = st.columns([1, 3])
        
        with explore_col1:
            show_columns = st.multiselect(
                "Select columns to display",
                options=raw_df.columns.tolist(),
                default=['model_name', 'latency_ms', 'input_tokens', 'output_tokens', 'cost_usd_total', 'status'] if all(col in raw_df.columns for col in ['model_name', 'latency_ms', 'input_tokens', 'output_tokens', 'cost_usd_total', 'status']) else raw_df.columns.tolist()[:6]
            )
            
            rows_to_show = st.slider("Rows to display", 10, min(1000, len(raw_df)), 100)
        
        with explore_col2:
            display_columns = [col for col in show_columns if col in raw_df.columns]
            if display_columns:
                st.dataframe(
                    raw_df[display_columns].head(rows_to_show),
                    use_container_width=True,
                    height=400
                )
        
        # Enhanced export options
        st.subheader("💾 Advanced Export")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if not agg_df.empty:
                csv_agg = agg_df.to_csv(index=False)
                st.download_button(
                    label="📊 Aggregated Data",
                    data=csv_agg,
                    file_name=f"aggregated_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with export_col2:
            if not raw_df.empty:
                csv_raw = raw_df.to_csv(index=False)
                st.download_button(
                    label="📈 Raw Metrics",
                    data=csv_raw,
                    file_name=f"raw_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with export_col3:
            # Export configuration
            if Path(config_path).exists():
                try:
                    config_data = open(config_path, 'r', encoding='utf-8').read()
                    st.download_button(
                        label="⚙️ Configuration",
                        data=config_data,
                        file_name="model_configuration.yaml",
                        mime="text/yaml",
                        use_container_width=True
                    )
                except:
                    st.button("⚙️ Configuration", disabled=True, use_container_width=True)
            else:
                st.button("⚙️ Configuration", disabled=True, use_container_width=True)

# Premium Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p style="font-size: 1.1rem; font-weight: 600; color: #667eea;">🚀 AI Cost Optimizer Pro</p>
    <p style="font-size: 0.9rem;">Enterprise-Grade LLM Performance & Cost Analytics</p>
    <p style="font-size: 0.85rem; margin-top: 0.5rem;">Built with Streamlit | Powered by AWS Bedrock</p>
</div>
""", unsafe_allow_html=True)
