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
import tempfile
import os


def extract_full_prompt_text(item: dict) -> str:
    """
    Extract the FULL prompt text (all user messages combined) from JSON structures.
    This is used for CSV conversion - it returns the complete text without extracting questions.
    """
    if not isinstance(item, dict):
        return str(item) if item else ""
    
    # Handle Bedrock CloudTrail format: input.inputBodyJson.messages[].content[].text
    if 'input' in item:
        input_data = item['input']
        if isinstance(input_data, dict):
            # Check for inputBodyJson
            if 'inputBodyJson' in input_data:
                input_body = input_data['inputBodyJson']
                if isinstance(input_body, dict) and 'messages' in input_body:
                    messages = input_body['messages']
                    if isinstance(messages, list):
                        # Combine all user messages
                        user_messages = []
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get('role') == 'user':
                                content = msg.get('content', [])
                                if isinstance(content, list):
                                    for content_item in content:
                                        if isinstance(content_item, dict) and 'text' in content_item:
                                            text = content_item['text']
                                            if isinstance(text, str) and text.strip():
                                                user_messages.append(text.strip())
                                        elif isinstance(content_item, str) and content_item.strip():
                                            user_messages.append(content_item.strip())
                                elif isinstance(content, str) and content.strip():
                                    user_messages.append(content.strip())
                        
                        if user_messages:
                            # Return ALL user messages combined (this is the full prompt)
                            return "\n\n".join(user_messages)
            
            # Check for direct messages array
            if 'messages' in input_data:
                messages = input_data['messages']
                if isinstance(messages, list):
                    user_messages = []
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            content = msg.get('content', [])
                            if isinstance(content, list):
                                for content_item in content:
                                    if isinstance(content_item, dict) and 'text' in content_item:
                                        text = content_item['text']
                                        if isinstance(text, str) and text.strip():
                                            user_messages.append(text.strip())
                                    elif isinstance(content_item, str) and content_item.strip():
                                        user_messages.append(content_item.strip())
                            elif isinstance(content, str) and content.strip():
                                user_messages.append(content.strip())
                    
                    if user_messages:
                        return "\n\n".join(user_messages)
    
    # Try direct prompt fields
    for field in ['prompt', 'input', 'text', 'question', 'query', 'message']:
        if field in item:
            value = item[field]
            if isinstance(value, str) and len(value.strip()) > 0:
                return value.strip()
            elif isinstance(value, dict):
                # Recursively search in nested dict
                nested = extract_full_prompt_text(value)
                if nested:
                    return nested
    
    return ""


def extract_prompt_from_json_item(item: dict) -> str:
    """
    Extract prompt text from various JSON structures.
    Handles Bedrock CloudTrail format and other common formats.
    Specifically handles NDJSON files with questions in messages.
    This version tries to extract individual questions if found.
    """
    if not isinstance(item, dict):
        return str(item) if item else ""
    
    # Try direct prompt fields first
    for field in ['prompt', 'input', 'text', 'question', 'query', 'message']:
        if field in item:
            value = item[field]
            if isinstance(value, str) and len(value.strip()) > 0:
                return value.strip()
            elif isinstance(value, dict):
                # Recursively search in nested dict
                nested = extract_prompt_from_json_item(value)
                if nested:
                    return nested
    
    # Handle Bedrock CloudTrail format: input.inputBodyJson.messages[].content[].text
    if 'input' in item:
        input_data = item['input']
        if isinstance(input_data, dict):
            # Check for inputBodyJson
            if 'inputBodyJson' in input_data:
                input_body = input_data['inputBodyJson']
                if isinstance(input_body, dict) and 'messages' in input_body:
                    messages = input_body['messages']
                    if isinstance(messages, list):
                        # Combine all user messages
                        user_messages = []
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get('role') == 'user':
                                content = msg.get('content', [])
                                if isinstance(content, list):
                                    for content_item in content:
                                        if isinstance(content_item, dict) and 'text' in content_item:
                                            text = content_item['text']
                                            if isinstance(text, str) and text.strip():
                                                # Try to extract questions from "Questions:" section
                                                questions_text = _extract_questions_from_text(text)
                                                if questions_text:
                                                    return questions_text
                                                user_messages.append(text.strip())
                                        elif isinstance(content_item, str) and content_item.strip():
                                            questions_text = _extract_questions_from_text(content_item)
                                            if questions_text:
                                                return questions_text
                                            user_messages.append(content_item.strip())
                                elif isinstance(content, str) and content.strip():
                                    questions_text = _extract_questions_from_text(content)
                                    if questions_text:
                                        return questions_text
                                    user_messages.append(content.strip())
                        
                        if user_messages:
                            # Return the last user message (usually contains the actual question)
                            return user_messages[-1] if len(user_messages) > 0 else "\n\n".join(user_messages)
            
            # Check for direct messages array
            if 'messages' in input_data:
                messages = input_data['messages']
                if isinstance(messages, list):
                    user_messages = []
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            content = msg.get('content', [])
                            if isinstance(content, list):
                                for content_item in content:
                                    if isinstance(content_item, dict) and 'text' in content_item:
                                        text = content_item['text']
                                        if isinstance(text, str) and text.strip():
                                            questions_text = _extract_questions_from_text(text)
                                            if questions_text:
                                                return questions_text
                                            user_messages.append(text.strip())
                                    elif isinstance(content_item, str) and content_item.strip():
                                        questions_text = _extract_questions_from_text(content_item)
                                        if questions_text:
                                            return questions_text
                                        user_messages.append(content_item.strip())
                            elif isinstance(content, str) and content.strip():
                                questions_text = _extract_questions_from_text(content)
                                if questions_text:
                                    return questions_text
                                user_messages.append(content.strip())
                    
                    if user_messages:
                        return user_messages[-1] if len(user_messages) > 0 else "\n\n".join(user_messages)
    
    # Try to find any string value that looks like a prompt (longer than 20 chars, not a timestamp)
    for key, value in item.items():
        if isinstance(value, str) and len(value.strip()) > 20:
            # Skip timestamps and IDs
            if not (key.lower() in ['timestamp', 'time', 'id', 'requestid', 'date'] or 
                    value.strip().startswith('202') or  # Dates like 2025-10-01
                    len(value.strip().split()) < 3):  # Too short
                questions_text = _extract_questions_from_text(value)
                if questions_text:
                    return questions_text
                return value.strip()
    
    # Last resort: try to extract from any nested structures
    for key, value in item.items():
        if isinstance(value, dict):
            nested = extract_prompt_from_json_item(value)
            if nested:
                return nested
        elif isinstance(value, list) and len(value) > 0:
            # Check first item in list
            if isinstance(value[0], dict):
                nested = extract_prompt_from_json_item(value[0])
                if nested:
                    return nested
    
    return ""


def _extract_questions_from_text(text: str) -> str:
    """
    Extract questions from text that contains a "Questions:" section with JSON array.
    Returns formatted questions or empty string if not found.
    """
    if not isinstance(text, str):
        return ""
    
    # Look for "Questions:" section (case-insensitive)
    questions_marker = "Questions:"
    questions_marker_lower = questions_marker.lower()
    text_lower = text.lower()
    
    if questions_marker_lower in text_lower:
        # Find the start of the questions array (case-insensitive search)
        start_idx = text_lower.find(questions_marker_lower)
        if start_idx >= 0:
            # Find the actual marker in original text (preserve case)
            actual_marker = text[start_idx:start_idx + len(questions_marker)]
            after_marker = text[start_idx + len(questions_marker):].strip()
            
            # Try to find the JSON array
            # Look for opening bracket
            bracket_start = after_marker.find('[')
            if bracket_start >= 0:
                # Find matching closing bracket (handle nested brackets and CSV double quotes)
                bracket_count = 0
                bracket_end = -1
                in_string = False
                escape_next = False
                # Track if we're in a CSV double-quote sequence (""")
                csv_double_quote = False
                
                i = bracket_start
                while i < len(after_marker):
                    char = after_marker[i]
                    
                    if escape_next:
                        escape_next = False
                        i += 1
                        continue
                    
                    # Check for CSV double quotes (""")
                    if i + 1 < len(after_marker) and char == '"' and after_marker[i+1] == '"':
                        # This is a CSV double quote - toggle string state
                        in_string = not in_string
                        i += 2  # Skip both quotes
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        i += 1
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        i += 1
                        continue
                    
                    if not in_string:
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                bracket_end = i + 1
                                break
                    
                    i += 1
                
                if bracket_end > 0:
                    # Extract the JSON array string
                    json_array_str = after_marker[bracket_start:bracket_end]
                    
                    # Try to parse the JSON array
                    # First, try direct parsing
                    try:
                        questions_array = json.loads(json_array_str)
                        if isinstance(questions_array, list) and len(questions_array) > 0:
                            return _format_questions_from_array(questions_array)
                    except (json.JSONDecodeError, ValueError) as e:
                        # If direct parsing fails, try to handle escaped quotes
                        # The JSON might have escaped quotes like \" or CSV double quotes like ""
                        try:
                            # Handle CSV-style double quotes ("" becomes ")
                            json_array_fixed = json_array_str.replace('""', '"')
                            questions_array = json.loads(json_array_fixed)
                            if isinstance(questions_array, list) and len(questions_array) > 0:
                                return _format_questions_from_array(questions_array)
                        except (json.JSONDecodeError, ValueError):
                            # If that fails, try to extract manually using regex
                            import re
                            # Look for patterns like "Question":"..." or 'Question':'...'
                            # Handle both single and double quotes, and CSV double-escaped quotes
                            
                            # First, try to handle CSV format with double quotes: ""Question"":""...""
                            # We need to match the entire JSON object and extract the Question field
                            question_texts = []
                            
                            # Pattern for CSV format: ""Question"":""...""
                            # Match: ""Question"":"" followed by question text (which may contain escaped quotes) followed by ""
                            csv_pattern = r'""Question""\s*:\s*""((?:[^"]|""|\\"")*)""'
                            csv_matches = re.findall(csv_pattern, json_array_str)
                            for match in csv_matches:
                                # Unescape: "" becomes ", \" becomes "
                                unescaped = match.replace('""', '"').replace('\\""', '"').replace('\\"', '"')
                                unescaped = unescaped.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
                                if unescaped.strip():
                                    question_texts.append(unescaped.strip())
                            
                            # If no CSV matches, try standard JSON format
                            if not question_texts:
                                question_patterns = [
                                    r'"Question"\s*:\s*"((?:[^"\\]|\\.)*)"',   # Standard: "Question":"..."
                                    r"'Question'\s*:\s*'((?:[^'\\]|\\.)*)'",   # Single quotes
                                    r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"',   # Lowercase
                                    r"'question'\s*:\s*'((?:[^'\\]|\\.)*)'",   # Lowercase single quotes
                                ]
                                
                                for pattern in question_patterns:
                                    matches = re.findall(pattern, json_array_str)
                                    for match in matches:
                                        # Unescape the matched string
                                        unescaped = match.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
                                        if unescaped.strip():
                                            question_texts.append(unescaped.strip())
                            
                            # Remove duplicates while preserving order
                            seen = set()
                            unique_questions = []
                            for q in question_texts:
                                if q not in seen:
                                    seen.add(q)
                                    unique_questions.append(q)
                            
                            if unique_questions:
                                return "\n\n".join([f"Q{i+1}: {q}" for i, q in enumerate(unique_questions)])
    
    # If no questions found, return empty string
    return ""


def _format_questions_from_array(questions_array: list) -> str:
    """Format questions from a parsed JSON array."""
    question_texts = []
    for q_item in questions_array:
        if isinstance(q_item, dict):
            # Try different field names for question (case-insensitive)
            question = None
            for key in ['Question', 'question', 'QuestionText', 'questionText', 'text', 'Text', 'prompt', 'Prompt']:
                if key in q_item:
                    val = q_item[key]
                    if isinstance(val, str) and len(val.strip()) > 0:
                        question = val.strip()
                        break
            
            if not question:
                # Try to find any string value that looks like a question
                for key, val in q_item.items():
                    if isinstance(val, str) and len(val.strip()) > 10:
                        # Skip LinkId and other non-question fields
                        if key.lower() not in ['linkid', 'link_id', 'id', 'link', 'questionid']:
                            question = val.strip()
                            break
            
            if question:
                question_texts.append(question)
        elif isinstance(q_item, str):
            if q_item.strip():
                question_texts.append(q_item.strip())
    
    if question_texts:
        # Return formatted questions
        return "\n\n".join([f"Q{i+1}: {q}" for i, q in enumerate(question_texts)])
    
    return ""

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
                        st.success(f"✅ Loaded {len(st.session_state.uploaded_prompts)} prompts from CSV file")
                        
                        # Initialize selected prompts if not exists
                        if 'selected_uploaded_prompts' not in st.session_state:
                            st.session_state.selected_uploaded_prompts = st.session_state.uploaded_prompts.copy()
                        
                        # Show checkbox list for prompt selection (same as JSON)
                        with st.expander("📋 Select Prompts to Test", expanded=True):
                            st.markdown(f"**Total prompts loaded:** {len(st.session_state.uploaded_prompts)}")
                            
                            # Select all / Deselect all buttons (stacked vertically)
                            if st.button("✅ Select All", key="select_all_csv", use_container_width=True):
                                st.session_state.selected_uploaded_prompts = st.session_state.uploaded_prompts.copy()
                                st.rerun()
                            if st.button("❌ Deselect All", key="deselect_all_csv", use_container_width=True):
                                st.session_state.selected_uploaded_prompts = []
                                st.rerun()
                            
                            st.markdown("---")
                            
                            # Show checkboxes for each prompt
                            selected_prompts = []
                            for idx, prompt in enumerate(st.session_state.uploaded_prompts):
                                # Create a readable preview of the prompt (show first 200 chars)
                                prompt_text = str(prompt).strip()
                                if len(prompt_text) > 200:
                                    prompt_preview = prompt_text[:200] + "..."
                                else:
                                    prompt_preview = prompt_text
                                
                                # Clean up the preview for display (remove extra whitespace, newlines)
                                prompt_preview = ' '.join(prompt_preview.split())
                                
                                # If prompt is empty or very short, show a default message
                                if not prompt_preview or len(prompt_preview.strip()) < 5:
                                    prompt_preview = f"[Empty prompt {idx + 1}]"
                                
                                # Checkbox for each prompt - show the actual prompt text
                                is_selected = st.checkbox(
                                    f"**Prompt {idx + 1}:** {prompt_preview}",
                                    value=prompt in st.session_state.selected_uploaded_prompts,
                                    key=f"csv_prompt_checkbox_{idx}",
                                    help=f"Full prompt: {prompt_text[:500] if len(prompt_text) > 500 else prompt_text}"
                                )
                                
                                if is_selected:
                                    selected_prompts.append(prompt)
                            
                            # Update session state
                            st.session_state.selected_uploaded_prompts = selected_prompts
                            
                            st.markdown("---")
                            st.info(f"**Selected:** {len(selected_prompts)} / {len(st.session_state.uploaded_prompts)} prompts")
                            
                            # Show preview of selected prompts
                            if selected_prompts:
                                with st.expander("👁️ Preview Selected Prompts", expanded=False):
                                    for idx, prompt in enumerate(selected_prompts[:5], 1):
                                        st.markdown(f"**Prompt {idx}:**")
                                        st.text_area(
                                            "",
                                            value=str(prompt)[:500] + ("..." if len(str(prompt)) > 500 else ""),
                                            height=100,
                                            key=f"csv_preview_prompt_{idx}",
                                            disabled=True,
                                            label_visibility="collapsed"
                                        )
                                    if len(selected_prompts) > 5:
                                        st.caption(f"... and {len(selected_prompts) - 5} more prompts")
                    else:
                        st.error("❌ CSV must have 'prompt' column")
                        st.session_state.uploaded_prompts = []
                        st.session_state.selected_uploaded_prompts = []
                        
                elif file_extension == 'json':
                    # Read file content as string (handles both bytes and text)
                    file_content = uploaded_file.read()
                    if isinstance(file_content, bytes):
                        file_content = file_content.decode('utf-8')
                    
                    # Check if this is NDJSON format (one JSON object per line)
                    # First, convert NDJSON to CSV format, then extract questions
                    lines = file_content.strip().split('\n')
                    is_ndjson = len(lines) > 1
                    ndjson_processed = False
                    
                    if is_ndjson:
                        # Try to parse first line as JSON to check if it's NDJSON format
                        try:
                            first_line_json = json.loads(lines[0].strip())
                            if isinstance(first_line_json, dict) and 'input' in first_line_json:
                                # This looks like NDJSON format - convert to CSV first
                                st.info("🔄 Detected NDJSON format. Converting to CSV format...")
                                
                                # Convert NDJSON to CSV format
                                csv_prompts = []
                                prompt_id = 1
                                
                                for line_num, line in enumerate(lines, 1):
                                    line = line.strip()
                                    if not line:
                                        continue
                                    
                                    try:
                                        record = json.loads(line)
                                        if not isinstance(record, dict):
                                            continue
                                        
                                        # Extract FULL prompt text (all user messages combined) for CSV conversion
                                        # Use extract_full_prompt_text to get the complete text, not just questions
                                        extracted_prompt = extract_full_prompt_text(record)
                                        if not extracted_prompt:
                                            continue
                                        
                                        # Detect if JSON is expected
                                        expected_json = (
                                            "json" in extracted_prompt.lower() or
                                            "return the result in a json" in extracted_prompt.lower() or
                                            "formatted as follows:" in extracted_prompt.lower()
                                        )
                                        
                                        # Extract category
                                        category = "json-gen" if expected_json else "general"
                                        operation = record.get("operation", "")
                                        if operation:
                                            category = operation.lower()
                                        
                                        csv_prompts.append({
                                            "prompt_id": prompt_id,
                                            "prompt": extracted_prompt,
                                            "expected_json": expected_json,
                                            "category": category
                                        })
                                        prompt_id += 1
                                    except json.JSONDecodeError:
                                        continue
                                
                                if csv_prompts:
                                    # Now extract individual questions from each CSV prompt
                                    st.success(f"✅ Converted {len(csv_prompts)} NDJSON records to CSV format")
                                    all_questions = []
                                    
                                    for csv_prompt in csv_prompts:
                                        prompt_text = csv_prompt["prompt"]
                                        # Extract questions from the "Questions:" section
                                        questions_text = _extract_questions_from_text(prompt_text)
                                        if questions_text:
                                            # Split into individual questions
                                            if "\n\nQ" in questions_text:
                                                question_parts = questions_text.split("\n\nQ")
                                                for i, part in enumerate(question_parts):
                                                    if part.strip():
                                                        if i == 0 and not part.startswith("Q"):
                                                            all_questions.append(part.strip())
                                                        else:
                                                            all_questions.append(f"Q{part.strip()}")
                                            else:
                                                all_questions.append(questions_text)
                                    
                                    if all_questions:
                                        st.session_state.uploaded_prompts = all_questions
                                        st.success(f"✅ Extracted {len(all_questions)} individual questions from CSV prompts")
                                        
                                        # Initialize selected prompts if not exists
                                        if 'selected_uploaded_prompts' not in st.session_state:
                                            st.session_state.selected_uploaded_prompts = all_questions.copy()
                                        
                                        # Show checkbox list for prompt selection
                                        with st.expander("📋 Select Prompts to Test", expanded=True):
                                            st.markdown(f"**Total prompts loaded:** {len(all_questions)}")
                                            
                                            # Select all / Deselect all buttons (stacked vertically)
                                            if st.button("✅ Select All", key="select_all_ndjson", use_container_width=True):
                                                st.session_state.selected_uploaded_prompts = all_questions.copy()
                                                st.rerun()
                                            if st.button("❌ Deselect All", key="deselect_all_ndjson", use_container_width=True):
                                                st.session_state.selected_uploaded_prompts = []
                                                st.rerun()
                                            
                                            st.markdown("---")
                                            
                                            # Show checkboxes for each question
                                            selected_prompts = []
                                            for idx, question in enumerate(all_questions):
                                                question_text = str(question).strip()
                                                # Remove "Q1:", "Q2:" prefix if present for cleaner display
                                                if question_text.startswith("Q") and ":" in question_text:
                                                    display_text = question_text.split(":", 1)[1].strip()
                                                else:
                                                    display_text = question_text
                                                
                                                if len(display_text) > 200:
                                                    prompt_preview = display_text[:200] + "..."
                                                else:
                                                    prompt_preview = display_text
                                                
                                                prompt_preview = ' '.join(prompt_preview.split())
                                                
                                                if not prompt_preview or len(prompt_preview.strip()) < 5:
                                                    prompt_preview = f"[Empty prompt {idx + 1}]"
                                                
                                                is_selected = st.checkbox(
                                                    f"**Prompt {idx + 1}:** {prompt_preview}",
                                                    value=question in st.session_state.selected_uploaded_prompts,
                                                    key=f"ndjson_prompt_checkbox_{idx}",
                                                    help=f"Full prompt: {question_text[:500] if len(question_text) > 500 else question_text}"
                                                )
                                                
                                                if is_selected:
                                                    selected_prompts.append(question)
                                            
                                            st.session_state.selected_uploaded_prompts = selected_prompts
                                            
                                            st.markdown("---")
                                            st.info(f"**Selected:** {len(selected_prompts)} / {len(all_questions)} prompts")
                                            if len(selected_prompts) > 0 and len(selected_prompts) <= 5:
                                                st.caption("**Selected prompts:**")
                                                for i, prompt in enumerate(selected_prompts, 1):
                                                    preview = str(prompt).strip()[:100] + "..." if len(str(prompt)) > 100 else str(prompt)
                                                    st.caption(f"{i}. {preview}")
                                            elif len(selected_prompts) > 5:
                                                st.caption("**First 5 selected prompts:**")
                                                for i, prompt in enumerate(selected_prompts[:5], 1):
                                                    preview = str(prompt).strip()[:100] + "..." if len(str(prompt)) > 100 else str(prompt)
                                                    st.caption(f"{i}. {preview}")
                                                st.caption(f"... and {len(selected_prompts) - 5} more prompts")
                                    else:
                                        st.warning("⚠️ No questions found in the converted CSV prompts. Displaying full prompts instead.")
                                        # Fall back to showing full prompts
                                        st.session_state.uploaded_prompts = [p["prompt"] for p in csv_prompts]
                                        if 'selected_uploaded_prompts' not in st.session_state:
                                            st.session_state.selected_uploaded_prompts = st.session_state.uploaded_prompts.copy()
                                else:
                                    st.error("❌ Could not extract prompts from NDJSON file")
                                    st.session_state.uploaded_prompts = []
                                
                                # Mark that NDJSON was processed
                                ndjson_processed = True
                        except (json.JSONDecodeError, ValueError, KeyError):
                            # Not NDJSON or error parsing, continue with regular JSON processing
                            pass
                    
                    # Only process as regular JSON if NDJSON was not processed
                    if not ndjson_processed:
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
                            
                            # Helper function to split multiple questions into individual prompts
                            def add_prompts_with_splitting(extracted_text):
                                """Add prompts, splitting multiple questions if needed."""
                                if "\n\nQ" in extracted_text and extracted_text.count("Q") > 1:
                                    # Split by question markers to get individual questions
                                    question_parts = extracted_text.split("\n\nQ")
                                    for i, part in enumerate(question_parts):
                                        if part.strip():
                                            if i == 0 and not part.startswith("Q"):
                                                # First part might not have Q prefix
                                                prompts.append(part.strip())
                                            else:
                                                prompts.append(f"Q{part.strip()}")
                                else:
                                    prompts.append(extracted_text)
                            
                            # If data is a list (from JSON array or JSONL)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict):
                                        # Use comprehensive extraction function
                                        extracted_prompt = extract_prompt_from_json_item(item)
                                        if extracted_prompt:
                                            add_prompts_with_splitting(extracted_prompt)
                                        else:
                                            # Fallback: try to stringify the whole item
                                            prompts.append(str(item))
                                    elif isinstance(item, str):
                                        prompts.append(item)
                            
                            # If data is a single dict
                            elif isinstance(data, dict):
                                if 'prompts' in data:
                                    prompt_list = data['prompts']
                                    if isinstance(prompt_list, list):
                                        # Extract prompts from each item in the list
                                        for item in prompt_list:
                                            if isinstance(item, dict):
                                                extracted = extract_prompt_from_json_item(item)
                                                if extracted:
                                                    add_prompts_with_splitting(extracted)
                                            elif isinstance(item, str):
                                                prompts.append(item)
                                    else:
                                        if isinstance(prompt_list, dict):
                                            extracted = extract_prompt_from_json_item(prompt_list)
                                        else:
                                            extracted = str(prompt_list)
                                        if extracted:
                                            add_prompts_with_splitting(extracted)
                                else:
                                    # Try to extract prompt from the dict
                                    extracted_prompt = extract_prompt_from_json_item(data)
                                    if extracted_prompt:
                                        add_prompts_with_splitting(extracted_prompt)
                                    else:
                                        st.error("❌ Could not extract prompt from JSON. Please ensure the JSON contains a 'prompt', 'input', or 'messages' field.")
                                        st.session_state.uploaded_prompts = []
                                        prompts = []
                            
                            st.session_state.uploaded_prompts = prompts
                            
                            if st.session_state.uploaded_prompts:
                                st.success(f"✅ Loaded {len(st.session_state.uploaded_prompts)} prompts from JSON/JSONL file")
                                
                                # Initialize selected prompts if not exists
                                if 'selected_uploaded_prompts' not in st.session_state:
                                    st.session_state.selected_uploaded_prompts = st.session_state.uploaded_prompts.copy()
                                
                                # Show checkbox list for prompt selection
                                with st.expander("📋 Select Prompts to Test", expanded=True):
                                    st.markdown(f"**Total prompts loaded:** {len(st.session_state.uploaded_prompts)}")
                                    
                                    # Select all / Deselect all buttons (stacked vertically)
                                    if st.button("✅ Select All", key="select_all_uploaded", use_container_width=True):
                                        st.session_state.selected_uploaded_prompts = st.session_state.uploaded_prompts.copy()
                                        st.rerun()
                                    if st.button("❌ Deselect All", key="deselect_all_uploaded", use_container_width=True):
                                        st.session_state.selected_uploaded_prompts = []
                                        st.rerun()
                                    
                                    st.markdown("---")
                                    
                                    # Show checkboxes for each prompt
                                    selected_prompts = []
                                    for idx, prompt in enumerate(st.session_state.uploaded_prompts):
                                        # Create a readable preview of the prompt (show first 200 chars)
                                        prompt_text = str(prompt).strip()
                                        if len(prompt_text) > 200:
                                            prompt_preview = prompt_text[:200] + "..."
                                        else:
                                            prompt_preview = prompt_text
                                        
                                        # Clean up the preview for display (remove extra whitespace, newlines)
                                        prompt_preview = ' '.join(prompt_preview.split())
                                        
                                        # If prompt is empty or very short, show a default message
                                        if not prompt_preview or len(prompt_preview.strip()) < 5:
                                            prompt_preview = f"[Empty prompt {idx + 1}]"
                                        
                                        # Checkbox for each prompt - show the actual prompt text
                                        is_selected = st.checkbox(
                                            f"**Prompt {idx + 1}:** {prompt_preview}",
                                            value=prompt in st.session_state.selected_uploaded_prompts,
                                            key=f"prompt_checkbox_{idx}",
                                            help=f"Full prompt: {prompt_text[:500] if len(prompt_text) > 500 else prompt_text}"
                                        )
                                        
                                        if is_selected:
                                            selected_prompts.append(prompt)
                                    
                                    # Update session state
                                    st.session_state.selected_uploaded_prompts = selected_prompts
                                    
                                    st.markdown("---")
                                    st.info(f"**Selected:** {len(selected_prompts)} / {len(st.session_state.uploaded_prompts)} prompts")
                                    
                                    # Show preview of selected prompts
                                    if selected_prompts:
                                        with st.expander("👁️ Preview Selected Prompts", expanded=False):
                                            for idx, prompt in enumerate(selected_prompts[:5], 1):
                                                st.markdown(f"**Prompt {idx}:**")
                                                st.text_area(
                                                    "",
                                                    value=prompt[:500] + ("..." if len(prompt) > 500 else ""),
                                                    height=100,
                                                    key=f"preview_prompt_{idx}",
                                                    disabled=True,
                                                    label_visibility="collapsed"
                                                )
                                            if len(selected_prompts) > 5:
                                                st.caption(f"... and {len(selected_prompts) - 5} more prompts")
                            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.uploaded_prompts = []
        else:
            st.session_state.uploaded_prompts = []
            st.session_state.selected_uploaded_prompts = []
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
    # Use selected prompts from uploaded file (if any)
    if st.session_state.get('selected_uploaded_prompts'):
        prompts_to_use.extend(st.session_state.selected_uploaded_prompts)
    
    if prompts_to_use:
        custom_count = 1 if user_prompt.strip() else 0
        selected_count = len(st.session_state.get('selected_uploaded_prompts', []))
        st.info(f"**📊 Total Prompts Selected:** {len(prompts_to_use)}")
        if len(prompts_to_use) > 1:
            parts = []
            if custom_count > 0:
                parts.append(f"{custom_count} custom")
            if selected_count > 0:
                parts.append(f"{selected_count} from file")
            if parts:
                st.caption(f"• {' + '.join(parts)}")
    
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
