"""Streamlit dashboard for visualizing LLM model comparison metrics."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(page_title="Bedrock LLM Model Comparison", layout="wide")
st.title("🚀 Bedrock LLM Model Comparison Dashboard")

# Sidebar configuration
st.sidebar.header("📁 Data Sources")
raw_path = st.sidebar.text_input(
    "Raw metrics CSV",
    value="data/runs/raw_metrics.csv",
    help="Path to raw metrics CSV file"
)
agg_path = st.sidebar.text_input(
    "Aggregated CSV",
    value="data/runs/model_comparison.csv",
    help="Path to aggregated comparison CSV file"
)

# Load data
@st.cache_data
def load_data(raw_path: str, agg_path: str):
    """Load and cache data files."""
    raw_df = pd.DataFrame()
    agg_df = pd.DataFrame()
    
    if Path(raw_path).exists():
        try:
            raw_df = pd.read_csv(raw_path)
        except Exception as e:
            st.sidebar.error(f"Error loading raw metrics: {e}")
    
    if Path(agg_path).exists():
        try:
            agg_df = pd.read_csv(agg_path)
        except Exception as e:
            st.sidebar.error(f"Error loading aggregated data: {e}")
    
    return raw_df, agg_df

raw_df, agg_df = load_data(raw_path, agg_path)

# Sidebar filters
st.sidebar.header("🔍 Filters")

if not raw_df.empty:
    models = sorted(raw_df["model_name"].unique().tolist())
    selected_models = st.sidebar.multiselect(
        "Model(s)",
        options=models,
        default=models,
        help="Select models to compare"
    )
    
    # Prompt ID filter
    if "prompt_id" in raw_df.columns:
        prompt_ids = sorted(raw_df["prompt_id"].unique().tolist())
        selected_prompts = st.sidebar.multiselect(
            "Prompt ID(s)",
            options=prompt_ids,
            default=prompt_ids,
            help="Select prompts to include"
        )
    else:
        selected_prompts = []
    
    # Status filter
    if "status" in raw_df.columns:
        statuses = sorted(raw_df["status"].unique().tolist())
        selected_statuses = st.sidebar.multiselect(
            "Status",
            options=statuses,
            default=["success"],
            help="Filter by request status"
        )
    else:
        selected_statuses = ["success"]
    
    # Apply filters to raw_df
    filtered_raw = raw_df.copy()
    if selected_models:
        filtered_raw = filtered_raw[filtered_raw["model_name"].isin(selected_models)]
    if selected_prompts:
        filtered_raw = filtered_raw[filtered_raw["prompt_id"].isin(selected_prompts)]
    if selected_statuses:
        filtered_raw = filtered_raw[filtered_raw["status"].isin(selected_statuses)]
else:
    selected_models = []
    filtered_raw = pd.DataFrame()

# Filter aggregated data
if not agg_df.empty and selected_models:
    filtered_agg = agg_df[agg_df["model_name"].isin(selected_models)].copy()
else:
    filtered_agg = agg_df.copy()

# Main content
if agg_df.empty and raw_df.empty:
    st.warning("⚠️ No data found. Please run evaluation first using:")
    st.code("python scripts/run_evaluation.py --models all --prompts data/test_prompts.csv")
    st.stop()

# Summary cards
if not filtered_agg.empty:
    st.header("📊 Summary")
    cols = st.columns(4)
    
    with cols[0]:
        total_evaluations = len(filtered_raw) if not filtered_raw.empty else 0
        st.metric("Total Evaluations", total_evaluations)
    
    with cols[1]:
        success_rate = 0
        if not filtered_raw.empty and "status" in filtered_raw.columns:
            total = len(filtered_raw)
            success = len(filtered_raw[filtered_raw["status"] == "success"])
            success_rate = (success / total * 100) if total > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with cols[2]:
        total_cost = filtered_agg["total_cost_usd"].sum() if "total_cost_usd" in filtered_agg.columns else 0
        st.metric("Total Cost (USD)", f"${total_cost:.4f}")
    
    with cols[3]:
        num_models = len(filtered_agg)
        st.metric("Models Compared", num_models)

# Aggregated comparison table
st.header("📈 Model Comparison")
if filtered_agg.empty:
    st.info("No aggregated results found. Run evaluation first.")
else:
    # Highlight best performers
    if len(filtered_agg) > 0:
        metric_cols = st.columns(3)
        with metric_cols[0]:
            if "p95_latency_ms" in filtered_agg.columns:
                best_p95_idx = filtered_agg["p95_latency_ms"].idxmin()
                best_p95 = filtered_agg.loc[best_p95_idx, "model_name"]
                best_p95_val = filtered_agg.loc[best_p95_idx, "p95_latency_ms"]
                st.success(f"⚡ Best p95 Latency: **{best_p95}** ({best_p95_val:.0f} ms)")
        
        with metric_cols[1]:
            if "avg_cost_usd_per_request" in filtered_agg.columns:
                best_cost_idx = filtered_agg["avg_cost_usd_per_request"].idxmin()
                best_cost = filtered_agg.loc[best_cost_idx, "model_name"]
                best_cost_val = filtered_agg.loc[best_cost_idx, "avg_cost_usd_per_request"]
                st.success(f"💰 Best Cost/Request: **{best_cost}** (${best_cost_val:.6f})")
        
        with metric_cols[2]:
            if "json_valid_pct" in filtered_agg.columns:
                best_valid_idx = filtered_agg["json_valid_pct"].idxmax()
                best_valid = filtered_agg.loc[best_valid_idx, "model_name"]
                best_valid_val = filtered_agg.loc[best_valid_idx, "json_valid_pct"]
                st.success(f"✅ Best JSON Validity: **{best_valid}** ({best_valid_val:.1f}%)")
    
    # Display comparison table
    st.dataframe(
        filtered_agg.style.format({
            "avg_input_tokens": "{:.1f}",
            "avg_output_tokens": "{:.1f}",
            "p50_latency_ms": "{:.1f}",
            "p95_latency_ms": "{:.1f}",
            "p99_latency_ms": "{:.1f}",
            "json_valid_pct": "{:.2f}%",
            "avg_cost_usd_per_request": "${:.6f}",
            "total_cost_usd": "${:.4f}",
        }),
        use_container_width=True,
        height=300
    )

# Visualizations
if not filtered_raw.empty and len(filtered_raw) > 0:
    st.header("📊 Visualizations")
    
    # Filter success only for visualizations
    success_df = filtered_raw[filtered_raw["status"] == "success"].copy()
    
    if not success_df.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["Latency", "Tokens", "Cost", "JSON Validity"])
        
        with tab1:
            st.subheader("Latency Distribution")
            if "latency_ms" in success_df.columns and "model_name" in success_df.columns:
                fig = px.box(
                    success_df,
                    x="model_name",
                    y="latency_ms",
                    title="Latency Distribution by Model",
                    labels={"latency_ms": "Latency (ms)", "model_name": "Model"},
                    color="model_name"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Token Usage")
            cols = st.columns(2)
            
            with cols[0]:
                if "input_tokens" in success_df.columns:
                    fig = px.bar(
                        success_df.groupby("model_name")["input_tokens"].mean().reset_index(),
                        x="model_name",
                        y="input_tokens",
                        title="Average Input Tokens",
                        labels={"input_tokens": "Avg Input Tokens", "model_name": "Model"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with cols[1]:
                if "output_tokens" in success_df.columns:
                    fig = px.bar(
                        success_df.groupby("model_name")["output_tokens"].mean().reset_index(),
                        x="model_name",
                        y="output_tokens",
                        title="Average Output Tokens",
                        labels={"output_tokens": "Avg Output Tokens", "model_name": "Model"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Cost Analysis")
            if "cost_usd_total" in success_df.columns:
                fig = px.box(
                    success_df,
                    x="model_name",
                    y="cost_usd_total",
                    title="Cost per Request Distribution",
                    labels={"cost_usd_total": "Cost (USD)", "model_name": "Model"},
                    color="model_name"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Cost breakdown
                if not filtered_agg.empty and "avg_cost_usd_per_request" in filtered_agg.columns:
                    fig2 = px.bar(
                        filtered_agg.sort_values("avg_cost_usd_per_request"),
                        x="model_name",
                        y="avg_cost_usd_per_request",
                        title="Average Cost per Request",
                        labels={"avg_cost_usd_per_request": "Avg Cost (USD)", "model_name": "Model"}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        
        with tab4:
            st.subheader("JSON Validity Rate")
            if "json_valid" in success_df.columns:
                json_stats = success_df.groupby("model_name")["json_valid"].agg(["sum", "count"])
                json_stats["validity_pct"] = (json_stats["sum"] / json_stats["count"] * 100).round(2)
                json_stats = json_stats.reset_index()
                
                fig = px.bar(
                    json_stats.sort_values("validity_pct", ascending=False),
                    x="model_name",
                    y="validity_pct",
                    title="JSON Validity Percentage",
                    labels={"validity_pct": "Validity %", "model_name": "Model"},
                    color="validity_pct",
                    color_continuous_scale="Greens"
                )
                st.plotly_chart(fig, use_container_width=True)

# Raw metrics table
with st.expander("📋 Raw Metrics", expanded=False):
    if filtered_raw.empty:
        st.info("No raw metrics found.")
    else:
        st.dataframe(
            filtered_raw,
            use_container_width=True,
            height=400
        )

# Export buttons
st.sidebar.header("💾 Export")
if not filtered_agg.empty:
    csv_agg = filtered_agg.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Aggregated CSV",
        data=csv_agg,
        file_name="model_comparison.csv",
        mime="text/csv"
    )

if not filtered_raw.empty:
    csv_raw = filtered_raw.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Raw Metrics CSV",
        data=csv_raw,
        file_name="raw_metrics.csv",
        mime="text/csv"
    )
