"""
Loop-BI - Historical Data Visualization

This Streamlit page provides time-series visualizations for historical data.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add the project root to the path so we can import modules
import sys
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

from core.supabase_client import get_supabase_client
from core.historical_analytics import (
    get_protocol_metrics_over_time,
    get_asset_metrics_over_time,
    get_user_segments_over_time,
    calculate_growth_metrics
)

# Set page configuration
st.set_page_config(
    page_title="Loop-BI Historical Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric .label {
        font-weight: 500;
    }
    .stDataFrame {
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Loop-BI Historical Analytics")

# Initialize connection to Supabase
@st.cache_resource
def init_connection():
    return get_supabase_client()

try:
    supabase = init_connection()
except Exception as e:
    st.error(f"Failed to connect to Supabase: {str(e)}")
    st.stop()

# Sidebar for controls
st.sidebar.title("Controls")

# Time period selector
st.sidebar.header("Time Period")
time_options = {
    "7 days": 7,
    "30 days": 30,
    "90 days": 90,
    "180 days": 180,
    "1 year": 365
}
selected_time = st.sidebar.selectbox(
    "Select Time Period", 
    list(time_options.keys()),
    index=1  # Default to 30 days
)
days_to_show = time_options[selected_time]

# Chain selector
st.sidebar.header("Chain")
chain_options = ["All Chains", "ETH", "BSC"]
selected_chain = st.sidebar.selectbox("Select Chain", chain_options, index=0)
chain_filter = None if selected_chain == "All Chains" else selected_chain.lower()

# Data refresh button
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Function to get protocol metrics with caching
@st.cache_data(ttl=3600)
def load_protocol_metrics(days, chain=None):
    return get_protocol_metrics_over_time(
        supabase, 
        days=days,
        chain_id=chain
    )

# Function to get asset metrics with caching
@st.cache_data(ttl=3600)
def load_asset_metrics(days, role, chain=None, top_n=5):
    return get_asset_metrics_over_time(
        supabase,
        role=role,
        days=days,
        chain_id=chain,
        top_n=top_n
    )

# Function to get user segments with caching
@st.cache_data(ttl=3600)
def load_user_segments(days, chain=None):
    return get_user_segments_over_time(
        supabase,
        days=days,
        chain_id=chain
    )

# Function to get growth metrics with caching
@st.cache_data(ttl=3600)
def load_growth_metrics(days, chain=None):
    return calculate_growth_metrics(
        supabase,
        days=days,
        chain_id=chain
    )

# Display protocol metrics over time
st.header("Protocol Metrics Over Time")
st.caption(f"Showing data for the past {days_to_show} days on {selected_chain}")

# Growth metrics
growth_metrics = load_growth_metrics(days_to_show, chain_filter)

# Display growth metrics in columns
col1, col2, col3 = st.columns(3)

with col1:
    tvl_growth = growth_metrics.get("tvl_growth", 0)
    growth_color = "green" if tvl_growth >= 0 else "red"
    st.metric(
        "TVL Growth", 
        f"{tvl_growth:.2f}%",
        delta=f"{tvl_growth:.2f}%",
        delta_color=growth_color
    )

with col2:
    user_growth = growth_metrics.get("user_growth", 0)
    growth_color = "green" if user_growth >= 0 else "red"
    st.metric(
        "User Growth", 
        f"{user_growth:.2f}%",
        delta=f"{user_growth:.2f}%",
        delta_color=growth_color
    )

with col3:
    util_change = growth_metrics.get("utilization_rate_change", 0) * 100  # Convert to percentage points
    growth_color = "green" if util_change >= 0 else "red"
    st.metric(
        "Utilization Rate Change", 
        f"{util_change:.2f} pp",  # percentage points
        delta=f"{util_change:.2f} pp",
        delta_color=growth_color
    )

# Load protocol metrics data
protocol_data = load_protocol_metrics(days_to_show, chain_filter)

if not protocol_data.empty:
    # Create tabs for different metrics
    metric_tabs = st.tabs(["TVL", "Users", "Utilization Rate"])
    
    with metric_tabs[0]:
        # TVL Chart
        st.subheader("Total Value Locked (TVL) Over Time")
        fig = px.line(
            protocol_data, 
            x='snapshot_date', 
            y='tvl',
            color='chain_id' if 'chain_id' in protocol_data.columns and selected_chain == "All Chains" else None,
            title=f"TVL Trend - Past {days_to_show} Days"
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="TVL (USD)")
        st.plotly_chart(fig, use_container_width=True)
    
    with metric_tabs[1]:
        # Users Chart
        st.subheader("Total Users Over Time")
        fig = px.line(
            protocol_data, 
            x='snapshot_date', 
            y='total_users',
            color='chain_id' if 'chain_id' in protocol_data.columns and selected_chain == "All Chains" else None,
            title=f"User Growth - Past {days_to_show} Days"
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Users")
        st.plotly_chart(fig, use_container_width=True)
    
    with metric_tabs[2]:
        # Utilization Rate Chart
        st.subheader("Utilization Rate Over Time")
        fig = px.line(
            protocol_data, 
            x='snapshot_date', 
            y='utilization_rate',
            color='chain_id' if 'chain_id' in protocol_data.columns and selected_chain == "All Chains" else None,
            title=f"Utilization Rate Trend - Past {days_to_show} Days"
        )
        fig.update_layout(
            xaxis_title="Date", 
            yaxis_title="Utilization Rate",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No historical protocol metrics available yet. Run pipeline with --historical flag to generate data.")

# Asset Trends Section
st.header("Asset Trends")
asset_tabs = st.tabs(["Lending Assets", "Looping Assets"])

with asset_tabs[0]:
    # Lending Assets Trends
    st.subheader("Top Lending Assets Over Time")
    lending_data = load_asset_metrics(days_to_show, role='lending', chain=chain_filter)
    
    if not lending_data.empty:
        fig = px.line(
            lending_data,
            x='snapshot_date',
            y='total_value',
            color='asset_symbol',
            title=f"Top Lending Assets - Past {days_to_show} Days",
            labels={'total_value': 'Total Value (USD)', 'snapshot_date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical lending asset data available yet.")

with asset_tabs[1]:
    # Looping Assets Trends
    st.subheader("Top Looping Assets Over Time")
    looping_data = load_asset_metrics(days_to_show, role='looping', chain=chain_filter)
    
    if not looping_data.empty:
        fig = px.line(
            looping_data,
            x='snapshot_date',
            y='total_value',
            color='asset_symbol',
            title=f"Top Looping Assets - Past {days_to_show} Days",
            labels={'total_value': 'Total Value (USD)', 'snapshot_date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical looping asset data available yet.")

# User Segments Section
st.header("User Segments Over Time")
segment_data = load_user_segments(days_to_show, chain_filter)

if not segment_data.empty:
    # Create a stacked area chart of user segments over time
    fig = px.area(
        segment_data,
        x='snapshot_date',
        y='total_value',
        color='segment',
        title=f"TVL by User Segment - Past {days_to_show} Days",
        labels={'total_value': 'TVL (USD)', 'snapshot_date': 'Date'}
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="TVL (USD)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a chart of user count by segment
    fig = px.line(
        segment_data,
        x='snapshot_date',
        y='user_count',
        color='segment',
        title=f"Users by Segment - Past {days_to_show} Days",
        labels={'user_count': 'Number of Users', 'snapshot_date': 'Date'}
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Users")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No historical user segment data available yet.")

# Footer
st.markdown("---")
st.caption(f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("Loop-BI: Historical Analytics Dashboard") 