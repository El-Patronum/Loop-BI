"""
Loop-BI - Analytics Dashboard for LoopFi Protocol

This Streamlit application provides comprehensive analytics for the LoopFi protocol,
including user behavior, asset distribution, risk analysis, and more.
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

from dotenv import load_dotenv
from supabase import create_client

# Add the project root to the path so we can import modules
import sys
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

from core.supabase_client import get_supabase_client
from core.analytics import get_assets_by_role as get_assets_by_role_analytics
from core.analytics import get_position_duration as get_position_duration_analytics
from core.analytics import calculate_loop_factors
from core.analytics import get_portfolio_strategy_analysis
from core.analytics import compare_chains_metrics
from core.analytics import analyze_user_behavior
from core.analytics import check_chain_data_quality

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase = create_client(supabase_url, supabase_key)

# Define display_chain_comparison function
def display_chain_comparison():
    st.header("Chain Comparison Analytics")
    
    # Run data quality check to detect duplication issues
    data_quality = check_chain_data_quality(supabase)
    
    # If duplication issues detected, show detailed warning
    if data_quality.get('has_duplication_issues', False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.warning(f"""
            ⚠️ **Data Quality Issue Detected**: Our analysis indicates nearly identical data across chains.
            
            * User count similarity: {data_quality.get('max_user_similarity', 0):.1f}%
            * TVL similarity: {data_quality.get('max_tvl_similarity', 0):.1f}%
            * Duplicate positions: {data_quality.get('duplicate_count', 0)}
            
            This issue is likely due to duplication in the data collection process. The Chain Comparison metrics 
            below may not accurately represent the differences between chains.
            """)
        
        with col2:
            # Add a button to repair the data
            if st.button("🛠️ Repair Data"):
                st.info("Running data repair process... This may take several minutes.")
                try:
                    import subprocess
                    result = subprocess.run(
                        ["python", "scripts/refresh_chains.py"], 
                        capture_output=True, 
                        text=True,
                        cwd="/Users/vertex/Documents/GitHub/Loop-BI"
                    )
                    if result.returncode == 0:
                        st.success("Data repair completed successfully! Refresh the page to see updated data.")
                    else:
                        st.error(f"Error during data repair: {result.stderr}")
                except Exception as e:
                    st.error(f"Failed to run data repair: {str(e)}")
    
    # Get chain comparison metrics
    comparison_df = compare_chains_metrics(supabase)
    
    if comparison_df.empty:
        st.warning("No chain comparison data available")
        return
    
    # Clean up the DataFrame for display
    display_cols = ['user_count', 'tvl', 'avg_deposit', 'utilization_rate', 'top_asset']
    display_names = {'user_count': 'Users', 'tvl': 'TVL', 'avg_deposit': 'Avg. Deposit', 
                     'utilization_rate': 'Utilization Rate', 'top_asset': 'Top Asset'}
    
    # Select only main columns for the key metrics table
    if all(col in comparison_df.columns for col in display_cols):
        display_df = comparison_df[display_cols].copy()
        display_df = display_df.rename(columns=display_names)
        
        # Format the columns
        display_df['TVL'] = display_df['TVL'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        display_df['Avg. Deposit'] = display_df['Avg. Deposit'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        display_df['Utilization Rate'] = display_df['Utilization Rate'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
        
        # Display the table
        st.subheader("Key Metrics by Chain")
        st.dataframe(display_df)
    
    # Extract strategy distribution data for visualization
    st.subheader("Strategy Distribution Comparison")
    
    # Find strategy columns
    strategy_cols = [col for col in comparison_df.columns if col.startswith('strategy_')]
    
    if strategy_cols:
        # Create a DataFrame for plotting
        chains = comparison_df.index.tolist()
        strategies = [col.replace('strategy_', '') for col in strategy_cols]
        
        # Prepare data for the visualization
        chart_data = []
        
        for chain in chains:
            for strategy, col in zip(strategies, strategy_cols):
                if col in comparison_df.columns:
                    value = comparison_df.loc[chain, col] if pd.notna(comparison_df.loc[chain, col]) else 0
                    chart_data.append({
                        'Chain': chain,
                        'Strategy': strategy,
                        'Percentage': value * 100  # Convert to percentage
                    })
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            
            # Create stacked bar chart
            fig = px.bar(
                chart_df,
                x='Chain',
                y='Percentage',
                color='Strategy',
                title='Strategy Distribution by Chain',
                labels={'Percentage': 'Percentage (%)'},
                height=400,
                text=chart_df['Percentage'].apply(lambda x: f'{x:.1f}%' if x > 5 else '')
            )
            
            fig.update_layout(
                xaxis_title='Chain',
                yaxis_title='Percentage (%)',
                legend_title='Strategy',
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No strategy distribution data available for visualization")
    else:
        st.info("No strategy data available to compare")
    
    # Additional visualizations for specific strategies
    if 'strategy_Leveraged Farming' in strategy_cols:
        st.subheader("Chain Distribution for Leveraged Farming")
        
        # Gather data for leveraged farming positions by chain
        leveraged_data = []
        for chain in chains:
            # Get position count for this strategy on this chain
            position_count = comparison_df.loc[chain, 'user_count'] * comparison_df.loc[chain, 'strategy_Leveraged Farming'] if pd.notna(comparison_df.loc[chain, 'strategy_Leveraged Farming']) else 0
            leveraged_data.append({
                'Chain': chain,
                'Positions': position_count
            })
        
        leveraged_df = pd.DataFrame(leveraged_data)
        
        if not leveraged_df.empty and leveraged_df['Positions'].sum() > 0:
            fig = px.bar(
                leveraged_df,
                y='Chain',
                x='Positions',
                color='Chain',
                orientation='h',
                title='Chain Distribution for Leveraged Farming',
                height=300,
                text='Positions'
            )
            
            # Add position count as text on bars
            fig.update_traces(texttemplate='Chain=%{y}<br>Positions=%{x:.0f}', textposition='outside')
            
            fig.update_layout(
                xaxis_title='Number of Positions',
                yaxis_title='Chain',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No leveraged farming position data available")

# Set page configuration
st.set_page_config(
    page_title="Loop-BI Analytics Dashboard",
    page_icon="📊",
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
st.title("Loop-BI Analytics Dashboard")

# Initialize connection to Supabase
@st.cache_resource
def init_connection():
    return get_supabase_client()

try:
    supabase = init_connection()
except Exception as e:
    st.error(f"Failed to connect to Supabase: {str(e)}")
    st.stop()

# Dashboard header
st.title("Loop-BI: LoopFi Protocol Analytics")

# Function to fetch protocol info
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_protocol_info():
    response = supabase.table("debank_protocols").select("*").eq("id", os.getenv("LOOPFI_PROTOCOL_ID", "loopfi")).execute()
    if response.data:
        return response.data[0]
    return None

# Function to get total user count
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_user_count():
    """
    Get the total number of unique users across all chains
    """
    response = supabase.table("debank_loopfi_users").select("user_address").execute()
    
    if response.data:
        # Get unique addresses to avoid counting users multiple times across chains
        unique_addresses = set(item['user_address'] for item in response.data)
        return len(unique_addresses)
    
    return 0

# Function to get average deposit size
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_average_deposit_size():
    # Fetch all positions and calculate average client-side
    response = supabase.table("debank_user_loopfi_positions").select("asset_usd_value").execute()
    
    if response.data and len(response.data) > 0:
        # Extract all asset values and calculate average
        asset_values = [item.get('asset_usd_value', 0) for item in response.data if item.get('asset_usd_value') is not None]
        if asset_values:
            return sum(asset_values) / len(asset_values)
    
    return 0

# Function to get utilization rate (borrowing vs. lending ratio)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_utilization_rate(by_asset=False, chain_filter=None):
    """
    Calculate the utilization rate (borrowing vs. lending ratio)
    
    Args:
        by_asset: If True, returns asset-specific utilization rates
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
    """
    # Client-side calculation instead of SQL aggregation
    response = supabase.table("debank_user_loopfi_positions").select(
        "debt_usd_value", "asset_usd_value", "asset_symbol", "chain_id"
    ).execute()
    
    if not response.data:
        return 0.0, pd.DataFrame()
    
    # Create DataFrame for easier filtering
    df = pd.DataFrame(response.data)
    # Convert potential None values to 0
    df['debt_usd_value'] = df['debt_usd_value'].fillna(0)
    df['asset_usd_value'] = df['asset_usd_value'].fillna(0)
    
    # Apply chain filter if specified
    if chain_filter and chain_filter.lower() != "all chains":
        df = df[df['chain_id'].str.lower() == chain_filter.lower()]
    
    # For overall utilization
    debt_sum = df['debt_usd_value'].sum()
    asset_sum = df['asset_usd_value'].sum()
    overall_rate = debt_sum / asset_sum if asset_sum > 0 else 0
    
    # For asset-specific utilization
    if by_asset:
        # Group by asset and chain
        grouped = df.groupby(['asset_symbol', 'chain_id'], observed=True).agg({
            'debt_usd_value': 'sum',
            'asset_usd_value': 'sum'
        }).reset_index()
        
        # Calculate utilization rate for each asset
        grouped['utilization_rate'] = grouped['debt_usd_value'] / grouped['asset_usd_value']
        grouped = grouped.replace([np.inf, -np.inf, np.nan], 0)
        
        # Sort by utilization rate
        grouped = grouped.sort_values('utilization_rate', ascending=False)
        
        return overall_rate, grouped
    
    return overall_rate, pd.DataFrame()

# Function to get most used assets
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_most_used_assets(limit=10, chain_filter=None):
    """
    Get the most used assets with optional chain filtering
    
    Args:
        limit: Maximum number of assets to return
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
    """
    # Direct query approach without using RPC
    try:
        # Try to fetch token holdings data
        response = supabase.table("debank_user_token_holdings").select(
            "token_symbol", "chain_id", "usd_value", "user_address"
        ).execute()
        
        if response.data:
            # Process the data in Python
            df = pd.DataFrame(response.data)
            
            # Apply chain filter if specified
            if chain_filter and chain_filter.lower() != "all chains":
                df = df[df['chain_id'].str.lower() == chain_filter.lower()]
            
            # Group by token and chain, compute aggregates
            if not df.empty:
                # Group by token_symbol and chain_id
                grouped = df.groupby(['token_symbol', 'chain_id'], observed=True)
                result = grouped.agg(
                    total_usd_value=('usd_value', 'sum'),
                    user_count=('user_address', 'nunique')
                ).reset_index()
                
                # Sort by total USD value
                result = result.sort_values('total_usd_value', ascending=False).head(limit)
                return result
    except Exception as e:
        st.error(f"Error processing token data: {str(e)}")
        
    # Return empty DataFrame if anything fails
    return pd.DataFrame()

# Function to analyze assets by role (lending vs looping)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_assets_by_role(chain_filter=None):
    """
    Get assets by role (lending vs looping) with optional chain filtering
    
    Args:
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
    """
    try:
        # Use the analytics module function but adapt the return format to what the app expects
        combined_assets = get_assets_by_role_analytics(supabase)
        
        if combined_assets is None or combined_assets.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Apply chain filter if specified
        if chain_filter and chain_filter.lower() != "all chains":
            combined_assets = combined_assets[
                combined_assets['chain_id'].str.lower() == chain_filter.lower()
            ]
        
        # Split the results into lending and looping assets
        lending_assets = combined_assets[combined_assets['role'] == 'lending'].copy()
        looping_assets = combined_assets[combined_assets['role'] == 'looping'].copy()
        
        # Rename columns to match expected format in the app
        if not lending_assets.empty:
            lending_assets = lending_assets.rename(columns={
                'total_usd_value': 'total_supplied'
            })
        
        if not looping_assets.empty:
            looping_assets = looping_assets.rename(columns={
                'total_usd_value': 'total_borrowed'
            })
        
        return lending_assets, looping_assets
    except Exception as e:
        st.error(f"Error in get_assets_by_role: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Function to get position duration statistics
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_position_duration():
    # Use the new analytics module function
    return get_position_duration_analytics(supabase)

# Function to get user net worth distribution
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_user_net_worth_distribution():
    response = supabase.table("debank_loopfi_users").select("total_net_worth_usd").execute()
    if response.data:
        df = pd.DataFrame(response.data)
        # Create bins for the distribution
        bins = [0, 1000, 10000, 100000, 1000000, float('inf')]
        labels = ['<$1K', '$1K-$10K', '$10K-$100K', '$100K-$1M', '>$1M']
        df['worth_category'] = pd.cut(df['total_net_worth_usd'], bins=bins, labels=labels)
        
        # Get value counts and rename properly to avoid duplicate column names
        worth_counts = df['worth_category'].value_counts()
        result_df = pd.DataFrame({
            'category': worth_counts.index,
            'count': worth_counts.values
        })
        
        return result_df
    return pd.DataFrame()

# Function to get other protocols used by LoopFi users
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_other_protocols_usage(limit=10, chain_filter=None):
    """
    Get other protocols used by LoopFi users with optional chain filtering
    
    Args:
        limit: Maximum number of protocols to return
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
    """
    response = supabase.table("debank_user_protocol_interactions").select(
        "protocol_id", "protocol_name", "chain_id", "user_address"
    ).execute()
    
    if response.data:
        df = pd.DataFrame(response.data)
        
        # Apply chain filter if specified
        if chain_filter and chain_filter.lower() != "all chains":
            df = df[df['chain_id'].str.lower() == chain_filter.lower()]
        
        # Only proceed if we have data after filtering
        if not df.empty:
            protocol_counts = df.groupby(['protocol_name', 'chain_id'], observed=True).agg(
                count=('user_address', 'nunique')
            ).reset_index()
            
            protocol_counts = protocol_counts.sort_values('count', ascending=False).head(limit)
            return protocol_counts
    
    return pd.DataFrame()

# Function to calculate loop factor distribution (risk tolerance)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_loop_factor_distribution(chain_filter=None):
    """
    Get the loop factor distribution with optional chain filtering
    
    Args:
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
    """
    # Use the analytics module function
    risk_counts, avg_loop_factor = calculate_loop_factors(supabase)
    
    # If we need to apply chain filter, we need to do it manually since
    # the analytics function doesn't support chain filtering directly
    if chain_filter and chain_filter.lower() != "all chains":
        # Fetch positions data with chain filter
        response = supabase.table("debank_user_loopfi_positions").select(
            "user_address", "asset_usd_value", "debt_usd_value", "chain_id"
        ).eq("chain_id", chain_filter.lower()).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            
            # Calculate loop factor (debt/asset ratio)
            df['loop_factor'] = df['debt_usd_value'] / df['asset_usd_value']
            
            # Remove extreme values or NaNs
            df = df[df['loop_factor'].between(0, 1)]
            
            # Create distribution bins
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
            df['risk_category'] = pd.cut(df['loop_factor'], bins=bins, labels=labels)
            
            # Get counts by category
            risk_counts = df['risk_category'].value_counts().reset_index()
            risk_counts.columns = ['factor_range', 'count']
            
            # Calculate average loop factor
            avg_loop_factor = df['loop_factor'].mean()
    
    return risk_counts, avg_loop_factor

# Function to analyze TVL distribution by user size
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_tvl_by_user_size():
    # Fetch users and their loopfi_usd_value
    response = supabase.table("debank_loopfi_users").select(
        "user_address", "loopfi_usd_value"
    ).execute()
    
    if response.data:
        df = pd.DataFrame(response.data)
        
        # Create user segments based on their TVL contribution
        bins = [0, 1000, 10000, 100000, float('inf')]
        labels = ['Small (<$1K)', 'Medium ($1K-$10K)', 'Large ($10K-$100K)', 'Whale (>$100K)']
        df['size_category'] = pd.cut(df['loopfi_usd_value'], bins=bins, labels=labels)
        
        # Calculate total TVL per category
        tvl_by_category = df.groupby('size_category', observed=True)['loopfi_usd_value'].sum().reset_index()
        
        # Calculate percentage of total
        total_tvl = tvl_by_category['loopfi_usd_value'].sum()
        if total_tvl > 0:
            tvl_by_category['percentage'] = (tvl_by_category['loopfi_usd_value'] / total_tvl * 100).round(2)
        
        return tvl_by_category
    
    return pd.DataFrame()

# Display TVL distribution by user size
def display_tvl_distribution():
    tvl_data = get_tvl_by_user_size()
    
    if not tvl_data.empty:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Create a pie chart of TVL distribution
            fig = px.pie(
                tvl_data,
                values='loopfi_usd_value',
                names='size_category',
                title='TVL Distribution by User Size',
                hole=0.4
            )
            
            # Customize the chart
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                insidetextorientation='radial'
            )
            
            # Add hover information with USD values
            fig.update_traces(
                hovertemplate='<b>%{label}</b><br>TVL: $%{value:,.2f}<br>Percentage: %{percent:.1%}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Convert to millions for easier reading
            tvl_data['tvl_millions'] = tvl_data['loopfi_usd_value'] / 1000000
            
            # Format the TVL values for display
            formatted_df = tvl_data.copy()
            formatted_df['loopfi_usd_value'] = formatted_df['loopfi_usd_value'].apply(lambda x: f"${x:,.2f}")
            formatted_df['percentage'] = formatted_df['percentage'].apply(lambda x: f"{x:.2f}%")
            formatted_df = formatted_df.rename(columns={
                'size_category': 'User Category',
                'loopfi_usd_value': 'TVL',
                'percentage': 'Percentage'
            })
            
            # Display as table
            st.dataframe(formatted_df[['User Category', 'TVL', 'Percentage']])
            
            # Define the labels here, same as in get_tvl_by_user_size function
            labels = ['Small (<$1K)', 'Medium ($1K-$10K)', 'Large ($10K-$100K)', 'Whale (>$100K)']
            
            # Create a horizontal bar chart
            fig = px.bar(
                tvl_data,
                x='tvl_millions',
                y='size_category',
                title='TVL by User Size (in millions $)',
                orientation='h',
                text=tvl_data['percentage'].apply(lambda x: f"{x:.1f}%")
            )
            
            fig.update_layout(
                xaxis_title='TVL (millions $)',
                yaxis_title='User Size Category',
                yaxis={'categoryorder': 'array', 'categoryarray': labels[::-1]}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No TVL distribution data available yet.")

# Function to get user distribution by chain
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_user_chain_distribution():
    # Use the positions table instead of token holdings for more accurate LoopFi user distribution
    response = supabase.table("debank_user_loopfi_positions").select(
        "chain_id", "user_address"
    ).execute()
    
    if response.data:
        df = pd.DataFrame(response.data)
        # Count unique users per chain
        chain_counts = df.groupby('chain_id', observed=True)['user_address'].nunique().reset_index()
        chain_counts.columns = ['chain_id', 'user_count']
        
        # Only include chains with users
        chain_counts = chain_counts[chain_counts['user_count'] > 0]
        chain_counts = chain_counts.sort_values('user_count', ascending=False)
        
        # Make sure chain_id is string
        chain_counts['chain_id'] = chain_counts['chain_id'].astype(str)
        
        return chain_counts
    return pd.DataFrame()

# Function to get portfolio strategy analysis
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_strategy_analysis():
    """Get portfolio strategy analysis from the analytics module"""
    return get_portfolio_strategy_analysis(supabase)

# Function to get user behavior analysis
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_user_behavior_analysis():
    """Get user behavior analysis and segment classification"""
    return analyze_user_behavior(supabase)

# Sidebar for data refresh control
st.sidebar.title("Controls")

if st.sidebar.button("Refresh Data"):
    # Clear all cached data
    st.cache_data.clear()
    st.rerun()

# Add chain filter
st.sidebar.header("Chain Filter")
chain_options = ["All Chains", "ETH", "BSC"]
selected_chain = st.sidebar.selectbox("Select Chain", chain_options, index=0)

# Add navigation section
st.sidebar.header("Navigation")
if st.sidebar.button("Historical Analytics"):
    # This will execute a script to run the historical view
    import subprocess
    import os
    historical_script = os.path.join(os.path.dirname(__file__), "historical_view.py")
    subprocess.Popen(["streamlit", "run", historical_script])
    st.sidebar.success("Launched Historical Analytics View")

st.sidebar.info(
    "This dashboard displays analytics for the LoopFi protocol based on data from DeBank API. "
    "Use the 'Refresh Data' button to fetch the latest information."
)

# Main dashboard content
protocol_info = get_protocol_info()
user_count = get_user_count()

# Protocol overview section
st.header("Protocol Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Protocol Name", protocol_info.get('name', 'LoopFi') if protocol_info else 'LoopFi')

with col2:
    tvl_value = protocol_info.get('tvl', 0) if protocol_info else 0
    st.metric("Total Value Locked (TVL)", f"${tvl_value:,.2f}")

with col3:
    st.metric("Total Users", f"{user_count:,}")

with col4:
    utilization_rate, _ = get_utilization_rate(chain_filter=selected_chain)
    st.metric("Utilization Rate", f"{utilization_rate:.2%}")

# Add utilization rate by asset section
st.header("Utilization Rate Analysis")
st.caption(f"Showing data for: {selected_chain}")
_, utilization_by_asset = get_utilization_rate(by_asset=True, chain_filter=selected_chain)

if not utilization_by_asset.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Asset Utilization Rates")
        # Only include assets with non-zero utilization
        util_chart_data = utilization_by_asset[utilization_by_asset['utilization_rate'] > 0].copy()
        
        if not util_chart_data.empty:
            # Format for better display
            util_chart_data['formatted_rate'] = util_chart_data['utilization_rate'].apply(lambda x: f"{x:.2%}")
            
            # Create the chart
            fig = px.bar(
                util_chart_data, 
                x='asset_symbol', 
                y='utilization_rate',
                color='chain_id' if 'chain_id' in util_chart_data.columns else None,
                title="Utilization Rate by Asset",
                labels={'utilization_rate': 'Utilization Rate', 'asset_symbol': 'Asset'},
                text='formatted_rate'
            )
            fig.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No asset utilization data available yet.")
    
    with col2:
        st.subheader("Supply vs Borrow by Asset")
        # Create a supply vs borrow comparison chart
        supply_borrow_data = utilization_by_asset.copy()
        
        if not supply_borrow_data.empty:
            fig = go.Figure()
            
            # Add supply bars
            fig.add_trace(go.Bar(
                x=supply_borrow_data['asset_symbol'],
                y=supply_borrow_data['asset_usd_value'],
                name='Supply',
                marker_color='blue'
            ))
            
            # Add borrow bars
            fig.add_trace(go.Bar(
                x=supply_borrow_data['asset_symbol'],
                y=supply_borrow_data['debt_usd_value'],
                name='Borrow',
                marker_color='red'
            ))
            
            # Update layout
            fig.update_layout(
                title="Supply vs Borrow by Asset",
                xaxis_title="Asset",
                yaxis_title="USD Value",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No supply vs borrow data available yet.")
else:
    st.info("No utilization data available yet.")

st.markdown("---")

# User metrics section
st.header("User Metrics")
st.caption(f"Showing data for: {selected_chain}")
col1, col2 = st.columns(2)

with col1:
    # Average deposit size
    avg_deposit = get_average_deposit_size()
    st.metric("Average Deposit Size", f"${avg_deposit:,.2f}")
    
    # Net worth distribution
    st.subheader("User Net Worth Distribution")
    net_worth_df = get_user_net_worth_distribution()
    if not net_worth_df.empty:
        fig = px.pie(net_worth_df, values='count', names='category', 
                     title='User Distribution by Net Worth')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No net worth data available yet.")

with col2:
    # Chain distribution - only show if "All Chains" is selected
    if selected_chain == "All Chains":
        st.subheader("Users by Chain")
        chain_df = get_user_chain_distribution()
        if not chain_df.empty:
            fig = px.bar(chain_df, x='chain_id', y='user_count', 
                         title='User Distribution by Chain')
            fig.update_layout(xaxis_title="Chain", yaxis_title="Number of Users")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chain distribution data available yet.")
    else:
        # Otherwise show another relevant metric
        st.subheader("User Activity")
        st.write(f"Showing activity for users on {selected_chain}")
        # Add relevant visualization here...

st.markdown("---")

# Asset metrics section
st.header("Asset Metrics")
st.caption(f"Showing data for: {selected_chain}")
col1, col2 = st.columns(2)

with col1:
    # Most used assets
    st.subheader("Most Used Assets")
    assets_df = get_most_used_assets(limit=10, chain_filter=selected_chain)
    if not assets_df.empty:
        if 'total_usd_value' in assets_df.columns and 'token_symbol' in assets_df.columns:
            # Only use chain_id for color if showing all chains
            color_column = 'chain_id' if selected_chain == "All Chains" and 'chain_id' in assets_df.columns else None
            
            fig = px.bar(assets_df, x='token_symbol', y='total_usd_value', 
                      color=color_column, title='Top 10 Assets by USD Value')
            fig.update_layout(xaxis_title="Token", yaxis_title="Total USD Value")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(assets_df)
    else:
        st.info("No asset data available yet.")

with col2:
    # Other protocols usage
    st.subheader("Other Protocols Usage")
    protocols_df = get_other_protocols_usage(limit=10, chain_filter=selected_chain)
    if not protocols_df.empty:
        # Only use chain_id for color if showing all chains
        color_column = 'chain_id' if selected_chain == "All Chains" and 'chain_id' in protocols_df.columns else None
        
        fig = px.bar(protocols_df, x='protocol_name', y='count',
                   color=color_column, title='Top Protocols Used by LoopFi Users')
        fig.update_layout(xaxis_title="Protocol", yaxis_title="Number of Users")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No protocol usage data available yet.")

# Lending and Looping Assets
st.header("Lending vs Looping Analysis")
col1, col2 = st.columns(2)

# Get lending and looping assets data with better error handling
try:
    lending_assets, looping_assets = get_assets_by_role(selected_chain)
except Exception as e:
    st.error(f"Error loading assets data: {str(e)}")
    lending_assets, looping_assets = pd.DataFrame(), pd.DataFrame()

with col1:
    # Lending assets
    st.subheader("Top Lending Assets")
    if not lending_assets.empty and 'chain_id' in lending_assets.columns:
        # Check if chain_id exists in the DataFrame
        color_column = 'chain_id' if 'chain_id' in lending_assets.columns else None
        
        fig = px.bar(lending_assets.head(10), x='asset_symbol', y='total_supplied',
                   color=color_column, title='Top 10 Lending Assets by USD Value')
        fig.update_layout(xaxis_title="Asset", yaxis_title="Total Supplied (USD)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No lending asset data available yet.")

with col2:
    # Looping assets
    st.subheader("Top Looping Assets")
    if not looping_assets.empty and 'chain_id' in looping_assets.columns:
        # Check if chain_id exists in the DataFrame
        color_column = 'chain_id' if 'chain_id' in looping_assets.columns else None
        
        fig = px.bar(looping_assets.head(10), x='asset_symbol', y='total_borrowed',
                   color=color_column, title='Top 10 Looping Assets by USD Value')
        fig.update_layout(xaxis_title="Asset", yaxis_title="Total Borrowed (USD)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No looping asset data available yet.")

# Add new section for Risk Analysis
st.header("Risk Analysis")
col1, col2 = st.columns(2)

with col1:
    # Loop Factor Distribution
    st.subheader("Loop Factor Distribution")
    loop_factor_df, avg_loop_factor = get_loop_factor_distribution(selected_chain)
    if not loop_factor_df.empty:
        st.metric("Average Loop Factor", f"{avg_loop_factor:.2%}")
        
        # Make sure we have the correct column names
        if 'factor_range' in loop_factor_df.columns and 'count' in loop_factor_df.columns:
            fig = px.pie(loop_factor_df, values='count', names='factor_range', 
                        title='User Distribution by Loop Factor (Leverage)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Loop factor data format is not as expected.")
    else:
        st.info("No loop factor data available yet.")

with col2:
    # Position Duration
    st.subheader("Position Duration Analysis")
    duration_data, avg_duration, has_real_data = get_position_duration()

    if not has_real_data:
        st.warning("⚠️ Position duration data is not yet available. Duration tracking will begin with the next data refresh cycle.")
        
        # Show placeholder statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Positions", f"{len(duration_data['count']) > 0 and sum(duration_data['count']) or 0}")
        with col2:
            st.metric("New Positions (0-7 days)", f"{duration_data['count'][0] if len(duration_data['count']) > 0 else 0}")
        
        # Display informational message about position durations
        st.info("Position duration tracking allows you to understand how long users maintain their positions before closing them. This metric will be available after the next data collection cycle.")
    else:
        # Real duration data is available
        st.metric("Average Position Duration", f"{avg_duration:.2f} days")
        
        if not duration_data.empty:
            # Create bar chart for duration distribution
            fig = px.bar(
                duration_data,
                x='duration_range',
                y='count',
                title='Position Duration Distribution'
            )
            fig.update_layout(xaxis_title="Duration", yaxis_title="Number of Positions")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No position duration data available yet.")

# Add new section for TVL Analysis
st.header("TVL Analysis")
tvl_distribution = get_tvl_by_user_size()
if not tvl_distribution.empty:
    fig = px.pie(tvl_distribution, values='loopfi_usd_value', names='size_category',
                 title='TVL Distribution by User Size', 
                 hover_data=['percentage'])
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show the distribution as a table too
    st.dataframe(tvl_distribution)
else:
    st.info("No TVL distribution data available yet.")

# TVL Distribution Analysis
with st.expander("TVL Distribution Analysis", expanded=False):
    display_tvl_distribution()

# Call the Chain Comparison Analytics section
display_chain_comparison()

# Portfolio Strategy Analysis
st.header("Portfolio Strategy Analysis")

strategy_counts, strategy_metrics = get_strategy_analysis()

if not strategy_counts.empty:
    # Create two columns for the strategy overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Pie chart of strategy distribution
        fig = px.pie(
            strategy_counts, 
            values='count', 
            names='strategy',
            title='Position Strategy Distribution',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Strategy count table with percentages
        st.subheader("Strategy Breakdown")
        # Format percentages to 1 decimal place
        strategy_counts['percentage'] = strategy_counts['percentage'].round(1).astype(str) + '%'
        st.dataframe(
            strategy_counts[['strategy', 'count', 'percentage']],
            hide_index=True,
            use_container_width=True
        )
    
    # Strategy Metrics
    st.subheader("Strategy-Specific Metrics")
    
    # Create tabs for each strategy
    if strategy_metrics:
        tabs = st.tabs(list(strategy_metrics.keys()))
        
        for i, (strategy_name, metrics) in enumerate(strategy_metrics.items()):
            with tabs[i]:
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Average Position Size", f"${metrics['avg_size']:,.2f}")
                
                with metric_col2:
                    st.metric("Unique Users", f"{metrics['user_count']:,}")
                
                with metric_col3:
                    leverage = metrics['avg_leverage'] * 100
                    st.metric("Leverage Ratio", f"{leverage:.1f}%" if leverage < 100 else f"{leverage/100:.2f}x")
                
                # Chain distribution for this strategy
                if metrics['chain_distribution']:
                    chain_df = pd.DataFrame({
                        'Chain': [k.upper() for k in metrics['chain_distribution'].keys()],
                        'Positions': list(metrics['chain_distribution'].values())
                    })
                    
                    # Create horizontal bar chart of chain distribution
                    fig = px.bar(
                        chain_df,
                        x='Positions',
                        y='Chain',
                        orientation='h',
                        title=f'Chain Distribution for {strategy_name}',
                        color='Chain',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(yaxis_title="", xaxis_title="Number of Positions")
                    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Portfolio strategy data is not available.")

# User Behavior Analysis
st.header("User Behavior Analysis")
user_segments_df, segment_insights = get_user_behavior_analysis()

if not user_segments_df.empty and segment_insights:
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show segment distribution
        segment_counts = user_segments_df['user_segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'User Count']
        
        # Pie chart of user segments
        fig = px.pie(
            segment_counts,
            values='User Count',
            names='Segment',
            title='User Segment Distribution',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show segment counts in a table
        st.subheader("User Segments")
        styled_counts = segment_counts.copy()
        
        # Calculate percentage
        total_users = styled_counts['User Count'].sum()
        styled_counts['Percentage'] = (styled_counts['User Count'] / total_users * 100).round(1).astype(str) + '%'
        
        st.dataframe(
            styled_counts,
            hide_index=True,
            use_container_width=True
        )
    
    # Segment metrics
    st.subheader("Segment Metrics")
    
    # Create metrics dataframe
    metrics_data = []
    for segment, insights in segment_insights.items():
        metrics_data.append({
            'Segment': segment,
            'User Count': insights.get('count', 0),
            'Total TVL': insights.get('total_tvl', 0),
            'Avg. TVL': insights.get('avg_tvl', 0),
            'Avg. Positions': insights.get('avg_positions', 0),
            'Avg. Net Worth': insights.get('avg_net_worth', 0)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Format numbers
    metrics_df['Total TVL'] = metrics_df['Total TVL'].apply(lambda x: f"${x:,.2f}")
    metrics_df['Avg. TVL'] = metrics_df['Avg. TVL'].apply(lambda x: f"${x:,.2f}")
    metrics_df['Avg. Positions'] = metrics_df['Avg. Positions'].apply(lambda x: f"{x:.1f}")
    metrics_df['Avg. Net Worth'] = metrics_df['Avg. Net Worth'].apply(lambda x: f"${x:,.2f}")
    
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    # TVL distribution by segment
    st.subheader("TVL Distribution by Segment")
    
    # Extract TVL data by segment
    tvl_by_segment = []
    for segment, insights in segment_insights.items():
        tvl_by_segment.append({
            'Segment': segment,
            'TVL': insights.get('total_tvl', 0)
        })
    
    tvl_segment_df = pd.DataFrame(tvl_by_segment)
    
    if not tvl_segment_df.empty:
        fig = px.bar(
            tvl_segment_df,
            x='Segment',
            y='TVL',
            title='TVL by User Segment',
            color='Segment',
            labels={'TVL': 'Total Value Locked ($)'},
            text_auto='.2s'
        )
        fig.update_traces(texttemplate='$%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment insights and recommendations
    st.subheader("Key Insights")
    
    # Add insights for each segment
    segment_descriptions = {
        'Power Users': "Users with multiple positions across different strategies, often diversified across chains.",
        'Whales': "High-value users who contribute significantly to the protocol's TVL.",
        'Risk-Takers': "Users with high leverage ratios, comfortable taking on more risk.",
        'Experimental Users': "Users trying multiple strategies with diversified portfolios.",
        'Multi-Chain Users': "Users who are active across multiple blockchain networks.",
        'Casual Users': "Users with fewer positions and lower value deposits.",
        'Other': "Users who don't fit neatly into other categories."
    }
    
    for segment in segment_insights.keys():
        if segment in segment_descriptions:
            st.markdown(f"**{segment}**: {segment_descriptions[segment]}")
            
            # Add specific insights based on the data
            if segment == 'Power Users':
                power_pct = segment_insights[segment].get('pct_of_users', 0)
                power_tvl = segment_insights[segment].get('total_tvl', 0)
                st.markdown(f"- Power users make up {power_pct:.1f}% of users but control ${power_tvl:,.2f} in TVL.")
            
            elif segment == 'Whales':
                whale_pct = segment_insights[segment].get('pct_of_users', 0)
                whale_tvl_pct = segment_insights[segment].get('total_tvl', 0) / sum(i.get('total_tvl', 0) for i in segment_insights.values()) * 100
                st.markdown(f"- Whales represent only {whale_pct:.1f}% of users but control {whale_tvl_pct:.1f}% of total TVL.")
    
    # Recommendations based on user segments
    st.subheader("Strategic Recommendations")
    
    # Check for specific patterns in the data
    power_user_count = segment_insights.get('Power Users', {}).get('count', 0)
    casual_user_count = segment_insights.get('Casual Users', {}).get('count', 0)
    total_users = sum(s.get('count', 0) for s in segment_insights.values())
    
    if power_user_count / total_users < 0.1:
        st.markdown("- **Increase Power User Conversion**: Create incentives to convert casual users into power users by offering rewards for multi-chain activity.")
    
    if 'Whales' in segment_insights and segment_insights['Whales'].get('count', 0) > 0:
        st.markdown("- **Whale Retention Program**: Develop a targeted program to retain whales and encourage increased engagement.")
    
    if 'Multi-Chain Users' in segment_insights and segment_insights['Multi-Chain Users'].get('count', 0) > 0:
        st.markdown("- **Promote Cross-Chain Activity**: Highlight the benefits of using LoopFi across multiple chains through educational content and incentives.")
    
    if casual_user_count / total_users > 0.5:
        st.markdown("- **User Education**: Most users are casual, indicating a need for better educational resources to increase confidence and engagement.")
else:
    st.info("User behavior analysis data is not available.")

# Footer with last update time
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("Data source: DeBank API") 