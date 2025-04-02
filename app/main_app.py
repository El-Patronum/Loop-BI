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
from core.analytics import (
    get_assets_by_role as get_assets_by_role_analytics,
    get_position_duration_analytics,
    calculate_loop_factors,
    get_portfolio_strategy_analysis,
    compare_chains_metrics,
    analyze_user_behavior,
    check_chain_data_quality,
    analyze_leverage_by_group,
    analyze_looping_pair_performance,
    calculate_utilization_rate as calculate_utilization_rate_analytics
)

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase = create_client(supabase_url, supabase_key)

# Set page configuration
st.set_page_config(
    page_title="LoopFi Analytics Dashboard",
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
st.title("LoopFi Analytics Dashboard")

# Initialize connection to Supabase
@st.cache_resource
def init_connection():
    return get_supabase_client()

try:
    supabase = init_connection()
except Exception as e:
    st.error(f"Failed to connect to Supabase: {str(e)}")
    st.stop()

# Function to fetch protocol info for specific LoopFi chains
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_target_loopfi_protocol_info():
    """Fetches protocol info entries for specific ETH and BSC LoopFi IDs."""
    eth_protocol_id = os.getenv("ETH_LOOPFI_PROTOCOL_ID", "loopfixyz").split('#')[0].strip().strip('"')
    bsc_protocol_id = os.getenv("BSC_LOOPFI_PROTOCOL_ID", "bsc_loopfixyz").split('#')[0].strip().strip('"')

    # Build specific filters for (id, chain_id) pairs
    eth_filter = f"and(id.eq.{eth_protocol_id},chain_id.eq.eth)"
    bsc_filter = f"and(id.eq.{bsc_protocol_id},chain_id.eq.bsc)"
    
    # Combine filters with OR
    combined_filter = f"or({eth_filter},{bsc_filter})"
    
    response = supabase.table("debank_protocols").select("*").or_(combined_filter).execute()
    
    if response.data:
        return response.data # Return list containing only ETH and BSC entries
    return []

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
def get_average_deposit_size(chain_filter=None):
    """
    Calculate the average deposit size with optional chain filtering
    
    Args:
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
    """
    # Build query with optional chain filter
    query = supabase.table("debank_user_loopfi_positions").select("asset_usd_value", "chain_id")
    
    # Apply chain filter if specified
    if chain_filter and chain_filter.lower() != "all chains":
        query = query.eq("chain_id", chain_filter.lower())
    
    response = query.execute()
    
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
    # Client-side calculation using the analytics function
    overall_rate, looping_specific_rate = calculate_utilization_rate_analytics(supabase)
    
    # Asset-specific calculation remains client-side for now, might be refactored
    asset_specific_df = pd.DataFrame()
    if by_asset:
        response = supabase.table("debank_user_loopfi_positions").select(
            "debt_usd_value", "asset_usd_value", "asset_symbol", "chain_id"
        ).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['debt_usd_value'] = df['debt_usd_value'].fillna(0)
            df['asset_usd_value'] = df['asset_usd_value'].fillna(0)
            
            # Apply chain filter if specified
            if chain_filter and chain_filter.lower() != "all chains":
                df = df[df['chain_id'].str.lower() == chain_filter.lower()]
            
            if not df.empty:
                grouped = df.groupby(['asset_symbol', 'chain_id'], observed=True).agg({
                    'debt_usd_value': 'sum',
                    'asset_usd_value': 'sum'
                }).reset_index()
                
                grouped['utilization_rate'] = grouped['debt_usd_value'] / grouped['asset_usd_value']
                grouped = grouped.replace([np.inf, -np.inf, np.nan], 0)
                asset_specific_df = grouped.sort_values('utilization_rate', ascending=False)
        
    return overall_rate, looping_specific_rate, asset_specific_df

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
            
            # Ensure chain_id is lowercase for consistent comparison
            if 'chain_id' in df.columns:
                df['chain_id'] = df['chain_id'].str.lower()
            
            # Apply chain filter if specified
            if chain_filter and chain_filter.lower() != "all chains":
                chain_filter_lower = chain_filter.lower()
                df = df[df['chain_id'] == chain_filter_lower]
            
            # Filter out tokens with empty or invalid symbols
            df = df[df['token_symbol'].notna() & (df['token_symbol'] != '') & 
                    (df['token_symbol'] != '""') & (df['token_symbol'] != '?')]
            
            # Ensure usd_value is numeric and positive
            df['usd_value'] = pd.to_numeric(df['usd_value'], errors='coerce').fillna(0)
            df = df[df['usd_value'] > 0]
            
            # Group by token and chain, compute aggregates
            if not df.empty:
                try:
                    # Try the observed=True method first (for newer pandas)
                    grouped = df.groupby(['token_symbol', 'chain_id'], observed=True)
                    result = grouped.agg(
                        total_usd_value=('usd_value', 'sum'),
                        user_count=('user_address', 'nunique')
                    ).reset_index()
                except:
                    # Fallback for older pandas versions
                    grouped = df.groupby(['token_symbol', 'chain_id'])
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
                'total_usd_value': 'total_collateral_usd_value'
            })
        
        return lending_assets, looping_assets
    except Exception as e:
        st.error(f"Error in get_assets_by_role: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Function to get position duration statistics
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_position_duration(chain_filter=None):
    # Use the new analytics module function
    return get_position_duration_analytics(supabase, chain_filter)

# Function to get user net worth distribution
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_user_net_worth_distribution(chain_filter=None):
    """
    Get the net worth distribution of LoopFi users with optional chain filtering
    
    Args:
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
    """
    if chain_filter and chain_filter.lower() != "all chains":
        # For chain-specific filtering, we need to join with positions table to get users by chain
        query_positions = supabase.table("debank_user_loopfi_positions").select(
            "user_address"
        ).eq("chain_id", chain_filter.lower()).execute()
        
        if not query_positions.data:
            return pd.DataFrame()
            
        # Get unique users with positions on this chain
        chain_users = list(set([item.get('user_address') for item in query_positions.data]))
        
        # Now get net worth for these users
        response = supabase.table("debank_loopfi_users").select(
            "total_net_worth_usd"
        ).in_("user_address", chain_users).execute()
    else:
        # If no chain filter, get all users
        response = supabase.table("debank_loopfi_users").select(
            "total_net_worth_usd"
        ).execute()
    
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
    # Extract LoopFi protocol IDs to exclude from results
    eth_protocol_id = os.getenv("ETH_LOOPFI_PROTOCOL_ID", "loopfixyz").split('#')[0].strip().strip('"')
    bsc_protocol_id = os.getenv("BSC_LOOPFI_PROTOCOL_ID", "bsc_loopfixyz").split('#')[0].strip().strip('"')
    loopfi_protocol_ids = [eth_protocol_id, bsc_protocol_id]
    
    response = supabase.table("debank_user_protocol_interactions").select(
        "protocol_id", "protocol_name", "chain_id", "user_address"
    ).execute()
    
    if response.data:
        df = pd.DataFrame(response.data)
        
        # Convert all chain_ids to lowercase for consistent comparison
        if 'chain_id' in df.columns:
            df['chain_id'] = df['chain_id'].str.lower()
        
        # Filter out LoopFi protocols (we want other protocols)
        df = df[~df['protocol_id'].isin(loopfi_protocol_ids)]
        
        # Apply chain filter if specified
        if chain_filter and chain_filter.lower() != "all chains":
            chain_filter_lower = chain_filter.lower()
            df = df[df['chain_id'] == chain_filter_lower]
        
        # Only proceed if we have data after filtering
        if not df.empty:
            # Make sure protocol_name and user_address are not null to avoid groupby issues
            df = df.dropna(subset=['protocol_name', 'user_address'])
            
            try:
                # Try the observed=True method first (for newer pandas)
                protocol_counts = df.groupby(['protocol_name', 'chain_id'], observed=True).agg(
                    count=('user_address', 'nunique')
                ).reset_index()
            except:
                # Fallback for older pandas versions
                protocol_counts = df.groupby(['protocol_name', 'chain_id']).agg(
                    count=('user_address', 'nunique')
                ).reset_index()
            
            if not protocol_counts.empty:
                protocol_counts = protocol_counts.sort_values('count', ascending=False).head(limit)
                return protocol_counts
    
    return pd.DataFrame()

# Function to calculate loop factor distribution (risk tolerance)
@st.cache_data(ttl=3600)
def get_loop_factor_distribution(chain_filter=None):
    """
    Calculates the loop factor distribution based on users and their overall positions.
    Loop Factor = Total Debt / Total Collateral per user.
    Returns the average loop factor and a DataFrame with percentage distribution per range.

    Args:
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC').

    Returns:
        tuple: (avg_loop_factor: float, distribution_df: pd.DataFrame)
               distribution_df columns: ['Range', 'Percentage']
    """
    try:
        # Fetch necessary data: user, collateral, debt, chain
        query = supabase.table("debank_user_loopfi_positions").select(
            "user_address", "asset_usd_value", "debt_usd_value", "chain_id"
        )

        # Apply chain filter if specified
        if chain_filter and chain_filter.lower() != "all chains":
            query = query.eq("chain_id", chain_filter.lower())

        response = query.execute()

        if not response.data:
            st.info(f"No position data found for filter: {chain_filter or 'All Chains'}")
            return 0.0, pd.DataFrame({'Range': [], 'Percentage': []})

        df = pd.DataFrame(response.data)
        df['asset_usd_value'] = pd.to_numeric(df['asset_usd_value'], errors='coerce').fillna(0)
        df['debt_usd_value'] = pd.to_numeric(df['debt_usd_value'], errors='coerce').fillna(0)

        # Group by user and sum collateral and debt
        user_totals = df.groupby('user_address').agg(
            total_collateral=('asset_usd_value', 'sum'),
            total_debt=('debt_usd_value', 'sum')
        ).reset_index()

        # Filter out users with zero debt (not looping) or zero collateral (cannot calculate factor)
        user_totals = user_totals[(user_totals['total_debt'] > 0) & (user_totals['total_collateral'] > 0)]

        if user_totals.empty:
            st.info(f"No users with valid loop factors found for filter: {chain_filter or 'All Chains'}")
            return 0.0, pd.DataFrame({'Range': [], 'Percentage': []})

        # Calculate loop factor per user
        user_totals['loop_factor'] = user_totals['total_debt'] / user_totals['total_collateral']

        # Calculate average loop factor
        avg_loop_factor = user_totals['loop_factor'].mean()

        # Define bins and labels based on the target pie chart
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0] # Up to 100%
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

        # Handle factors potentially > 1.0: assign them to the last bin '80-100%' for this chart
        # Alternatively, could add a '>100%' bin if needed. Sticking to screenshot for now.
        user_totals['loop_factor_clipped'] = user_totals['loop_factor'].clip(upper=1.0) # Clip at 1 for binning

        # Create categorical bins
        # include_lowest=True ensures 0 is included in the first bin
        user_totals['factor_range'] = pd.cut(
            user_totals['loop_factor_clipped'],
            bins=bins,
            labels=labels,
            right=True, # Intervals are (...] e.g., (0.0, 0.2]
            include_lowest=True # Include 0 in the first bin [0, 0.2]
        )

        # Count users per range
        distribution_counts = user_totals['factor_range'].value_counts()

        # Calculate percentage
        total_users = distribution_counts.sum()
        distribution_percentage = (distribution_counts / total_users * 100).round(2) # Round to 2 decimal places

        # Format into DataFrame for the pie chart
        distribution_df = distribution_percentage.reset_index()
        distribution_df.columns = ['Range', 'Percentage']

        # Ensure the order is correct for the chart legend
        distribution_df['Range'] = pd.Categorical(distribution_df['Range'], categories=labels, ordered=True)
        distribution_df = distribution_df.sort_values('Range')

        return avg_loop_factor, distribution_df

    except Exception as e:
        st.error(f"Error calculating loop factor distribution: {e}")
        return 0.0, pd.DataFrame({'Range': [], 'Percentage': []})

# Function to analyze TVL distribution by user size
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_tvl_by_user_size(chain_filter=None):
    """
    Get TVL distribution by user size with optional chain filtering
    
    Args:
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
    """
    if chain_filter and chain_filter.lower() != "all chains":
        # For chain-specific filtering, we need to join with positions table to get users by chain
        query_positions = supabase.table("debank_user_loopfi_positions").select(
            "user_address"
        ).eq("chain_id", chain_filter.lower()).execute()
        
        if not query_positions.data:
            return pd.DataFrame()
            
        # Get unique users with positions on this chain
        chain_users = list(set([item.get('user_address') for item in query_positions.data]))
        
        # Now get TVL for these users
        response = supabase.table("debank_loopfi_users").select(
            "user_address", "loopfi_usd_value"
        ).in_("user_address", chain_users).execute()
    else:
        # If no chain filter, get all users
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
def display_tvl_distribution(chain_filter=None):
    tvl_data = get_tvl_by_user_size(chain_filter)
    
    if not tvl_data.empty:
        # Use columns for better layout, but maybe not needed if charts are distinct
        st.subheader("TVL Distribution Analysis") # Add subheader for clarity
        
        # Display the pie chart first
        fig_pie = px.pie(
            tvl_data,
            values='loopfi_usd_value',
            names='size_category',
            title='TVL Distribution by User Size',
            hole=0.4
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextorientation='radial',
            hovertemplate='<b>%{label}</b><br>TVL: $%{value:,.2f}<br>Percentage: %{percent:.1%}'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Display the table and bar chart side-by-side if space allows, or below
        col1, col2 = st.columns([1, 2]) # Adjust ratio as needed

        with col1:
            # Format the TVL values for display in the table
            formatted_df = tvl_data.copy()
            formatted_df['loopfi_usd_value'] = formatted_df['loopfi_usd_value'].apply(lambda x: f"${x:,.2f}")
            formatted_df['percentage'] = formatted_df['percentage'].apply(lambda x: f"{x:.2f}%")
            formatted_df = formatted_df.rename(columns={
                'size_category': 'User Category',
                'loopfi_usd_value': 'TVL',
                'percentage': 'Percentage'
            })
            st.dataframe(formatted_df[['User Category', 'TVL', 'Percentage']])

        with col2:
            # Create a horizontal bar chart
            tvl_data['tvl_millions'] = tvl_data['loopfi_usd_value'] / 1000000
            labels = ['Small (<$1K)', 'Medium ($1K-$10K)', 'Large ($10K-$100K)', 'Whale (>$100K)']
            fig_bar = px.bar(
                tvl_data,
                x='tvl_millions',
                y='size_category',
                title='TVL by User Size (in millions $)',
                orientation='h',
                text=tvl_data['percentage'].apply(lambda x: f"{x:.1f}%")
            )
            fig_bar.update_layout(
                xaxis_title='TVL (millions $)',
                yaxis_title='User Size Category',
                yaxis={'categoryorder': 'array', 'categoryarray': labels[::-1]} # Ensure correct order
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
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
def get_strategy_analysis(selected_chain):
    """Get portfolio strategy analysis from the analytics module"""
    return get_portfolio_strategy_analysis(supabase, selected_chain)

# Function to get user behavior analysis
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_user_behavior_analysis(chain_filter=None):
    """Get user behavior analysis and segment classification with optional chain filtering"""
    return analyze_user_behavior(supabase, chain_filter)

# Function to fetch and calculate lending vs looping stats
@st.cache_data(ttl=3600)
def get_lending_looping_stats(chain_filter=None):
    """Fetches position data and calculates stats for lending vs looping."""
    query = supabase.table("debank_user_loopfi_positions").select(
        "asset_usd_value", "debt_usd_value", "chain_id"
    )
    if chain_filter and chain_filter.lower() != "all chains":
        query = query.eq("chain_id", chain_filter.lower())

    response = query.execute()

    # Default stats
    stats = {
        "lending_count": 0, "lending_avg_size": 0, "lending_median_size": 0,
        "looping_count": 0, "looping_avg_size": 0, "looping_median_size": 0
    }

    if not response.data:
        return stats

    df = pd.DataFrame(response.data)
    df['asset_usd_value'] = pd.to_numeric(df['asset_usd_value'], errors='coerce').fillna(0)
    df['debt_usd_value'] = pd.to_numeric(df['debt_usd_value'], errors='coerce').fillna(0)

    # Lending: debt_usd_value is zero or very small (handle potential float issues)
    lending_df = df[df['debt_usd_value'] < 0.01]
    # Looping: debt_usd_value is greater than zero
    looping_df = df[df['debt_usd_value'] >= 0.01]

    stats["lending_count"] = len(lending_df)
    stats["lending_avg_size"] = lending_df['asset_usd_value'].mean() if not lending_df.empty else 0
    stats["lending_median_size"] = lending_df['asset_usd_value'].median() if not lending_df.empty else 0
    stats["looping_count"] = len(looping_df)
    # Use asset_usd_value for looping size (collateral)
    stats["looping_avg_size"] = looping_df['asset_usd_value'].mean() if not looping_df.empty else 0
    stats["looping_median_size"] = looping_df['asset_usd_value'].median() if not looping_df.empty else 0
    
    return stats

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
view_options = ["Current Dashboard", "Historical Analytics"]
selected_view = st.sidebar.radio("Select View", view_options)

st.sidebar.info(
    "This dashboard displays analytics for the LoopFi protocol based on data from DeBank API. "
    "Use the 'Refresh Data' button to fetch the latest information."
)

# Import historical view functions if needed
if selected_view == "Historical Analytics":
    from core.historical_analytics import (
        get_protocol_metrics_over_time,
        get_asset_metrics_over_time,
        get_user_segments_over_time,
        calculate_growth_metrics
    )
    
    # Display historical view
    st.title("Historical Analytics")

    # Time period selector in sidebar
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
    
    # Historical view functions with caching
    @st.cache_data(ttl=3600)
    def load_protocol_metrics(days, chain=None):
        return get_protocol_metrics_over_time(
            supabase, 
            days=days,
            chain_id=chain
        )

    @st.cache_data(ttl=3600)
    def load_asset_metrics(days, role, chain=None, top_n=5):
        return get_asset_metrics_over_time(
            supabase,
            role=role,
            days=days,
            chain_id=chain,
            top_n=top_n
        )

    @st.cache_data(ttl=3600)
    def load_user_segments(days, chain=None):
        return get_user_segments_over_time(
            supabase,
            days=days,
            chain_id=chain
        )

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

    # Chain filter for historical view
    chain_filter = None if selected_chain == "All Chains" else selected_chain.lower()
    
    # Growth metrics
    growth_metrics = load_growth_metrics(days_to_show, chain_filter)

    # Display growth metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        tvl_growth = growth_metrics.get("tvl_growth", 0)
        st.metric(
            "TVL Growth", 
            f"{tvl_growth:.2f}%",
            delta=f"{tvl_growth:.2f}%",
            delta_color='normal'
        )

    with col2:
        user_growth = growth_metrics.get("user_growth", 0)
        st.metric(
            "User Growth", 
            f"{user_growth:.2f}%",
            delta=f"{user_growth:.2f}%",
            delta_color='normal'
        )

    with col3:
        util_change = growth_metrics.get("utilization_rate_change", 0) * 100  # Convert to percentage points
        st.metric(
            "Utilization Rate Change", 
            f"{util_change:.2f} pp",  # percentage points
            delta=f"{util_change:.2f} pp",
            delta_color='normal'
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
    
else:  # Current Dashboard view
    # Main dashboard content
    target_protocol_info = get_target_loopfi_protocol_info()
    user_count = get_user_count()

    # Protocol overview section
    st.header("Protocol Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Get name from the first entry if available
        protocol_name = target_protocol_info[0].get('name', 'LoopFi') if target_protocol_info else 'LoopFi'
        st.metric("Protocol Name", protocol_name)

    with col2:
        # Calculate total TVL by summing across the specifically fetched ETH and BSC entries
        total_tvl = sum(info.get('tvl', 0) for info in target_protocol_info if info.get('tvl'))
        st.metric("Total Value Locked (TVL)", f"${total_tvl:,.2f}")

    with col3:
        st.metric("Total Users", f"{user_count:,}")

    with col4:
        overall_util_rate_local, looping_util_rate_local, utilization_by_asset = get_utilization_rate(by_asset=True, chain_filter=selected_chain)
        st.metric("Overall Utilization", f"{overall_util_rate_local:.2%}")
        st.metric("Looping Utilization", f"{looping_util_rate_local:.2%}", delta_color="off",
                  help="Utilization calculated only for looping positions (Debt > 0)")

    # Add utilization rate by asset section
    st.header("Utilization Rate Analysis")
    st.caption(f"Showing data for: {selected_chain}")

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
        avg_deposit = get_average_deposit_size(selected_chain)
        st.metric("Average Deposit Size", f"${avg_deposit:,.2f}")
        
        # Net worth distribution
        st.subheader("User Net Worth Distribution")
        net_worth_df = get_user_net_worth_distribution(selected_chain)
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

    # Looping and Lending Assets
    st.header("Lending vs Looping Analysis")
    st.caption(f"Position Stats for: {selected_chain}") # Add caption for context

    # Fetch and display lending/looping stats
    ll_stats = get_lending_looping_stats(selected_chain)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lending Positions")
        c1, c2, c3 = st.columns(3)
        c1.metric("Count", f"{ll_stats['lending_count']:,}")
        c2.metric("Avg Size", f"${ll_stats['lending_avg_size']:,.2f}")
        c3.metric("Median Size", f"${ll_stats['lending_median_size']:,.2f}")

    with col2:
        st.subheader("Looping Positions")
        c4, c5, c6 = st.columns(3)
        c4.metric("Count", f"{ll_stats['looping_count']:,}")
        c5.metric("Avg Collateral", f"${ll_stats['looping_avg_size']:,.2f}")
        c6.metric("Median Collateral", f"${ll_stats['looping_median_size']:,.2f}")

    # Keep Top Lending Assets Chart
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

    # Keep Looping Pair Performance Expander
    # ADDED: Looping Pair Performance
    with st.expander("Looping Pair Performance (Avg. Net Value)", expanded=False):
        pair_perf_df = analyze_looping_pair_performance(supabase, selected_chain)
        if not pair_perf_df.empty:
            st.dataframe(
                pair_perf_df.head(15), # Show top 15 pairs
                column_config={
                    "collateral_symbol": "Collateral",
                    "borrowed_symbol": "Borrowed",
                    "chain_id": "Chain",
                    "avg_net_usd_value": st.column_config.NumberColumn("Avg. Net Value ($)", format="$%.2f"),
                    "position_count": "Positions"
                },
                hide_index=True
            )
        else:
            st.info("No data available for looping pair performance analysis.")

    # Add new section for Risk Analysis
    st.header("Risk Analysis")
    col1, col2 = st.columns(2)

    with col1:
        # Loop Factor Distribution
        st.subheader("Loop Factor Distribution")
        avg_loop_factor, distribution_df = get_loop_factor_distribution(selected_chain)

        # Display metric only if avg is valid, otherwise show N/A
        avg_display = f"{avg_loop_factor:.2%}"
        st.metric("Average Loop Factor", avg_display)

        if not distribution_df.empty:
            try:
                fig = px.pie(distribution_df, values='Percentage', names='Range',
                             title='User Distribution by Loop Factor (Leverage)',
                             category_orders={'Range': ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']} # Ensure order
                            )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as plot_err:
                 st.warning(f"Could not plot loop factor distribution: {plot_err}")
                 st.dataframe(distribution_df) # Show data instead
        else:
            st.info("No loop factor data available yet.")

    with col2:
        # Position Duration
        st.subheader("Position Duration Analysis")
        duration_data, avg_duration, has_real_data = get_position_duration(selected_chain)

        if not has_real_data:
            st.warning("⚠️ Position duration data is not yet available. Duration tracking will begin with the next data refresh cycle.")
            
            # Show placeholder statistics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Total Positions", f"{len(duration_data['count']) > 0 and sum(duration_data['count']) or 0}")
            with metric_col2:
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

    # ADDED: Leverage by Group Section
    with st.expander("Average Leverage by Group (Looping Positions)", expanded=False):
        leverage_by_asset, leverage_by_chain = analyze_leverage_by_group(supabase, selected_chain)
        
        group_col1, group_col2 = st.columns(2)
        
        with group_col1:
            st.subheader("By Collateral Asset")
            if not leverage_by_asset.empty:
                # Format leverage as percentage or multiplier
                leverage_by_asset['avg_leverage_formatted'] = leverage_by_asset['avg_leverage'].apply(
                    lambda x: f"{x*100:.1f}%" if x < 1 else f"{x:.2f}x"
                )
                st.dataframe(
                    leverage_by_asset[['asset_symbol', 'avg_leverage_formatted', 'position_count']].head(10),
                    column_config={
                        "asset_symbol": "Collateral Asset",
                        "avg_leverage_formatted": "Avg. Leverage",
                        "position_count": "Positions"
                    },
                    hide_index=True
                )
            else:
                st.info("No leverage data by asset available.")
            
        with group_col2:
            st.subheader("By Chain")
            if not leverage_by_chain.empty:
                # Format leverage as percentage or multiplier
                leverage_by_chain['avg_leverage_formatted'] = leverage_by_chain['avg_leverage'].apply(
                    lambda x: f"{x*100:.1f}%" if x < 1 else f"{x:.2f}x"
                )
                st.dataframe(
                    leverage_by_chain[['chain_id', 'avg_leverage_formatted', 'position_count']],
                    column_config={
                        "chain_id": "Chain",
                        "avg_leverage_formatted": "Avg. Leverage",
                        "position_count": "Positions"
                    },
                    hide_index=True
                )
            else:
                st.info("No leverage data by chain available.")

    # TVL Distribution Analysis
    with st.expander("TVL Distribution Analysis", expanded=False):
        display_tvl_distribution(selected_chain)

    # Portfolio Strategy Analysis
    st.header("Portfolio Strategy Analysis")

    strategy_counts, strategy_metrics = get_strategy_analysis(selected_chain)

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
                        st.metric("Average Position Size", f"${metrics.get('avg_size', 0):,.2f}")
                    
                    with metric_col2:
                        st.metric("Unique Users", f"{metrics.get('user_count', 0):,}")
                    
                    with metric_col3:
                        # Display both overall and looping-specific leverage
                        leverage_overall = metrics.get('avg_overall_leverage', 0)
                        leverage_looping = metrics.get('avg_looping_leverage', 0)
                        st.metric("Avg Leverage (Overall)", f"{leverage_overall*100:.1f}%" if leverage_overall < 1 else f"{leverage_overall:.2f}x")
                        st.metric("Avg Leverage (Looping)", f"{leverage_looping*100:.1f}%" if leverage_looping < 1 else f"{leverage_looping:.2f}x")
                    
                    # Add Avg Looping Net Value
                    st.metric("Avg Net Value (Looping)", f"${metrics.get('avg_looping_net_value', 0):,.2f}",
                              help="Average Net Value (Collateral - Debt) for looping positions within this strategy")

                    # Chain distribution for this strategy
                    chain_dist = metrics.get('chain_distribution', {})
                    if chain_dist:
                        chain_df = pd.DataFrame({
                            'Chain': [k.upper() for k in chain_dist.keys()],
                            'Positions': list(chain_dist.values())
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
    user_segments_df, segment_insights = get_user_behavior_analysis(selected_chain)

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
                'Avg. Net Worth': insights.get('avg_net_worth', 0),
                'Avg. Leverage': insights.get('avg_leverage', 0)
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Format numbers
        metrics_df['Total TVL'] = metrics_df['Total TVL'].apply(lambda x: f"${x:,.2f}")
        metrics_df['Avg. TVL'] = metrics_df['Avg. TVL'].apply(lambda x: f"${x:,.2f}")
        metrics_df['Avg. Positions'] = metrics_df['Avg. Positions'].apply(lambda x: f"{x:.1f}")
        metrics_df['Avg. Net Worth'] = metrics_df['Avg. Net Worth'].apply(lambda x: f"${x:,.2f}")
        metrics_df['Avg. Leverage'] = metrics_df['Avg. Leverage'].apply(
            lambda x: f"{x*100:.1f}%" if x < 1 else f"{x:.2f}x"
        )
        
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
    else:
        st.info("User behavior analysis data is not available.")

    # Footer with last update time
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Data source: DeBank API") 