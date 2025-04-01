"""
Historical Analytics Module for Loop-BI

This module contains functions for analyzing historical data from the data stored in Supabase.
These functions are used by the Streamlit app for time-series visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from supabase import Client


def get_protocol_metrics_over_time(
    supabase: Client, 
    days: int = 30,
    chain_id: Optional[str] = None,
    metric: str = "tvl"
) -> pd.DataFrame:
    """
    Get protocol-level metrics over time.
    
    Args:
        supabase: A Supabase client instance
        days: Number of days to look back
        chain_id: Optional chain ID to filter by
        metric: Metric to retrieve (tvl, utilization_rate, total_users, etc.)
        
    Returns:
        DataFrame with dates and the requested metric
    """
    # Calculate the start date
    start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
    
    # Build the query
    query = supabase.table("historical_protocol_metrics").select(
        "snapshot_date", metric, "chain_id"
    ).gte("snapshot_date", start_date)
    
    # Apply chain filter if specified
    if chain_id:
        query = query.eq("chain_id", chain_id)
        
    # Execute the query
    response = query.order("snapshot_date").execute()
    
    if not response.data:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["snapshot_date", metric, "chain_id"])
    
    # Convert to DataFrame
    df = pd.DataFrame(response.data)
    
    # Ensure date column is datetime
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    return df


def get_asset_metrics_over_time(
    supabase: Client,
    asset_symbol: Optional[str] = None,
    role: Optional[str] = None,
    days: int = 30,
    chain_id: Optional[str] = None,
    top_n: int = 5,
    metric: str = "total_value"
) -> pd.DataFrame:
    """
    Get asset-specific metrics over time.
    
    Args:
        supabase: A Supabase client instance
        asset_symbol: Optional specific asset to filter by
        role: Optional role to filter by ('lending' or 'looping')
        days: Number of days to look back
        chain_id: Optional chain ID to filter by
        top_n: If no specific asset is provided, return top N assets
        metric: Metric to retrieve (total_value, user_count, avg_position_size)
        
    Returns:
        DataFrame with dates, assets, and the requested metric
    """
    # Calculate the start date
    start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
    
    # Build the query
    query = supabase.table("historical_asset_metrics").select(
        "snapshot_date", "asset_symbol", "role", "chain_id", metric
    ).gte("snapshot_date", start_date)
    
    # Apply filters
    if asset_symbol:
        query = query.eq("asset_symbol", asset_symbol)
    
    if role:
        query = query.eq("role", role)
        
    if chain_id:
        query = query.eq("chain_id", chain_id)
        
    # Execute the query
    response = query.order("snapshot_date").execute()
    
    if not response.data:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["snapshot_date", "asset_symbol", "role", "chain_id", metric])
    
    # Convert to DataFrame
    df = pd.DataFrame(response.data)
    
    # Ensure date column is datetime
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    # If no specific asset was requested, get the top N assets by the metric
    if not asset_symbol:
        # Get the most recent date
        latest_date = df['snapshot_date'].max()
        
        # Filter to just the latest date for ranking
        latest_df = df[df['snapshot_date'] == latest_date]
        
        # Get top N assets by the metric
        top_assets = latest_df.sort_values(metric, ascending=False).head(top_n)['asset_symbol'].unique()
        
        # Filter the full dataset to just those top assets
        df = df[df['asset_symbol'].isin(top_assets)]
    
    return df


def get_user_segments_over_time(
    supabase: Client,
    days: int = 30,
    chain_id: Optional[str] = None,
    segment: Optional[str] = None,
    metric: str = "user_count"
) -> pd.DataFrame:
    """
    Get user segment metrics over time.
    
    Args:
        supabase: A Supabase client instance
        days: Number of days to look back
        chain_id: Optional chain ID to filter by
        segment: Optional segment to filter by
        metric: Metric to retrieve (user_count, total_value, percentage_of_tvl)
        
    Returns:
        DataFrame with dates, segments, and the requested metric
    """
    # Calculate the start date
    start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
    
    # Build the query
    query = supabase.table("historical_user_segments").select(
        "snapshot_date", "segment", "chain_id", metric
    ).gte("snapshot_date", start_date)
    
    # Apply filters
    if segment:
        query = query.eq("segment", segment)
        
    if chain_id:
        query = query.eq("chain_id", chain_id)
        
    # Execute the query
    response = query.order("snapshot_date").execute()
    
    if not response.data:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["snapshot_date", "segment", "chain_id", metric])
    
    # Convert to DataFrame
    df = pd.DataFrame(response.data)
    
    # Ensure date column is datetime
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    return df


def calculate_growth_metrics(
    supabase: Client,
    chain_id: Optional[str] = None,
    days: int = 30
) -> Dict[str, float]:
    """
    Calculate growth metrics by comparing current values to historical values.
    
    Args:
        supabase: A Supabase client instance
        chain_id: Optional chain ID to filter by
        days: Number of days for growth calculation
        
    Returns:
        Dictionary with growth percentages for key metrics
    """
    # Get historical protocol metrics
    metrics_df = get_protocol_metrics_over_time(
        supabase, 
        days=days,
        chain_id=chain_id
    )
    
    if metrics_df.empty:
        return {
            "tvl_growth": 0,
            "user_growth": 0,
            "utilization_rate_change": 0
        }
    
    # Get first and last dates with data
    first_date = metrics_df['snapshot_date'].min()
    last_date = metrics_df['snapshot_date'].max()
    
    # Filter to just these dates
    first_day = metrics_df[metrics_df['snapshot_date'] == first_date]
    last_day = metrics_df[metrics_df['snapshot_date'] == last_date]
    
    # Handle empty data frames
    if first_day.empty or last_day.empty:
        return {
            "tvl_growth": 0,
            "user_growth": 0,
            "utilization_rate_change": 0
        }
    
    # Calculate growth metrics
    tvl_start = first_day['tvl'].sum()
    tvl_end = last_day['tvl'].sum()
    tvl_growth = ((tvl_end - tvl_start) / tvl_start * 100) if tvl_start > 0 else 0
    
    users_start = first_day['total_users'].sum()
    users_end = last_day['total_users'].sum()
    user_growth = ((users_end - users_start) / users_start * 100) if users_start > 0 else 0
    
    util_start = first_day['utilization_rate'].mean()
    util_end = last_day['utilization_rate'].mean()
    util_change = util_end - util_start
    
    return {
        "tvl_growth": tvl_growth,
        "user_growth": user_growth,
        "utilization_rate_change": util_change
    } 