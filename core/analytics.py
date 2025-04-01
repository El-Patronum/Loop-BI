"""
Analytics Module for Loop-BI

This module contains functions for computing analytics from the data stored in Supabase.
These functions are used by the Streamlit app but are kept separate for modularity.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from supabase import Client


def calculate_average_deposit_size(supabase: Client) -> float:
    """
    Calculate the average deposit size across all LoopFi positions.
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        The average deposit size (asset_usd_value) in USD
    """
    response = supabase.table("debank_user_loopfi_positions").select("avg(asset_usd_value)").execute()
    if response.data and response.data[0] and 'avg' in response.data[0]:
        return response.data[0]['avg']
    return 0.0


def get_most_used_assets(supabase: Client, limit: int = 10) -> pd.DataFrame:
    """
    Get the most used assets across all LoopFi users.
    
    Args:
        supabase: A Supabase client instance
        limit: Maximum number of assets to return
        
    Returns:
        DataFrame with token_symbol, chain_id, total_usd_value, and user_count
    """
    # Try to use a stored procedure if available
    try:
        response = supabase.rpc('get_most_used_assets', {'limit_val': limit}).execute()
        if response.data:
            return pd.DataFrame(response.data)
    except:
        pass
    
    # Fallback: use a regular query
    # Note: This simplified query doesn't get user_count accurately
    response = supabase.table("debank_user_token_holdings").select(
        "token_symbol", "chain_id", "usd_value"
    ).order("usd_value", desc=True).limit(limit).execute()
    
    if response.data:
        df = pd.DataFrame(response.data)
        # Aggregate by token_symbol and chain_id
        agg_df = df.groupby(["token_symbol", "chain_id"]).agg({
            "usd_value": "sum"
        }).reset_index()
        
        agg_df = agg_df.rename(columns={"usd_value": "total_usd_value"})
        agg_df = agg_df.sort_values("total_usd_value", ascending=False).head(limit)
        return agg_df
    
    return pd.DataFrame()


def calculate_loop_factors(supabase: Client) -> Tuple[pd.DataFrame, float]:
    """
    Calculate the loop factors (leverage) for positions with debt.
    Loop factor = debt_usd_value / asset_usd_value
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        Tuple containing (DataFrame with loop_factor distribution, average loop factor)
    """
    response = supabase.table("debank_user_loopfi_positions").select(
        "user_address", "asset_usd_value", "debt_usd_value"
    ).gt("debt_usd_value", 0).execute()
    
    if not response.data:
        return pd.DataFrame(), 0
    
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


def get_user_token_distribution(supabase: Client, chain_id: Optional[str] = None) -> pd.DataFrame:
    """
    Get the distribution of users by token holdings.
    
    Args:
        supabase: A Supabase client instance
        chain_id: Optional chain ID to filter by
        
    Returns:
        DataFrame with token_symbol and user_count
    """
    query = supabase.table("debank_user_token_holdings").select(
        "token_symbol", "user_address", "chain_id"
    )
    
    if chain_id:
        query = query.eq("chain_id", chain_id)
        
    response = query.execute()
    
    if response.data:
        df = pd.DataFrame(response.data)
        # Count unique users per token
        token_counts = df.groupby('token_symbol')['user_address'].nunique().reset_index()
        token_counts.columns = ['token_symbol', 'user_count']
        
        # Get total user count
        total_users = len(df['user_address'].unique())
        
        # Calculate percentage
        token_counts['percentage'] = (token_counts['user_count'] / total_users) * 100
        
        return token_counts.sort_values('user_count', ascending=False)
    
    return pd.DataFrame()


def get_other_protocol_usage(supabase: Client, limit: int = 10) -> pd.DataFrame:
    """
    Get the other protocols used by LoopFi users.
    
    Args:
        supabase: A Supabase client instance
        limit: Maximum number of protocols to return
        
    Returns:
        DataFrame with protocol_name, chain_id, user_count, and percentage
    """
    response = supabase.table("debank_user_protocol_interactions").select(
        "protocol_id", "protocol_name", "chain_id", "user_address"
    ).execute()
    
    if response.data:
        df = pd.DataFrame(response.data)
        
        # Count users per protocol
        protocol_counts = df.groupby(['protocol_name', 'chain_id'])['user_address'].nunique().reset_index()
        protocol_counts.columns = ['protocol_name', 'chain_id', 'user_count']
        
        # Get total LoopFi user count
        total_users_response = supabase.table("debank_loopfi_users").select("count", count="exact").execute()
        total_users = total_users_response.count if hasattr(total_users_response, 'count') else 1
        
        # Calculate percentage
        protocol_counts['percentage'] = (protocol_counts['user_count'] / total_users) * 100
        
        return protocol_counts.sort_values('user_count', ascending=False).head(limit)
    
    return pd.DataFrame()


def calculate_utilization_rate(supabase: Client) -> float:
    """
    Calculate the utilization rate (borrowing vs. lending ratio).
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        The utilization rate as a fraction
    """
    # Client-side calculation instead of SQL aggregation
    response = supabase.table("debank_user_loopfi_positions").select(
        "debt_usd_value", "asset_usd_value"
    ).execute()
    
    if response.data:
        debt_sum = sum(item.get('debt_usd_value', 0) for item in response.data)
        asset_sum = sum(item.get('asset_usd_value', 0) for item in response.data)
        if asset_sum > 0:
            return debt_sum / asset_sum
    
    return 0.0


def calculate_average_net_worth(supabase: Client) -> float:
    """
    Calculate the average net worth of LoopFi users.
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        The average net worth in USD
    """
    response = supabase.table("debank_loopfi_users").select("avg(total_net_worth_usd)").execute()
    
    if response.data and response.data[0] and 'avg' in response.data[0]:
        return response.data[0]['avg']
    
    return 0.0


def calculate_whale_distribution(supabase: Client, whale_threshold: float = 100000) -> Dict[str, Any]:
    """
    Calculate the distribution of TVL between whales and smaller users.
    
    Args:
        supabase: A Supabase client instance
        whale_threshold: USD threshold to consider a user a whale
        
    Returns:
        Dictionary with whale_tvl, small_user_tvl, whale_count, small_user_count, and percentages
    """
    response = supabase.table("debank_loopfi_users").select(
        "user_address", "loopfi_usd_value"
    ).execute()
    
    if not response.data:
        return {
            "whale_tvl": 0,
            "small_user_tvl": 0,
            "whale_count": 0,
            "small_user_count": 0,
            "whale_tvl_percentage": 0,
            "small_user_tvl_percentage": 0
        }
    
    df = pd.DataFrame(response.data)
    
    # Split into whales and small users
    whales = df[df['loopfi_usd_value'] >= whale_threshold]
    small_users = df[df['loopfi_usd_value'] < whale_threshold]
    
    # Calculate TVL sums
    whale_tvl = whales['loopfi_usd_value'].sum()
    small_user_tvl = small_users['loopfi_usd_value'].sum()
    total_tvl = whale_tvl + small_user_tvl
    
    # Calculate percentages
    whale_tvl_percentage = (whale_tvl / total_tvl * 100) if total_tvl > 0 else 0
    small_user_tvl_percentage = (small_user_tvl / total_tvl * 100) if total_tvl > 0 else 0
    
    return {
        "whale_tvl": whale_tvl,
        "small_user_tvl": small_user_tvl,
        "whale_count": len(whales),
        "small_user_count": len(small_users),
        "whale_tvl_percentage": whale_tvl_percentage,
        "small_user_tvl_percentage": small_user_tvl_percentage
    }


def get_assets_by_role(supabase: Client, role: str = 'both') -> pd.DataFrame:
    """
    Get analysis of assets based on their role (lending or looping/debt).
    
    Args:
        supabase: A Supabase client instance
        role: Which role to analyze - 'lending', 'looping', or 'both'
        
    Returns:
        DataFrame with asset analysis by their role
    """
    # Fetch positions data
    response = supabase.table("debank_user_loopfi_positions").select(
        "user_address", "asset_symbol", "debt_symbol", "supplied_tokens", "debt_tokens", 
        "asset_usd_value", "debt_usd_value", "chain_id"
    ).execute()
    
    if not response.data:
        return pd.DataFrame()
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(response.data)
    
    # Debug: Print counts of assets
    asset_count = df['asset_symbol'].notna().sum()
    debt_count = df['debt_symbol'].notna().sum()
    print(f"Asset symbols: {asset_count}, Debt symbols: {debt_count} out of {len(df)} positions")
    
    # Handle missing asset_symbol or debt_symbol by extracting from JSON
    for index, row in df.iterrows():
        if pd.isna(row['asset_symbol']) and row['supplied_tokens']:
            try:
                supplied = json.loads(row['supplied_tokens']) if isinstance(row['supplied_tokens'], str) else row['supplied_tokens']
                if supplied and len(supplied) > 0:
                    df.at[index, 'asset_symbol'] = supplied[0].get('symbol')
                    if pd.isna(row['chain_id']) and supplied[0].get('chain'):
                        df.at[index, 'chain_id'] = supplied[0].get('chain')
            except Exception as e:
                print(f"Error extracting asset symbol: {e}")
                
        if pd.isna(row['debt_symbol']) and row['debt_tokens']:
            try:
                # Handle both string and already-parsed JSON
                debt = json.loads(row['debt_tokens']) if isinstance(row['debt_tokens'], str) else row['debt_tokens']
                if debt and len(debt) > 0:
                    df.at[index, 'debt_symbol'] = debt[0].get('symbol')
                    if pd.isna(row['chain_id']) and debt[0].get('chain'):
                        df.at[index, 'chain_id'] = debt[0].get('chain')
            except Exception as e:
                print(f"Error extracting debt symbol: {e}")
    
    # Count assets after extraction
    asset_count_after = df['asset_symbol'].notna().sum()
    debt_count_after = df['debt_symbol'].notna().sum()
    print(f"After extraction - Asset symbols: {asset_count_after}, Debt symbols: {debt_count_after}")
    
    # Create the analysis based on the role
    lending_analysis = pd.DataFrame()
    looping_analysis = pd.DataFrame()
    
    if role == 'lending' or role == 'both':
        lending_df = df.dropna(subset=['asset_symbol'])
        if len(lending_df) > 0:
            lending_analysis = lending_df.groupby(['asset_symbol', 'chain_id']).agg({
                'asset_usd_value': 'sum',
                'user_address': 'nunique'
            }).reset_index()
            lending_analysis.columns = ['asset_symbol', 'chain_id', 'total_usd_value', 'user_count']
            lending_analysis = lending_analysis.sort_values('total_usd_value', ascending=False)
            lending_analysis['role'] = 'lending'
        
    if role == 'looping' or role == 'both':
        # For looping assets, check for both debt_symbol and debt_usd_value
        looping_df = df.dropna(subset=['debt_symbol'])
        looping_df = looping_df[looping_df['debt_usd_value'] > 0]
        
        if len(looping_df) > 0:
            looping_analysis = looping_df.groupby(['debt_symbol', 'chain_id']).agg({
                'debt_usd_value': 'sum',
                'user_address': 'nunique'
            }).reset_index()
            looping_analysis.columns = ['asset_symbol', 'chain_id', 'total_usd_value', 'user_count']
            looping_analysis = looping_analysis.sort_values('total_usd_value', ascending=False)
            looping_analysis['role'] = 'looping'
    
    # Return based on requested role
    if role == 'lending':
        return lending_analysis
    elif role == 'looping':
        return looping_analysis
    else:
        # Combine both analyses
        if len(lending_analysis) > 0 and len(looping_analysis) > 0:
            return pd.concat([lending_analysis, looping_analysis], ignore_index=True)
        elif len(lending_analysis) > 0:
            return lending_analysis
        elif len(looping_analysis) > 0:
            return looping_analysis
        else:
            return pd.DataFrame()


def get_position_duration(supabase: Client) -> Tuple[pd.DataFrame, float, bool]:
    """
    Get the distribution of position durations.
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        Tuple containing (DataFrame with duration distribution, average duration in days, has_real_data flag)
    """
    response = supabase.table("debank_user_loopfi_positions").select(
        "entry_timestamp", "last_updated_timestamp"
    ).execute()
    
    if not response.data or len(response.data) == 0:
        # Return placeholder data as fallback
        placeholder_data = {
            'duration_range': ['0-7 days', '7-30 days', '30-90 days', '90+ days'],
            'count': [0, 0, 0, 0]
        }
        return pd.DataFrame(placeholder_data), 0, False
    
    df = pd.DataFrame(response.data)
    
    # Check if we have identical timestamps for all entries (data migration artifact)
    entry_timestamps = pd.to_datetime(df['entry_timestamp'], errors='coerce')
    update_timestamps = pd.to_datetime(df['last_updated_timestamp'], errors='coerce')
    
    # If all timestamps are within 1 hour of each other, it's likely migration data
    unique_entries = entry_timestamps.nunique()
    unique_updates = update_timestamps.nunique()
    
    # Debug info
    print(f"Position timestamps - Unique entry timestamps: {unique_entries}, Unique update timestamps: {unique_updates}")
    
    # Check if the timestamps represent real duration data or migration artifacts
    has_real_data = False
    
    # If we have very few unique timestamps or they're all clustered together, likely not real data
    if unique_entries <= 3 or unique_updates <= 3:
        print("Position duration data appears to be migration artifacts rather than real position durations")
        # Generate a synthetic distribution as a placeholder
        # For display purposes, create a distribution that shows most positions are new
        duration_counts = pd.DataFrame({
            'duration_range': ['0-7 days', '7-30 days', '30-90 days', '90+ days'],
            'count': [len(df), 0, 0, 0]  # All positions classified as "new"
        })
        return duration_counts, 0, False
    
    # If we're here, we have real timestamp data
    # Convert timestamps to datetime objects
    df['entry_timestamp'] = entry_timestamps
    df['last_updated_timestamp'] = update_timestamps
    
    # Remove rows with invalid timestamps
    df = df.dropna(subset=['entry_timestamp', 'last_updated_timestamp'])
    
    if len(df) == 0:
        # If no valid data after cleaning, return placeholder
        placeholder_data = {
            'duration_range': ['0-7 days', '7-30 days', '30-90 days', '90+ days'],
            'count': [0, 0, 0, 0]
        }
        return pd.DataFrame(placeholder_data), 0, False
    
    # Force entry_timestamp to be before or equal to last_updated_timestamp
    df = df[df['entry_timestamp'] <= df['last_updated_timestamp']]
    
    # Calculate duration in days
    df['duration_days'] = (df['last_updated_timestamp'] - df['entry_timestamp']).dt.total_seconds() / (60*60*24)
    
    # Check for negative durations (shouldn't happen after the above filter, but just in case)
    df = df[df['duration_days'] >= 0]
    
    # Create duration bins
    bins = [0, 7, 30, 90, float('inf')]
    labels = ['0-7 days', '7-30 days', '30-90 days', '90+ days']
    df['duration_range'] = pd.cut(df['duration_days'], bins=bins, labels=labels)
    
    # Get counts by duration range and ensure all bins are represented
    all_ranges = pd.Series(labels, name='duration_range')
    counts = df['duration_range'].value_counts()
    duration_counts = pd.DataFrame({
        'duration_range': all_ranges,
        'count': [counts.get(r, 0) for r in labels]
    })
    
    # Calculate average duration (limit to reasonable values, e.g., max 1 year)
    reasonable_df = df[df['duration_days'] <= 365]
    avg_duration = reasonable_df['duration_days'].mean() if len(reasonable_df) > 0 else 0
    
    return duration_counts, avg_duration, True


def get_portfolio_strategy_analysis(supabase: Client) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Analyzes positions by portfolio strategy type (Staked, Yield, Leveraged Farming).
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        Tuple containing (DataFrame with strategy distribution, Dict with strategy-specific metrics)
    """
    response = supabase.table("debank_user_loopfi_positions").select(
        "portfolio_item_name", "asset_usd_value", "debt_usd_value", "user_address", "chain_id"
    ).execute()
    
    if not response.data or len(response.data) == 0:
        return pd.DataFrame(), {}
    
    df = pd.DataFrame(response.data)
    
    # Ensure required columns exist
    for col in ['portfolio_item_name', 'asset_usd_value', 'debt_usd_value']:
        if col not in df.columns:
            df[col] = 0
    
    # Fill missing values
    df['asset_usd_value'] = df['asset_usd_value'].fillna(0)
    df['debt_usd_value'] = df['debt_usd_value'].fillna(0)
    
    # Get strategy counts
    strategy_counts = df['portfolio_item_name'].value_counts().reset_index()
    strategy_counts.columns = ['strategy', 'count']
    
    # If empty, return placeholder
    if strategy_counts.empty:
        return pd.DataFrame({'strategy': [], 'count': []}), {}
    
    # Calculate strategy-specific metrics
    metrics = {}
    
    # Group by strategy
    grouped = df.groupby('portfolio_item_name')
    
    # Compute metrics for each strategy
    for strategy, group in grouped:
        metrics[strategy] = {
            'avg_size': group['asset_usd_value'].mean(),
            'total_value': group['asset_usd_value'].sum(),
            'user_count': group['user_address'].nunique(),
            'avg_leverage': group['debt_usd_value'].sum() / max(group['asset_usd_value'].sum(), 1),
            'chain_distribution': group['chain_id'].value_counts().to_dict()
        }
    
    # Calculate percentage for each strategy
    total_positions = strategy_counts['count'].sum()
    strategy_counts['percentage'] = strategy_counts['count'] / total_positions * 100
    
    return strategy_counts, metrics


def compare_chains_metrics(supabase: Client) -> pd.DataFrame:
    """
    Generate a comprehensive side-by-side comparison of metrics between chains.
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        DataFrame containing comparative metrics for each chain
    """
    # Debug: Check DB contents directly
    positions_response = supabase.table("debank_user_loopfi_positions").select(
        "chain_id", "user_address", "asset_usd_value", "debt_usd_value"
    ).execute()
    
    print("\n=========== RAW POSITION DATA ANALYSIS ===========")
    if positions_response.data and len(positions_response.data) > 0:
        df_debug = pd.DataFrame(positions_response.data)
        # Count unique chains
        unique_chains = df_debug['chain_id'].unique()
        print(f"Unique chain IDs in positions table: {unique_chains}")
        
        # Create a frequency table for chain_id values
        chain_counts = df_debug['chain_id'].value_counts()
        print("\nChain ID Counts:")
        for chain, count in chain_counts.items():
            print(f"  {chain}: {count} positions")
        
        # Check total value distribution by chain
        chain_tvl = df_debug.groupby('chain_id')['asset_usd_value'].sum()
        print("\nTVL by chain:")
        for chain, tvl in chain_tvl.items():
            print(f"  {chain}: ${tvl:,.2f}")
        
        # Check case variations in chain IDs
        lowercase_chains = df_debug['chain_id'].str.lower().unique()
        print(f"\nUnique chain IDs after lowercase conversion: {lowercase_chains}")
        
        # Check for duplicate positions (same user, same values, different chain)
        duplicates = df_debug[df_debug.duplicated(['user_address', 'asset_usd_value', 'debt_usd_value'], keep=False)]
        print(f"\nFound {len(duplicates)} potentially duplicated positions")
        if len(duplicates) > 0:
            print("Sample of duplicates:")
            print(duplicates.head(5))
    
    print("=====================================================\n")
    
    # Continue with the existing implementation
    # Fetch all the data we need for comparison
    chain_metrics = {}
    all_chains = []
    
    # 1. Get unique users per chain
    response = supabase.table("debank_user_loopfi_positions").select(
        "chain_id", "user_address"
    ).execute()
    
    if response.data:
        user_df = pd.DataFrame(response.data)
        
        # Debug: Check original user distribution 
        print("\n========== USER DISTRIBUTION BEFORE NORMALIZATION ==========")
        original_chain_users = user_df.groupby('chain_id')['user_address'].nunique()
        for chain, count in original_chain_users.items():
            print(f"  {chain}: {count} users")
        print("============================================================\n")
        
        # Normalize chain IDs to lowercase for consistent comparison
        user_df['chain_id'] = user_df['chain_id'].str.lower()
        
        # Count unique users per chain
        chain_users = user_df.groupby('chain_id')['user_address'].nunique()
        
        # Debug: Check user distribution after normalization
        print("\n========== USER DISTRIBUTION AFTER NORMALIZATION ==========")
        for chain, count in chain_users.items():
            print(f"  {chain}: {count} users")
        print("===========================================================\n")
        
        for chain, count in chain_users.items():
            if chain not in chain_metrics:
                chain_metrics[chain] = {}
                all_chains.append(chain)
            chain_metrics[chain]['user_count'] = count
    
    # 2. Get TVL per chain
    response = supabase.table("debank_user_loopfi_positions").select(
        "chain_id", "asset_usd_value"
    ).execute()
    
    if response.data:
        tvl_df = pd.DataFrame(response.data)
        # Normalize chain IDs to lowercase
        tvl_df['chain_id'] = tvl_df['chain_id'].str.lower()
        
        # Calculate TVL per chain
        chain_tvl = tvl_df.groupby('chain_id')['asset_usd_value'].sum()
        
        # Debug: Show TVL by chain
        print("\n========== TVL BY CHAIN AFTER NORMALIZATION ==========")
        for chain, tvl in chain_tvl.items():
            print(f"  {chain}: ${tvl:,.2f}")
        print("=======================================================\n")
        
        for chain, tvl in chain_tvl.items():
            if chain not in chain_metrics:
                chain_metrics[chain] = {}
                all_chains.append(chain)
            chain_metrics[chain]['tvl'] = tvl
    
    # 3. Get average deposit size per chain
    for chain in all_chains:
        if 'tvl' in chain_metrics[chain] and 'user_count' in chain_metrics[chain]:
            chain_metrics[chain]['avg_deposit'] = chain_metrics[chain]['tvl'] / chain_metrics[chain]['user_count']
        else:
            chain_metrics[chain]['avg_deposit'] = 0
    
    # 4. Get utilization rate per chain
    response = supabase.table("debank_user_loopfi_positions").select(
        "chain_id", "asset_usd_value", "debt_usd_value"
    ).execute()
    
    if response.data:
        util_df = pd.DataFrame(response.data)
        # Normalize chain IDs to lowercase
        util_df['chain_id'] = util_df['chain_id'].str.lower()
        
        # Group by normalized chain ID
        chain_groups = util_df.groupby('chain_id')
        
        for chain, group in chain_groups:
            if chain not in chain_metrics:
                chain_metrics[chain] = {}
                all_chains.append(chain)
            
            asset_sum = group['asset_usd_value'].sum()
            debt_sum = group['debt_usd_value'].sum()
            
            util_rate = debt_sum / asset_sum if asset_sum > 0 else 0
            chain_metrics[chain]['utilization_rate'] = util_rate
    
    # 5. Get portfolio strategy distribution per chain
    response = supabase.table("debank_user_loopfi_positions").select(
        "chain_id", "portfolio_item_name"
    ).execute()
    
    if response.data:
        strategy_df = pd.DataFrame(response.data)
        # Normalize chain IDs to lowercase
        strategy_df['chain_id'] = strategy_df['chain_id'].str.lower()
        
        # For each chain, count strategies
        for chain in all_chains:
            chain_strategies = strategy_df[strategy_df['chain_id'] == chain]['portfolio_item_name'].value_counts()
            
            # Calculate strategy percentages
            total_positions = chain_strategies.sum()
            if total_positions > 0:
                for strategy, count in chain_strategies.items():
                    chain_metrics[chain][f'strategy_{strategy}'] = count / total_positions
    
    # 6. Get popular assets per chain
    response = supabase.table("debank_user_loopfi_positions").select(
        "chain_id", "asset_symbol", "asset_usd_value"
    ).execute()
    
    if response.data:
        asset_df = pd.DataFrame(response.data)
        # Normalize chain IDs to lowercase
        asset_df['chain_id'] = asset_df['chain_id'].str.lower()
        
        for chain in all_chains:
            chain_assets = asset_df[asset_df['chain_id'] == chain]
            if not chain_assets.empty:
                # Get top asset by value
                top_assets = chain_assets.groupby('asset_symbol')['asset_usd_value'].sum().sort_values(ascending=False)
                if not top_assets.empty:
                    top_asset = top_assets.index[0]
                    chain_metrics[chain]['top_asset'] = top_asset
    
    # Deduplicate the all_chains list to handle any duplicates from different sections
    all_chains = list(set(all_chains))
    
    # Convert to DataFrame for easy comparison
    comparison_df = pd.DataFrame(chain_metrics).T
    comparison_df.index.name = 'chain_id'
    
    # Make sure we have all the same columns for all chains
    all_columns = set()
    for chain in comparison_df.index:
        all_columns.update(comparison_df.loc[chain].keys())
    
    # Fill in missing values
    for col in all_columns:
        if col not in comparison_df.columns:
            comparison_df[col] = np.nan
    
    # Restore original chain ID casing for display purposes
    # Map lowercase chain IDs back to their display case (e.g., 'eth' -> 'ETH')
    chain_display_names = {'eth': 'ETH', 'bsc': 'BSC', 'arbitrum': 'Arbitrum', 'optimism': 'Optimism'}
    comparison_df = comparison_df.rename(index=chain_display_names)
    
    # Add debug logging
    print(f"Chain comparison metrics calculated for chains: {list(comparison_df.index)}")
    for chain in comparison_df.index:
        print(f"Chain {chain} - Users: {comparison_df.loc[chain].get('user_count', 0)}, TVL: {comparison_df.loc[chain].get('tvl', 0)}")
    
    return comparison_df


def analyze_user_behavior(supabase: Client) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze user behavior patterns to identify different user segments.
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        Tuple containing (DataFrame with user classifications, Dict with segment insights)
    """
    # 1. Fetch user positions data
    position_response = supabase.table("debank_user_loopfi_positions").select(
        "user_address", "asset_usd_value", "debt_usd_value", "chain_id", "portfolio_item_name"
    ).execute()
    
    # 2. Fetch user protocol interactions 
    protocol_response = supabase.table("debank_user_protocol_interactions").select(
        "user_address", "protocol_id"
    ).execute()
    
    # 3. Fetch user token holdings
    token_response = supabase.table("debank_user_token_holdings").select(
        "user_address", "token_symbol", "usd_value"
    ).execute()
    
    # 4. Fetch user net worth
    user_response = supabase.table("debank_loopfi_users").select(
        "user_address", "loopfi_usd_value", "total_net_worth_usd", "chain_id"
    ).execute()
    
    # Initialize dictionaries to store user behavior metrics
    user_metrics = {}
    
    # Process position data
    if position_response.data:
        positions_df = pd.DataFrame(position_response.data)
        
        # Count positions per user
        position_counts = positions_df.groupby('user_address').size()
        
        # Calculate avg position size per user
        position_sizes = positions_df.groupby('user_address')['asset_usd_value'].mean()
        
        # Count unique chains per user
        chain_counts = positions_df.groupby('user_address')['chain_id'].nunique()
        
        # Calculate leverage ratio (debt/asset) per user
        positions_df['leverage_ratio'] = positions_df['debt_usd_value'] / positions_df['asset_usd_value'].replace(0, np.nan)
        leverage_ratios = positions_df.groupby('user_address')['leverage_ratio'].mean()
        
        # Count different strategy types per user
        strategy_counts = positions_df.groupby('user_address')['portfolio_item_name'].nunique()
        
        # Store metrics for each user
        for user in position_counts.index:
            if user not in user_metrics:
                user_metrics[user] = {}
            
            user_metrics[user]['position_count'] = position_counts.get(user, 0)
            user_metrics[user]['avg_position_size'] = position_sizes.get(user, 0)
            user_metrics[user]['chain_count'] = chain_counts.get(user, 0)
            user_metrics[user]['avg_leverage'] = leverage_ratios.get(user, 0)
            user_metrics[user]['strategy_count'] = strategy_counts.get(user, 0)
    
    # Process protocol interaction data
    if protocol_response.data:
        protocols_df = pd.DataFrame(protocol_response.data)
        
        # Count unique protocols per user
        protocol_counts = protocols_df.groupby('user_address')['protocol_id'].nunique()
        
        for user in protocol_counts.index:
            if user not in user_metrics:
                user_metrics[user] = {}
            
            user_metrics[user]['protocol_count'] = protocol_counts.get(user, 0)
    
    # Process token holding data
    if token_response.data:
        tokens_df = pd.DataFrame(token_response.data)
        
        # Count unique tokens per user
        token_counts = tokens_df.groupby('user_address')['token_symbol'].nunique()
        
        # Calculate portfolio diversity
        portfolio_diversity = {}
        for user in token_counts.index:
            user_tokens = tokens_df[tokens_df['user_address'] == user]
            total_value = user_tokens['usd_value'].sum()
            if total_value > 0:
                # Count tokens that make up at least 5% of portfolio
                significant = (user_tokens['usd_value'] / total_value) >= 0.05
                portfolio_diversity[user] = significant.sum()
            else:
                portfolio_diversity[user] = 0
        
        for user in token_counts.index:
            if user not in user_metrics:
                user_metrics[user] = {}
            
            user_metrics[user]['token_count'] = token_counts.get(user, 0)
            user_metrics[user]['portfolio_diversity'] = portfolio_diversity.get(user, 0)
    
    # Process user net worth data
    if user_response.data:
        users_df = pd.DataFrame(user_response.data)
        
        # Get TVL and net worth for each user
        user_tvl = users_df.groupby('user_address')['loopfi_usd_value'].sum()
        user_networth = users_df.groupby('user_address')['total_net_worth_usd'].first()
        
        # Calculate LoopFi engagement ratio
        engagement_ratios = {}
        for user in user_tvl.index:
            net_worth = user_networth.get(user, 0)
            tvl = user_tvl.get(user, 0)
            engagement_ratios[user] = tvl / net_worth if net_worth > 0 else 0
        
        for user in user_tvl.index:
            if user not in user_metrics:
                user_metrics[user] = {}
            
            user_metrics[user]['tvl'] = user_tvl.get(user, 0)
            user_metrics[user]['net_worth'] = user_networth.get(user, 0)
            user_metrics[user]['engagement_ratio'] = engagement_ratios.get(user, 0)
    
    # Convert user metrics to DataFrame
    user_df = pd.DataFrame.from_dict(user_metrics, orient='index')
    
    # Fill NaN values
    user_df = user_df.fillna(0)
    
    # Calculate user segments based on activity and value
    segments = {}
    
    # Power Users: Multiple positions, multiple chains, high engagement
    power_user_criteria = (
        (user_df['position_count'] >= 3) & 
        ((user_df['chain_count'] >= 2) | (user_df['engagement_ratio'] >= 0.3))
    )
    segments['Power Users'] = user_df[power_user_criteria].index.tolist()
    
    # Whales: High TVL
    whale_threshold = user_df['tvl'].quantile(0.9) if len(user_df) > 10 else 100000
    whale_criteria = user_df['tvl'] >= whale_threshold
    segments['Whales'] = user_df[whale_criteria].index.tolist()
    
    # Multi-Chain Users: Active on multiple chains
    multichain_criteria = user_df['chain_count'] >= 2
    segments['Multi-Chain Users'] = user_df[multichain_criteria].index.tolist()
    
    # Casual Users: Low position count, low TVL
    casual_criteria = (
        (user_df['position_count'] <= 2) & 
        (user_df['tvl'] < (whale_threshold * 0.1))
    )
    segments['Casual Users'] = user_df[casual_criteria].index.tolist()
    
    # Experimental Users: High portfolio diversity
    experimental_criteria = (
        (user_df['portfolio_diversity'] >= 5) | 
        (user_df['strategy_count'] >= 2)
    )
    segments['Experimental Users'] = user_df[experimental_criteria].index.tolist()
    
    # Risk-Takers: High leverage ratio
    risk_criteria = user_df['avg_leverage'] >= 0.7
    segments['Risk-Takers'] = user_df[risk_criteria].index.tolist()
    
    # Add segment classification to DataFrame with priority
    user_df['user_segment'] = 'Other'
    
    segment_priority = [
        'Power Users', 
        'Whales', 
        'Risk-Takers',
        'Experimental Users',
        'Multi-Chain Users',
        'Casual Users'
    ]
    
    for segment in segment_priority:
        if segment in segments:
            user_df.loc[user_df.index.isin(segments[segment]), 'user_segment'] = segment
    
    # Calculate segment insights
    segment_insights = {}
    
    segment_counts = user_df['user_segment'].value_counts()
    for segment, count in segment_counts.items():
        segment_users = user_df[user_df['user_segment'] == segment]
        segment_insights[segment] = {
            'count': count,
            'total_tvl': segment_users['tvl'].sum(),
            'avg_positions': segment_users['position_count'].mean(),
            'avg_tvl': segment_users['tvl'].mean(),
            'avg_net_worth': segment_users['net_worth'].mean(),
            'pct_of_users': (count / len(user_df) * 100) if len(user_df) > 0 else 0
        }
    
    return user_df, segment_insights


def check_chain_data_quality(supabase: Client) -> Dict[str, Any]:
    """
    Check the quality of chain data for potential duplication issues.
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        Dictionary with data quality metrics and flags
    """
    # Fetch position data to analyze
    response = supabase.table("debank_user_loopfi_positions").select(
        "chain_id", "user_address", "asset_usd_value", "debt_usd_value"
    ).execute()
    
    if not response.data or len(response.data) == 0:
        return {"has_duplication_issues": False}
    
    df = pd.DataFrame(response.data)
    
    # Normalize chain IDs
    df['chain_id'] = df['chain_id'].str.lower()
    
    # Check for unique chains
    unique_chains = df['chain_id'].unique()
    if len(unique_chains) <= 1:
        return {"has_duplication_issues": False}
    
    # Calculate user counts per chain
    user_counts = df.groupby('chain_id')['user_address'].nunique()
    
    # Calculate TVL per chain
    tvl_by_chain = df.groupby('chain_id')['asset_usd_value'].sum()
    
    # Check for duplicated positions (same user, same asset value, same debt value)
    duplicates = df[df.duplicated(['user_address', 'asset_usd_value', 'debt_usd_value'], keep=False)]
    duplicate_count = len(duplicates) // 2  # Divide by 2 because each duplicate appears twice
    
    # Calculate similarity metrics
    # If user counts are nearly identical across chains, that's suspicious
    max_user_similarity = 0
    if len(user_counts) >= 2:
        for i, chain1 in enumerate(user_counts.index):
            for chain2 in user_counts.index[i+1:]:
                similarity = min(user_counts[chain1], user_counts[chain2]) / max(user_counts[chain1], user_counts[chain2]) * 100
                max_user_similarity = max(max_user_similarity, similarity)
    
    # If TVL is nearly identical across chains, that's also suspicious
    max_tvl_similarity = 0
    if len(tvl_by_chain) >= 2:
        for i, chain1 in enumerate(tvl_by_chain.index):
            for chain2 in tvl_by_chain.index[i+1:]:
                similarity = min(tvl_by_chain[chain1], tvl_by_chain[chain2]) / max(tvl_by_chain[chain1], tvl_by_chain[chain2]) * 100
                max_tvl_similarity = max(max_tvl_similarity, similarity)
    
    # Determine if we have a data duplication issue
    # High thresholds indicate potential duplication
    has_duplication_issues = (
        max_user_similarity > 90 or 
        max_tvl_similarity > 90 or 
        duplicate_count > 10
    )
    
    return {
        "has_duplication_issues": has_duplication_issues,
        "max_user_similarity": max_user_similarity,
        "max_tvl_similarity": max_tvl_similarity,
        "duplicate_count": duplicate_count,
        "user_counts": user_counts.to_dict(),
        "tvl_by_chain": tvl_by_chain.to_dict()
    } 