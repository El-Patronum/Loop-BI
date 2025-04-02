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


def calculate_loop_factors(supabase: Client) -> pd.DataFrame:
    """
    Calculate the distribution of loop factors (leverage ratios).
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        DataFrame with loop factor distribution, including all categories, sorted by range.
    """
    # Define structure for empty DataFrame to ensure consistency
    labels = ['0-25%', '25-50%', '50-75%', '75-100%', '100-150%', '150-200%', '200-300%', '300%+']
    empty_df = pd.DataFrame({
        'loop_factor_range': pd.Categorical(labels, categories=labels, ordered=True),
        'count': pd.Series([0]*len(labels), dtype='int')
    }).sort_values('loop_factor_range') # Ensure empty df is sorted too

    try:
        response = supabase.table("debank_user_loopfi_positions").select(
            "asset_usd_value", "debt_usd_value"
        ).execute()
        
        if not response.data:
            return empty_df
            
        df = pd.DataFrame(response.data)
        df['asset_usd_value'] = pd.to_numeric(df['asset_usd_value'], errors='coerce').fillna(0)
        df['debt_usd_value'] = pd.to_numeric(df['debt_usd_value'], errors='coerce').fillna(0)
        
        # Consider only positions with debt (looping positions)
        df = df[df['debt_usd_value'] > 0].copy() # Use .copy() to avoid SettingWithCopyWarning
        if df.empty:
            return empty_df
        
        # Calculate loop factor, handling potential division by zero
        df['loop_factor'] = df['debt_usd_value'] / df['asset_usd_value'].replace(0, np.nan)
        
        # Handle cases where loop_factor calculation resulted in NaN
        df.dropna(subset=['loop_factor'], inplace=True)
        if df.empty:
            return empty_df

        # Create bins for loop factors
        bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, float('inf')]
        # Explicitly use Categorical with all defined labels/categories
        df['loop_factor_range'] = pd.cut(df['loop_factor'], bins=bins, labels=labels, right=False)
        df['loop_factor_range'] = df['loop_factor_range'].astype('category').cat.set_categories(labels, ordered=True)

        # Get counts by loop factor range
        counts = df['loop_factor_range'].value_counts()

        # Reindex with all labels to ensure all categories are present, fill missing with 0
        counts = counts.reindex(labels, fill_value=0)

        # Create the final DataFrame directly from the reindexed Series
        loop_factor_counts_df = pd.DataFrame({
            'loop_factor_range': counts.index, # Index contains the labels
            'count': counts.values
        })

        # Explicitly set the categorical type and order on the column
        loop_factor_counts_df['loop_factor_range'] = pd.Categorical(
            loop_factor_counts_df['loop_factor_range'],
            categories=labels,
            ordered=True
        )

        # Sort by the categorical range for consistent order before returning
        return loop_factor_counts_df.sort_values('loop_factor_range')

    except Exception as e:
        # Log the error (replace with proper logging if available)
        print(f"Error in calculate_loop_factors: {e}")
        # Return the predefined empty DataFrame on any error
        return empty_df


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


def calculate_utilization_rate(supabase: Client) -> Tuple[float, float]:
    """
    Calculate the overall and looping-specific utilization rates.
    Utilization = Total Debt / Total Collateral (Assets)
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        Tuple containing (overall_utilization_rate, looping_specific_utilization_rate)
    """
    # Client-side calculation instead of SQL aggregation
    response = supabase.table("debank_user_loopfi_positions").select(
        "debt_usd_value", "asset_usd_value"
    ).execute()
    
    overall_utilization_rate = 0.0
    looping_specific_utilization_rate = 0.0
    
    if response.data:
        df = pd.DataFrame(response.data)
        df['debt_usd_value'] = df['debt_usd_value'].fillna(0)
        df['asset_usd_value'] = df['asset_usd_value'].fillna(0)

        # Overall rate
        total_debt = df['debt_usd_value'].sum()
        total_asset = df['asset_usd_value'].sum()
        overall_utilization_rate = total_debt / total_asset if total_asset > 0 else 0.0
        
        # Looping-specific rate
        looping_df = df[df['debt_usd_value'] > 0]
        looping_debt = looping_df['debt_usd_value'].sum()
        looping_asset = looping_df['asset_usd_value'].sum()
        looping_specific_utilization_rate = looping_debt / looping_asset if looping_asset > 0 else 0.0
            
    return overall_utilization_rate, looping_specific_utilization_rate


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


def get_assets_by_role(supabase: Client, role: str = 'both', chain_filter=None) -> pd.DataFrame:
    """
    Get analysis of assets based on their role (lending or looping/debt).
    
    Args:
        supabase: A Supabase client instance
        role: Which role to analyze - 'lending', 'looping', or 'both'
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
        
    Returns:
        DataFrame containing asset data for the specified role(s).
        Columns vary based on role.
    """
    # Build query with optional chain filter
    query = supabase.table("debank_user_loopfi_positions").select(
        "asset_symbol", "asset_usd_value", "debt_usd_value", 
        "chain_id", "supplied_tokens", "debt_tokens"
    )
    
    # Apply chain filter if specified
    if chain_filter and chain_filter.lower() != "all chains":
        query = query.eq("chain_id", chain_filter.lower())
    
    response = query.execute()
    
    if not response.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(response.data)
    
    # Ensure required columns exist and are properly formatted
    for col in ['asset_symbol', 'asset_usd_value', 'debt_usd_value', 'chain_id']:
        if col not in df.columns:
            df[col] = 'Unknown' if col == 'asset_symbol' or col == 'chain_id' else 0
        else:
            if col in ['asset_usd_value', 'debt_usd_value']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Filter by role as needed
    if role == 'lending':
        # For lending, get positions with no debt or very small debt (might be dust)
        df = df[df['debt_usd_value'] <= 0.01].copy()
        
        # Group by asset and chain
        grouped = df.groupby(['asset_symbol', 'chain_id'], observed=True).agg(
            total_usd_value=('asset_usd_value', 'sum'),
            position_count=('asset_usd_value', 'count')
        ).reset_index()
        
        # Add role column
        grouped['role'] = 'lending'
        
        return grouped.sort_values('total_usd_value', ascending=False)
    
    elif role == 'looping':
        # For looping, get positions with debt
        df = df[df['debt_usd_value'] > 0.01].copy()
        
        # This is a bit more complex as we need to extract the borrowed asset from debt_tokens
        looping_positions = []
        
        for _, row in df.iterrows():
            # Try to extract the borrowed token
            borrowed_asset = 'Unknown'
            borrowed_amount = 0
            
            try:
                if row.get('debt_tokens') and isinstance(row['debt_tokens'], str):
                    debt_tokens = json.loads(row['debt_tokens'])
                    if isinstance(debt_tokens, list) and len(debt_tokens) > 0:
                        first_token = debt_tokens[0]
                        borrowed_asset = first_token.get('symbol', 'Unknown')
                        borrowed_amount = first_token.get('amount', 0)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            
            # Add to positions list
            looping_positions.append({
                'asset_symbol': row.get('asset_symbol', 'Unknown'),
                'borrowed_asset_symbol': borrowed_asset,
                'chain_id': row.get('chain_id', 'Unknown'),
                'asset_usd_value': row.get('asset_usd_value', 0),
                'debt_usd_value': row.get('debt_usd_value', 0),
                'net_usd_value': row.get('asset_usd_value', 0) - row.get('debt_usd_value', 0)
            })
        
        if not looping_positions:
            return pd.DataFrame()
            
        looping_df = pd.DataFrame(looping_positions)
        
        # Group by collateral asset and chain
        grouped = looping_df.groupby(['asset_symbol', 'chain_id'], observed=True).agg(
            total_usd_value=('asset_usd_value', 'sum'),
            total_debt_value=('debt_usd_value', 'sum'),
            position_count=('asset_symbol', 'count')
        ).reset_index()
        
        # Add role column
        grouped['role'] = 'looping'
        
        return grouped.sort_values('total_usd_value', ascending=False)
        
    else:  # both
        # Get both lending and looping positions
        lending_df = get_assets_by_role(supabase, 'lending', chain_filter)
        looping_df = get_assets_by_role(supabase, 'looping', chain_filter)
        
        # Combine the results
        if lending_df.empty and looping_df.empty:
            return pd.DataFrame()
        elif lending_df.empty:
            return looping_df
        elif looping_df.empty:
            return lending_df
        else:
            # Ensure columns match before concatenating
            common_cols = ['asset_symbol', 'chain_id', 'total_usd_value', 'position_count', 'role']
            return pd.concat([
                lending_df[common_cols], 
                looping_df[common_cols]
            ], ignore_index=True)


def get_position_duration_analytics(supabase: Client, chain_filter=None) -> Tuple[pd.DataFrame, float, bool]:
    """
    Calculate position duration distribution statistics.
    
    Args:
        supabase: A Supabase client instance
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
        
    Returns:
        Tuple containing:
            - DataFrame with duration distribution (bins and counts)
            - Average duration in days
            - Boolean indicating if real data is available
    """
    # Default empty data with standard bins
    bins = ["0-7 days", "7-30 days", "1-3 months", "3-6 months", "6-12 months", ">12 months"]
    duration_data = pd.DataFrame({
        'duration_range': bins,
        'count': [0] * len(bins)
    })
    
    # Check if we have real duration data
    has_real_data = False
    avg_duration = 0
    
    try:
        # Build query with optional chain filter
        query = supabase.table("debank_position_duration").select(
            "position_id", "user_address", "chain_id", "duration_days", "is_active"
        )
        
        # Apply chain filter if specified
        if chain_filter and chain_filter.lower() != "all chains":
            query = query.eq("chain_id", chain_filter.lower())
        
        response = query.execute()
        
        if response.data and len(response.data) > 0:
            df = pd.DataFrame(response.data)
            
            # Check if we have real duration_days data
            if 'duration_days' in df.columns and not df['duration_days'].isnull().all():
                has_real_data = True
                
                # Fill NaNs
                df['duration_days'] = df['duration_days'].fillna(0)
                
                # Calculate average duration
                avg_duration = df['duration_days'].mean()
                
                # Create bins for the chart
                conditions = [
                    (df['duration_days'] < 7),
                    (df['duration_days'] >= 7) & (df['duration_days'] < 30),
                    (df['duration_days'] >= 30) & (df['duration_days'] < 90),
                    (df['duration_days'] >= 90) & (df['duration_days'] < 180),
                    (df['duration_days'] >= 180) & (df['duration_days'] < 365),
                    (df['duration_days'] >= 365)
                ]
                
                # Apply binning
                df['duration_range'] = np.select(conditions, bins, default="Unknown")
                
                # Count positions per duration range
                duration_counts = df['duration_range'].value_counts().reset_index()
                duration_counts.columns = ['duration_range', 'count']
                
                # Ensure all categories are represented in the correct order
                duration_data = pd.DataFrame({'duration_range': bins})
                duration_data = duration_data.merge(duration_counts, on='duration_range', how='left')
                duration_data['count'] = duration_data['count'].fillna(0).astype(int)
                
                # Add categorical type to ensure correct order
                duration_data['duration_range'] = pd.Categorical(
                    duration_data['duration_range'], 
                    categories=bins,
                    ordered=True
                )
                
                # Sort by duration range
                duration_data = duration_data.sort_values('duration_range')
    except Exception as e:
        # Handle the case where the table doesn't exist or other errors
        print(f"Error getting position duration data: {str(e)}")
        # Return default empty data with has_real_data = False
    
    return duration_data, avg_duration, has_real_data


def get_portfolio_strategy_analysis(supabase: Client, chain_filter=None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Analyzes positions by portfolio strategy type (Staked, Yield, Leveraged Farming).
    
    Args:
        supabase: A Supabase client instance
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
        
    Returns:
        Tuple containing (DataFrame with strategy distribution, Dict with strategy-specific metrics)
    """
    # Build query with optional chain filter
    query = supabase.table("debank_user_loopfi_positions").select(
        "portfolio_item_name", "asset_usd_value", "debt_usd_value", "user_address", "chain_id"
    )
    
    # Apply chain filter if specified
    if chain_filter and chain_filter.lower() != "all chains":
        query = query.eq("chain_id", chain_filter.lower())
    
    response = query.execute()
    
    if not response.data or len(response.data) == 0:
        return pd.DataFrame(), {}
    
    df = pd.DataFrame(response.data)
    
    # Ensure required columns exist and fill missing values
    for col in ['portfolio_item_name', 'asset_usd_value', 'debt_usd_value']:
        if col not in df.columns:
            df[col] = 0 # Or appropriate default
        else:
            # Ensure numeric columns are numeric, coercing errors
            if col in ['asset_usd_value', 'debt_usd_value']:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    # Fill NaNs in portfolio_item_name with 'Unknown'
    df['portfolio_item_name'] = df['portfolio_item_name'].fillna('Unknown')
    
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
        # Filter for looping positions within this strategy group
        looping_group = group[group['debt_usd_value'] > 0]
        
        # Calculate looping-specific average leverage
        total_looping_debt = looping_group['debt_usd_value'].sum()
        total_looping_asset = looping_group['asset_usd_value'].sum()
        avg_looping_leverage = total_looping_debt / total_looping_asset if total_looping_asset > 0 else 0
        
        # Calculate looping-specific average net value
        looping_group['net_usd_value'] = looping_group['asset_usd_value'] - looping_group['debt_usd_value']
        avg_looping_net_value = looping_group['net_usd_value'].mean() if not looping_group.empty else 0

        # Calculate overall average leverage for the strategy (includes non-looping)
        total_strategy_debt = group['debt_usd_value'].sum()
        total_strategy_asset = group['asset_usd_value'].sum()
        avg_overall_leverage = total_strategy_debt / total_strategy_asset if total_strategy_asset > 0 else 0

        metrics[strategy] = {
            'avg_size': group['asset_usd_value'].mean(),
            'total_value': total_strategy_asset, # Use pre-calculated sum
            'user_count': group['user_address'].nunique(),
            'avg_overall_leverage': avg_overall_leverage, # Leverage across all positions in strategy
            'avg_looping_leverage': avg_looping_leverage, # Leverage specifically for looping positions in strategy
            'avg_looping_net_value': avg_looping_net_value, # Added avg net value for looping positions
            'chain_distribution': group['chain_id'].value_counts().to_dict()
        }
    
    # Calculate percentage for each strategy
    total_positions = strategy_counts['count'].sum()
    strategy_counts['percentage'] = (strategy_counts['count'] / total_positions * 100).round(2)
    
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
            
            # Calculate looping-specific metrics
            looping_group = group[group['debt_usd_value'] > 0]
            looping_asset_sum = looping_group['asset_usd_value'].sum()
            looping_debt_sum = looping_group['debt_usd_value'].sum()
            
            looping_tvl_percentage = (looping_asset_sum / asset_sum * 100) if asset_sum > 0 else 0
            looping_utilization_rate = looping_debt_sum / looping_asset_sum if looping_asset_sum > 0 else 0
            
            chain_metrics[chain]['looping_tvl_percentage'] = looping_tvl_percentage
            chain_metrics[chain]['looping_utilization_rate'] = looping_utilization_rate
    
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


def analyze_user_behavior(supabase: Client, chain_filter=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyzes user behavior and categorizes users into segments.
    
    Args:
        supabase: A Supabase client instance
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
        
    Returns:
        Tuple containing (DataFrame with user segments, Dict with segment insights)
    """
    # Fetch position data for analysis
    position_query = supabase.table("debank_user_loopfi_positions").select(
        "user_address", "portfolio_item_name", "asset_usd_value", 
        "debt_usd_value", "chain_id"
    )
    
    # Apply chain filter to positions if specified
    if chain_filter and chain_filter.lower() != "all chains":
        position_query = position_query.eq("chain_id", chain_filter.lower())
    
    position_response = position_query.execute()
    
    # Fetch protocol interaction data
    protocol_query = supabase.table("debank_user_protocol_interactions").select(
        "user_address", "protocol_id", "chain_id"
    )
    
    # Apply chain filter to protocol interactions if specified
    if chain_filter and chain_filter.lower() != "all chains":
        protocol_query = protocol_query.eq("chain_id", chain_filter.lower())
    
    protocol_response = protocol_query.execute()
    
    # Fetch user data (including loopfi users)
    # Note: For user data, we'll need to filter after fetching since we don't have chain_id in this table
    user_response = supabase.table("debank_loopfi_users").select(
        "user_address", "loopfi_usd_value", "total_net_worth_usd"
    ).execute()
    
    # Initialize user metrics dictionary
    user_metrics = {}
    
    # Process position data
    if position_response.data:
        positions_df = pd.DataFrame(position_response.data)
        
        # If we have a chain filter, get the list of users who have positions on this chain
        chain_filtered_users = None
        if chain_filter and chain_filter.lower() != "all chains":
            chain_filtered_users = set(positions_df['user_address'].unique())
        
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
    
    # Process user net worth data
    if user_response.data:
        users_df = pd.DataFrame(user_response.data)
        
        # If we have chain filtered users, only keep those users
        if chain_filter and chain_filter.lower() != "all chains" and chain_filtered_users is not None:
            users_df = users_df[users_df['user_address'].isin(chain_filtered_users)]
        
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
    
    # Risk-Takers: High leverage
    risk_criteria = user_df['avg_leverage'] >= 0.7
    segments['Risk-Takers'] = user_df[risk_criteria].index.tolist()
    
    # Experimental Users: Try multiple strategies
    exp_criteria = user_df['strategy_count'] >= 2
    segments['Experimental Users'] = user_df[exp_criteria].index.tolist()
    
    # Casual Users: Small TVL, few positions
    casual_criteria = (
        (user_df['tvl'] < whale_threshold / 10) & 
        (user_df['position_count'] < 3)
    )
    segments['Casual Users'] = user_df[casual_criteria].index.tolist()
    
    # Assign segment to each user (prioritizing more specific segments)
    user_df['user_segment'] = 'Other'
    
    # Order of priority for segment assignment
    segment_priority = [
        'Power Users', 'Whales', 'Risk-Takers', 
        'Experimental Users', 'Multi-Chain Users', 'Casual Users'
    ]
    
    for segment in segment_priority:
        user_df.loc[user_df.index.isin(segments[segment]), 'user_segment'] = segment
    
    # Calculate segment insights
    segment_insights = {}
    
    for segment, users in segments.items():
        if users:
            segment_df = user_df.loc[users]
            
            segment_insights[segment] = {
                'count': len(users),
                'pct_of_users': len(users) / len(user_df) * 100,
                'total_tvl': segment_df['tvl'].sum(),
                'avg_tvl': segment_df['tvl'].mean(),
                'avg_positions': segment_df['position_count'].mean(),
                'avg_net_worth': segment_df['net_worth'].mean(),
                'avg_leverage': segment_df['avg_leverage'].mean(),
                'multi_chain_pct': (segment_df['chain_count'] > 1).mean() * 100
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


def calculate_looping_overview_stats(supabase: Client) -> Dict[str, float]:
    """
    Calculate overview statistics about looping positions.
    
    Args:
        supabase: A Supabase client instance
        
    Returns:
        Dictionary containing looping percentage, avg debt, median debt.
    """
    response = supabase.table("debank_user_loopfi_positions").select(
        "debt_usd_value"
    ).execute()
    
    if not response.data:
        return {
            "looping_percentage": 0,
            "average_looping_debt": 0,
            "median_looping_debt": 0
        }
        
    df = pd.DataFrame(response.data)
    df['debt_usd_value'] = df['debt_usd_value'].fillna(0)
    
    total_positions = len(df)
    if total_positions == 0:
        return {
            "looping_percentage": 0,
            "average_looping_debt": 0,
            "median_looping_debt": 0
        }
        
    looping_positions = df[df['debt_usd_value'] > 0]
    looping_count = len(looping_positions)
    
    looping_percentage = (looping_count / total_positions) * 100 if total_positions > 0 else 0
    
    if looping_count > 0:
        average_looping_debt = looping_positions['debt_usd_value'].mean()
        median_looping_debt = looping_positions['debt_usd_value'].median()
    else:
        average_looping_debt = 0
        median_looping_debt = 0
        
    return {
        "looping_percentage": looping_percentage,
        "average_looping_debt": average_looping_debt,
        "median_looping_debt": median_looping_debt
    }


# Function to analyze leverage by different groupings
def analyze_leverage_by_group(supabase: Client, chain_filter=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates average leverage for looping positions, grouped by collateral asset and by chain.
    
    Args:
        supabase: A Supabase client instance
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
        
    Returns:
        Tuple containing:
            - DataFrame with avg leverage per collateral asset.
            - DataFrame with avg leverage per chain.
    """
    # Build query with optional chain filter
    query = supabase.table("debank_user_loopfi_positions").select(
        "asset_symbol", "chain_id", "asset_usd_value", "debt_usd_value"
    ).gt("debt_usd_value", 0) # Only fetch looping positions
    
    # Apply chain filter if specified
    if chain_filter and chain_filter.lower() != "all chains":
        query = query.eq("chain_id", chain_filter.lower())
    
    response = query.execute()
    
    if not response.data:
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.DataFrame(response.data)
    df['asset_usd_value'] = df['asset_usd_value'].fillna(0)
    df['debt_usd_value'] = df['debt_usd_value'].fillna(0)
    df['asset_symbol'] = df['asset_symbol'].fillna('Unknown')
    df['chain_id'] = df['chain_id'].fillna('Unknown')
    
    # --- Group by Collateral Asset ---
    asset_groups = df.groupby('asset_symbol')
    leverage_by_asset_list = []
    for asset, group in asset_groups:
        total_asset_val = group['asset_usd_value'].sum()
        total_debt_val = group['debt_usd_value'].sum()
        avg_leverage = total_debt_val / total_asset_val if total_asset_val > 0 else 0
        leverage_by_asset_list.append({
            'asset_symbol': asset,
            'avg_leverage': avg_leverage,
            'position_count': len(group),
            'total_collateral_value': total_asset_val,
            'total_debt_value': total_debt_val
        })
    
    leverage_by_asset_df = pd.DataFrame(leverage_by_asset_list)
    leverage_by_asset_df = leverage_by_asset_df.sort_values('avg_leverage', ascending=False)

    # --- Group by Chain ---
    chain_groups = df.groupby('chain_id')
    leverage_by_chain_list = []
    for chain, group in chain_groups:
        total_asset_val = group['asset_usd_value'].sum()
        total_debt_val = group['debt_usd_value'].sum()
        avg_leverage = total_debt_val / total_asset_val if total_asset_val > 0 else 0
        leverage_by_chain_list.append({
            'chain_id': chain,
            'avg_leverage': avg_leverage,
            'position_count': len(group),
            'total_collateral_value': total_asset_val,
            'total_debt_value': total_debt_val
        })
        
    leverage_by_chain_df = pd.DataFrame(leverage_by_chain_list)
    leverage_by_chain_df = leverage_by_chain_df.sort_values('avg_leverage', ascending=False)
    
    return leverage_by_asset_df, leverage_by_chain_df


# Function to analyze performance of looping pairs
def analyze_looping_pair_performance(supabase: Client, chain_filter=None) -> pd.DataFrame:
    """
    Calculates average net USD value for looping positions, grouped by collateral and borrowed asset pairs.
    
    Args:
        supabase: A Supabase client instance
        chain_filter: Optional chain to filter by (e.g., 'ETH', 'BSC')
        
    Returns:
        DataFrame with avg net value per collateral/borrow asset pair.
    """
    # Use the get_assets_by_role function to get the detailed looping data
    # This avoids repeating the JSON parsing logic
    looping_data = get_assets_by_role(supabase, role='looping', chain_filter=chain_filter)
    
    if looping_data.empty or 'borrowed_asset_symbol' not in looping_data.columns:
        return pd.DataFrame()
        
    # Calculate net_usd_value for each position record
    # Note: get_assets_by_role returns aggregated data, we need raw positions
    # Re-fetch raw data for this specific analysis
    query = supabase.table("debank_user_loopfi_positions").select(
        "user_address", "supplied_tokens", "debt_tokens", 
        "asset_usd_value", "debt_usd_value", "chain_id"
    ).gt("debt_usd_value", 0) # Only looping positions
    
    # Apply chain filter if specified
    if chain_filter and chain_filter.lower() != "all chains":
        query = query.eq("chain_id", chain_filter.lower())
    
    response = query.execute()
    
    if not response.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(response.data)

    # --- Re-parse minimal info needed: collateral & borrowed symbols --- 
    parsed_positions = []
    for _, row in df.iterrows():
        try:
            # Try to extract collateral and borrowed symbols
            collateral_symbol = 'Unknown'
            borrowed_symbol = 'Unknown'
            
            # Handle supplied token case - get first token as collateral
            if row.get('supplied_tokens') and isinstance(row['supplied_tokens'], str):
                try:
                    supplied_tokens = json.loads(row['supplied_tokens'])
                    if isinstance(supplied_tokens, list) and len(supplied_tokens) > 0:
                        first_token = supplied_tokens[0]
                        collateral_symbol = first_token.get('symbol', 'Unknown')
                except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                    pass
                    
            # Handle debt token case - get first token as borrowed
            if row.get('debt_tokens') and isinstance(row['debt_tokens'], str):
                try:
                    debt_tokens = json.loads(row['debt_tokens'])
                    if isinstance(debt_tokens, list) and len(debt_tokens) > 0:
                        first_token = debt_tokens[0]
                        borrowed_symbol = first_token.get('symbol', 'Unknown')
                except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                    pass
                
            # Calculate net value    
            asset_value = row.get('asset_usd_value', 0) or 0
            debt_value = row.get('debt_usd_value', 0) or 0
            net_value = asset_value - debt_value
            
            # Add to parsed positions list
            parsed_positions.append({
                'collateral_symbol': collateral_symbol,
                'borrowed_symbol': borrowed_symbol,
                'chain_id': row.get('chain_id', 'Unknown'),
                'net_usd_value': net_value
            })
                
        except Exception as e:
            # Skip positions that can't be parsed
            continue
    
    # Create DataFrame from parsed positions
    if not parsed_positions:
        return pd.DataFrame()
        
    perf_df = pd.DataFrame(parsed_positions)
    
    # --- Group by Pair and Chain ---
    pair_groups = perf_df.groupby(['collateral_symbol', 'borrowed_symbol', 'chain_id'], observed=True).agg(
        avg_net_usd_value = ('net_usd_value', 'mean'),
        position_count = ('net_usd_value', 'count') # Count positions for this pair
    ).reset_index()
    
    pair_performance_df = pair_groups.sort_values('avg_net_usd_value', ascending=False)
    
    return pair_performance_df 