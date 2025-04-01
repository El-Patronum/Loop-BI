"""
Historical Data Module for Loop-BI

This module handles storing and retrieving historical data snapshots,
enabling time-series analysis of key metrics.
"""

import logging
import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from core.supabase_client import get_supabase_client
from core.analytics import (
    calculate_utilization_rate,
    calculate_average_deposit_size,
    calculate_loop_factors,
    get_assets_by_role,
    calculate_whale_distribution
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('historical_data')

class HistoricalDataStorage:
    """Handles storing historical data snapshots in Supabase."""
    
    def __init__(self, verbose=True):
        """
        Initialize the historical data storage handler.
        
        Args:
            verbose: Whether to show progress bars and detailed output
        """
        self.supabase = get_supabase_client()
        self.today = date.today().isoformat()
        self.verbose = verbose
        
    def store_protocol_metrics(self, chain_id: str = "all"):
        """
        Store protocol-level metrics for today's snapshot.
        
        Args:
            chain_id: Chain ID to snapshot, or "all" for all chains
        """
        try:
            # Get chains to process
            chains = [chain_id] if chain_id != "all" else ["eth", "bsc"]
            
            with tqdm(total=len(chains), desc="Protocol metrics", disable=not self.verbose) as pbar:
                for chain in chains:
                    logger.info(f"Storing protocol metrics snapshot for chain {chain}")
                    
                    # Get protocol info for TVL
                    protocol_response = self.supabase.table("debank_protocols").select(
                        "tvl"
                    ).eq("chain_id", chain).execute()
                    
                    tvl = 0
                    if protocol_response.data:
                        tvl = protocol_response.data[0].get('tvl', 0)
                    
                    # Get user count
                    user_response = self.supabase.table("debank_loopfi_users").select(
                        "count", "chain_id"
                    ).eq("chain_id", chain).execute()
                    
                    total_users = len(user_response.data) if user_response.data else 0
                    
                    # Get new users in the last 24 hours
                    yesterday = (datetime.now() - pd.Timedelta(days=1)).isoformat()
                    new_users_response = self.supabase.table("debank_loopfi_users").select(
                        "count", "chain_id"
                    ).eq("chain_id", chain).gt("first_seen_timestamp", yesterday).execute()
                    
                    new_users_24h = len(new_users_response.data) if new_users_response.data else 0
                    
                    # Calculate utilization rate
                    utilization_rate = calculate_utilization_rate(self.supabase)
                    
                    # Calculate average deposit size
                    avg_deposit_size = calculate_average_deposit_size(self.supabase)
                    
                    # Calculate average loop factor
                    _, avg_loop_factor = calculate_loop_factors(self.supabase)
                    
                    # Store the metrics
                    metrics = {
                        'snapshot_date': self.today,
                        'chain_id': chain,
                        'tvl': tvl,
                        'utilization_rate': utilization_rate,
                        'total_users': total_users,
                        'new_users_24h': new_users_24h,
                        'avg_deposit_size': avg_deposit_size,
                        'avg_loop_factor': avg_loop_factor
                    }
                    
                    # Upsert to avoid duplicates
                    self.supabase.table("historical_protocol_metrics").upsert(
                        [metrics],
                        on_conflict='snapshot_date,chain_id'
                    ).execute()
                    
                    pbar.update(1)
                    pbar.set_postfix({"chain": chain, "tvl": f"${tvl:,.2f}", "users": total_users})
                    
                    logger.info(f"Stored protocol metrics for chain {chain} on {self.today}")
                
        except Exception as e:
            logger.error(f"Failed to store protocol metrics: {str(e)}")
    
    def store_asset_metrics(self, chain_id: str = "all"):
        """
        Store asset-specific metrics for today's snapshot.
        
        Args:
            chain_id: Chain ID to snapshot, or "all" for all chains
        """
        try:
            print("Processing asset metrics...")
            
            # Get asset analysis by role
            combined_assets = get_assets_by_role(self.supabase, role='both')
            
            if combined_assets.empty:
                logger.warning("No asset data available for historical snapshot")
                print("✗ No asset data available")
                return
            
            # Filter by chain if specified
            if chain_id != "all":
                combined_assets = combined_assets[
                    combined_assets['chain_id'].str.lower() == chain_id.lower()
                ]
            
            # Process each asset and store metrics
            asset_metrics = []
            asset_count = {'lending': 0, 'looping': 0}
            
            with tqdm(total=len(combined_assets), desc="Processing assets", disable=not self.verbose) as pbar:
                for _, row in combined_assets.iterrows():
                    asset_symbol = row['asset_symbol']
                    role = row['role']
                    chain = row['chain_id']
                    total_value = row['total_usd_value']
                    user_count = row['user_count']
                    
                    # Calculate average position size
                    avg_position_size = total_value / user_count if user_count > 0 else 0
                    
                    metric = {
                        'snapshot_date': self.today,
                        'chain_id': chain,
                        'asset_symbol': asset_symbol,
                        'role': role,
                        'total_value': total_value,
                        'user_count': user_count,
                        'avg_position_size': avg_position_size
                    }
                    
                    asset_metrics.append(metric)
                    asset_count[role] += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({"asset": asset_symbol, "role": role, "chain": chain})
            
            # Batch store all metrics
            if asset_metrics:
                with tqdm(total=1, desc="Storing asset metrics", disable=not self.verbose) as pbar:
                    self.supabase.table("historical_asset_metrics").upsert(
                        asset_metrics,
                        on_conflict='snapshot_date,chain_id,asset_symbol,role'
                    ).execute()
                    pbar.update(1)
                
                print(f"✓ Stored metrics for {asset_count['lending']} lending assets and {asset_count['looping']} looping assets")
            
        except Exception as e:
            logger.error(f"Failed to store asset metrics: {str(e)}")
            print(f"✗ Error storing asset metrics: {str(e)}")
    
    def store_user_segments(self, chain_id: str = "all"):
        """
        Store user segment metrics for today's snapshot.
        
        Args:
            chain_id: Chain ID to snapshot, or "all" for all chains
        """
        try:
            print("Processing user segments...")
            
            # Get chains to process
            chains = [chain_id] if chain_id != "all" else ["eth", "bsc"]
            
            with tqdm(total=len(chains), desc="User segments", disable=not self.verbose) as pbar:
                for chain in chains:
                    logger.info(f"Storing user segments for chain {chain}")
                    
                    # Get users for this chain
                    users_response = self.supabase.table("debank_loopfi_users").select(
                        "user_address", "loopfi_usd_value", "chain_id"
                    ).eq("chain_id", chain).execute()
                    
                    if not users_response.data:
                        logger.warning(f"No users found for chain {chain}")
                        pbar.update(1)
                        continue
                    
                    # Create DataFrame for analysis
                    df = pd.DataFrame(users_response.data)
                    
                    # Get total TVL for this chain
                    total_tvl = df['loopfi_usd_value'].sum()
                    
                    # Define user segments
                    bins = [0, 1000, 10000, 100000, float('inf')]
                    labels = ['Small (<$1K)', 'Medium ($1K-$10K)', 'Large ($10K-$100K)', 'Whale (>$100K)']
                    df['segment'] = pd.cut(df['loopfi_usd_value'], bins=bins, labels=labels)
                    
                    # Calculate metrics by segment
                    segment_metrics = []
                    
                    for segment in labels:
                        segment_df = df[df['segment'] == segment]
                        segment_count = len(segment_df)
                        segment_tvl = segment_df['loopfi_usd_value'].sum()
                        segment_pct = (segment_tvl / total_tvl * 100) if total_tvl > 0 else 0
                        
                        metric = {
                            'snapshot_date': self.today,
                            'chain_id': chain,
                            'segment': segment,
                            'user_count': segment_count,
                            'total_value': segment_tvl,
                            'percentage_of_tvl': segment_pct
                        }
                        
                        segment_metrics.append(metric)
                    
                    # Store all segment metrics
                    if segment_metrics:
                        self.supabase.table("historical_user_segments").upsert(
                            segment_metrics,
                            on_conflict='snapshot_date,chain_id,segment'
                        ).execute()
                        
                        users_total = len(df)
                        pbar.set_postfix({"chain": chain, "users": users_total, "segments": len(segment_metrics)})
                    
                    pbar.update(1)
                    logger.info(f"Stored {len(segment_metrics)} user segment metrics for chain {chain} on {self.today}")
            
        except Exception as e:
            logger.error(f"Failed to store user segment metrics: {str(e)}")
            print(f"✗ Error storing user segments: {str(e)}")
    
    def store_all_historical_data(self):
        """Store a complete snapshot of all metrics for today."""
        print(f"\n{'='*40}")
        print(f"  HISTORICAL DATA SNAPSHOT - {self.today}")
        print(f"{'='*40}\n")
        
        try:
            print("1. Storing protocol metrics...")
            self.store_protocol_metrics()
            
            print("\n2. Storing asset metrics...")
            self.store_asset_metrics()
            
            print("\n3. Storing user segments...")
            self.store_user_segments()
            
            print(f"\n✅ Historical data snapshot for {self.today} completed successfully!")
            
        except Exception as e:
            logger.error(f"Failed to complete historical data snapshot: {str(e)}")
            print(f"\n❌ Error completing historical data snapshot: {str(e)}")


# Example usage
if __name__ == "__main__":
    historical_storage = HistoricalDataStorage()
    historical_storage.store_all_historical_data() 