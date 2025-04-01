#!/usr/bin/env python

"""
Debug script to analyze chain data in the Supabase database
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
import pathlib

# Add project root to path
ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

from core.supabase_client import get_supabase_client

def main():
    """Main debug function"""
    print("Debugging Chain Data in Supabase")
    print("===============================\n")

    # Load environment variables and connect to Supabase
    load_dotenv()
    supabase = get_supabase_client()

    # Fetch positions data
    print("Fetching positions data...")
    response = supabase.table("debank_user_loopfi_positions").select(
        "chain_id", "user_address", "asset_usd_value", "debt_usd_value", "asset_symbol", "debt_symbol"
    ).execute()

    if not response.data:
        print("No data found in positions table!")
        return

    # Convert to dataframe for analysis
    df = pd.DataFrame(response.data)
    print(f"Found {len(df)} position records")

    # 1. Analyze chain IDs
    print("\n=== Chain ID Analysis ===")
    unique_chains = df['chain_id'].unique()
    print(f"Unique chain IDs: {unique_chains}")

    # Count by chain ID (case sensitive)
    chain_counts = df['chain_id'].value_counts()
    print("\nPosition count by chain ID:")
    for chain, count in chain_counts.items():
        print(f"  {chain}: {count} positions")

    # 2. Check for case variations
    lowercase_chains = df['chain_id'].str.lower().unique()
    print(f"\nUnique chain IDs (lowercase): {lowercase_chains}")

    # 3. Check TVL by chain
    print("\n=== TVL Analysis ===")
    # Group by original chain_id
    tvl_by_chain = df.groupby('chain_id')['asset_usd_value'].sum()
    print("TVL by chain ID (original case):")
    for chain, tvl in tvl_by_chain.items():
        print(f"  {chain}: ${tvl:,.2f}")

    # Group by lowercase chain_id
    df['chain_id_lower'] = df['chain_id'].str.lower()
    tvl_by_chain_lower = df.groupby('chain_id_lower')['asset_usd_value'].sum()
    print("\nTVL by chain ID (lowercase):")
    for chain, tvl in tvl_by_chain_lower.items():
        print(f"  {chain}: ${tvl:,.2f}")

    # 4. Check for duplicate position records
    print("\n=== Duplicate Analysis ===")
    # Look for positions with same user, asset value, and debt value
    dup_columns = ['user_address', 'asset_usd_value', 'debt_usd_value']
    duplicates = df[df.duplicated(dup_columns, keep=False)]
    duplicates = duplicates.sort_values(by=dup_columns)

    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} potentially duplicated position records")
        print("Sample of duplicates:")
        display_cols = ['chain_id', 'user_address', 'asset_usd_value', 'debt_usd_value']
        print(duplicates[display_cols].head(10))

        # Analyze chain distribution in duplicates
        dup_by_chain = duplicates['chain_id'].value_counts()
        print("\nDuplicates by chain:")
        for chain, count in dup_by_chain.items():
            print(f"  {chain}: {count} duplicated positions")

        # Count users with multi-chain duplicates
        user_chain_counts = df.groupby('user_address')['chain_id'].nunique()
        multi_chain_users = user_chain_counts[user_chain_counts > 1]
        print(f"\nUsers with positions on multiple chains: {len(multi_chain_users)}")

        if len(multi_chain_users) > 0:
            print("Sample of users with multi-chain positions:")
            for user in multi_chain_users.index[:5]:
                user_positions = df[df['user_address'] == user]
                print(f"\nUser {user}:")
                for _, position in user_positions.iterrows():
                    asset_symbol = position['asset_symbol'] if pd.notna(position['asset_symbol']) else "Unknown"
                    print(f"  Chain: {position['chain_id']}, Asset: {asset_symbol}, Value: ${position['asset_usd_value']:,.2f}")
    else:
        print("No duplicated position records found.")

    # 5. Check for users that exist on both chains with identical values
    print("\n=== Cross-Chain User Analysis ===")
    user_chains = df.groupby(['user_address', 'chain_id_lower'])['asset_usd_value'].sum().reset_index()
    multi_chain_users = user_chains.groupby('user_address')['chain_id_lower'].nunique()
    users_on_multiple_chains = multi_chain_users[multi_chain_users > 1].index

    print(f"Users on multiple chains: {len(users_on_multiple_chains)}")
    
    if len(users_on_multiple_chains) > 0:
        print("\nAnalyzing the first 5 users with positions on multiple chains:")
        for i, user in enumerate(users_on_multiple_chains[:5]):
            user_data = user_chains[user_chains['user_address'] == user]
            print(f"\nUser {i+1}: {user}")
            for _, row in user_data.iterrows():
                print(f"  Chain: {row['chain_id_lower']}, TVL: ${row['asset_usd_value']:,.2f}")
        
        # Check for exact value duplicates across chains 
        print("\n=== Exact Value Duplicate Analysis ===")
        suspicious_users = []
        
        for user in users_on_multiple_chains:
            user_values = {}
            for chain in lowercase_chains:
                # Get total value for this user on this chain
                chain_data = user_chains[(user_chains['user_address'] == user) & 
                                        (user_chains['chain_id_lower'] == chain)]
                if not chain_data.empty:
                    user_values[chain] = chain_data['asset_usd_value'].values[0]
            
            # Check if values are almost identical across chains
            if len(user_values) > 1:
                values = list(user_values.values())
                max_diff_pct = max([abs(a-b)/max(a,b)*100 for a in values for b in values if a != b], default=0)
                if max_diff_pct < 1.0:  # Less than 1% difference
                    suspicious_users.append((user, user_values, max_diff_pct))
        
        if suspicious_users:
            print(f"Found {len(suspicious_users)} users with nearly identical values across chains:")
            for user, values, diff_pct in suspicious_users[:10]:
                print(f"User {user}: Difference {diff_pct:.2f}%")
                for chain, value in values.items():
                    print(f"  {chain}: ${value:,.2f}")
        else:
            print("No users found with suspiciously identical values across chains.")

    # 6. Check for data duplication between chains
    print("\n=== Aggregate Similarity Analysis ===")
    # Compare total users and TVL across chains
    eth_users = df[df['chain_id_lower'] == 'eth']['user_address'].nunique()
    bsc_users = df[df['chain_id_lower'] == 'bsc']['user_address'].nunique()
    eth_tvl = df[df['chain_id_lower'] == 'eth']['asset_usd_value'].sum()
    bsc_tvl = df[df['chain_id_lower'] == 'bsc']['asset_usd_value'].sum()
    
    print(f"ETH users: {eth_users}, BSC users: {bsc_users}")
    print(f"ETH TVL: ${eth_tvl:,.2f}, BSC TVL: ${bsc_tvl:,.2f}")
    
    user_similarity = min(eth_users, bsc_users) / max(eth_users, bsc_users) * 100
    tvl_similarity = min(eth_tvl, bsc_tvl) / max(eth_tvl, bsc_tvl) * 100
    
    print(f"User count similarity: {user_similarity:.2f}%")
    print(f"TVL similarity: {tvl_similarity:.2f}%")
    
    if user_similarity > 95 and tvl_similarity > 95:
        print("\nWARNING: Chain data appears to be duplicated! Both chains have nearly identical metrics.")
        print("This suggests the data was not correctly differentiated by chain during collection or processing.")
    
    # 7. Check data modification dates
    try:
        print("\n=== Data Freshness Analysis ===")
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            most_recent = df['created_at'].max()
            oldest = df['created_at'].min()
            print(f"Data timestamp range: {oldest} to {most_recent}")
            
            # Check if all the data was created at the same time
            date_range = (most_recent - oldest).total_seconds()
            if date_range < 60:  # All data created within 1 minute
                print("WARNING: All position data was created within a 1-minute window.")
                print("This suggests possible bulk data duplication during migration.")
    except Exception as e:
        print(f"Could not analyze timestamps: {e}")

if __name__ == "__main__":
    main() 