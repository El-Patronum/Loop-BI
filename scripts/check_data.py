#!/usr/bin/env python
"""
Script to check the processed data in Supabase

This script verifies that data has been properly processed and stored in Supabase.
"""
import os
import logging
from tabulate import tabulate
from core.supabase_client import get_supabase_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_check')

def check_tables():
    """Check the data in all tables"""
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Check protocol info
        logger.info("Checking protocol info...")
        protocol_info = supabase.table("debank_protocols").select("*").execute()
        if protocol_info.data:
            protocol = protocol_info.data[0]
            print(f"\nProtocol: {protocol.get('name', 'Unknown')}")
            print(f"TVL: ${protocol.get('tvl', 0):,.2f}")
            print(f"Chain: {protocol.get('chain', 'Unknown')}")
        else:
            print("No protocol info found")
        
        # Check users
        logger.info("Checking users...")
        users = supabase.table("debank_loopfi_users").select("*").order("loopfi_usd_value", desc=True).limit(10).execute()
        if users.data:
            print("\nTop 10 Users:")
            user_data = []
            for user in users.data:
                loopfi_value = user.get('loopfi_usd_value', 0) or 0
                net_worth = user.get('total_net_worth_usd', 0) or 0
                user_data.append([
                    user.get('user_address', '')[:10] + "...",
                    f"${loopfi_value:,.2f}",
                    f"${net_worth:,.2f}"
                ])
            print(tabulate(user_data, headers=["Address", "LoopFi Value", "Net Worth"]))
        else:
            print("No users found")
        
        # Check positions
        logger.info("Checking positions...")
        positions = supabase.table("debank_user_loopfi_positions").select("*").limit(10).execute()
        if positions.data:
            print("\nSample Positions:")
            position_data = []
            for pos in positions.data:
                asset_value = pos.get('asset_usd_value', 0) or 0
                debt_value = pos.get('debt_usd_value', 0) or 0
                position_data.append([
                    (pos.get('user_address', '') or '')[:10] + "...",
                    pos.get('portfolio_item_name', 'Unknown'),
                    pos.get('asset_symbol', 'Unknown'),
                    pos.get('debt_symbol', 'Unknown') or 'None',
                    f"${asset_value:,.2f}",
                    f"${debt_value:,.2f}"
                ])
            print(tabulate(position_data, headers=["Address", "Position", "Asset", "Debt", "Asset Value", "Debt Value"]))
        else:
            print("No positions found")
        
        # Check tokens
        logger.info("Checking tokens...")
        tokens = supabase.table("debank_user_token_holdings").select("*").limit(10).execute()
        if tokens.data:
            print("\nSample Token Holdings:")
            token_data = []
            for token in tokens.data:
                usd_value = token.get('usd_value', 0) or 0
                token_data.append([
                    (token.get('user_address', '') or '')[:10] + "...",
                    token.get('token_symbol', 'Unknown'),
                    token.get('chain_id', 'Unknown'),
                    f"${usd_value:,.2f}"
                ])
            print(tabulate(token_data, headers=["Address", "Token", "Chain", "Value"]))
        else:
            print("No token holdings found")
        
        # Check protocol interactions
        logger.info("Checking protocol interactions...")
        interactions = supabase.table("debank_user_protocol_interactions").select("*").limit(10).execute()
        if interactions.data:
            print("\nSample Protocol Interactions:")
            interaction_data = []
            for interaction in interactions.data:
                net_value = interaction.get('net_usd_value', 0) or 0
                interaction_data.append([
                    (interaction.get('user_address', '') or '')[:10] + "...",
                    interaction.get('protocol_name', 'Unknown'),
                    interaction.get('chain_id', 'Unknown'),
                    f"${net_value:,.2f}"
                ])
            print(tabulate(interaction_data, headers=["Address", "Protocol", "Chain", "Value"]))
        else:
            print("No protocol interactions found")
            
    except Exception as e:
        logger.error(f"Error checking data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        check_tables()
    except Exception as e:
        logger.error(f"Data check failed: {str(e)}") 