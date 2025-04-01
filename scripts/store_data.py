"""
Data Storage Module for Loop-BI

This module handles storing data fetched from the DeBank API into Supabase.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Union
from tqdm import tqdm
from supabase import Client

from core.supabase_client import get_supabase_client
from scripts.fetch_debank_data import DebankAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('debank_storage')

class DebankDataStorage:
    """Handles storing DeBank data into Supabase."""
    
    def __init__(self, verbose=True):
        """
        Initialize the data storage handlers.
        
        Args:
            verbose: Whether to show detailed progress information
        """
        self.supabase = get_supabase_client()
        self.debank_api = DebankAPI()
        self.timestamp = datetime.utcnow().isoformat()
        self.verbose = verbose
        
        # Flag to control chain validation during data processing
        # When enabled, prevents duplicate position data across chains
        self.verify_chain_data = False
        
        logger.info(f"Initialized DeBank data storage handler with timestamp: {self.timestamp}")
    
    def set_chain_validation(self, enabled=True):
        """
        Enable or disable chain data validation.
        
        Args:
            enabled: Whether to enable chain validation
        """
        self.verify_chain_data = enabled
        if enabled:
            logger.info("Chain validation ENABLED - Will verify position chains to prevent duplicates")
        else:
            logger.info("Chain validation DISABLED - Chain data will not be verified for duplicates")
    
    def store_protocol_info(self) -> Dict[str, Any]:
        """
        Fetch and store LoopFi protocol information for all supported chains.
        
        Returns:
            The stored protocol data
        """
        protocol_data_by_chain = {}
        
        # Fetch protocol info for each supported chain
        for chain_id in self.debank_api.supported_chains:
            try:
                logger.info(f"Fetching protocol info for chain: {chain_id}")
                protocol_data = self.debank_api.get_protocol_info(chain_id=chain_id)
                
                # Check if we got valid data back
                if not protocol_data:
                    logger.error(f"Failed to get protocol info from DeBank API for chain: {chain_id}")
                    continue
                
                # Add timestamp and handle chain vs chain_id
                protocol_data['last_updated_timestamp'] = self.timestamp
                
                # Remove 'chain' and set 'chain_id' properly
                if 'chain' in protocol_data:
                    original_chain = protocol_data.pop('chain')
                    logger.info(f"Removed 'chain' field with value '{original_chain}' and using '{chain_id}' for chain_id")
                
                protocol_data['chain_id'] = chain_id
                
                # Upsert the protocol data based on protocol_id and chain_id
                result = self.supabase.table('debank_protocols').upsert(
                    [protocol_data],
                    on_conflict='id,chain_id'
                ).execute()
                
                logger.info(f"Stored protocol info for {protocol_data.get('name', 'Unknown')} on chain {chain_id}")
                protocol_data_by_chain[chain_id] = protocol_data
                
            except Exception as e:
                logger.error(f"Failed to store protocol info for chain {chain_id}: {str(e)}")
        
        # Return combined protocol data
        return protocol_data_by_chain
    
    def store_protocol_users(self) -> List[Dict[str, Any]]:
        """
        Fetch and store all users of the LoopFi protocol across all supported chains.
        
        Returns:
            List of stored user records
        """
        try:
            # Fetch users from all supported chains
            all_chains_users = self.debank_api.fetch_all_chains_protocol_users()
            
            # Initialize list of all formatted users
            all_formatted_users = []
            
            # Process users for each chain
            for chain_id, users_data in all_chains_users.items():
                # Format the data for Supabase insertion
                formatted_users = []
                for user_address, loopfi_usd_value in users_data:
                    formatted_users.append({
                        'user_address': user_address,
                        'loopfi_usd_value': loopfi_usd_value,
                        'chain_id': chain_id,  # Store the chain ID
                        'first_seen_timestamp': self.timestamp,
                        'last_updated_timestamp': self.timestamp
                    })
                
                # Collect for combined result
                all_formatted_users.extend(formatted_users)
                
                # Batch insert users (100 at a time to avoid request size limits)
                batch_size = 100
                batches = [formatted_users[i:i+batch_size] for i in range(0, len(formatted_users), batch_size)]
                
                # Show progress bar for storing users
                with tqdm(total=len(batches), desc=f"Storing users for {chain_id}", disable=not self.verbose) as pbar:
                    for i, batch in enumerate(batches):
                        # Upsert based on user_address and chain_id (composite key)
                        self.supabase.table('debank_loopfi_users').upsert(
                            batch,
                            on_conflict='user_address,chain_id'
                        ).execute()
                        
                        pbar.update(1)
                        pbar.set_postfix({"batch": f"{i+1}/{len(batches)}", "users": len(batch)})
                
                logger.info(f"Stored {len(formatted_users)} users for chain {chain_id}")
            
            logger.info(f"Stored {len(all_formatted_users)} users in total across all chains")
            return all_formatted_users
            
        except Exception as e:
            logger.error(f"Failed to store protocol users: {str(e)}")
            raise
    
    def process_user_data(self, user_address: str, chain_id: str = "eth"):
        """
        Process complete data for a single user on a specific chain
        
        Args:
            user_address: The user's address
            chain_id: The chain ID to process data for
        """
        try:
            logger.info(f"Processing user {user_address[:10]}... on chain {chain_id}")
            
            # 1. Get user's LoopFi positions on this chain
            positions = self.debank_api.get_user_protocol_positions(user_address, chain_id=chain_id)
            
            # 2. Get user's total balance across all chains
            total_balance = self.debank_api.get_user_total_balance(user_address)
            
            # 3. Store user data with total net worth and chain ID
            self.store_user_data(user_address, positions, total_balance, chain_id)
            
            # 4. Get and store user's token holdings for this chain
            tokens = self.debank_api.get_user_token_list(user_address, chain_id=chain_id)
            self.store_user_tokens(user_address, tokens, chain_id)
            
            # 5. Get and store user's other protocol interactions on this chain
            protocols = self.debank_api.get_user_protocol_list(user_address, chain_id=chain_id)
            self.store_user_protocols(user_address, protocols, chain_id)
            
            logger.info(f"Completed processing data for user {user_address[:10]}... on chain {chain_id}")
            
        except Exception as e:
            logger.error(f"Error processing user {user_address[:10]}... on chain {chain_id}: {str(e)}")
            raise
    
    def store_user_data(self, user_address: str, positions: Dict[str, Any], total_balance: Dict[str, Any], chain_id: str):
        """
        Store user profile and position data
        
        Args:
            user_address: The user's address
            positions: The user's positions data
            total_balance: The user's total balance data
            chain_id: The chain ID the positions belong to
        """
        # Calculate total value in LoopFi on this chain
        loopfi_value = 0
        if positions and 'portfolio_item_list' in positions:
            loopfi_value = sum(
                item.get('stats', {}).get('net_usd_value', 0) 
                for item in positions.get('portfolio_item_list', [])
            )
        
        # Store user profile
        user_data = {
            'user_address': user_address,
            'chain_id': chain_id,
            'loopfi_usd_value': loopfi_value,
            'total_net_worth_usd': total_balance.get('total_usd_value', 0) if total_balance else 0,
            'last_updated_timestamp': self.timestamp
        }
        
        # Check if user exists on this chain
        existing_user = self.supabase.table("debank_loopfi_users").select("user_address").eq(
            "user_address", user_address).eq("chain_id", chain_id).execute()
        
        if not existing_user.data:
            # New user - add first_seen_timestamp
            user_data['first_seen_timestamp'] = self.timestamp
        
        # Upsert user data
        self.supabase.table("debank_loopfi_users").upsert(user_data, on_conflict='user_address,chain_id').execute()
        
        # Store position data
        if positions and 'portfolio_item_list' in positions:
            for position in positions['portfolio_item_list']:
                self.store_user_position(user_address, position, chain_id)
    
    def store_user_position(self, user_address: str, position: Dict[str, Any], chain_id: str):
        """
        Store a single user position with improved data extraction and timestamp tracking
        
        Args:
            user_address: The user's address
            position: The position data to store
            chain_id: The chain ID the position belongs to
        """
        # Extract position details
        portfolio_item_name = position.get('name', '')
        
        stats = position.get('stats', {})
        asset_usd_value = stats.get('asset_usd_value', 0)
        debt_usd_value = stats.get('debt_usd_value', 0)
        net_usd_value = stats.get('net_usd_value', 0)
        
        # Extract tokens data from details
        detail = position.get('detail', {})
        supplied_tokens = detail.get('supply_token_list', [])
        
        # CRITICAL FIX: Check both possible debt token field names in the DeBank API
        debt_tokens = detail.get('debt_token_list', detail.get('debt_tokens', []))
        rewards_tokens = detail.get('reward_token_list', [])
        
        # Debug logging for debt tokens
        logger.info(f"Position {portfolio_item_name} debt tokens: {debt_tokens[:200] if debt_tokens else 'None'}")
        
        # Extract asset_symbol and debt_symbol
        asset_symbol = None
        if supplied_tokens and len(supplied_tokens) > 0:
            asset_symbol = supplied_tokens[0].get('symbol')
        
        debt_symbol = None
        if debt_tokens and len(debt_tokens) > 0:
            debt_symbol = debt_tokens[0].get('symbol')
            if debt_symbol:
                logger.info(f"Found debt symbol: {debt_symbol} for position {portfolio_item_name}")
            
        # CHAIN VALIDATION: Only perform when verify_chain_data is enabled
        if self.verify_chain_data:
            # Extract position chain information from tokens to verify
            # this position really belongs to the specified chain
            position_chain = None
            if supplied_tokens and len(supplied_tokens) > 0 and 'chain' in supplied_tokens[0]:
                position_chain = supplied_tokens[0].get('chain')
            elif debt_tokens and len(debt_tokens) > 0 and 'chain' in debt_tokens[0]:
                position_chain = debt_tokens[0].get('chain')
                
            # Compare against expected chain and log any mismatches
            if position_chain and position_chain.lower() != chain_id.lower():
                logger.warning(
                    f"Chain mismatch for position - Expected: {chain_id}, Found: {position_chain}, "
                    f"User: {user_address[:10]}..., Position: {portfolio_item_name}"
                )
                
                # Correct the chain_id to prevent duplicates
                chain_id = position_chain.lower()
                
                # Skip storing this position if we detect it's likely a duplicate
                existing_position = self.supabase.table("debank_user_loopfi_positions").select("position_id").eq(
                    "user_address", user_address).eq("portfolio_item_name", portfolio_item_name).execute()
                    
                if existing_position.data:
                    logger.info(f"Skipping potential duplicate position for user {user_address[:10]}...")
                    return
        
        # Check if position exists to preserve entry timestamp
        existing_position = self.supabase.table("debank_user_loopfi_positions").select("position_id", "entry_timestamp").eq(
            "user_address", user_address).eq("portfolio_item_name", portfolio_item_name).eq("chain_id", chain_id).execute()
        
        position_data = {
            'user_address': user_address,
            'portfolio_item_name': portfolio_item_name,
            'asset_usd_value': asset_usd_value,
            'debt_usd_value': debt_usd_value,
            'net_usd_value': net_usd_value,
            'chain_id': chain_id,
            'supplied_tokens': json.dumps(supplied_tokens),
            'debt_tokens': json.dumps(debt_tokens),
            'rewards_tokens': json.dumps(rewards_tokens),
            'asset_symbol': asset_symbol,
            'debt_symbol': debt_symbol,
            'last_updated_timestamp': self.timestamp
        }
        
        # For new positions, add an entry timestamp
        if not existing_position.data:
            # Log new position creation
            logger.info(f"Creating new position for user {user_address[:10]}... on chain {chain_id}: {portfolio_item_name}")
            
            # Set creation timestamp
            position_data['entry_timestamp'] = self.timestamp
            
            self.supabase.table('debank_user_loopfi_positions').insert(
                position_data
            ).execute()
        else:
            # For existing positions, use upsert to update but preserve the original entry_timestamp
            logger.debug(f"Updating existing position for user {user_address[:10]}... on chain {chain_id}: {portfolio_item_name}")
            
            # Important: Don't include entry_timestamp in the update data
            # This ensures the original entry timestamp is preserved
            self.supabase.table('debank_user_loopfi_positions').upsert(
                position_data, 
                on_conflict='user_address,portfolio_item_name,chain_id'
            ).execute()
    
    def store_user_tokens(self, user_address: str, tokens: List[Dict[str, Any]], chain_id: str):
        """
        Store a user's token holdings for a specific chain
        
        Args:
            user_address: The user's address
            tokens: The token data to store
            chain_id: The chain ID the tokens belong to
        """
        # Delete existing token holdings for this user on this chain
        self.supabase.table("debank_user_token_holdings").delete().eq(
            "user_address", user_address).eq("chain_id", chain_id).execute()
        
        # Process and store each token holding
        batch_size = 100
        token_batches = [tokens[i:i+batch_size] for i in range(0, len(tokens), batch_size)]
        
        for batch in token_batches:
            holding_records = []
            
            for token in batch:
                token_chain = token.get('chain', chain_id)  # Use specified chain as fallback
                
                holding_record = {
                    'user_address': user_address,
                    'token_address': token.get('id', ''),
                    'chain_id': token_chain,
                    'token_symbol': token.get('symbol', ''),
                    'amount': token.get('amount', 0),
                    'usd_value': token.get('amount', 0) * token.get('price', 0),
                    'last_updated_timestamp': self.timestamp
                }
                
                holding_records.append(holding_record)
            
            # Batch insert token holdings
            if holding_records:
                self.supabase.table('debank_user_token_holdings').insert(
                    holding_records
                ).execute()
        
        logger.info(f"Stored {len(tokens)} token holdings for user {user_address[:10]}... on chain {chain_id}")
    
    def store_user_protocols(self, user_address: str, protocols: List[Dict[str, Any]], chain_id: str):
        """
        Store a user's protocol interactions for a specific chain
        
        Args:
            user_address: The user's address
            protocols: The protocol data to store
            chain_id: The chain ID the protocols belong to
        """
        # Delete existing protocol interactions for this user on this chain
        self.supabase.table("debank_user_protocol_interactions").delete().eq(
            "user_address", user_address).eq("chain_id", chain_id).execute()
        
        # Process and store each protocol interaction
        batch_size = 100
        protocol_batches = [protocols[i:i+batch_size] for i in range(0, len(protocols), batch_size)]
        
        for batch in protocol_batches:
            interaction_records = []
            
            for protocol in batch:
                # Skip LoopFi itself (we already have dedicated tables for that)
                current_chain_protocol_id = self.debank_api.get_protocol_id(chain_id)
                if protocol.get('id') == current_chain_protocol_id:
                    continue
                
                protocol_chain = protocol.get('chain', chain_id)  # Use specified chain as fallback
                
                # Calculate net value from portfolio items if available
                net_usd_value = 0
                portfolio_items = protocol.get('portfolio_item_list', [])
                if portfolio_items:
                    net_usd_value = sum(
                        item.get('stats', {}).get('net_usd_value', 0) 
                        for item in portfolio_items
                    )
                
                interaction_record = {
                    'user_address': user_address,
                    'protocol_id': protocol.get('id', ''),
                    'chain_id': protocol_chain,
                    'protocol_name': protocol.get('name', ''),
                    'net_usd_value': net_usd_value,
                    'last_updated_timestamp': self.timestamp
                }
                
                interaction_records.append(interaction_record)
            
            # Batch insert protocol interactions
            if interaction_records:
                self.supabase.table('debank_user_protocol_interactions').insert(
                    interaction_records
                ).execute()
        
        logger.info(f"Stored {len(protocols)} protocol interactions for user {user_address[:10]}... on chain {chain_id}")
    
    def process_all_data(self):
        """
        Full data processing pipeline: fetch and store all necessary data across all chains.
        
        This method implements the entire data pipeline described in the development plan:
        1. Store protocol info for all chains
        2. Store protocol users for all chains
        3. For each user on each chain:
           a. Store positions
           b. Update total balance
           c. Store token holdings
           d. Store protocol interactions
        """
        try:
            # Step 1: Store protocol info for all supported chains
            print("Step 1: Storing protocol information...")
            self.store_protocol_info()
            
            # Step 2: Store protocol users for all supported chains
            print("Step 2: Storing protocol users...")
            users = self.store_protocol_users()
            
            # Group users by chain for more efficient processing
            users_by_chain = {}
            for user in users:
                chain_id = user.get('chain_id', 'eth')  # Default to eth if missing
                if chain_id not in users_by_chain:
                    users_by_chain[chain_id] = []
                users_by_chain[chain_id].append(user)
            
            # Step 3: Process each user's detailed data by chain
            print("Step 3: Processing detailed user data...")
            for chain_id, chain_users in users_by_chain.items():
                print(f"\nProcessing {len(chain_users)} users for chain {chain_id}")
                
                # Use tqdm to show progress
                with tqdm(total=len(chain_users), desc=f"Processing users on {chain_id}", disable=not self.verbose) as pbar:
                    for i, user_record in enumerate(chain_users):
                        user_address = user_record['user_address']
                        
                        try:
                            # Process user data for this chain
                            self.process_user_data(user_address, chain_id=chain_id)
                            
                            # Update progress
                            pbar.update(1)
                            pbar.set_postfix({"user": f"{i+1}/{len(chain_users)}", "address": user_address[:10]+"..."})
                            
                        except Exception as e:
                            logger.error(f"Error processing data for user {user_address[:10]}... on chain {chain_id}: {str(e)}")
                            pbar.update(1)
                
                print(f"Completed processing users for chain {chain_id}")
            
            print("Data pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Failed to process all data: {str(e)}")
            raise


# Example usage (can be used for testing)
if __name__ == "__main__":
    try:
        storage = DebankDataStorage()
        
        # Test storing protocol info
        protocol_info = storage.store_protocol_info()
        print(f"Stored protocol info for: {protocol_info.get('name', 'Unknown')}")
        
        # For testing, process just a few users to avoid API rate limits
        # In production, use storage.process_all_data() for the full pipeline
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}") 