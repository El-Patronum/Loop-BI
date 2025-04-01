#!/usr/bin/env python
"""
Test script to run the pipeline for the top 10 users

This script fetches data for the top 10 users of LoopFi protocol and processes it
through the data pipeline.
"""
import os
import time
import logging
from datetime import datetime
from scripts.store_data import DebankDataStorage
from scripts.fetch_debank_data import DebankAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_pipeline')

def run_test():
    """Run a test pipeline for the top 10 users"""
    try:
        start_time = time.time()
        logger.info("Starting test pipeline for top 10 users")
        
        # Initialize API and storage
        debank_api = DebankAPI()
        storage = DebankDataStorage()
        
        # Step 1: Store protocol info
        logger.info("Fetching and storing protocol info")
        protocol_info = storage.store_protocol_info()
        logger.info(f"Stored protocol info for: {protocol_info.get('name', 'Unknown')}")
        
        # Step 2: Get top 10 users
        logger.info("Fetching top 10 users")
        top_users = debank_api.get_protocol_top_holders(limit=10)
        
        if not top_users:
            logger.error("Failed to get top users")
            return
            
        logger.info(f"Got {len(top_users)} users")
        
        # Format users for storage
        formatted_users = []
        for user_address, loopfi_usd_value in top_users:
            formatted_users.append({
                'user_address': user_address,
                'loopfi_usd_value': loopfi_usd_value,
                'first_seen_timestamp': datetime.utcnow().isoformat(),
                'last_updated_timestamp': datetime.utcnow().isoformat()
            })
        
        # Step 3: Store users
        logger.info("Storing user records")
        storage.supabase.table('debank_loopfi_users').upsert(formatted_users).execute()
        
        # Step 4: Process each user's data
        for i, user in enumerate(formatted_users):
            user_address = user['user_address']
            logger.info(f"Processing user {i+1}/{len(formatted_users)}: {user_address[:10]}...")
            
            try:
                # Store user positions
                storage.store_user_positions(user_address)
                
                # Update user balance
                storage.update_user_balance(user_address)
                
                # Store token holdings
                storage.store_user_token_holdings(user_address)
                
                # Store protocol interactions
                storage.store_user_protocol_interactions(user_address)
                
                logger.info(f"Completed processing for user {user_address[:10]}...")
                
            except Exception as e:
                logger.error(f"Error processing user {user_address[:10]}: {str(e)}")
                continue
        
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Test pipeline completed in {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Test pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_test() 