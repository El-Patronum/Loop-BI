#!/usr/bin/env python

"""
Utility script to fix chain data duplication issues and refresh the data with proper chain validation.

This script:
1. Updates the data collection pipeline with improved chain validation
2. Resets the chain data in the database to avoid duplication
3. Refreshes the data with the updated pipeline
"""

import os
import sys
import logging
import pathlib
from dotenv import load_dotenv
import pandas as pd

# Add project root to path
ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

from core.supabase_client import get_supabase_client
from scripts.run_pipeline import run_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_repair.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chain_data_repair')

def reset_chain_data():
    """
    Reset chain-related data in the database to ensure clean collection.
    """
    logger.info("Resetting chain-specific data in the database...")
    
    # Connect to Supabase
    load_dotenv()
    supabase = get_supabase_client()
    
    # Get chain validation status before reset
    logger.info("Analyzing current data state...")
    try:
        response = supabase.table("debank_user_loopfi_positions").select(
            "chain_id", "count", count="exact"
        ).execute()
        
        if response.data:
            logger.info(f"Current position counts by chain:")
            df = pd.DataFrame(response.data)
            chain_counts = df['chain_id'].value_counts()
            for chain, count in chain_counts.items():
                logger.info(f"  {chain}: {count} positions")
    except Exception as e:
        logger.warning(f"Error analyzing current data: {str(e)}")
    
    # Clear chain-specific data tables
    try:
        # Clear positions
        supabase.table("debank_user_loopfi_positions").delete().neq("chain_id", "invalid").execute()
        logger.info("Reset positions data")
        
        # Clear token holdings
        supabase.table("debank_user_token_holdings").delete().neq("chain_id", "invalid").execute()
        logger.info("Reset token holdings data")
        
        # Clear protocol interactions
        supabase.table("debank_user_protocol_interactions").delete().neq("chain_id", "invalid").execute()
        logger.info("Reset protocol interactions data")
        
        # Clear users (chain-specific)
        supabase.table("debank_loopfi_users").delete().neq("chain_id", "invalid").execute()
        logger.info("Reset user chain data")
        
        logger.info("Database reset completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        return False

def main():
    """
    Main function to execute the chain data repair process.
    """
    logger.info("Starting chain data repair process...")
    
    # Step 1: Reset chain data in the database
    if not reset_chain_data():
        logger.error("Failed to reset chain data. Aborting.")
        return
    
    # Step 2: Run the full data pipeline with improved chain validation
    logger.info("Running full data pipeline with improved chain validation...")
    try:
        # Always enable chain validation to prevent duplicates
        run_pipeline(full_pipeline=True, verify_chains=True)
        logger.info("Data pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Error running data pipeline: {str(e)}")
        return
    
    # Step 3: Verify the results
    logger.info("Verifying data quality after repair...")
    
    # Import check_chain_data_quality dynamically to avoid circular imports
    try:
        from core.analytics import check_chain_data_quality
        
        # Connect to Supabase
        load_dotenv()
        supabase = get_supabase_client()
        
        # Run data quality check
        data_quality = check_chain_data_quality(supabase)
        
        if data_quality.get('has_duplication_issues', False):
            logger.warning("""
            ⚠️ Data duplication issues still detected after repair. 
            Manual intervention may be required to resolve this issue.
            """)
            logger.warning(f"User similarity: {data_quality.get('max_user_similarity')}%")
            logger.warning(f"TVL similarity: {data_quality.get('max_tvl_similarity')}%")
            logger.warning(f"Duplicate positions: {data_quality.get('duplicate_count')}")
            
            # Print detailed chain statistics
            if 'user_counts' in data_quality:
                logger.warning("User counts by chain:")
                for chain, count in data_quality['user_counts'].items():
                    logger.warning(f"  {chain}: {count} users")
            
            if 'tvl_by_chain' in data_quality:
                logger.warning("TVL by chain:")
                for chain, tvl in data_quality['tvl_by_chain'].items():
                    logger.warning(f"  {chain}: ${tvl:,.2f}")
        else:
            logger.info("✅ Data repair successful! Chain data now appears to be correctly differentiated.")
            
            # Print summary statistics
            if 'user_counts' in data_quality:
                logger.info("User counts by chain:")
                for chain, count in data_quality['user_counts'].items():
                    logger.info(f"  {chain}: {count} users")
            
            if 'tvl_by_chain' in data_quality:
                logger.info("TVL by chain:")
                for chain, tvl in data_quality['tvl_by_chain'].items():
                    logger.info(f"  {chain}: ${tvl:,.2f}")
    except Exception as e:
        logger.error(f"Error verifying data quality: {str(e)}")
    
    logger.info("Chain data repair process completed.")

if __name__ == "__main__":
    main() 