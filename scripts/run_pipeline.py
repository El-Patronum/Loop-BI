#!/usr/bin/env python
"""
Loop-BI Data Pipeline Runner

This script runs the complete data collection pipeline, fetching data from DeBank API
and storing it in Supabase. It can be run manually or scheduled using cron.

Usage:
    python -m scripts.run_pipeline [--full] [--user ADDRESS] [--protocol-only] [--batch-size SIZE] [--batch-delay SECONDS] [--historical] [--quiet]

Options:
    --full              Run the full pipeline (all users and their data)
    --user ADDRESS      Process a single user by Ethereum address
    --protocol-only     Only fetch and store protocol information, not user data
    --batch-size SIZE   Number of users to process in a batch (default: 10)
    --batch-delay SEC   Seconds to wait between batches (default: 5)
    --historical        Create historical data snapshot after data collection
    --chain CHAIN       Specific chain to process (eth, bsc). If omitted, process all chains
    --quiet             Run in quiet mode (no progress bars or detailed output)
"""

import argparse
import logging
import time
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime

# Check if tqdm is installed, if not install it
try:
    import tqdm
except ImportError:
    print("Installing tqdm for progress visualization...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        import tqdm
    except Exception as e:
        print(f"Failed to install tqdm: {e}")
        print("Continuing without progress visualization")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger('pipeline')

# Import after logging is configured
from scripts.store_data import DebankDataStorage
from scripts.historical_data import HistoricalDataStorage


def process_user_batch(
    storage: DebankDataStorage,
    users: List[Dict[str, Any]],
    batch_idx: int,
    batch_size: int
) -> None:
    """
    Process a batch of users.
    
    Args:
        storage: The DebankDataStorage instance
        users: List of user records to process
        batch_idx: Current batch index
        batch_size: Size of each batch
    """
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(users))
    batch = users[start_idx:end_idx]
    
    logger.info(f"Processing batch {batch_idx+1} ({start_idx}-{end_idx-1}, {len(batch)} users)")
    
    for i, user_record in enumerate(batch):
        user_address = user_record['user_address']
        try:
            logger.info(f"[{start_idx+i+1}/{len(users)}] Processing user {user_address[:10]}...")
            
            # Store user positions in LoopFi
            storage.store_user_positions(user_address)
            
            # Update user balance
            storage.update_user_balance(user_address)
            
            # Store user's token holdings
            storage.store_user_token_holdings(user_address)
            
            # Store user's interactions with other protocols
            storage.store_user_protocol_interactions(user_address)
            
        except Exception as e:
            logger.error(f"Error processing user {user_address[:10]}...: {str(e)}")
            # Continue with next user
    
    logger.info(f"Completed batch {batch_idx+1}")


def run_pipeline(
    full_pipeline: bool = False,
    single_user: Optional[str] = None,
    protocol_only: bool = False,
    batch_size: int = 10,
    batch_delay: int = 5,
    chain_id: Optional[str] = None,
    create_historical_snapshot: bool = False,
    verbose: bool = True,
    verify_chains: bool = True
) -> None:
    """
    Run the data collection pipeline.
    
    Args:
        full_pipeline: Whether to run the full pipeline (all users)
        single_user: Ethereum address of a single user to process
        protocol_only: Only fetch and store protocol information
        batch_size: Number of users to process in a batch
        batch_delay: Seconds to wait between batches
        chain_id: Optional specific chain to process (e.g., 'eth', 'bsc')
        create_historical_snapshot: Whether to create a historical snapshot after data collection
        verbose: Whether to show detailed progress information
        verify_chains: Whether to enable chain validation to prevent duplicates across chains
    """
    try:
        start_time = time.time()
        
        # Print a nice header
        print("\n" + "="*80)
        print(f"  LOOP-BI DATA PIPELINE - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Get supported chains
        supported_chains = ["eth", "bsc"]
        
        # Limit to specified chain if provided
        if chain_id:
            if chain_id in supported_chains:
                chains_to_process = [chain_id]
                print(f"Chain Selection: {chain_id.upper()} only")
            else:
                logger.error(f"Unsupported chain: {chain_id}. Please use one of {supported_chains}")
                return
        else:
            chains_to_process = supported_chains
            print(f"Chain Selection: All supported chains ({', '.join([c.upper() for c in chains_to_process])})")
        
        # Show mode
        if protocol_only:
            print("Mode: Protocol-only")
        elif single_user:
            print(f"Mode: Single user ({single_user})")
        elif full_pipeline:
            print("Mode: Full pipeline (all users and data)")
        else:
            print("Mode: Basic (protocol and user list only)")
            
        if verify_chains:
            print("Chain Validation: Enabled (will prevent duplicate positions across chains)")
        else:
            print("Chain Validation: Disabled (duplicate positions may occur)")
        
        # Add an empty line
        print()
        
        # Initialize the storage handler
        storage = DebankDataStorage(verbose=verbose)
        
        # Enable chain validation if requested
        if verify_chains:
            storage.set_chain_validation(enabled=True)
        else:
            storage.set_chain_validation(enabled=False)
        
        # Always get protocol info for all chains
        logger.info("Fetching protocol information for all chains")
        protocol_info = storage.store_protocol_info()
        
        protocol_summary = []
        for chain, info in protocol_info.items():
            protocol_summary.append(f"• {info.get('name', 'Unknown')} on {chain.upper()}: ${info.get('tvl', 0):,.2f} TVL")
        
        if protocol_summary:
            print("\nProtocol Information:")
            for line in protocol_summary:
                print(line)
        
        if protocol_only:
            logger.info("Protocol-only mode: skipping user data")
        else:
            if single_user:
                print(f"\nProcessing user: {single_user}")
                
                # Process the user for each chain
                for chain in chains_to_process:
                    try:
                        logger.info(f"Processing user {single_user} on chain {chain}")
                        storage.process_user_data(single_user, chain_id=chain)
                        print(f"✓ Processed user data on {chain.upper()}")
                    except Exception as e:
                        logger.error(f"Error processing user {single_user} on chain {chain}: {str(e)}")
                        print(f"✗ Failed to process user data on {chain.upper()}: {str(e)}")
                
                print(f"\nCompleted processing user {single_user} across all chains")
                
            elif full_pipeline:
                print("\nRunning full pipeline (all users on all chains)")
                
                # Run the multi-chain data processing
                storage.process_all_data()
                
            else:
                print("\nFetching protocol users only")
                
                # Just fetch and store users, not their detailed data
                users = storage.store_protocol_users()
                print(f"✓ Stored {len(users)} users across all chains")
        
        # Create historical snapshot if requested
        if create_historical_snapshot:
            print("\nCreating historical data snapshot...")
            historical_storage = HistoricalDataStorage()
            historical_storage.store_all_historical_data()
            print("✓ Historical snapshot completed")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print summary footer
        print("\n" + "="*80)
        print(f"  PIPELINE SUMMARY")
        print("="*80)
        print(f"• Started at:  {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"• Finished at: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"• Duration:    {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        
        if create_historical_snapshot:
            print(f"• Historical snapshot: Created")
        
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loop-BI Data Pipeline Runner")
    parser.add_argument("--full", action="store_true", help="Run the full pipeline (all users and their data)")
    parser.add_argument("--user", type=str, help="Process a single user by Ethereum address")
    parser.add_argument("--protocol-only", action="store_true", help="Only fetch and store protocol information")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of users to process in a batch")
    parser.add_argument("--batch-delay", type=int, default=5, help="Seconds to wait between batches")
    parser.add_argument("--chain", type=str, help="Specific chain to process (eth, bsc). If omitted, process all chains")
    parser.add_argument("--historical", action="store_true", help="Create historical data snapshot after collection")
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode (no progress bars)")
    parser.add_argument("--no-verify-chains", action="store_true", help="Disable chain validation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.full and args.user:
        logger.error("Cannot specify both --full and --user")
        sys.exit(1)
    
    if args.full and args.protocol_only:
        logger.error("Cannot specify both --full and --protocol-only")
        sys.exit(1)
    
    if args.user and args.protocol_only:
        logger.error("Cannot specify both --user and --protocol-only")
        sys.exit(1)
    
    # Run the pipeline with the specified options
    run_pipeline(
        full_pipeline=args.full,
        single_user=args.user,
        protocol_only=args.protocol_only,
        batch_size=args.batch_size,
        batch_delay=args.batch_delay,
        chain_id=args.chain,
        create_historical_snapshot=args.historical,
        verbose=not args.quiet,
        verify_chains=not args.no_verify_chains
    ) 