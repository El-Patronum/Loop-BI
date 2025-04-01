"""
DeBank API Interaction Module for Loop-BI

This module handles fetching data from the DeBank API for the Loop-BI application.
It includes functions to retrieve protocol users, user positions, and other relevant data.
"""

import os
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('debank_api')

# Load environment variables
load_dotenv()

# Constants
DEBANK_API_BASE_URL = "https://pro-openapi.debank.com"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 30
MAX_REQUESTS_PER_SECOND = 100  # DeBank rate limit

class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""
    
    def __init__(self, max_per_second: int = MAX_REQUESTS_PER_SECOND):
        """
        Initialize the rate limiter.
        
        Args:
            max_per_second: Maximum number of requests allowed per second
        """
        self.max_per_second = max_per_second
        self.tokens = max_per_second  # Start with a full bucket
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
    
    def wait_for_token(self) -> None:
        """
        Wait until a token is available for a new request.
        This ensures we don't exceed the rate limit.
        """
        with self.lock:
            current_time = time.time()
            time_passed = current_time - self.last_refill_time
            
            # Refill tokens based on time passed
            if time_passed > 0:
                new_tokens = time_passed * self.max_per_second
                self.tokens = min(self.max_per_second, self.tokens + new_tokens)
                self.last_refill_time = current_time
            
            # If no tokens are available, wait
            if self.tokens < 1:
                # Calculate wait time to get a token
                wait_time = (1 - self.tokens) / self.max_per_second
                with self.lock:
                    self.lock.release()  # Release lock during sleep
                    time.sleep(wait_time)
                    self.lock.acquire()  # Reacquire lock after sleep
                    
                    # Update tokens after waiting
                    current_time = time.time()
                    time_passed = current_time - self.last_refill_time
                    new_tokens = time_passed * self.max_per_second
                    self.tokens = min(self.max_per_second, self.tokens + new_tokens)
                    self.last_refill_time = current_time
            
            # Consume a token
            self.tokens -= 1

class DebankAPI:
    """Handles interactions with the DeBank API."""
    
    def __init__(self):
        """Initialize the DeBank API client."""
        self.access_key = os.getenv("DEBANK_ACCESS_KEY")
        
        # Load protocol IDs from environment variables
        eth_protocol_id = os.getenv("ETH_LOOPFI_PROTOCOL_ID", "loopfixyz").split('#')[0].strip().strip('"')
        bsc_protocol_id = os.getenv("BSC_LOOPFI_PROTOCOL_ID", "bsc_loopfixyz").split('#')[0].strip().strip('"')
        
        # Chain-specific protocol IDs
        self.protocol_ids = {
            "eth": eth_protocol_id,
            "bsc": bsc_protocol_id
        }
        
        # Define supported chains
        self.supported_chains = ["eth", "bsc"]
        
        logger.info(f"Initialized DeBank API client with protocol IDs:")
        for chain, pid in self.protocol_ids.items():
            logger.info(f"  - {chain}: {pid}")
        logger.info(f"Supported chains: {', '.join(self.supported_chains)}")
        
        if not self.access_key:
            raise ValueError(
                "DeBank API key not found. Ensure DEBANK_ACCESS_KEY is set in your .env file."
            )
        
        self.headers = {
            "accept": "application/json",
            "AccessKey": self.access_key
        }
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET"
    ) -> Dict[str, Any]:
        """
        Make a request to the DeBank API with retry logic and error handling.
        
        Args:
            endpoint: API endpoint path (without the base URL)
            params: Query parameters for the request
            method: HTTP method (GET, POST, etc.)
            
        Returns:
            Response data as a dictionary
            
        Raises:
            requests.RequestException: If the request fails after retries
        """
        url = f"{DEBANK_API_BASE_URL}{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                # Wait for a token from the rate limiter before making the request
                self.rate_limiter.wait_for_token()
                
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    params=params,
                    timeout=REQUEST_TIMEOUT_SECONDS
                )
                
                # Handle rate limiting responses (429 Too Many Requests)
                if response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', RETRY_DELAY_SECONDS * 2))
                    logger.warning(f"Rate limited. Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
                
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff for retries
                    wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to {method} {url} after {MAX_RETRIES} attempts")
                    raise
    
    def get_protocol_id(self, chain_id: str) -> str:
        """
        Get the correct protocol ID for the given chain.
        
        Args:
            chain_id: Chain ID (e.g., 'eth', 'bsc')
            
        Returns:
            Protocol ID for the specified chain
        """
        # Default to ETH protocol ID if chain not found
        return self.protocol_ids.get(chain_id.lower(), self.protocol_ids.get('eth'))
    
    def get_protocol_info(self, chain_id=None):
        """
        Get information about the LoopFi protocol.
        
        Args:
            chain_id: Optional chain ID (e.g., 'eth', 'bsc'). If None, returns default chain.
        
        Returns:
            Protocol information including TVL, name, and other details
        """
        # Get the correct protocol ID for this chain
        protocol_id = self.get_protocol_id(chain_id) if chain_id else self.protocol_ids.get('eth')
        
        logger.info(f"Fetching protocol info with id: {protocol_id} on chain: {chain_id or 'default'}")
        try:
            response = self._make_request(
                endpoint="/v1/protocol",
                params={"id": protocol_id}
            )
            logger.info(f"Protocol API response received, data size: {len(str(response)) if response else 0} bytes")
            # Log the first 500 characters of the response for debugging
            logger.info(f"Protocol data structure (first 500 chars): {str(response)[:500]}")
            return response
        except Exception as e:
            logger.error(f"Error fetching protocol info: {str(e)}")
            return {}
    
    def get_protocol_top_holders(
        self, 
        start: int = 0,
        limit: int = 100,
        chain_id: str = "eth"
    ) -> List[List[Union[str, float]]]:
        """
        Get top holders of the LoopFi protocol.
        
        Args:
            start: Offset for pagination
            limit: Number of results to return per page (max 100)
            chain_id: Chain ID (e.g., 'eth', 'bsc')
            
        Returns:
            List of [address, usd_value] pairs
        """
        # Get the correct protocol ID for this chain
        protocol_id = self.get_protocol_id(chain_id)
        
        logger.info(f"Fetching top holders for protocol {protocol_id} on chain {chain_id} (start={start}, limit={limit})")
        return self._make_request(
            endpoint="/v1/protocol/top_holders",
            params={
                "id": protocol_id,
                "start": start,
                "limit": min(limit, 100)  # Ensure we don't exceed API limit
            }
        )
    
    def get_user_protocol_positions(self, user_address: str, chain_id: str = "eth") -> Dict[str, Any]:
        """
        Get user's positions in the LoopFi protocol.
        
        Args:
            user_address: Ethereum address of the user
            chain_id: Chain ID (e.g., 'eth', 'bsc')
            
        Returns:
            User's positions in the LoopFi protocol
        """
        # Get the correct protocol ID for this chain
        protocol_id = self.get_protocol_id(chain_id)
        
        logger.info(f"Fetching positions for user {user_address[:10]}... on chain {chain_id} with protocol ID {protocol_id}")
        return self._make_request(
            endpoint="/v1/user/protocol",
            params={
                "id": user_address,
                "protocol_id": protocol_id
            }
        )
    
    def get_user_total_balance(self, user_address: str) -> Dict[str, Any]:
        """
        Get user's total balance across all chains.
        
        Args:
            user_address: Ethereum address of the user
            
        Returns:
            User's total balance information
        """
        logger.info(f"Fetching total balance for user {user_address[:10]}...")
        return self._make_request(
            endpoint="/v1/user/total_balance",
            params={"id": user_address}
        )
    
    def get_user_token_list(self, user_address: str, chain_id: str = "eth") -> List[Dict[str, Any]]:
        """
        Get all tokens held by a user on a specific chain.
        
        Args:
            user_address: Ethereum address of the user
            chain_id: Chain ID (e.g., 'eth', 'bsc')
            
        Returns:
            List of token holdings
        """
        logger.info(f"Fetching token list for user {user_address[:10]}... on chain {chain_id}")
        return self._make_request(
            endpoint="/v1/user/token_list",
            params={
                "id": user_address, 
                "chain_id": chain_id,
                "is_all": "true"
            }
        )
    
    def get_user_protocol_list(self, user_address: str, chain_id: str = "eth") -> List[Dict[str, Any]]:
        """
        Get all protocols a user interacts with on a specific chain.
        
        Args:
            user_address: Ethereum address of the user
            chain_id: Chain ID (e.g., 'eth', 'bsc')
            
        Returns:
            List of protocol interactions
        """
        logger.info(f"Fetching protocol list for user {user_address[:10]}... on chain {chain_id}")
        return self._make_request(
            endpoint="/v1/user/complex_protocol_list",
            params={
                "id": user_address,
                "chain_id": chain_id
            }
        )
    
    def fetch_all_protocol_users(self, chain_id: str = "eth") -> List[List[Union[str, float]]]:
        """
        Fetch all users of the LoopFi protocol by paginating through the top holders endpoint.
        
        Args:
            chain_id: Chain ID (e.g., 'eth', 'bsc')
            
        Returns:
            Combined list of all [address, usd_value] pairs
        """
        all_users = []
        start = 0
        limit = 100
        max_start = 1000  # DeBank API limit
        max_users = 500  # Limit total users to prevent excessive data
        
        # Get correct protocol ID for this chain
        protocol_id = self.get_protocol_id(chain_id)
        logger.info(f"Fetching protocol users for chain {chain_id} with protocol ID {protocol_id}")
        
        # Create a progress bar
        with tqdm(total=max_start//limit, desc=f"Fetching users on {chain_id}", unit="batch") as pbar:
            while start < max_start and len(all_users) < max_users:
                try:
                    batch = self.get_protocol_top_holders(start=start, limit=limit, chain_id=chain_id)
                    
                    if not batch:
                        # No more users to fetch
                        logger.info(f"No more users found for chain {chain_id} at offset {start}")
                        break
                        
                    # Log the first few users for debugging
                    if start == 0:
                        sample_users = batch[:3]
                        logger.info(f"Sample users from {chain_id} (first batch):")
                        for user, value in sample_users:
                            logger.info(f"  - {user[:10]}... : ${value:,.2f}")
                    
                    all_users.extend(batch)
                    pbar.set_postfix({"users": len(all_users), "batch_size": len(batch)})
                    pbar.update(1)
                    
                    if len(batch) < limit:
                        # Less than requested limit means we've reached the end
                        logger.info(f"Reached end of users for chain {chain_id} (batch size {len(batch)} < limit {limit})")
                        break
                        
                    start += limit
                    
                except Exception as e:
                    logger.error(f"Error fetching users at offset {start} on chain {chain_id}: {str(e)}")
                    break
        
        # Check for users
        if len(all_users) == 0:
            logger.warning(f"No users found for chain {chain_id} with protocol ID {protocol_id}")
        else:
            logger.info(f"Total users fetched on chain {chain_id}: {len(all_users)}")
            
        return all_users
    
    def fetch_all_chains_protocol_users(self) -> Dict[str, List[List[Union[str, float]]]]:
        """
        Fetch all users of the LoopFi protocol across all supported chains.
        
        Returns:
            Dictionary mapping chain_id to list of [address, usd_value] pairs
        """
        all_chains_users = {}
        
        print(f"Fetching protocol users across {len(self.supported_chains)} chains")
        
        for chain_id in self.supported_chains:
            try:
                chain_users = self.fetch_all_protocol_users(chain_id=chain_id)
                all_chains_users[chain_id] = chain_users
                print(f"✓ Chain {chain_id.upper()}: Fetched {len(chain_users)} users")
            except Exception as e:
                logger.error(f"Error fetching users for chain {chain_id}: {str(e)}")
                all_chains_users[chain_id] = []
                print(f"✗ Chain {chain_id.upper()}: Error fetching users")
        
        total_users = sum(len(users) for users in all_chains_users.values())
        print(f"Total users fetched across all chains: {total_users}")
        
        return all_chains_users


# Example usage (can be used for testing)
if __name__ == "__main__":
    try:
        debank_api = DebankAPI()
        
        # Test the protocol info endpoint
        protocol_info = debank_api.get_protocol_info()
        print(f"Protocol: {protocol_info.get('name', 'Unknown')}")
        print(f"TVL: ${protocol_info.get('tvl', 0):,.2f}")
        
        # Get a sample of top protocol users
        top_users = debank_api.get_protocol_top_holders(limit=5)
        print(f"\nTop 5 users:")
        for user_address, usd_value in top_users:
            print(f"  {user_address[:10]}... : ${usd_value:,.2f}")
            
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}") 