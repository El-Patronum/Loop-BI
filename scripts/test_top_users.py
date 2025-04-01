#!/usr/bin/env python
"""
Test script to fetch the top 10 users of LoopFi

This script fetches the top 10 users from the LoopFi protocol using the DeBank API
and displays their addresses and values.
"""
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DeBank API base URL
DEBANK_API_BASE_URL = "https://pro-openapi.debank.com"
DEBANK_ACCESS_KEY = os.getenv("DEBANK_ACCESS_KEY")

# Get protocol ID and clean it (remove any quotes and comments)
raw_protocol_id = os.getenv("LOOPFI_PROTOCOL_ID", "loopfixyz")
LOOPFI_PROTOCOL_ID = raw_protocol_id.split('#')[0].strip().strip('"')

print(f"Using cleaned protocol ID: '{LOOPFI_PROTOCOL_ID}'")

# Headers for API requests
headers = {
    "accept": "application/json",
    "AccessKey": DEBANK_ACCESS_KEY
}

def get_top_users(limit=10):
    """Fetch the top users from the LoopFi protocol"""
    url = f"{DEBANK_API_BASE_URL}/v1/protocol/top_holders"
    params = {
        "id": LOOPFI_PROTOCOL_ID,
        "limit": limit
    }
    
    print(f"Making request to {url} with params: {params}")
    
    response = requests.get(url, headers=headers, params=params, timeout=10)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching top users: {response.status_code}")
        print(response.text)
        return []

if __name__ == "__main__":
    print(f"Fetching top 10 users for protocol: {LOOPFI_PROTOCOL_ID}")
    
    users = get_top_users(10)
    
    if users:
        print(f"\nFound {len(users)} users:")
        for i, (address, value) in enumerate(users):
            print(f"{i+1}. Address: {address}")
            print(f"   Value: ${value:,.2f}")
        
        # Save to file for reference
        with open("top_users.txt", "w") as f:
            for address, value in users:
                f.write(f"{address},{value}\n")
        print("\nUser addresses saved to top_users.txt")
    else:
        print("No users found or API request failed.") 