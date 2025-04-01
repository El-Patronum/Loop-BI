#!/usr/bin/env python
"""
Test script for DeBank API

This script tests direct connections to the DeBank API to verify credentials and protocol IDs.
"""
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DeBank API base URL
DEBANK_API_BASE_URL = "https://pro-openapi.debank.com"

# Read API key from .env
DEBANK_ACCESS_KEY = os.getenv("DEBANK_ACCESS_KEY")
if not DEBANK_ACCESS_KEY:
    raise ValueError("DEBANK_ACCESS_KEY not found in .env file")

print(f"Using DeBank Access Key: {DEBANK_ACCESS_KEY[:5]}...{DEBANK_ACCESS_KEY[-5:]}")

# Headers for all requests
headers = {
    "accept": "application/json",
    "AccessKey": DEBANK_ACCESS_KEY
}

# Test protocol IDs to try
protocol_ids = [
    "loopfixyz",   # Current value in .env
    "loopfi",      # Alternative common format
    "loop",        # Shorter alternative
    "compound",    # Known working protocol as control
    "aave-v3"      # Another known working protocol
]

print("\n=== Testing Protocol Info Endpoint ===")
for protocol_id in protocol_ids:
    print(f"\nTrying protocol ID: {protocol_id}")
    try:
        response = requests.get(
            f"{DEBANK_API_BASE_URL}/v1/protocol",
            headers=headers,
            params={"id": protocol_id},
            timeout=10
        )
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Got data for {data.get('name', 'unknown')}")
            print(f"TVL: ${data.get('tvl', 0):,.2f}")
            print(f"Chain: {data.get('chain', 'unknown')}")
        else:
            print(f"ERROR: Status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")

print("\n=== Testing Top Holders Endpoint ===")
# Test with a working protocol ID (from above tests)
working_id = "compound"  # Replace with a working ID from above if needed
try:
    response = requests.get(
        f"{DEBANK_API_BASE_URL}/v1/protocol/top_holders",
        headers=headers,
        params={"id": working_id, "limit": 5},
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"SUCCESS: Got {len(data)} top holders for {working_id}")
        for i, holder in enumerate(data[:5]):
            print(f"  {i+1}. {holder[0][:10]}... : ${holder[1]:,.2f}")
    else:
        print(f"ERROR: Status code {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"EXCEPTION: {str(e)}")

print("\nAPI Testing Complete") 