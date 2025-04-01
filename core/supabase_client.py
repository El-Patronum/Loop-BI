"""
Supabase Client Module for Loop-BI

This module provides a centralized connection to Supabase database
for the Loop-BI application.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def get_supabase_client() -> Client:
    """
    Initialize and return a Supabase client instance.
    
    Returns:
        Client: A configured Supabase client.
    
    Raises:
        ValueError: If Supabase URL or key is not found in environment variables.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase credentials not found. Ensure SUPABASE_URL and SUPABASE_KEY "
            "are set in your .env file."
        )
    
    return create_client(supabase_url, supabase_key)


def initialize_tables(client: Client) -> None:
    """
    Initialize required tables in Supabase if they don't exist.
    
    Args:
        client (Client): A configured Supabase client.
    """
    # This function would contain the table creation SQL statements
    # For now, this is a placeholder - we'll implement the actual table creation
    # once we finalize the schema design
    pass 