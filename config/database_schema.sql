-- Database schema for Loop-BI Supabase tables

-- Table for storing protocol information
CREATE TABLE IF NOT EXISTS debank_protocols (
    id TEXT,
    chain_id TEXT,
    name TEXT,
    site_url TEXT,
    logo_url TEXT,
    has_supported_portfolio BOOLEAN,
    tvl DECIMAL,
    last_updated_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    PRIMARY KEY (id, chain_id)
);

-- Table for storing LoopFi users
CREATE TABLE IF NOT EXISTS debank_loopfi_users (
    user_address TEXT,
    chain_id TEXT,
    loopfi_usd_value DECIMAL,
    total_net_worth_usd DECIMAL,
    first_seen_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    last_updated_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    PRIMARY KEY (user_address, chain_id)
);

-- Table for storing user positions in LoopFi
CREATE TABLE IF NOT EXISTS debank_user_loopfi_positions (
    position_id SERIAL,
    user_address TEXT,
    chain_id TEXT,
    portfolio_item_name TEXT,
    asset_usd_value DECIMAL,
    debt_usd_value DECIMAL,
    net_usd_value DECIMAL,
    supplied_tokens JSONB,
    debt_tokens JSONB,
    rewards_tokens JSONB,
    asset_symbol TEXT,
    debt_symbol TEXT,
    entry_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    last_updated_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    PRIMARY KEY (position_id),
    UNIQUE (user_address, portfolio_item_name, chain_id),
    FOREIGN KEY (user_address, chain_id) REFERENCES debank_loopfi_users(user_address, chain_id)
);

-- Table for storing user token holdings
CREATE TABLE IF NOT EXISTS debank_user_token_holdings (
    holding_id SERIAL PRIMARY KEY,
    user_address TEXT,
    chain_id TEXT,
    token_address TEXT,
    token_symbol TEXT,
    amount DECIMAL,
    usd_value DECIMAL,
    last_updated_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    FOREIGN KEY (user_address, chain_id) REFERENCES debank_loopfi_users(user_address, chain_id)
);

-- Table for storing user protocol interactions (other than LoopFi)
CREATE TABLE IF NOT EXISTS debank_user_protocol_interactions (
    interaction_id SERIAL PRIMARY KEY,
    user_address TEXT,
    chain_id TEXT,
    protocol_id TEXT,
    protocol_name TEXT,
    net_usd_value DECIMAL,
    last_updated_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    FOREIGN KEY (user_address, chain_id) REFERENCES debank_loopfi_users(user_address, chain_id)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_protocols_chain ON debank_protocols(chain_id);
CREATE INDEX IF NOT EXISTS idx_users_chain ON debank_loopfi_users(chain_id);

CREATE INDEX IF NOT EXISTS idx_positions_user_chain ON debank_user_loopfi_positions(user_address, chain_id);
CREATE INDEX IF NOT EXISTS idx_positions_timestamps ON debank_user_loopfi_positions(entry_timestamp, last_updated_timestamp);
CREATE INDEX IF NOT EXISTS idx_positions_chain ON debank_user_loopfi_positions(chain_id);
CREATE INDEX IF NOT EXISTS idx_positions_asset ON debank_user_loopfi_positions(asset_symbol);
CREATE INDEX IF NOT EXISTS idx_positions_debt ON debank_user_loopfi_positions(debt_symbol);

CREATE INDEX IF NOT EXISTS idx_holdings_user_chain ON debank_user_token_holdings(user_address, chain_id);
CREATE INDEX IF NOT EXISTS idx_holdings_chain ON debank_user_token_holdings(chain_id);
CREATE INDEX IF NOT EXISTS idx_holdings_chain_token ON debank_user_token_holdings(chain_id, token_symbol);

CREATE INDEX IF NOT EXISTS idx_interactions_user_chain ON debank_user_protocol_interactions(user_address, chain_id);
CREATE INDEX IF NOT EXISTS idx_interactions_chain ON debank_user_protocol_interactions(chain_id);
CREATE INDEX IF NOT EXISTS idx_interactions_protocol ON debank_user_protocol_interactions(protocol_id);