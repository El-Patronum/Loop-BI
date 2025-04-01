-- Schema for historical data tracking in Loop-BI

-- Table for tracking daily protocol-level metrics
CREATE TABLE IF NOT EXISTS historical_protocol_metrics (
    snapshot_date DATE NOT NULL,
    chain_id TEXT NOT NULL,
    tvl DECIMAL,
    utilization_rate DECIMAL,
    total_users INTEGER,
    new_users_24h INTEGER,
    avg_deposit_size DECIMAL,
    avg_loop_factor DECIMAL,
    PRIMARY KEY (snapshot_date, chain_id)
);

-- Table for tracking daily asset-specific metrics
CREATE TABLE IF NOT EXISTS historical_asset_metrics (
    snapshot_date DATE NOT NULL,
    chain_id TEXT NOT NULL,
    asset_symbol TEXT NOT NULL,
    role TEXT NOT NULL, -- 'lending' or 'looping'
    total_value DECIMAL,
    user_count INTEGER,
    avg_position_size DECIMAL,
    PRIMARY KEY (snapshot_date, chain_id, asset_symbol, role)
);

-- Table for tracking user segment metrics over time
CREATE TABLE IF NOT EXISTS historical_user_segments (
    snapshot_date DATE NOT NULL,
    chain_id TEXT NOT NULL,
    segment TEXT NOT NULL, -- 'Small', 'Medium', 'Large', 'Whale'
    user_count INTEGER,
    total_value DECIMAL,
    percentage_of_tvl DECIMAL,
    PRIMARY KEY (snapshot_date, chain_id, segment)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_protocol_metrics_date ON historical_protocol_metrics(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_protocol_metrics_chain ON historical_protocol_metrics(chain_id);

CREATE INDEX IF NOT EXISTS idx_asset_metrics_date ON historical_asset_metrics(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_asset_metrics_chain ON historical_asset_metrics(chain_id);
CREATE INDEX IF NOT EXISTS idx_asset_metrics_symbol ON historical_asset_metrics(asset_symbol);
CREATE INDEX IF NOT EXISTS idx_asset_metrics_role ON historical_asset_metrics(role);

CREATE INDEX IF NOT EXISTS idx_user_segments_date ON historical_user_segments(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_user_segments_chain ON historical_user_segments(chain_id);
CREATE INDEX IF NOT EXISTS idx_user_segments_segment ON historical_user_segments(segment); 