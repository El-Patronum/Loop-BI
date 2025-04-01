# Getting Started with Loop-BI

This guide will help you set up and run the Loop-BI analytics platform for LoopFi.

## Prerequisites

Before you begin, make sure you have:

1. **Python 3.9+** installed on your system
2. **DeBank Pro API Key** - You need access to the DeBank Pro API
3. **Supabase Account** - Create a free account at [supabase.com](https://supabase.com)

## Quick Setup

### 1. Clone the repository and create a virtual environment

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd Loop-BI

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your environment variables

```bash
# Copy the template file
cp .env.template .env

# Edit the .env file with your credentials
# Replace the placeholders with your actual values:
# - DEBANK_ACCESS_KEY: Your DeBank Pro API key
# - SUPABASE_URL: Your Supabase project URL
# - SUPABASE_KEY: Your Supabase anon/public key
# - LOOPFI_PROTOCOL_ID: The correct DeBank protocol ID for LoopFi
```

### 4. Set up Supabase

1. Log in to your [Supabase Dashboard](https://app.supabase.com)
2. Create a new project
3. Go to the SQL Editor
4. Copy the contents of `config/database_schema.sql` and execute it in the SQL Editor
5. This will create all the necessary tables for Loop-BI

## Running the Application

### Step 1: Fetch data from DeBank

You have several options for fetching data:

```bash
# Fetch only the protocol information
python -m scripts.run_pipeline --protocol-only

# Fetch protocol information and basic user list (lighter operation)
python -m scripts.run_pipeline

# Fetch protocol information, user list, and detailed user data (heavy operation)
python -m scripts.run_pipeline --full

# Process a single user (for testing)
python -m scripts.run_pipeline --user 0x123...  # Replace with an actual Ethereum address

# Customize batch processing to manage rate limits
python -m scripts.run_pipeline --full --batch-size 5 --batch-delay 10
```

The `--batch-size` and `--batch-delay` options are particularly useful for managing DeBank's rate limit of 100 requests per second:

- `--batch-size`: Number of users to process in a batch (default: 10)
- `--batch-delay`: Seconds to wait between batches (default: 5)

For larger datasets, consider reducing the batch size and increasing the delay between batches.

### Step 2: Start the Streamlit dashboard

```bash
streamlit run app/main_app.py
```

This will launch the Streamlit server, and your browser should automatically open the Loop-BI dashboard (typically at http://localhost:8501).

## Development Workflow

1. **Data Pipeline**: Modify scripts in the `scripts/` directory to adjust data fetching
2. **Analytics**: Add or modify analytics functions in `core/analytics.py`
3. **Dashboard**: Update the Streamlit app in `app/main_app.py`

## Troubleshooting

### API Key Issues

- If you see "DeBank API key not found" errors, make sure your `.env` file contains the correct `DEBANK_ACCESS_KEY`.
- For Supabase connection errors, verify your `SUPABASE_URL` and `SUPABASE_KEY` in the `.env` file.

### Database Setup

If your Streamlit app shows "No data available yet" for all metrics:

1. Check the console output from your `run_pipeline.py` script for errors
2. Verify that data was successfully stored in your Supabase tables
3. Ensure you're using the correct LoopFi protocol ID in your `.env` file

### Rate Limiting

The DeBank API has rate limits of 100 requests per second. We've implemented rate limiting in the code to manage this, but if you still experience issues:

1. Start with `--protocol-only` to test your connection
2. Use `--user` with a single address for testing
3. For large datasets, reduce batch size (`--batch-size 5`) and increase the delay between batches (`--batch-delay 15`)
4. Check the logs for any rate limiting errors
5. For very large datasets, consider running the pipeline in multiple sessions with different subsets of users

## Next Steps

Once your basic setup is working:

1. Customize the dashboard layout and visualizations
2. Set up scheduled data collection (e.g., using cron jobs)
3. Add authentication to your dashboard if hosting publicly
4. Implement monitoring for the data collection process

For more details, see the [Development Plan](Development_Plan) and [README.md](README.md). 