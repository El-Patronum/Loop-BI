# Loop-BI: LoopFi Analytics Platform

A comprehensive analytics platform for the LoopFi DeFi protocol, providing insights into user behavior, asset distribution, risk analysis, and more.

## Project Overview

Loop-BI fetches data from the DeBank API, stores it in Supabase, and presents analytics via a Streamlit application. It offers a modular, scalable, and extensible solution that can be hosted for external access.

## Features

- **Protocol Insights**: TVL, utilization rate, user count, and more
- **User Analytics**: Net worth distribution, average deposit size
- **Asset Distribution**: Most used assets, chain distribution
- **Risk Analysis**: Loop factor distribution, leverage metrics
- **Cross-Protocol Insights**: Other protocols used by LoopFi users
- **Multi-chain Support**: Analytics across all chains supported by LoopFi

## Setup & Installation

### Prerequisites

- Python 3.9+ 
- A Supabase account and project
- DeBank Pro API access key

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Loop-BI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file by copying the template:
   ```bash
   cp .env.template .env
   ```

5. Edit the `.env` file with your credentials:
   ```
   DEBANK_ACCESS_KEY="your_debank_api_key"
   SUPABASE_URL="your_supabase_url"
   SUPABASE_KEY="your_supabase_anon_key"
   LOOPFI_PROTOCOL_ID="correct_loopfi_protocol_id"
   ```

6. Set up the Supabase database schema:
   - Log in to your Supabase dashboard
   - Navigate to the SQL Editor
   - Execute the SQL in `config/database_schema.sql`

## Usage

### Data Collection

Run the data collection script to fetch and store data from DeBank:

```bash
cd Loop-BI
python -m scripts.store_data
```

Note: For production use, consider scheduling this script to run periodically.

### Running the Dashboard

Start the Streamlit app with:

```bash
cd Loop-BI
streamlit run app/main_app.py
```

The dashboard will be available at http://localhost:8501 by default.

## Project Structure

```
Loop-BI/
├── app/                  # Streamlit application
│   └── main_app.py       # Main dashboard app
├── core/                 # Core functionality
│   └── supabase_client.py  # Supabase connection handling
├── scripts/              # Data processing scripts
│   ├── fetch_debank_data.py  # DeBank API interaction
│   └── store_data.py     # Data storage logic
├── config/               # Configuration files
│   └── database_schema.sql  # Supabase table definitions
├── .env                  # Environment variables (not in Git)
├── .env.template         # Template for environment variables
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Development Plan

See the [Development Plan](Development_Plan) for detailed information on the project's implementation phases and roadmap.

## License

[MIT License](LICENSE)

## Contributors

Loop-BI is maintained by the LoopFi team and contributors. 