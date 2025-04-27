# DATASETS README

This project requires two data files to be manually provided inside the `datasets` folder:

1. `key_fred.txt`
  - Purpose: Contains your personal [FRED API Key](https://fred.stlouisfed.org/docs/api/fred/) required to download macroeconomic data.
  - How to obtain:
    - Register for a free account on the Federal Reserve Economic Data (FRED) website.
    - Generate an API key from your FRED account settings.
    - Save the API key as plain text in a file named `key_fred.txt` (no spaces, no new lines).
  - Where to place:
    - `datasets/key_fred.txt`

2. `portfolio.csv`
  - Purpose: Defines the list of tickers and associated portfolio weights.
  - How to obtain:
    - Download the [Berkshire Hathaway Q4 2019 portfolio](https://valuesider.com/guru/warren-buffett-berkshire-hathaway/portfolio/2019/4?sort=-percent_portfolio&sells_page=1&page=1)
    - Filters only tickers with sufficient historical data (2004–2021).
    - Ensure the CSV file contains at least two columns: `ticker` and `weight` (weights expressed as percentages).
  - Where to place:
    - `datasets/portfolio.csv`

Other data:
- Market data (daily prices, SP500 returns, VIX, FX rates) and macroeconomic indicators are automatically downloaded by the script `download_data.py` using [Yahoo Finance API](https://developer.yahoo.com/api/) and [FRED API](https://fred.stlouisfed.org/docs/api/fred/).
- No manual download of historical price series or macroeconomic series is needed.

Folder structure:
```lua
YOUR_FOLDER
├── datasets
│   ├── key_fred.txt
│   ├── portfolio.csv
│   └── README_data.md
```