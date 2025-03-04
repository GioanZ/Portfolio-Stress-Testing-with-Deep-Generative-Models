# Portfolio Stress Testing

## Overview
This project explores portfolio stress testing using deep generative models, specifically:
- Conditional Variational Autoencoders (cVAE)
- Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)

The goal is to generate synthetic market scenarios conditioned on macroeconomic indicators and evaluate the risk exposure of a financial portfolio. Risk is assessed using:
- Value at Risk (VaR)
- Expected Shortfall (ES)
- Distance metrics such as Earth Mover's Distance, Jensen-Shannon Divergence, and the Kolmogorov-Smirnov test

### Why Stress Testing?
Financial markets experience sudden crashes, such as the 2008 financial crisis or the COVID-19 pandemic. Traditional risk models assume normally distributed returns, which fail to capture extreme events. Generative models allow for simulating realistic crises and evaluating potential losses.

## Data

### Market Data
- Source: [Yahoo Finance API](https://developer.yahoo.com/api/)
- Portfolio: [Warren Buffett's 2019 Q4 portfolio](https://valuesider.com/guru/warren-buffett-berkshire-hathaway/portfolio/2019/4?sort=-percent_portfolio&sells_page=1&page=1)
- Timeframe: Data is collected from 2000 to the end of 2024, but only the period from 2018 onward is considered, as many assets did not exist before that time

### Macroeconomic Indicators
- Source: Federal Reserve Economic Data ([FRED API](https://fred.stlouisfed.org/docs/api/fred/))
   - Variables:
      - Inflation Rate (CPIAUCSL)
      - Federal Funds Rate (FEDFUNDS)
      - Unemployment Rate (UNRATE)
      - 3-Month Treasury Rate (DGS3MO)
      - 10-Year Treasury Rate (DGS10)
- Source: Yahoo Finance
   - Variables:
      - VIX Index (from Yahoo Finance)
- Preprocessing: Forward-filled missing values and standardized features.

## Goals

1. Train a Conditional Variational Autoencoder (cVAE)
   - Encode asset return distributions into a latent space
   - Generate synthetic return scenarios based on macroeconomic conditions

2. Train a Conditional WGAN-GP
   - Generate synthetic latent representations conditioned on macroeconomic indicators
   - Ensure realistic synthetic distributions through adversarial training with gradient penalty

3. Perform Portfolio Stress Testing
   - Generate 1000 synthetic market scenarios
   - Evaluate portfolio risk under stress using Value at Risk (VaR) and Expected Shortfall (ES)
   - Compare synthetic and historical distributions with Earth Mover's Distance (EMD), Jensen-Shannon Divergence (JSD), and the Kolmogorov-Smirnov test

4. Cluster Extreme Scenarios in Latent Space
   - Use K-Means clustering to detect extreme financial regimes
   - Identify outliers representing market crashes

## Running the Code

### Requirements
To run the code, install the required Python packages:
```bash
pip install numpy pandas matplotlib yfinance fredapi tensorflow scikit-learn scipy
```