# Portfolio Stress Testing

## Overview

This project implements portfolio stress testing using advanced deep generative models. It combines a Conditional Variational Autoencoder (cVAE) and a Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate synthetic market scenarios conditioned on macroeconomic indicators. 

The goal is to assess portfolio risk under extreme market conditions by simulating crisis scenarios that traditional models might miss.

## Project Components

- **Data Acquisition and Preprocessing:**  
  Modules are provided for downloading market data via Yahoo Finance and macroeconomic data from the FRED API (and additional sources). Preprocessing functions handle scaling, forward-filling missing values, and aligning time series.

- **Model Implementation:**  
  - Conditional VAE: Encodes asset return distributions into a latent space and decodes conditional on macroeconomic indicators using custom layers (SamplingLayer, ReconstructionLossLayer, and KLDivergenceLayer).  
  - Conditional WGAN-GP: Generates latent representations conditioned on macroeconomic indicators through adversarial training, with gradient penalty enforcing Lipschitz continuity.

- **Stress Testing and Backtesting:**  
  Functions for generating synthetic scenarios for individual dates, performing rolling backtests, and evaluating risk metrics at both portfolio and individual asset levels. Risk is measured using Value at Risk (VaR) and Expected Shortfall (ES).

- **Evaluation Metrics:**  
  The project assesses model performance using Earth Mover’s Distance (EMD), Jensen-Shannon Divergence (JSD), and the Kolmogorov-Smirnov (KS) test.

- **Visualization:**  
  A suite of plotting utilities is available to visualize training progress, risk metrics over time, latent space clustering, and scenario distributions.

## Data Sources

- **Market Data:**  
  Downloaded via the [Yahoo Finance API](https://developer.yahoo.com/api/), this data includes asset prices and trading volumes.

- **Macroeconomic Indicators:**  
  Sourced from the Federal Reserve Economic Data ([FRED API](https://fred.stlouisfed.org/docs/api/fred/)) for variables such as inflation, federal funds rate, unemployment rate, and treasury rates. Additional indicators like the VIX and exchange rates (EUR/USD) are obtained from Yahoo Finance.

- **Portfolio:**  
  The analysis uses a portfolio based on [Warren Buffett’s Q4 2019 holdings](https://valuesider.com/guru/warren-buffett-berkshire-hathaway/portfolio/2019/4?sort=-percent_portfolio&sells_page=1&page=1).

## Installation

To install the required dependencies, run:

```bash
pip install numpy pandas matplotlib seaborn yfinance fredapi tensorflow scikit-learn scipy
```