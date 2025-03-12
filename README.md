# Portfolio Stress Testing

## Overview

This project implements **Portfolio Stress Testing** using **deep generative models** to assess portfolio risk under extreme market conditions. It combines:

- **Conditional Variational Autoencoder (cVAE)** to learn a latent representation of financial returns conditioned on macroeconomic indicators.
- **Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)** to generate synthetic financial return scenarios, with a strong emphasis on **negative return roll scenarios**.

The goal is to measure **Value at Risk (VaR)** and **Expected Shortfall (ES)** to analyze the portfolio's behavior under stress.

---

## Project Workflow

### 1. Data Acquisition & Preprocessing
- **Portfolio** used for analysis would be [Warren Buffett’s Q4 2019 holdings](https://valuesider.com/guru/warren-buffett-berkshire-hathaway/portfolio/2019/4?sort=-percent_portfolio&sells_page=1&page=1).
- **Market Data** is collected from [Yahoo Finance](https://developer.yahoo.com/api/).
- **Macroeconomic Indicators** are obtained from [FRED API](https://fred.stlouisfed.org/docs/api/fred/), including:
  - Federal Funds Rate
  - Inflation
  - VIX Index
  - Unemployment Rate
  - Exchange Rates
  - **5-day rolling worst-case return (`returns_sp500_roll_5`)**, which emphasizes **negative market scenarios in model training**.

- **Data Preprocessing:**
  - **Log returns** are computed for financial assets.
  - **Shifting of macroeconomic data:** To prevent lookahead bias, **macroeconomic indicators are shifted forward by one day**.
  - **Feature scaling using StandardScaler**.
  - **Additional shift for `returns_sp500_roll_5`**:  
    - On day X, the final value represents **the worst return observed over the 5-day window from X-1 to X+3**.
    - This ensures that the GAN model is **heavily conditioned on the worst observed market conditions**.

---

### 2. Model Training
#### Conditional Variational Autoencoder (cVAE)
- **Encoder:** compresses financial returns into a latent space.
- **Decoder:** reconstructs returns based on macroeconomic conditions.
- **Loss Function:**
  - **Reconstruction Loss (MSE + Tail Sensitivity):** optimized to reproduce extreme scenarios accurately.
  - **KL Divergence Loss:** regularizes the latent space distribution.

#### Conditional WGAN-GP
- **Generator:** learns a **realistic latent representation of financial returns**, conditioned on macroeconomic indicators.
- **Critic:** evaluates the quality of generated samples.
- **Gradient Penalty:** enforces **Lipschitz continuity** for stable training.
- **Emphasis on worst-case scenarios:**  
  - The model **assigns higher weight to negative values in `returns_sp500_roll_5`**.
  - **Worst-Case Quantile Training:** The generator learns to prioritize **the worst observed market downturns**.

---

### 3. Scenario Generation & Stress Testing
- The trained model generates **NUM_SCENARIOS** synthetic return paths.
- Stress testing modifies only **specific** macroeconomic inputs while keeping others unchanged.
- Portfolio risk is measured through:
  - **Value at Risk (VaR)** (5% worst-case simulated returns)
  - **Expected Shortfall (ES)**

#### Example: Custom Stress Scenario

```python
stress_values = {
    "returns_sp500_roll_5": -0.2,
    "vix": 99.0,
    "unemployment": 20.0,
}
```

- **Only specified variables are altered.**
- **All other real-world macroeconomic indicators remain unchanged.**
- The model then **recalculates VaR and ES based on these new conditions**.

---

### 4. Backtesting & Risk Analysis
- The model is validated against real market data using two approaches:
  - **Unbiased Backtesting:** Uses historical macroeconomic values.
  - **Stressed Scenario Backtesting:** Manually alters selected macroeconomic factors.
- **Performance Metrics:**
  - **Kolmogorov-Smirnov (KS) Test**
  - **Jensen-Shannon Divergence (JSD)**
  - **Earth Mover’s Distance (EMD)**

---

## Project Components
### 1. Preprocessing Modules
- `preprocess_data.py`: Handles data acquisition, missing values, and feature engineering.
- **`returns_sp500_roll_5`** is crucial for generating crisis scenarios.

### 2. Model Implementation
- `custom_vae.py`: Implements **cVAE** (encoder, decoder, loss layers).
- `custom_wgangp.py`: Defines the **Conditional WGAN-GP** (generator, critic, loss functions).

### 3. Stress Testing & Evaluation
- `stress_backtesting.py`: Simulates extreme market conditions.
- `metrics_validation.py`: Computes **VaR, ES, and divergence scores**.
- `utils_stress_testing.py`: Validates synthetic scenarios.

### 4. Visualization & Analysis
- `utils_plot.py`: Provides plotting utilities for **latent space clustering, risk distributions, and backtesting performance**.

---

## Installation & GPU Setup

To install the required dependencies, run:

```bash
pip install numpy pandas matplotlib seaborn yfinance fredapi tensorflow scikit-learn scipy
```

---

## Conclusion
This project provides an **advanced framework for Portfolio Stress Testing** using **deep generative models**.  
The approach **prioritizes worst-case scenarios**, leveraging `returns_sp500_roll_5` to condition the models on **extreme negative events**, making it particularly useful for **risk management and volatility assessment**.

