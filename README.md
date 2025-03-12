# Portfolio Stress Testing

## Overview

This project implements Portfolio Stress Testing using deep generative models to assess portfolio risk under extreme market conditions. It combines:

- **Conditional Variational Autoencoder (cVAE)** to learn a latent representation of financial returns conditioned on macroeconomic indicators.
- **Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)** to generate synthetic financial return scenarios, with a strong emphasis on negative return roll scenarios.

The goal is to measure Value at Risk (VaR) and Expected Shortfall (ES) to analyze the portfolio's behavior under stress.

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
  - 5-day rolling worst-case return (`returns_sp500_roll_5`), which emphasizes negative market scenarios in model training.

- **Data Preprocessing:**
  - Log returns are computed for financial assets.
  - Shifting of macroeconomic data: To prevent lookahead bias, macroeconomic indicators are shifted forward by one day.
  - Feature scaling using StandardScaler.
  - Additional shift for `returns_sp500_roll_5`:  
    - On day X, the final value represents the worst return observed over the 5-day window from X-1 to X+3.
    - This ensures that the GAN model is heavily conditioned on the worst observed market conditions.

### 2. Model Training
#### Conditional Variational Autoencoder (cVAE)
- **Encoder:** compresses financial returns into a latent space.
- **Decoder:** reconstructs returns based on macroeconomic conditions.
- **Loss Function:**
  - **Reconstruction Loss (MSE + Tail Sensitivity):** optimized to reproduce extreme scenarios accurately.
  - **KL Divergence Loss:** regularizes the latent space distribution.

#### Conditional WGAN-GP
- **Generator:** learns a realistic latent representation of financial returns, conditioned on macroeconomic indicators.
- **Critic:** evaluates the quality of generated samples.
- **Gradient Penalty:** enforces Lipschitz continuity for stable training.
- **Emphasis on worst-case scenarios:**  
  - The model assigns higher weight to negative values in `returns_sp500_roll_5`.
  - **Worst-Case Quantile Training:** The generator learns to prioritize the worst observed market downturns.

### 3. Scenario Generation & Stress Testing
- The trained model generates `NUM_SCENARIOS` synthetic return paths.
- Stress testing modifies only specific macroeconomic inputs while keeping others unchanged.
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

- Only specified variables are altered.
- All other real-world macroeconomic indicators remain unchanged.
- The model then recalculates VaR and ES based on these new conditions.

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
### 1. GPU Setup & Optimization
- `custom_libraries/start_gpu.py`: Configures GPU memory allocation and limits TensorFlow GPU usage.

### 2. Preprocessing Modules
- `custom_libraries/download_data.py`: Handles data collection from Yahoo Finance (market data, VIX, FX, S&P500) and FRED API (inflation, unemployment, interest rates).
- `custom_libraries/preprocess_data.py`: Processes and scales the collected data, handling missing values and computing `returns_sp500_roll_5` for worst-case scenario modeling.

### 3. Model Implementation
- `custom_libraries/custom_vae.py`: Implements cVAE (encoder, decoder, loss layers).
- `custom_libraries/custom_wgangp.py`: Defines the Conditional WGAN-GP (generator, critic, loss functions).
- `custom_libraries/custome_layer.py`: Defines custom layers for the VAE, including Sampling, KL Divergence, and Reconstruction Loss layers.
- `custom_libraries/utils.py`: Handles model saving/loading and data formatting functions.

### 4. Stress Testing & Evaluation
- `custom_libraries/stress_backtesting.py`: Simulates extreme market conditions.
- `custom_libraries/metrics_validation.py`: Computes VaR, ES, and divergence scores.
- `custom_libraries/utils_stress_testing.py`: Validates synthetic scenarios.

### 5. Visualization & Analysis
- `custom_libraries/utils_plot.py`: Provides plotting utilities for latent space clustering, risk distributions, and backtesting performance.

---

## Explanation of Key Parameters

### General Settings
- `SEED_RANDOM = 101`  
  Ensures reproducibility across runs. By fixing a random seed, operations like weight initialization, data shuffling, and noise generation will produce consistent results.

- `LOAD_MODEL = False`  
  Determines whether to load pre-trained models from disk. If `False`, training starts from scratch. If `True`, saved models are loaded for evaluation or scenario generation.

- `LOGGING_ENABLED = False`  
  Enables detailed logging during training. This can be useful for debugging but may be turned off for faster execution.

- `USE_GPU = True`  
  Enables GPU acceleration for training, significantly improving computational efficiency.

- `FOLDER_MODELS = "models"`  
  Directory where trained models are saved or loaded from.

### Model Training Parameters
- `EPOCHS_WGAN = 1000`  
  Number of training epochs for the Conditional WGAN-GP. A higher value is chosen to allow adversarial training to converge properly.

- `EPOCHS_CVAE = 500`  
  Number of training epochs for the Conditional VAE. This is lower than for the WGAN-GP, as VAEs generally converge faster.

- `LATENT_DIM = 8`  
  The size of the latent space in both models. This value ensures a rich representation of return distributions while keeping model complexity manageable.

- `NOISE_DIM = 10`  
  The dimension of the noise vector input for the WGAN-GP generator ensures diverse scenario generation.

- `WGANGP_PATIENCE = 400`  
  Early stopping patience for WGAN-GP training. If no validation improvement is observed for 400 epochs, training stops to prevent overfitting and save resources.

- `NUM_SCENARIOS = 10000`  
  The number of synthetic market scenarios generated for stress testing. A large sample size improves reliability in risk metrics.

### Backtesting & Data Periods
- `START_DATE = "2004-01-01"`  
  The starting point for the dataset used in the analysis. 2004 is selected to ensure sufficient data history without using the very early part of the series.

- `START_BACKTEST = "2020-02-20"`  
  The backtesting start date was chosen just before the COVID-19 downturn. This ensures that stress testing captures how the market behaved during a severe crisis.

- `END_BACKTEST = "2020-05-01"`  
  The backtesting end date covers the COVID-19 market crash. This allows evaluation of portfolio behavior in extreme conditions.

- `END_DATE = "2021-01-01"`  
  The final date set provides a post-crisis context and allowing for a broader analysis of market recovery.

---

## Installation

To install the required dependencies, run:

```bash
pip install numpy pandas matplotlib seaborn yfinance fredapi tensorflow scikit-learn scipy
```

---

## Execution

### Set Up FRED API Key
This project requires an **API key** from the [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html) to access macroeconomic data.  

#### Steps to Configure
1. Register for an API key on [FRED’s website](https://fred.stlouisfed.org/docs/api/api_key.html).  
2. Create a text file named `fred_api_key.txt` in the root directory of the project.  
3. Save your API key inside the file without any extra characters or spaces.

Example:  
```txt
your_fred_api_key_here
```
Now you are ready to use the code! The script `download_data.py` will automatically read the key from this file.

---

## Conclusion
This project provides an advanced framework for Portfolio Stress Testing using deep generative models.  
The approach prioritizes worst-case scenarios, leveraging `returns_sp500_roll_5` to condition the models on extreme negative events, making it particularly useful for risk management and volatility assessment.