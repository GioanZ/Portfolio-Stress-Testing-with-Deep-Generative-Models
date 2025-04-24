# custom_libraries/utils.py

"""
Copyright:
    Portfolio Stress Testing with Deep Generative Models
    github.com/GioanZ
Disclaimer:
    This software is for research and educational purposes only.
    It is NOT intended for actual financial decision-making or investment strategies.
    The authors assume no liability for any losses or damages arising from the use
    of this code. Users should conduct their own due diligence before making financial
    decisions.

    This project utilizes deep generative models to simulate financial stress testing.
    The models are trained on historical market and macroeconomic data, but all results
    should be interpreted with caution.
"""

import os
import pandas as pd

from tensorflow.keras.models import load_model


def save_models(encoder, decoder, cvae, generator, critic, folder_name="models"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    encoder.save(f"{folder_name}/encoder_model.keras")
    decoder.save(f"{folder_name}/decoder_model.keras")
    cvae.save(f"{folder_name}/cvae_model.keras")
    generator.save(f"{folder_name}/generator_model.keras")
    critic.save(f"{folder_name}/critic_model.keras")


def load_models(custom_objects, folder_name="models"):
    # Load the models with the custom layer
    encoder = load_model(
        f"{folder_name}/encoder_model.keras", custom_objects=custom_objects
    )
    decoder = load_model(
        f"{folder_name}/decoder_model.keras", custom_objects=custom_objects
    )
    cvae = load_model(f"{folder_name}/cvae_model.keras", custom_objects=custom_objects)
    generator = load_model(
        f"{folder_name}/generator_model.keras", custom_objects=custom_objects
    )
    critic = load_model(
        f"{folder_name}/critic_model.keras", custom_objects=custom_objects
    )

    return encoder, decoder, cvae, generator, critic


def rename_col_yf(df, ticker, ticker_ren, name_col="Close"):
    if isinstance(df.columns, pd.MultiIndex):
        return df[(name_col, ticker)].rename(ticker_ren)

    return df[name_col].rename(ticker_ren)
