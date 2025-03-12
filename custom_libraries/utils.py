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
