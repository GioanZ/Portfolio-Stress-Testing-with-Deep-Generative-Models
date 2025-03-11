from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.callbacks import EarlyStopping

from custom_libraries.custome_layer import (
    KLDivergenceLayer,
    SamplingLayer,
    ReconstructionLossLayer,
)


def build_conditional_vae(input_dim, macro_dim, intermediate_dim=256, latent_dim=2):
    # Encoder
    x_input = layers.Input(shape=(input_dim,), name="returns_input")
    h = layers.Dense(intermediate_dim, activation="relu")(x_input)
    h = layers.Dropout(0.1)(h)
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
    z_mean = KLDivergenceLayer()([z_mean, z_log_var])
    z = SamplingLayer(name="z")([z_mean, z_log_var])
    encoder = Model(x_input, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    cond_input = layers.Input(shape=(macro_dim,), name="macro_input")
    decoder_input = layers.Concatenate(name="concat_decoder")([z, cond_input])
    d = layers.Dense(intermediate_dim, activation="relu")(decoder_input)
    d = layers.Dropout(0.1)(d)
    d = layers.Dense(intermediate_dim // 2, activation="relu")(d)
    outputs = layers.Dense(input_dim, activation="linear")(d)
    decoder = Model([z, cond_input], outputs, name="decoder")

    # Conditional VAE Model
    vae_output = decoder([encoder(x_input)[2], cond_input])
    x_pred = ReconstructionLossLayer()([x_input, vae_output])
    cvae = Model([x_input, cond_input], x_pred, name="cvae")

    # Compile the model
    cvae.compile(optimizer="adam")

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    return encoder, decoder, cvae, early_stop
