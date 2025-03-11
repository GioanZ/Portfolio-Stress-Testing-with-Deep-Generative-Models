import logging
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_conditional_generator(noise_dim, macro_dim, latent_dim):
    noise_input = layers.Input(shape=(noise_dim,), name="noise_input")
    macro_input = layers.Input(shape=(macro_dim,), name="macro_input_gen")
    x = layers.Concatenate(name="concat_gen")([noise_input, macro_input])
    x = layers.Dense(256, activation="relu")(x)
    #    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation="relu")(x)
    #    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    output_latent = layers.Dense(latent_dim, name="latent_generated")(x)
    model = Model(
        [noise_input, macro_input], output_latent, name="conditional_generator"
    )
    return model


def build_conditional_critic(latent_dim, macro_dim):
    latent_input = layers.Input(shape=(latent_dim,), name="latent_input")
    macro_input = layers.Input(shape=(macro_dim,), name="macro_input_critic")
    x = layers.Concatenate(name="concat_critic")([latent_input, macro_input])
    x = layers.Dense(256, activation="relu")(x)
    #    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation="relu")(x)
    #    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    output_score = layers.Dense(1, name="critic_score")(x)
    model = Model([latent_input, macro_input], output_score, name="conditional_critic")
    return model


def gradient_penalty_cond(critic, real_latent, fake_latent, macro, batch_size):
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    interpolated_latent = alpha * real_latent + (1 - alpha) * fake_latent
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated_latent)
        interpolated_output = critic([interpolated_latent, macro])
    grads = gp_tape.gradient(interpolated_output, [interpolated_latent])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


def train_conditional_wgan_gp(
    generator,
    critic,
    real_latent,
    real_input,
    input_test,
    noise_dim,
    batch_size=32,
    epochs=100,
    critic_iterations=5,
    lambda_gp=20.0,
    important_index=None,  # name of the input characteristics column that should be most important
    validation_split=0.3,  # fraction of data to use for validation
    worst_case_quantile=0.2,  # e.g., worst 10% cases
    worst_case_weight=3.0,  # additional weight for worst-case samples in validation
    patience=100,  # epochs with no improvement before early stopping
):
    """
    Train a conditional WGAN-GP with early stopping and weighted validation
    """
    input_test_indices = {col: idx for idx, col in enumerate(input_test.columns)}
    if important_index is None:
        raise ValueError(
            "put the name of the input characteristics column that should be most important"
        )
    imp_idx = input_test_indices[important_index]

    gen_optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-5, beta_1=0.5, beta_2=0.9, clipnorm=1.0
    )
    critic_optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-5, beta_1=0.5, beta_2=0.9, clipnorm=1.0
    )

    # Split the data into training and validation sets
    num_samples = real_input.shape[0]
    n_val = int(validation_split * num_samples)
    train_latent = real_latent[:-n_val]
    train_input = real_input[:-n_val]
    val_latent = real_latent[-n_val:]
    val_input = real_input[-n_val:]

    # Create the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_latent.astype("float32"), train_input.astype("float32"))
    )
    train_dataset = train_dataset.shuffle(1024).batch(batch_size)

    # Initialize lists for tracking losses and early stopping
    gen_losses = []
    critic_losses = []
    val_losses = []
    best_val_loss = np.inf
    epochs_no_improve = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_gen_losses = []
        epoch_critic_losses = []

        for latent_batch, input_batch in train_dataset:
            current_batch_size = tf.shape(input_batch)[0]
            # Update critic for critic_iterations steps
            for _ in range(critic_iterations):
                noise = tf.random.normal([current_batch_size, noise_dim])
                with tf.GradientTape() as tape:
                    fake_latent = generator([noise, input_batch])
                    critic_real = critic([latent_batch, input_batch])
                    critic_fake = critic([fake_latent, input_batch])
                    gp = gradient_penalty_cond(
                        critic,
                        latent_batch,
                        fake_latent,
                        input_batch,
                        current_batch_size,
                    )
                    critic_loss = (
                        tf.reduce_mean(critic_fake)
                        - tf.reduce_mean(critic_real)
                        + lambda_gp * gp
                    )
                grads = tape.gradient(critic_loss, critic.trainable_variables)
                critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))
                epoch_critic_losses.append(critic_loss.numpy())

            # Update generator
            noise = tf.random.normal([current_batch_size, noise_dim])
            with tf.GradientTape() as tape:
                fake_latent = generator([noise, input_batch])
                gen_loss = -tf.reduce_mean(critic([fake_latent, input_batch]))
            grads = tape.gradient(gen_loss, generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
            epoch_gen_losses.append(gen_loss.numpy())

        avg_gen_loss = np.mean(epoch_gen_losses)
        avg_critic_loss = np.mean(epoch_critic_losses)
        total_train_loss = avg_gen_loss + avg_critic_loss

        gen_losses.append(avg_gen_loss)
        critic_losses.append(avg_critic_loss)

        # --- Validation Evaluation ---
        # Convert validation input to tensor
        val_input_tensor = tf.convert_to_tensor(val_input, dtype=tf.float32)
        # Generate noise for validation samples
        val_noise = tf.random.normal([n_val, noise_dim])
        # Compute generator loss on the entire validation set
        val_fake_latent = generator([val_noise, val_input_tensor])
        val_gen_loss_all = -tf.reduce_mean(
            critic([val_fake_latent, val_input_tensor])
        ).numpy()

        # Identify worst-case samples in the validation set using the SP500 indicator
        sp500_vals = val_input[:, imp_idx]
        threshold = np.percentile(sp500_vals, worst_case_quantile * 100)
        worst_mask = sp500_vals < threshold
        if np.sum(worst_mask) > 0:
            worst_input = val_input[worst_mask]
            worst_input_tensor = tf.convert_to_tensor(worst_input, dtype=tf.float32)
            worst_noise = tf.random.normal([worst_input.shape[0], noise_dim])
            worst_fake_latent = generator([worst_noise, worst_input_tensor])
            val_gen_loss_worst = -tf.reduce_mean(
                critic([worst_fake_latent, worst_input_tensor])
            ).numpy()
        else:
            val_gen_loss_worst = 0.0

        # Combined validation loss: weighted average of overall and worst-case losses
        combined_val_loss = (
            val_gen_loss_all + worst_case_weight * val_gen_loss_worst
        ) / (1.0 + worst_case_weight)
        val_losses.append(combined_val_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Critic Loss = {avg_critic_loss:.4f}, Gen Loss = {avg_gen_loss:.4f}, Val Loss = {combined_val_loss:.4f}"
            )
        logging.info(f"Epoch {epoch}")

        # Early stopping check on the validation loss
        if combined_val_loss < best_val_loss:
            best_val_loss = combined_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    logging.info(f"Done!")

    return generator, gen_losses, critic_losses, val_losses
