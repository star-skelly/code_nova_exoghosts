#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon
    
latent_dim = 8

# ENCODER
T = 4320
encoder_inputs = keras.Input(shape=(T, 1))  # 1D sequence, single channel
x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

encoder.summary()

# DECODER
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense((T // 4) * 64, activation="relu")(latent_inputs)  # match downsampling in encoder
x = layers.Reshape((T // 4, 64))(x)
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
decoder_outputs = layers.Conv1D(1, 3, activation="linear", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.summary()


# In[37]:


# --- VAE Model ---
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        noisy, clean = data  # unpack tuple (noisy input, clean target)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(noisy)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(keras.losses.mean_squared_error(clean, reconstruction), axis=1)
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# In[35]:


# --- Load noise dataset ---
import numpy as np
with np.load("star_noise_cohort.npz") as npz:
    data = npz["data"]  # shape (NUM, N)
    n_samples = int(npz["n_samples"])
    fs = float(npz["fs"])

# Add transits
def add_transit_batch(data, fs, depth=0.01, duration=50, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    data_with_transits = []
    for signal in data:
        n_samples = len(signal)
        t = np.arange(n_samples) / fs
        t0 = rng.uniform(0.2, 0.8) * t[-1]
        dur = duration / fs
        mask = np.abs(t - t0) < (dur / 2)
        signal_transit = signal.copy()
        signal_transit[mask] -= depth
        data_with_transits.append(signal_transit)
    return np.array(data_with_transits)

data_with_transits = add_transit_batch(data, fs)

# Reshape for CNN (num_samples, length, channels)
X_clean = data[..., np.newaxis].astype("float32")
X_noisy = data_with_transits[..., np.newaxis].astype("float32")


# In[62]:


import matplotlib.pyplot as plt
# --- Helper: plot reconstructions ---
def plot_reconstructions(model, noisy, clean, n=5):
    idxs = np.random.choice(len(noisy), n, replace=False)
    reconstructed = model.decoder(model.encoder(noisy[idxs])[2]).numpy()
    plt.figure(figsize=(15, 2 * n))
    for i, idx in enumerate(idxs):
        plt.subplot(n, 1, i + 1)
        plt.plot(noisy[idx, :, 0], label="Noisy (with transit)", alpha=0.6)
        plt.plot(clean[idx, :, 0], label="Target (clean star noise)", alpha=0.6)
        plt.plot(reconstructed[i, :, 0], label="Reconstructed", alpha=0.8)
        plt.legend()
    plt.tight_layout()
    plt.show()

# --- Load noise dataset ---
with np.load("star_noise_cohort.npz") as npz:
    data = npz["data"]  # shape (NUM, N)
    n_samples = int(npz["n_samples"])
    fs = float(npz["fs"])

N = 4320
t = np.arange(N)
true_period = 320
true_dur = 20
alpha = 0.2
from noise_generator import boxtransit
template_true = boxtransit(t, period=true_period, dur=true_dur, t0=0, alpha=alpha)

print(data.shape)


# In[ ]:


# lc += planet
# highlight transit in time series like in noise_sim
planet_mask = template_true != 0
transited_planet = data[0] + template_true

plt.figure(figsize=(10, 3))
plt.plot(np.arange(N), data[0], color='blue', alpha=0.5, label='light curve (no transit)')
plt.plot(np.arange(N), transited_planet, color='red', alpha=0.5, label='light curve (transit)')

#plt.scatter(np.arange(N)[planet_mask], transited_planet[planet_mask], color='red', s=6, zorder=3, label='Planet (in-transit)')
plt.legend()
plt.tight_layout()
plt.show()

# test to see transits effect


# In[69]:


# Add transits
def add_transit_batch(data, fs, depth=0.01, duration=50, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    data_with_transits = []
    for signal in data:
        N = 4320
        t = np.arange(N)
        true_period = 320
        true_dur = 20
        alpha = 0.25 * np.random.randn() + 0.5 # should be range 0.25 to 0.75
        from noise_generator import boxtransit
        transit = boxtransit(t, period=true_period, dur=true_dur, t0=0, alpha=alpha)
        signal_transit = signal.copy()
        signal_transit += transit
        data_with_transits.append(signal_transit)
    return np.array(data_with_transits)

data_with_transits = add_transit_batch(data, fs)

# Reshape for CNN (num_samples, length, channels)
X_clean = data[..., np.newaxis].astype("float32")
X_noisy = data_with_transits[..., np.newaxis].astype("float32")


# In[71]:


# look at some transits

plt.figure(figsize=(10, 3))
plt.plot(np.arange(N), X_clean[10], color='blue', alpha=0.5, label='light curve (no transit)')
plt.plot(np.arange(N), X_noisy[10], color='red', alpha=0.5, label='light curve (transit)')

#plt.scatter(np.arange(N)[planet_mask], transited_planet[planet_mask], color='red', s=6, zorder=3, label='Planet (in-transit)')
plt.legend()
plt.tight_layout()
plt.show()

# test to see transits effect


# In[72]:


# --- Train ---
vae = VAE(encoder, decoder)
vae.compile(optimizer="adam")

history = vae.fit(
    X_noisy, X_clean,  # input noisy → target clean
    epochs=50,
    batch_size=128,
)


# In[73]:


# Plot loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['kl_loss'], label='kl loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()


# In[74]:


plot_reconstructions(vae, X_noisy, X_clean, n=5)

