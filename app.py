import tensorflow as tf
import numpy as np
import tf_keras
import matplotlib.pyplot as plt
import streamlit as st


def plot_multiple_images(images, n_cols=8):
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = images.squeeze(axis=-1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

    # Si solo hay una fila
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for idx, image in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.imshow(image, cmap="binary")
        ax.axis("off")

    # Ocultar ejes vacíos
    total_plots = n_rows * n_cols
    for idx in range(len(images), total_plots):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis("off")

    plt.tight_layout()
    return fig


def variance_schedule(T, s=0.008, max_beta=0.999):
    t = np.arange(T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha = np.clip(f[1:] / f[:-1], 1 - max_beta, 1)
    alpha = np.append(1, alpha).astype(np.float32)  # alpha[0] = 1
    beta = 1 - alpha
    alpha_cumprod = np.cumprod(alpha)
    return alpha, alpha_cumprod, beta


def generate(model, placeholder, batch_size=32):
    X = tf.random.normal([batch_size, 28, 28, 1])
    for t in range(T - 1, 0, -1):
        noise = (tf.random.normal if t > 1 else tf.zeros)(tf.shape(X))
        X_noise = model({"X_noisy": X, "time": tf.constant([t] * batch_size)})
        X = (
            1 / alpha[t] ** 0.5
            * (X - beta[t] / (1 - alpha_cumprod[t]) ** 0.5 * X_noise)
            + (1 - alpha[t]) ** 0.5 * noise
        )

        if t % 100 == 0:
            contador.write(f"t = {t}")
            fig = plot_multiple_images(X.numpy(), n_cols=8)
            placeholder.pyplot(fig)
            plt.close(fig)
    
    contador.write(f"t = {t}")
    fig = plot_multiple_images(X.numpy(), n_cols=8)
    placeholder.pyplot(fig)
    plt.close(fig)

    return X


st.title("🧠 Modelo de Diffusion en Vivo")

model = tf_keras.models.load_model('cap_17/my_diffusion_model')

T = 4000
alpha, alpha_cumprod, beta = variance_schedule(T)

if st.toggle("🚀 Ejecutar modelo"):
    placeholder = st.empty()
    contador = st.empty()
    X_gen = generate(model, placeholder)
