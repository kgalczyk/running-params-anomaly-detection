# Autoencoder for Running Params Anomaly Detection (MATLAB)

This project implements an unsupervised anomaly detection system for running parameters (such as heart rate and cadence) using an autoencoder based on an LSTM (Long Short-Term Memory) network built entirely in MATLAB.

The system learns the baseline pattern of a runner's normal training parameters and automatically flags deviations that could indicate equipment malfunctions (such as incorrect heart rate monitor or GPS measurements) or signal injury risk (for example, where overload may cause an abnormally higher heart rate).

## 🚀 Overview

An Autoencoder is a type of neural network trained to reconstruct its input data. In this project, the model is trained exclusively on data representing a runner's "normal" state (e.g., from the early, stable phases of a run or from baseline recovery sessions).

### How it Works:
1. **Compression & Reconstruction:** The network compresses the multi-dimensional running parameters into a lower-dimensional latent space and then attempts to reconstruct the original signal.
2. **Reconstruction Error:** For standard activities, the reconstruction error remains minimal.
3. **Anomaly Detection:** When abnormal running parameters occur (e.g., an unexpectedly low or high heart rate), the Autoencoder fails to reconstruct the data accurately. A spike in the **Reconstruction Error** exceeding a predefined threshold signals an anomaly.

## 🛠️ Requirements & Toolboxes

To run this project, you need **MATLAB** (R2021a or newer recommended) along with the following toolboxes:
* **Deep Learning Toolbox** (for designing and training the Autoencoder network)
* **Statistics and Machine Learning Toolbox** (for data normalization and threshold definition)

## 📋 Project Structure

```text
├── activity_data_preprocessing.m   # Data cleaning, noise filtering, and normalization
├── autoencoder.m                   # Network architecture definition
├── train_lstm.m                    # Script to train the autoencoder on normal data
├── analyze_full_training.m         # Anomaly detection logic and visualization
└── README.md                       # Project documentation