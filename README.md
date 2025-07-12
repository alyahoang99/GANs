# Project Overview
This project implements an advanced Recurrent Generative Adversarial Network (RGAN) for the synthesis of high-resolution smart home energy consumption data. The core innovation lies in leveraging recurrent neural networks (GRU in generator, LSTM with attention in discriminator) to capture complex temporal dependencies inherent in time-series data, specifically focusing on 'Value', 'Value_HeatPump', and 'Value_OtherAppliances' features. The work directly addresses critical challenges in smart home energy management systems (HEMS), such as data scarcity, privacy concerns, and the high cost of collecting real-world energy data. By generating realistic synthetic data, this model aims to facilitate the development and testing of HEMS without relying solely on sensitive user data.

Motivation and Research Context
The ability to accurately model and forecast energy consumption is pivotal for efficient smart home energy management. However, obtaining sufficient and diverse real-world energy data is often hampered by privacy regulations, operational expenses, and the difficulty in capturing a wide range of consumption behaviors, including rare events. This project offers a robust data-driven solution through GANs.
Traditional model-based synthetic data generation often requires extensive domain knowledge and struggles to emulate dynamic user habits. Data-driven approaches, particularly unsupervised machine learning techniques like GANs and RGANs, offer a flexible alternative. This research specifically highlights the advantages of RGANs in handling sequential data, making them ideal for time-series synthesis. The integration of an attention mechanism in the discriminator further refines its ability to discern and learn critical temporal patterns, mitigating common GAN training instabilities such as mode collapse.

# Key Features
- Recurrent GAN (RGAN) Architecture: Employs GRU layers in the generator for effective sequence generation and LSTM layers coupled with a custom attention mechanism in the discriminator for robust sequence evaluation and pattern recognition.

- Time Series Specific Design: Tailored for sequential data generation, demonstrated effectively on smart meter data.

- Wasserstein GAN with Gradient Penalty (WGAN-GP): Utilizes WGAN-GP for stable and robust training, ensuring high-quality synthetic data generation by promoting Lipschitz continuity in the discriminator.

- Comprehensive Data Preprocessing: Includes robust pipelines for loading, cleaning, normalizing (Min-Max scaling), and transforming raw time-series data into suitable sequences via a sliding window approach (SEQ_LEN = 7).

- Quantitative Evaluation: Incorporates rigorous statistical metrics (KL, JS, Wasserstein divergences) and practical utility tests (Train-on-Synthetic, Test-on-Real - TSTR) to validate the quality and applicability of the generated data.

# Getting Started
To run the project, follow these instructions to set up your environment and execute the code.

## Prerequisites
Python 3.x

TensorFlow 2.x

Numpy

Pandas

Matplotlib

Seaborn

Scikit-learn

## Installation
Clone the repository:

Bash

git clone https://github.com/alyahoang99/GANs.git
cd GANs
Install dependencies:
It is highly recommended to use a virtual environment for dependency management. Create a requirements.txt file (e.g., by running pip freeze > requirements.txt after installing the above libraries) and then install them:

Bash

pip install -r requirements.txt




## Prepare your dataset:
Ensure your smart meter data (or analogous time-series data) is formatted as a CSV file, including a 'Timestamp' column and the features 'Value', 'Value_HeatPump', and 'Value_OtherAppliances'.



Execute the notebook:
Open RGAN1.ipynb in a Jupyter-compatible environment (e.g., Jupyter Lab, VS Code with Jupyter extension, or Google Colab) and execute all cells sequentially. The notebook will guide you through data loading, preprocessing, model construction, training, and a detailed evaluation of the generated synthetic data.

Bash

jupyter notebook RGAN1.ipynb
For direct execution in Google Colab:

## Dataset
The research utilized a comprehensive smart meter dataset collected from November 2019 to August 2022, encompassing four distinct households. The dataset includes fine-grained energy consumption data (at 15-minute intervals, downsampled to daily for training stability) for the main meter ('Value') and disaggregated consumption for specific appliances like 'Value_HeatPump' and 'Value_OtherAppliances'.

## Key preprocessing steps applied to the data include:

- Normalization: Feature values are scaled to a [0, 1] range using MinMaxScaler.

- Sequence Generation: Data is transformed into fixed-length overlapping sequences (SEQ_LEN = 7) using a sliding window technique to capture temporal patterns.

- Baseload Estimation and Switching Event Detection: These techniques were employed to further analyze and annotate consumption patterns, although the primary model relies on the raw feature sequences after normalization.

# Model Architecture
The RGAN architecture is designed for optimal performance on time-series data.

1. Generator
The generator is responsible for learning the distribution of the real data and producing synthetic sequences.

- Input: A latent noise vector with dimensions (SEQ_LEN, LATENT_DIM, where LATENT_DIM is typically 100).
Layers:

- Two stacked Bidirectional GRU layers, each with GRU_UNITS (128 units), enabling the capture of dependencies from both forward and backward directions in the sequence.

- A TimeDistributed(Dense) layer with GELU activation, applied to each time step independently, to transform the recurrent output.

- A final TimeDistributed(Dense) layer to output sequences with the FEATURE_DIM (3), corresponding to 'Value', 'Value_HeatPump', and 'Value_OtherAppliances'.

2. Discriminator with Attention Mechanism
The discriminator evaluates the authenticity of sequences, distinguishing between real and synthetic data. Its design is crucial for stable GAN training and accurate feature learning.

- Input: Either real or generated time sequences with dimensions (SEQ_LEN, FEATURE_DIM).

- Layers:

  - A Bidirectional LSTM layer with GRU_UNITS (128 units) to process temporal dependencies.

  - A custom Attention Block (AttentionLayer), which dynamically weighs the importance of different time steps in the input sequence. This mechanism allows the discriminator to focus on the most salient features or events within the time series.

  - A Dense layer with GRU_UNITS (128 units) and Tanh activation for further feature extraction.

  - A final Dense output layer with linear activation, suitable for the Wasserstein loss function, predicting the "realness" score of the input sequence.

## Training Details
The RGAN is trained using the Wasserstein GAN with Gradient Penalty (WGAN-GP) objective for enhanced stability and performance.

- Optimizers: Adam optimizer is used for both generator and discriminator networks.

- Generator Learning Rate (GEN_LR): 2e-5

- Discriminator Learning Rate (DISC_LR): 1e-5

- Beta 1 (BETA_1): 0.5 (for both optimizers)

- Loss Function: Wasserstein loss is employed, which provides a more stable gradient than traditional GAN losses. A Gradient Penalty (LAMBDA_GP = 12.0) is added to enforce the Lipschitz constraint on the discriminator.

- Training Ratio: The discriminator is updated DISC_UPDATES_PER_GEN (6) times for every single generator update, ensuring the discriminator is sufficiently strong before the generator trains.

- Epochs: The model is trained for EPOCHS = 500.

- Batch Size: Each training iteration processes BATCH_SIZE = 100 sequences.

Here's a revised and more comprehensive README for your GANs repository, integrating key details and insights from your MSc Dissertation. This version aims to reflect the depth and academic rigor of your work.

Recurrent Generative Adversarial Network (RGAN) for Smart Home Energy Data Simulation
This repository presents the implementation of a Recurrent Generative Adversarial Network (RGAN) with an attention mechanism, specifically designed for generating realistic synthetic time-series data related to smart home energy consumption. This project directly stems from the research conducted for the MSc Dissertation, "Generative Adversarial Networks (GANs) for Smart Home Energy Data Simulation," which achieved a score of 90/100.

Table of Contents
Project Overview

Motivation and Research Context

Key Features

Getting Started

Prerequisites

Installation

Usage

Dataset

Model Architecture

Generator

Discriminator with Attention Mechanism

Training Details

Evaluation Methodology and Metrics

Results

Contributing

License

Acknowledgements

Reference to Dissertation

Project Overview
This project implements an advanced Recurrent Generative Adversarial Network (RGAN) for the synthesis of high-resolution smart home energy consumption data. The core innovation lies in leveraging recurrent neural networks (GRU in generator, LSTM with attention in discriminator) to capture complex temporal dependencies inherent in time-series data, specifically focusing on 'Value', 'Value_HeatPump', and 'Value_OtherAppliances' features. The work directly addresses critical challenges in smart home energy management systems (HEMS), such as data scarcity, privacy concerns, and the high cost of collecting real-world energy data. By generating realistic synthetic data, this model aims to facilitate the development and testing of HEMS without relying solely on sensitive user data.

Motivation and Research Context
The ability to accurately model and forecast energy consumption is pivotal for efficient smart home energy management. However, obtaining sufficient and diverse real-world energy data is often hampered by privacy regulations, operational expenses, and the difficulty in capturing a wide range of consumption behaviors, including rare events. This project offers a robust data-driven solution through GANs.

As detailed in the associated MSc Dissertation, traditional model-based synthetic data generation often requires extensive domain knowledge and struggles to emulate dynamic user habits. Data-driven approaches, particularly unsupervised machine learning techniques like GANs and RGANs, offer a flexible alternative. This research specifically highlights the advantages of RGANs in handling sequential data, making them ideal for time-series synthesis. The integration of an attention mechanism in the discriminator further refines its ability to discern and learn critical temporal patterns, mitigating common GAN training instabilities such as mode collapse.

Key Features
Recurrent GAN (RGAN) Architecture: Employs GRU layers in the generator for effective sequence generation and LSTM layers coupled with a custom attention mechanism in the discriminator for robust sequence evaluation and pattern recognition.

Time Series Specific Design: Tailored for sequential data generation, demonstrated effectively on smart meter data.

Wasserstein GAN with Gradient Penalty (WGAN-GP): Utilizes WGAN-GP for stable and robust training, ensuring high-quality synthetic data generation by promoting Lipschitz continuity in the discriminator.

Comprehensive Data Preprocessing: Includes robust pipelines for loading, cleaning, normalizing (Min-Max scaling), and transforming raw time-series data into suitable sequences via a sliding window approach (SEQ_LEN = 7).

Quantitative Evaluation: Incorporates rigorous statistical metrics (KL, JS, Wasserstein divergences) and practical utility tests (Train-on-Synthetic, Test-on-Real - TSTR) to validate the quality and applicability of the generated data.

Getting Started
To run the project, follow these instructions to set up your environment and execute the code.

Prerequisites
Python 3.x

TensorFlow 2.x

Numpy

Pandas

Matplotlib

Seaborn

Scikit-learn

Installation
Clone the repository:

Bash

git clone https://github.com/alyahoang99/GANs.git
cd GANs
Install dependencies:
It is highly recommended to use a virtual environment for dependency management. Create a requirements.txt file (e.g., by running pip freeze > requirements.txt after installing the above libraries) and then install them:

Bash

pip install -r requirements.txt
Usage
The primary implementation and demonstration of the RGAN model are contained within the RGAN1.ipynb Jupyter notebook.

Prepare your dataset:
Ensure your smart meter data (or analogous time-series data) is formatted as a CSV file, including a 'Timestamp' column and the features 'Value', 'Value_HeatPump', and 'Value_OtherAppliances'.

Important: The notebook currently expects the dataset to be located at /content/drive/MyDrive/Colab Notebooks/GANs/resampleddata (1).csv. You will need to adjust this path within the notebook to match your data's location.

Execute the notebook:
Open RGAN1.ipynb in a Jupyter-compatible environment (e.g., Jupyter Lab, VS Code with Jupyter extension, or Google Colab) and execute all cells sequentially. The notebook will guide you through data loading, preprocessing, model construction, training, and a detailed evaluation of the generated synthetic data.

Bash

jupyter notebook RGAN1.ipynb
For direct execution in Google Colab:

Dataset
The research utilized a comprehensive smart meter dataset collected from November 2019 to August 2022, encompassing four distinct households. The dataset includes fine-grained energy consumption data (at 15-minute intervals, downsampled to daily for training stability) for the main meter ('Value') and disaggregated consumption for specific appliances like 'Value_HeatPump' and 'Value_OtherAppliances'.

Key preprocessing steps applied to the data include:

Normalization: Feature values are scaled to a [0, 1] range using MinMaxScaler.

Sequence Generation: Data is transformed into fixed-length overlapping sequences (SEQ_LEN = 7) using a sliding window technique to capture temporal patterns.

Baseload Estimation and Switching Event Detection: These techniques were employed to further analyze and annotate consumption patterns, although the primary model relies on the raw feature sequences after normalization.

Model Architecture
The RGAN architecture is designed for optimal performance on time-series data.

Generator
The generator is responsible for learning the distribution of the real data and producing synthetic sequences.

Input: A latent noise vector with dimensions (SEQ_LEN, LATENT_DIM, where LATENT_DIM is typically 100).

Layers:

Two stacked Bidirectional GRU layers, each with GRU_UNITS (128 units), enabling the capture of dependencies from both forward and backward directions in the sequence.

A TimeDistributed(Dense) layer with GELU activation, applied to each time step independently, to transform the recurrent output.

A final TimeDistributed(Dense) layer to output sequences with the FEATURE_DIM (3), corresponding to 'Value', 'Value_HeatPump', and 'Value_OtherAppliances'.

Discriminator with Attention Mechanism
The discriminator evaluates the authenticity of sequences, distinguishing between real and synthetic data. Its design is crucial for stable GAN training and accurate feature learning.

Input: Either real or generated time sequences with dimensions (SEQ_LEN, FEATURE_DIM).

Layers:

A Bidirectional LSTM layer with GRU_UNITS (128 units) to process temporal dependencies.

A custom Attention Block (AttentionLayer), which dynamically weighs the importance of different time steps in the input sequence. This mechanism allows the discriminator to focus on the most salient features or events within the time series.

A Dense layer with GRU_UNITS (128 units) and Tanh activation for further feature extraction.

A final Dense output layer with linear activation, suitable for the Wasserstein loss function, predicting the "realness" score of the input sequence.

Training Details
The RGAN is trained using the Wasserstein GAN with Gradient Penalty (WGAN-GP) objective for enhanced stability and performance.

Optimizers: Adam optimizer is used for both generator and discriminator networks.

Generator Learning Rate (GEN_LR): 2e-5

Discriminator Learning Rate (DISC_LR): 1e-5

Beta 1 (BETA_1): 0.5 (for both optimizers)

Loss Function: Wasserstein loss is employed, which provides a more stable gradient than traditional GAN losses. A Gradient Penalty (LAMBDA_GP = 12.0) is added to enforce the Lipschitz constraint on the discriminator.

Training Ratio: The discriminator is updated DISC_UPDATES_PER_GEN (6) times for every single generator update, ensuring the discriminator is sufficiently strong before the generator trains.

Epochs: The model is trained for EPOCHS = 500.

Batch Size: Each training iteration processes BATCH_SIZE = 100 sequences.

# Evaluation Methodology and Metrics
The quality and utility of the synthetic data are rigorously assessed using a combination of statistical and practical evaluation methodologies, as detailed in the dissertation.

- Statistical Fidelity Metrics: These quantify the distributional similarity between real and generated data.

- Kullback-Leibler (KL) Divergence: Measures the difference between two probability distributions. Lower values indicate better similarity.

- Jensen-Shannon (JS) Divergence: A symmetrized and smoothed version of KL divergence, providing a more robust measure of similarity.

- Wasserstein Distance (Earth Mover's Distance): Measures the minimum "cost" of transforming one distribution into another. A smaller Wasserstein distance signifies closer distributions.

These metrics are computed for each feature ('Value', 'Value_HeatPump', 'Value_OtherAppliances') independently.

- Practical Utility: Train-on-Synthetic, Test-on-Real (TSTR):
This crucial evaluation method assesses the practical applicability of the synthetic data. Predictive models (e.g., LSTM, XGBoost) are trained exclusively on the generated synthetic dataset and subsequently tested on a real, unseen hold-out dataset. Performance metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² (Coefficient of Determination) are then used to gauge how well models generalize from synthetic to real data. This approach directly validates the synthetic data's utility in downstream analytical tasks.


