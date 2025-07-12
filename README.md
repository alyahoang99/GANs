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

