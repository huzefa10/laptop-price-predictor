# Laptop Price Predictor

A machine learning web app that predicts laptop prices based on hardware specifications. Built with a VotingRegressor ensemble (RF + ExtraTrees + XGBoost + GradientBoosting) achieving **R² = 0.9158** on the test set.

## Demo

![App Screenshot](screenshot.png)

## Features

- Predicts price from 13 hardware inputs: brand, type, RAM, weight, screen specs, CPU, storage, GPU, and OS
- Derives Pixels Per Inch (PPI) internally from screen resolution and size
- Ensemble model trained on 1303 laptops from the Kaggle laptop dataset

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| ML Model | scikit-learn VotingRegressor (RF + ExtraTrees + XGB + GBM) |
| Feature Engineering | OHE, PPI calculation, memory parsing, CPU/GPU brand extraction |
| Language | Python |

## Model Details

- **Dataset:** laptop_data.csv — 1303 laptops, 12 original features
- **Target:** log(Price) → exponentiated at prediction time
- **Best model:** VotingRegressor with weights [3, 1, 1, 3] (RF and GBM weighted higher)
- **R² Score:** 0.9158 | **Preprocessing:** OneHotEncoding + StandardScaler in sklearn Pipeline
- **11 models benchmarked** before selecting the ensemble

## Run Locally

```bash
git clone https://github.com/huzefa10/laptop-price-predictor.git
cd laptop-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

[laptop_data.csv](laptop_data.csv) — sourced from Kaggle (SmartPrix laptop listings). Includes brand, type, RAM, storage, CPU, GPU, OS, screen specs, and price.

## Project Structure

```
laptop-price-predictor/
├── app.py              # Streamlit app
├── pipe.pkl            # Trained VotingRegressor pipeline
├── df.pkl              # Preprocessed dataframe (for dropdown values)
├── laptop_data.csv     # Raw dataset
└── requirements.txt
```
