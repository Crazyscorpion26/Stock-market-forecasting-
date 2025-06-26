---
title: Real Time Stock Forecasting Using Machine Learning
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference



# 📈 Stock Market Forecasting

A Python-based project to predict stock prices using historical market data. This repository applies machine learning and time-series forecasting models to analyze trends and generate future price predictions.

---

## 🚀 Project Overview

This project aims to forecast stock prices using techniques like Linear Regression, Random Forest, and LSTM, based on cleaned and structured stock market data. It includes data preprocessing, visualization, model training, and evaluation.

---

## 🔍 Features

- 📊 **Data Preprocessing**: Cleaning, sorting, and preparing historical stock data.
- 🧠 **Machine Learning Models**: Linear Regression, Random Forest, and optional LSTM.
- 📉 **Time-Series Models**: ARIMA, SARIMA (if applicable).
- 📈 **Visualization**: Actual vs predicted prices, moving averages, and trends.
- 📦 **Modular Code**: Easy to extend, structured in folders for reusability.
- 🧪 **Evaluation Metrics**: MSE, RMSE, MAE.

---

## 🧰 Technologies Used

- **Python 3.8+**
- **pandas**, **numpy** – data manipulation
- **matplotlib**, **seaborn** – visualization
- **scikit-learn** – machine learning models
- **statsmodels**, **keras/tensorflow** – optional for ARIMA/LSTM

---

## 📁 Directory Structure

├── data/ # Raw and processed stock CSVs
├── notebooks/ # Jupyter notebooks for interactive development
├── src/ # Python modules (cleaned + documented)
│ ├── data_processing.py
│ ├── train_model.py
│ ├── evaluate.py
├── models/ # Saved trained models
├── results/ # Output graphs, evaluation plots
├── README.md # This file
├── requirements.txt # List of dependencies
├── .gitignore # Files/folders to ignore in git
└── LICENSE # MIT License file


---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Crazyscorpion26/Stock-market-forecasting-.git
   cd Stock-market-forecasting-

    Create virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
