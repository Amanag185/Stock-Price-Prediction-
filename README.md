# StockSense: Deep Learning-Based Stock Price Forecasting using LSTM

StockSense is a deep learning project focused on forecasting stock market trends using Long Short-Term Memory (LSTM) neural networks. The project leverages historical stock price data from major S&P 500 companies to model temporal dependencies and predict future stock closing prices.

The primary objective is to develop a robust time-series forecasting framework capable of capturing long-term market patterns and generating accurate price predictions for Apple Inc. (AAPL).

---

# Executive Summary

Financial markets generate highly dynamic and sequential data, making traditional regression approaches inadequate for capturing long-term dependencies.

This project utilizes Recurrent Neural Networks (RNNs), specifically LSTM architectures, to learn complex temporal patterns from historical stock prices and forecast future market movements.

The workflow includes:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Time-series feature engineering
- Sequence generation
- LSTM model training
- Performance evaluation
- Forecast visualization

---

# Objectives

- Develop a deep learning framework for stock price forecasting.
- Analyze historical market trends across multiple S&P 500 companies.
- Learn temporal dependencies using LSTM networks.
- Predict future closing prices of Apple stock.
- Evaluate model performance using RMSE, MSE, and R² metrics.
- Visualize actual vs predicted market movements.

---

# Dataset

### Source

Kaggle:
Price & Volume Data for All US Stocks & ETFs

### Dataset Statistics

- 2.5M+ historical stock records
- 5 years of daily trading data
- 500+ listed companies
- Multiple market sectors

### Features

| Feature | Description |
|----------|-------------|
| Date | Trading Date |
| Open | Opening Price |
| High | Highest Price |
| Low | Lowest Price |
| Close | Closing Price |
| Volume | Trading Volume |
| Name | Stock Ticker |

### Companies Analyzed

- Apple (AAPL)
- Amazon (AMZN)
- Google (GOOGL)
- NVIDIA (NVDA)
- Meta (FB)
- AMD
- IBM
- Cisco
- eBay

---

# Exploratory Data Analysis

Performed extensive market analysis including:

### Price Trend Analysis

- Open vs Close price behavior
- Long-term stock growth patterns
- Company-wise performance comparison

### Volume Analysis

- Trading volume fluctuations
- Market activity trends
- Liquidity assessment

### Time-Series Visualization

- Multi-company stock movement comparison
- Historical Apple stock trend analysis
- Temporal market behavior exploration

---

# Data Preprocessing

### Data Cleaning

- Missing value handling
- Date conversion and formatting
- Company-level filtering

### Feature Scaling

Applied:

```python
MinMaxScaler(feature_range=(0,1))
```

to normalize stock prices before model training.

### Sequence Generation

Created rolling-window sequences:

- Lookback Window = 60 Days
- Forecast Horizon = Next Trading Day

This enables the model to learn historical dependencies in stock prices.

---

# Deep Learning Architecture

## Model Type

Stacked Long Short-Term Memory (LSTM)

### Network Architecture

Input Layer

↓

LSTM (64 Units, Return Sequences=True)

↓

LSTM (64 Units)

↓

Dense Layer (32 Units)

↓

Dropout Layer (0.5)

↓

Output Layer (1 Unit)

### Why LSTM?

LSTM networks effectively capture:

- Long-term dependencies
- Sequential market behavior
- Temporal price patterns
- Non-linear financial trends

making them highly suitable for stock market forecasting.

---

# Training Configuration

| Parameter | Value |
|------------|---------|
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Epochs | 10 |
| Batch Processing | TensorFlow |
| Training Split | 95% |
| Lookback Window | 60 Days |

---

# Performance Metrics

Model performance was evaluated using:

### Mean Squared Error (MSE)

Measures average prediction error magnitude.

### Root Mean Squared Error (RMSE)

Provides interpretable forecasting error in stock price units.

### R² Score

Measures explanatory power of the model.

### Directional Accuracy

Evaluates the model's ability to correctly predict market movement direction.

---

# Results

### Forecasting Performance

- RMSE: **1.87**
- MSE: **6.54**
- R² Score: **0.97**
- Directional Accuracy: **91%**

### Key Outcomes

- Successfully captured long-term stock price trends.
- Achieved high predictive accuracy on unseen data.
- Demonstrated strong correlation between predicted and actual closing prices.
- Generated stable forecasts with minimal overfitting.

---

# Visualizations

### Stock Market Analytics

- Company-wise Open vs Close Trends
- Trading Volume Analysis
- Historical Price Evolution

### Forecast Evaluation

- Actual vs Predicted Stock Prices
- Trend Validation Curves
- Forecast Performance Analysis

---

# Technologies Used

## Programming

- Python

## Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- TensorFlow
- Keras
- Scikit-Learn

## Machine Learning

- Time Series Forecasting
- Deep Learning
- Recurrent Neural Networks
- Long Short-Term Memory (LSTM)

---

# Project Workflow

Raw Stock Data
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Scaling
      │
      ▼
Sequence Generation
      │
      ▼
LSTM Model Training
      │
      ▼
Prediction
      │
      ▼
Evaluation
      │
      ▼
Visualization

---

# Key Learnings

- Time-series forecasting using deep learning.
- Sequential data modeling with LSTM networks.
- Financial data preprocessing and normalization.
- Hyperparameter tuning and overfitting control.
- Performance evaluation of forecasting systems.
- End-to-end deployment workflow for predictive analytics.

---

# Future Improvements

- Integrate technical indicators (RSI, MACD, Bollinger Bands).
- Add multi-feature forecasting using OHLCV data.
- Implement GRU and Transformer architectures.
- Incorporate sentiment analysis from financial news.
- Build a real-time forecasting dashboard using Streamlit.

---

## 👥 Contributors

* **Aman Agrawal** - [@Amanag185](https://github.com/Amanag185)

## 📁 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/stock-price-lstm.git
cd stock-price-lstm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the script
python stock_price_prediction.py


