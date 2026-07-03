# 📈 Stock Price Prediction using LSTM

This project uses Long Short-Term Memory (LSTM) neural networks in TensorFlow to predict the stock closing prices of Apple Inc. based on historical time-series data. The dataset consists of 5 years of daily stock prices from multiple major companies.

---

## 🔍 Overview

- **Objective**: Predict future closing prices of Apple (AAPL) stock using historical price and volume data.
- **Approach**: Build and train a stacked LSTM network on 60-day rolling windows of scaled closing prices.
- **Visualization**: Compare predicted vs. actual stock prices using Matplotlib.

---

## 📊 Dataset

- **Source**: [all_stocks_5yr.csv](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
- **Size**: ~2.5 million rows
- **Companies Included**: AAPL, GOOGL, AMZN, NVDA, FB, IBM, CSCO, EBAY, AMD, etc.

Each row includes:  
`date, open, high, low, close, volume, Name (ticker)`

---

## 🛠️ Libraries Used

- `pandas` – Data manipulation  
- `matplotlib`, `seaborn` – Visualization  
- `numpy` – Numerical computation  
- `tensorflow` / `keras` – Model building  
- `sklearn` – Preprocessing (MinMaxScaler)  

---

## 🧠 Model Architecture

- **Input**: 60-day lookback of scaled closing prices
- **Layers**:
  - LSTM Layer (64 units, return sequences)
  - LSTM Layer (64 units)
  - Dense Layer (32 units)
  - Dropout Layer (rate = 0.5)
  - Dense Output Layer (1 unit)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Epochs**: 10

---

## ✅ Results

- **Test RMSE**: ~1.87  
- **Directional Accuracy**: ~91%  
- **Output**: Predicted prices plotted alongside actual prices for visual validation.

> *(Insert prediction plot here if saved as image)*  
> `![Prediction Plot](path_to_your_plot.png)`

---

## 📉 Visualizations

- Multi-company time-series plots of open/close prices and volume  
- Apple’s historical price trends  
- Predicted vs. actual prices using LSTM  

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


