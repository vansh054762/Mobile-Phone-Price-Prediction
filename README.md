# 📱 Mobile Phone Price Prediction

A machine learning project that predicts the price range of a mobile phone based on its technical specifications.

## 🎯 Objective
Build and compare ML models to predict mobile phone price categories using hardware and feature specifications.

## 📁 Project Structure
```
mobile-phone-price-prediction/
│
├── data/
│   └── mobile_data.csv          # Dataset
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb   # Jupyter notebook for exploration & modeling
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning & feature engineering
│   ├── model_training.py        # Train & compare ML models
│   ├── model_evaluation.py      # Evaluation metrics & plots
│   └── predict.py               # Single prediction utility
│
├── models/
│   └── best_model.pkl           # Saved best model (generated after training)
│
├── app/
│   └── app.py                   # Streamlit web application
│
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack
- **Python 3.9+**
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — visualization
- **Scikit-learn** — ML models
- **Streamlit** — web interface
- **Joblib** — model serialization

## 🤖 Models Compared
| Model | Type |
|---|---|
| Linear Regression | Baseline |
| Decision Tree | Tree-based |
| Random Forest ⭐ | Ensemble |
| Gradient Boosting | Boosting |

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset & train models
```bash
python src/data_preprocessing.py
python src/model_training.py
```

### 3. Run the Streamlit app
```bash
streamlit run app/app.py
```

## 📊 Features Used
| Feature | Description |
|---|---|
| RAM (MB) | RAM capacity |
| Internal Storage (GB) | Storage space |
| Battery Capacity (mAh) | Battery size |
| Screen Size (inches) | Display size |
| Primary Camera (MP) | Main camera resolution |
| Processor Speed (GHz) | CPU speed |
| 5G Support | 5G connectivity (Yes/No) |
| Weight (grams) | Phone weight |
| Screen Resolution (PPI) | Pixels per inch |
| Brand | Manufacturer |

## 🏷️ Price Range Labels
| Label | Price Range |
|---|---|
| 0 | Budget (< ₹10,000) |
| 1 | Mid-Range (₹10,000 – ₹20,000) |
| 2 | Upper Mid (₹20,000 – ₹40,000) |
| 3 | Premium (> ₹40,000) |
