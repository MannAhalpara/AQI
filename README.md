# Air Quality Index (AQI) Prediction Project

A machine learning project to predict Air Quality Index (AQI) values based on various air pollutant measurements across multiple Indian cities.

## 📋 Project Overview

This project analyzes air quality data and builds predictive models to forecast AQI values using multiple regression algorithms. The dataset contains daily measurements of various pollutants (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene) from different cities in India.

## 📊 Dataset

**File:** `city_day.csv`

### Features:
- **City**: Urban location (e.g., Ahmedabad)
- **Date**: Daily timestamp
- **Pollutants**: 
  - PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Target Variable**: AQI (Air Quality Index)
- **AQI_Bucket**: Quality classification (Poor, Moderate, Very Poor, Severe)

### Data Characteristics:
- Multiple years of daily measurements
- Some missing values in pollutant measurements
- AQI values with categorical quality buckets

## 🔧 Installation

### Prerequisites:
- Python 3.7+
- pip or conda

### Required Libraries:
Install dependencies using:
```bash
pip install -r requirements.txt
```

**Main Dependencies:**
- pandas
- numpy
- scikit-learn
- catboost
- xgboost
- lightgbm
- matplotlib
- seaborn

## 📁 Project Structure

```
Data Mining/Project/
├── README.md                          # This file
├── app.py                             # Streamlit web application
├── main.py                            # Standalone training script
├── main.ipynb                         # Jupyter notebook with EDA & models
├── city_day.csv                       # Dataset
├── requirements.txt                   # Python dependencies
├── model.pkl                          # Trained CatBoost model
├── scaler.pkl                         # StandardScaler for feature scaling
├── encoder.pkl                        # LabelEncoder for city encoding
└── catboost_info/                     # CatBoost training logs
```

## 🚀 Usage

### Option 1: Jupyter Notebook (Recommended for Exploration)
```bash
jupyter notebook main.ipynb
```
This notebook includes:
- Data exploration and visualization
- Missing value analysis
- Correlation analysis with AQI
- Comparison of multiple regression algorithms
- Model training and evaluation

### Option 2: Standalone Script
```bash
python main.py
```
This script:
- Trains a CatBoost model
- Evaluates performance metrics
- Saves the trained model and preprocessing objects

### Option 3: Streamlit Web Application
```bash
streamlit run app.py
```
Interactive web interface for:
- Model predictions
- Real-time feature exploration
- AQI forecasting

## 🤖 Implemented Models

The project compares multiple regression algorithms:

1. **Linear Regression** - Baseline linear model
2. **Decision Tree Regressor** - Tree-based single model
3. **Random Forest Regressor** - Ensemble of decision trees
4. **Gradient Boosting Regressor** - Sequential boosting approach
5. **Support Vector Regressor (SVR)** - Non-parametric kernel method
6. **K-Nearest Neighbors (KNN)** - Instance-based learning
7. **XGBoost** - Extreme gradient boosting
8. **LightGBM** - Fast gradient boosting framework
9. **CatBoost** - Categorical boosting (best performing)
10. **ElasticNet** - Regularized linear regression
11. **AdaBoost Regressor** - Adaptive boosting for regression

## 📈 Model Performance Metrics

Each model is evaluated using:
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Squared Error)**: Standard deviation of errors

## 🔄 Data Preprocessing Pipeline

1. **Missing Value Handling**:
   - Numeric columns: Filled with median values
   - Categorical columns: Filled with mode values
   - Xylene column: Dropped due to excessive missing values

2. **Feature Engineering**:
   - Convert Date to datetime format
   - Extract Year, Month, Day features
   - Encode categorical City variable using LabelEncoder

3. **Feature Scaling**:
   - StandardScaler applied to training data
   - Same scaler used for test data normalization

4. **Train-Test Split**:
   - 70% training, 30% testing
   - Random state: 42 (for reproducibility)

## 📊 Key Findings

- **Strong Correlations with AQI**: CO, PM2.5, NO2, SO2, PM10
- **Best Performing Model**: CatBoost (based on R² score)
- **Feature Importance**: Pollutant concentrations (PM2.5, PM10, CO) are primary AQI drivers
- **Seasonal Patterns**: AQI varies significantly across seasons

## 🛠️ Configuration

### Feature Set Used:
```python
features = ['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 
            'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
```

### Model Parameters (CatBoost):
- Depth: 8
- Learning Rate: 0.05
- Iterations: 500
- Verbose: 0

## 📝 Files Description

### `app.py`
Streamlit web application for interactive AQI prediction and visualization.

### `main.py`
Standalone script for model training with CatBoost. Saves model artifacts for production use.

### `main.ipynb`
Comprehensive Jupyter notebook containing:
- Data loading and exploration
- Missing value analysis
- Correlation heatmaps
- Model training for all algorithms
- Performance comparison
- Model serialization

## 🎯 Future Improvements

- [ ] Add time series forecasting models (ARIMA, LSTM)
- [ ] Implement ensemble voting mechanism
- [ ] Deploy model as REST API
- [ ] Add real-time weather data integration
- [ ] Develop mobile application
- [ ] Implement automated retraining pipeline


## 👤 Author

Mann Ahalpara

---

**Last Updated:** November 11, 2025
