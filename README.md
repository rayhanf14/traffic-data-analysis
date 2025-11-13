# Traffic Congestion Prediction

This project predicts **traffic congestion levels (in percentage)** using real-world urban traffic indicators such as traffic volume, road capacity utilization, average speed, and incident reports. It demonstrates a **complete end-to-end machine learning workflow** — including data preprocessing, feature scaling, model training, hyperparameter tuning, learning curve analysis, and feature importance evaluation.  

The project highlights how **data-driven modeling** can assist in **urban planning, traffic management, and congestion control strategies**.

---

## Project Overview

The goal of this project is to build a **regression model** capable of accurately estimating **traffic congestion (%)** based on multiple transport and environmental features.  
The workflow emphasizes:
- Predictive performance  
- Model interpretability  
- Generalization capability  

This project serves as a **portfolio-ready demonstration of applied data science and regression modeling**.

---

## ⚙️ Key Features

### **1. Data Preprocessing**
- Handled missing values and outliers.  
- Scaled numerical features using **StandardScaler**.  
- Performed feature engineering and correlation analysis.

### **2. Model Training & Hyperparameter Tuning**
- Trained and compared multiple regression algorithms:
  - Linear Regression  
  - Ridge Regression  
  - LASSO Regression  
  - Polynomial Regression  
  - Random Forest Regression  
- Applied **GridSearchCV** for hyperparameter tuning (e.g., polynomial degree, number of estimators, regularization strength).  
- Evaluated models using **R², MAE, RMSE, MAPE, and MAX Error** metrics.

### **3. Feature Importance Analysis**
- Extracted and ranked influential features from:
  - Linear/Ridge/LASSO coefficients  
  - Random Forest feature importances  
- Visualized top contributing variables impacting congestion.

### **4. Learning Curve Evaluation**
- Plotted learning curves for each model to assess **bias–variance trade-offs**.  
- Diagnosed overfitting/underfitting behavior visually.

### **5. Model Comparison Dashboard**
- Compared all models side-by-side using MAPE, RMSE, and R².  
- Highlighted the best-performing model using bar charts.

---

## Results & Insights

| Model | Test R² | RMSE | MAPE | Key Observation |
|--------|---------|------|------|----------------|
| Linear Regression | 0.89 | 6.52 | 0.053 | Simple, interpretable but limited flexibility |
| Ridge Regression | 0.89 | 6.48 | 0.052 | Reduced overfitting slightly |
| LASSO Regression | 0.89 | 6.51 | 0.053 | Sparse model; less feature redundancy |
| Polynomial Regression | 0.96 | 5.20 | 0.043 | Captured non-linear trends effectively |
| Random Forest Regression | **0.95** | **5.21** | **0.045** | Best balance of bias and variance |

### **Key Insights**
- **Incident Reports**, **Road Capacity Utilization** and **Traffic Volume** were the most influential predictors of congestion.  
- **Polynomial and Random Forest** models achieved the highest generalization accuracy.  
- **MAPE ≈ 0.05** implies an average prediction error of just **5%**, indicating strong performance.

---

## Technologies Used
- **Python**  
- **Pandas, NumPy** — Data processing  
- **Scikit-learn** — Modeling and evaluation  
- **Matplotlib, Seaborn** — Data visualization  
- **Jupyter Notebook** — Interactive workflow

---

## Evaluation Metrics
- **R² Score** — Measures model’s explanatory power  
- **RMSE (Root Mean Squared Error)** — Quantifies error magnitude  
- **MAE (Mean Absolute Error)** — Average deviation from true values  
- **MAPE (Mean Absolute Percentage Error)** — Percentage-based interpretability  

---

## Market & Policy Insights
- Congestion levels correlate strongly with **incident frequency** and **capacity usage**, suggesting infrastructure optimization can significantly reduce delays.  
- Predictive modeling can help **city planners prioritize high-risk zones** and **optimize signal timing**.

---

## Future Enhancements
- Integrate **real-time traffic sensor data** or **Google Maps API** for dynamic prediction.  
- Apply **Gradient Boosting / XGBoost** for enhanced performance.  
- Deploy the model as a **Flask or Streamlit dashboard** for live congestion forecasting.

---

## Project Structure
```
Traffic-Congestion-Prediction/
│
├── notebook.ipynb              # Main Jupyter Notebook workflow  
├── README.md                   # Documentation file  
├── requirements.txt             # Dependencies  
└── data/                       # Dataset files
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/traffic-congestion-prediction.git
cd traffic-congestion-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

---

## Data Source & Credits

**Dataset:** [Bangalore City Traffic Dataset](https://www.kaggle.com/datasets/preethamgouda/banglore-city-traffic-dataset)  
**Collected by:** Preetham Gouda  
**License:** CC0: Public Domain  
**Source:** Kaggle — maintained by [Preetham Gouda](https://www.kaggle.com/preethamgouda)

> The dataset contains real-time traffic statistics across various areas of Bangalore, including factors like traffic volume, incidents, public transport usage, and road capacity.  
> All rights and ownership belong to the original dataset creator and contributors.

## Author
**Rayhan Feroz**  
*B.Tech CSE (Data Science) @ BMS College of Engineering* 


---
