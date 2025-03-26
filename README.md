# California Housing Price Prediction

## Project Overview
This project aims to predict housing prices in California using the **California Housing Dataset**. The dataset contains information about various features like median income, number of rooms, and population per block group. We employ multiple regression models and evaluate their performances to determine the most effective approach.

## Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Jupyter Notebook** (for data exploration and model training)
- **Machine Learning Models** (Linear Regression, Decision Trees, Random Forests, Gradient Boosting)


---

## 1. Data Preprocessing
### Data Cleaning
- Handling missing values (imputation or removal)
- Removing duplicates

### Data Exploration
- Summary statistics (mean, median, standard deviation, etc.)
- Visualizations:
  - Histograms
  - Boxplots
  - Correlation heatmaps

### Data Transformation
- **Categorical Encoding:**
  - **One-Hot Encoding** for **Linear Regression**
  - **Label Encoding** for **Tree-based Models**
- **Feature Scaling:** Standardization using `StandardScaler`
- **Data Splitting:** Training (80%) / Testing (20%)

---

## 2. Modeling
### Model Selection
The following regression models were tested:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

### Model Training
Each model was trained on its respective preprocessed dataset:
- **Linear Regression:** Uses **One-Hot Encoded** features
- **Tree-based Models:** Use **Label Encoded** features

### Hyperparameter Optimization
We used **GridSearchCV** for tuning hyperparameters:
- **Decision Tree:** `max_depth` from 1 to 20
- **Random Forest:** `n_estimators` from 1 to 200, `max_depth` from 1 to 20
- **Gradient Boosting:** `n_estimators` from 1 to 200, `learning_rate` from 0.01 to 0.9

---

## 3. Model Evaluation
### Evaluation Metrics
The models were evaluated using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

### Visualization
- Residual Plots
- Actual vs. Predicted Values Comparison

---

## 4. Results Analysis
### Best Performing Model
The model with the highest **R² score** and lowest error metrics was determined.

### Insights & Discussion
- Comparison of models' strengths and weaknesses
- Potential improvements such as feature engineering and advanced hyperparameter tuning

---

## Getting Started
### 🔹 Installation
```bash
pip install -r requirements.txt
```

### 🔹 Running the Notebook
Launch Jupyter Notebook and open `notebooks/housing_price_prediction.ipynb` to run the project.
```bash
jupyter notebook
```

---

## License
This project is licensed under the **MIT License**.

## Contributing
Feel free to open issues or submit pull requests to improve the project! 🙌

