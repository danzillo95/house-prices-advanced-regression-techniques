**Project Description
The goal of this project is to build a machine learning model that can predict house prices based on their characteristics. The House Prices - Advanced Regression Techniques dataset from Kaggle was used for this.**

**Technologies Used**
Python
Pandas, Numpy (data processing)
Matplotlib, Seaborn (data visualization)
Scikit-learn (machine learning)

**Execution Steps
1. Problem Statement:**
  Predict house prices.
  Use numeric and categorical features such as house area, number of rooms, year of construction, etc.
**2. Data Download and Exploration:**
  Data was downloaded from Kaggle.
  Data structure, missing values, and descriptive statistics were analyzed.
**EDA (Exploratory Data Analysis):**
  Missing values ​​were checked.
  Target variable (house prices) was analyzed.
  Correlation of features with house price was analyzed.
**Data preprocessing:**
  Missing values ​​of numerical features were replaced with the median.
  Missing values ​​of categorical features were replaced with the most frequent values.
  Categorical features were transformed using OneHotEncoder.
**Model training:**
  The GradientBoostingRegressor model was trained.
  The model was evaluated using the MAE (Mean Absolute Error) metric.
**Results visualization:**
  Comparison of predicted prices with real values.
  Preparing the file for Kaggle:
  A file with predictions was created for submission to the Kaggle platform.

**Results**
The Gradient Boosting model showed good prediction accuracy with a minimum mean absolute error (MAE).
Visualization showed a strong correlation between real and predicted values.
