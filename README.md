# Store-Sales-Forecasting

**Research Question**

 - How can data mining and machine learning models be leveraged to forecast store sales accurately across various time periods using the provided multivariate dataset?

**Motivation**

Main motivation for this project includes:

- Decision Making – Enabling data-driven decisions for store operations.

- Pattern Recognition – Identifying trends and seasonal variations in sales data.

- Forecasting – Predicting future sales for better inventory and resource management.

- Data Analysis – Understanding key factors affecting sales through multivariate analysis.

- Teamwork – Collaborative learning and real-world problem-solving.

- Real-World Application – Applying machine learning to a practical business challenge.

**Dataset Description**

Used five datasets to train and evaluate our forecasting model:

- Stores Data – Provides details about each store, including city, state, type, and cluster classification.

- Train Data – Contains historical sales records, showing total sales for each product family at a specific store on a given date.

- Transaction Data – Highly correlated with the sales data, helps in identifying store-wise sales patterns.

- Oil Price Data – Includes fluctuations in oil prices, helping us analyze how different product families are affected by economic changes.

- Holidays & Events Data – Provides insights into past sales trends and seasonal effects due to holidays and special events.

**Model Selection**

By using the Random Forest Regressor for store sales forecasting. This model is chosen due to its:

- Effectiveness in complex regression tasks

- Robustness to noisy data and outliers

- Minimal hyperparameter tuning compared to other models

- Capability to handle multivariate data efficiently

**Data Preparation & Feature Engineering**

To convert the time series data into a supervised learning problem:

- Merged the oil price, holidays, and stores data with historical sales records.

- Created additional features to enhance model performance.

- Handled missing values and visualized data trends for better understanding.

- Performed multiple iterations of feature engineering and model retraining to optimize accuracy.

**Model Evaluation**

Used Root Mean Squared Log Error (RMSLE) as our evaluation metric because:

- It is scale-invariant, making it independent of the target variable's scale.

- It encourages relative accuracy, prioritizing the proportional difference between predicted and actual values.

- The ideal RMSLE value is 0, with lower values indicating better model performance.

**Why Not Use RMSE?**

While RMSE is a common regression metric, RMSLE is preferred because:

- It handles differences in magnitude better.

- It mitigates the effect of large absolute errors, focusing on relative differences.

**Key Insights & Challenges**

- Data Preparation is Time-Consuming – Cleaning and transforming the data took a significant effort.

- Minimal Impact of Normalization/Standardization – Random Forest performed well without extensive scaling.

- Iterative Modeling Approach – improved model accuracy by continuously refining feature selection and training strategies.

**Conclusion**

This project demonstrates how machine learning can effectively forecast store sales. By integrating diverse datasets and using Random Forest Regressor I was able achieve a model that is robust, interpretable, and practical for real-world applications.
