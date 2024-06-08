import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Set up Kaggle API credentials
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)

kaggle_json = '{"username":"melisanyh","key":"fd660fc7a7f36f794eb2399af3299c0d"}'

with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    f.write(kaggle_json)

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
dataset = 'nikhil7280/student-performance-multiple-linear-regression'
destination_folder = 'Student_Performance'
api.dataset_download_files(dataset, path=destination_folder, unzip=True)

# Load data
file_path = os.path.join(destination_folder, 'Student_Performance.csv')  # Sesuaikan dengan nama file yang benar
data = pd.read_csv(file_path)

# Extract relevant columns
X = data[['Hours Studied', 'Sample Question Papers Practiced']]
y = data['Performance Index']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Calculate RMS error for linear model
rms_linear = sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

# Plotting Linear Model
plt.scatter(X_test['Hours Studied'], y_test, color='blue', label='Data points')
plt.plot(X_test['Hours Studied'], y_pred_linear, color='orange', label='Linear regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Exponential Model
# Apply log transformation to the target variable
y_log_train = np.log(y_train)
exp_model = LinearRegression()
exp_model.fit(X_train, y_log_train)
y_pred_log = exp_model.predict(X_test)
y_pred_exp = np.exp(y_pred_log)

# Calculate RMS error for exponential model
rms_exp = sqrt(mean_squared_error(y_test, y_pred_exp))
r2_exp = r2_score(y_test, y_pred_exp)

# Plotting Exponential Model
plt.scatter(X_test['Hours Studied'], y_test, color='blue', label='Data points')
plt.plot(X_test['Hours Studied'], y_pred_exp, color='orange', label='Exponential regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Exponential Regression')
plt.legend()
plt.show()

# Print RMS errors and R-squared values
print(f'RMS Error (Linear Model): {rms_linear}')
print(f'R-squared (Linear Model): {r2_linear}')
print(f'RMS Error (Exponential Model): {rms_exp}')
print(f'R-squared (Exponential Model): {r2_exp}')
