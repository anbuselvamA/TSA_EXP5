# developer name : A.Anbuselvam 
# reg no : 212222240009
# Ex.No: 05  IMPLEMENTATION OF TIME SERIES ANALYSIS AND DECOMPOSITION
### Date: 


### AIM:
To Illustrates how to perform time series analysis and decomposition on the monthly average temperature of a city/country and for student performance dataset.

### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the decomposition process for the required data.
4. Plot the data according to need, either seasonal_decomposition or trend plot.
5. Display the overall results.

### PROGRAM:

``
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('/content/student_performance.csv')

# Create a DataFrame
df = pd.DataFrame(data)

# Simulate monthly average attendance rate over 24 months
months = pd.date_range(start='2021-01-01', periods=24, freq='M')
attendance_data = df['AttendanceRate'].mean() + np.random.normal(0, 5, 24)  # Simulated attendance rates

# Create a DataFrame for time series analysis
attendance_df = pd.DataFrame({
    'Month': months,
    'Average_AttendanceRate': attendance_data
})

# Set the Month as the index
attendance_df.set_index('Month', inplace=True)

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(attendance_df.index, attendance_df['Average_AttendanceRate'], marker='o', linestyle='-', color='b')
plt.title('Monthly Average Attendance Rate (Simulated)')
plt.xlabel('Date')
plt.ylabel('Average Attendance Rate (%)')
plt.grid()
plt.show()

# Perform time series decomposition
decomposition = seasonal_decompose(attendance_df['Average_AttendanceRate'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposition
plt.figure(figsize=(12, 10))
plt.subplot(4, 1, 1)
plt.plot(attendance_df.index, attendance_df['Average_AttendanceRate'], label='Original', color='b')
plt.legend(loc='upper left')
plt.title('Original Time Series')

plt.subplot(4, 1, 2)
plt.plot(attendance_df.index, trend, label='Trend', color='orange')
plt.legend(loc='upper left')
plt.title('Trend Component')

plt.subplot(4, 1, 3)
plt.plot(attendance_df.index, seasonal, label='Seasonal', color='green')
plt.legend(loc='upper left')
plt.title('Seasonal Component')

plt.subplot(4, 1, 4)
plt.plot(attendance_df.index, residual, label='Residual', color='red')
plt.legend(loc='upper left')
plt.title('Residual Component')

plt.tight_layout()
plt.show()
``

















# OUTPUT:


# Monthy Average Attendance Rate
# trend component
# seasona component 
# Residual component
![image](https://github.com/user-attachments/assets/455e4ead-39af-4633-b7f1-2771e15ddf6a)




### RESULT:
 we have created the python code for the time series analysis and decomposition.
