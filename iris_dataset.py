import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  

# Task 1: Load and Explore the Dataset  

# Load the Iris dataset  
try:  
    # Using seaborn to load the iris dataset  
    iris = sns.load_dataset('iris')  
except Exception as e:  
    print(f"Error loading dataset: {e}")  

# Display the first few rows of the dataset  
print("First few rows of the dataset:")  
print(iris.head())  

# Explore the structure of the dataset  
print("\nData types and missing values:")  
print(iris.info())  

# Check for missing values  
missing_values = iris.isnull().sum()  
print("\nMissing values in each column:")  
print(missing_values)  

# Clean the dataset (in this case, there are no missing values to fill or drop)  
# If there were missing values, we could use:  
# iris.dropna(inplace=True)  # To drop rows with missing values  
# or  
# iris.fillna(value=0, inplace=True)  # To fill missing values with 0  

# Task 2: Basic Data Analysis  

# Compute basic statistics of the numerical columns  
print("\nBasic statistics of the numerical columns:")  
print(iris.describe())  

# Perform groupings on the 'species' column and compute the mean of 'petal_length'  
grouped_data = iris.groupby('species')['petal_length'].mean().reset_index()  
print("\nAverage petal length per species:")  
print(grouped_data)  

# Identify any patterns or interesting findings  
print("\nInteresting findings:")  
print("The average petal length varies significantly among different species.")  

# Task 3: Data Visualization  

# Set the style for seaborn  
sns.set(style="whitegrid")  

# 1. Line chart (not directly applicable to the Iris dataset, so we'll skip this)  
# 2. Bar chart showing average petal length per species  
plt.figure(figsize=(8, 5))  
sns.barplot(data=grouped_data, x='species', y='petal_length', palette='viridis')  
plt.title('Average Petal Length per Species')  
plt.xlabel('Species')  
plt.ylabel('Average Petal Length (cm)')  
plt.legend(title='Species')  
plt.show()  

# 3. Histogram of 'sepal_length' to understand its distribution  
plt.figure(figsize=(8, 5))  
sns.histplot(iris['sepal_length'], bins=10, kde=True)  
plt.title('Distribution of Sepal Length')  
plt.xlabel('Sepal Length (cm)')  
plt.ylabel('Frequency')  
plt.show()  

# 4. Scatter plot to visualize the relationship between 'sepal_length' and 'petal_length'  
plt.figure(figsize=(8, 5))  
sns.scatterplot(data=iris, x='sepal_length', y='petal_length', hue='species', style='species', s=100)  
plt.title('Sepal Length vs Petal Length')  
plt.xlabel('Sepal Length (cm)')  
plt.ylabel('Petal Length (cm)')  
plt.legend(title='Species')  
plt.show()