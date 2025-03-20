import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load the dataset
housing = pd.read_csv("AmesHousing.csv", na_values=[""], keep_default_na=False)

# Customize the output of all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Display the first few rows and dataset information
print(housing.head())
print(housing.info())
print(housing.describe())

# ---------------------- Handle missing values ---------------------- #
print(housing.isnull().sum())

# Fill missing values in 'Lot Frontage' with 0 (assumption of no frontage)
housing['Lot Frontage'] = housing['Lot Frontage'].fillna(0)

# Fill missing values in both columns 'Mas Vnr Type' and 'Mas Vnr Area'
housing.loc[housing['Mas Vnr Type'].isnull() & housing['Mas Vnr Area'].isnull(), ['Mas Vnr Type', 'Mas Vnr Area']] = ['None', 0]

# Fill missing values in 'Garage Yr Blt' with 0
housing['Garage Yr Blt'] = housing['Garage Yr Blt'].fillna(0)

# Display unique values in basement-related columns
print(f"Unique values in column 'Bsmt Qual':")
print(housing['Bsmt Qual'].unique())
print(f"Unique values in column 'Bsmt Cond':")
print(housing['Bsmt Cond'].unique())
print(f"Unique values in column 'Bsmt Exposure':")
print(housing['Bsmt Exposure'].unique())
print(f"Unique values in column 'BsmtFin Type 1':")
print(housing['BsmtFin Type 1'].unique())
print(f"Unique values in column 'BsmtFin SF 1':")
print(housing['BsmtFin SF 1'].unique())
print(f"Unique values in column 'BsmtFin Type 2':")
print(housing['BsmtFin Type 2'].unique())
print(f"Unique values in column 'BsmtFin SF 2':")
print(housing['BsmtFin SF 2'].unique())

# Find rows where there is a basement, but the quality is not specified
missing_bsmt_qual = housing[(housing['Total Bsmt SF'] > 0) & (housing['Bsmt Qual'].isnull())]
print("Rows where there is a basement, but the quality is not specified:")
print(missing_bsmt_qual)

# Related columns for basement
related_cols = ['Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1', 'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF']
print(missing_bsmt_qual[related_cols])

# List of categorical variables related to the basement
bsmt_cat_cols = ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']

# Fill missing values in categorical basement variables with 'NA'
housing[bsmt_cat_cols] = housing[bsmt_cat_cols].fillna('NA')

# List of numerical variables related to the basement
bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF']

# Fill missing values in numerical basement variables with 0
housing[bsmt_num_cols] = housing[bsmt_num_cols].fillna(0)

# Display unique values in garage-related columns
print(f"Unique values in column 'Garage Type':")
print(housing['Garage Type'].unique())
print(f"Unique values in column 'Garage Finish':")
print(housing['Garage Finish'].unique())
print(f"Unique values in column 'Garage Cars':")
print(housing['Garage Cars'].unique())
print(f"Unique values in column 'Garage Qual':")
print(housing['Garage Qual'].unique())
print(f"Unique values in column 'Garage Cond':")
print(housing['Garage Cond'].unique())

# List of categorical variables related to the garage
garage_cat_cols = ['Garage Finish', 'Garage Qual', 'Garage Cond']

# List of numerical variables related to the garage
garage_num_cols = ['Garage Cars', 'Garage Area']

# If 'Garage Type' indicates no garage, fill missing values
housing.loc[housing['Garage Type'] == 'NA', garage_cat_cols] = 'NA'
housing.loc[housing['Garage Type'] == 'NA', garage_num_cols] = 0

# Find rows with missing values in garage-related variables
missing_rows = housing[housing[['Garage Finish', 'Garage Cars', 'Garage Area', 'Garage Qual', 'Garage Cond']].isnull().any(axis=1)]
print("Rows with missing values in garage-related variables:")
print(missing_rows)

# Replace the year the garage was built with the year the house was built if the garage type is not 'NA' and there are missing values
housing.loc[(housing['Garage Type'] != 'NA') & (housing['Garage Yr Blt'].isnull()), 'Garage Yr Blt'] = housing.loc[(housing['Garage Type'] != 'NA') & (housing['Garage Yr Blt'].isnull()), 'Year Built']

# Replace NaN in 'Garage Cars' with the mode
housing['Garage Cars'] = housing['Garage Cars'].fillna(housing['Garage Cars'].mode()[0])

# Replace NaN in categorical garage variables with 'NA'
housing[garage_cat_cols] = housing[garage_cat_cols].fillna('NA')

# Identify the row with missing 'Garage Area'
missing_garage_area = housing[housing['Garage Area'].isnull()]

# Get the number of cars for this row
garage_cars_value = missing_garage_area['Garage Cars'].iloc[0]

# Calculate the median 'Garage Area' for this number of cars
median_garage_area = housing[housing['Garage Cars'] == garage_cars_value]['Garage Area'].median()

# Fill the missing 'Garage Area' with the calculated median
housing.loc[housing['Garage Area'].isnull(), 'Garage Area'] = median_garage_area

# Find rows with missing values in garage-related variables
missing_rows = housing[housing[['Garage Finish', 'Garage Cars', 'Garage Area', 'Garage Qual', 'Garage Cond']].isnull().any(axis=1)]
print("Rows with missing values in garage-related variables:")
print(missing_rows)

# Display unique values in basement bathroom-related columns
print(f"Unique values in column 'Bsmt Full Bath':")
print(housing['Bsmt Full Bath'].unique())
print(f"Unique values in column 'Bsmt Half Bath':")
print(housing['Bsmt Half Bath'].unique())

# Replace NaN with 0 in 'Bsmt Full Bath' and 'Bsmt Half Bath'
housing['Bsmt Full Bath'] = housing['Bsmt Full Bath'].fillna(0)
housing['Bsmt Half Bath'] = housing['Bsmt Half Bath'].fillna(0)

# Display unique values in 'Electrical' column
print(housing['Electrical'].unique())

# Calculate the mode for the 'Electrical' column
mode_electrical = housing['Electrical'].mode()[0]

# Replace missing values with the mode
housing['Electrical'] = housing['Electrical'].fillna(mode_electrical)

# Display the count of missing values
print(housing.isnull().sum())

# ----------------------------- Normalize variables ----------------------------- #
# Since I will use a linear regression model to predict the target variable, I will apply standardization
# Numerical variables in the dataset:
# Large range: Need to standardize.
# Lot Frontage, Lot Area, Mas Vnr Area, 1st Flr SF, 2nd Flr SF, Low Qual Fin SF, Gr Liv Area, Garage Area, Wood Deck SF, Open Porch SF, Enclosed Porch, SalePrice.
# Small range (may not need standardization):
# Overall Qual, Overall Cond, Bsmt Full Bath, Bsmt Half Bath, Full Bath, Half Bath, Bedroom AbvGr, Kitchen AbvGr, TotRms AbvGrd, Fireplaces, Garage Cars.
# Categorical variables:
# MS SubClass, MS Zoning, Street, Alley, Lot Shape, Land Contour, Neighborhood, Garage Type, etc.

# Select numerical variables for standardization
numeric_cols = ['Lot Frontage', 'Lot Area', 'Mas Vnr Area', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', 'SalePrice']

# Initialize StandardScaler
scaler = StandardScaler()

# Standardize numerical variables
housing[numeric_cols] = scaler.fit_transform(housing[numeric_cols])

# Display the first few rows and dataset description after standardization
print(housing.head())
print(housing.describe())

# Check the mean and standard deviation of standardized variables
print(housing[numeric_cols].mean())  # Should be close to 0
print(housing[numeric_cols].std())   # Should be close to 1

# --------------------------- Feature engineering --------------------------- #
# 1. Create new features
# 1. Price per square foot (or meter):
# Price per SqFt = SalePrice / Gr Liv Area
housing['Price per SqFt'] = housing['SalePrice'] / housing['Gr Liv Area']

# 2. Age of the house at the time of sale:
# House Age = Yr Sold - Year Built
housing['House Age'] = housing['Yr Sold'] - housing['Year Built']

# 3. Time since last remodel:
# Since Remodel = Yr Sold - Year Remod/Add
housing['Since Remodel'] = housing['Yr Sold'] - housing['Year Remod/Add']

# 4. Total house area:
# Total House Area = 1st Flr SF + 2nd Flr SF + Total Bsmt SF
housing['Total House Area'] = housing['1st Flr SF'] + housing['2nd Flr SF'] + housing['Total Bsmt SF']

# 5. Total outdoor area:
housing['Total Outdoor Area'] = housing['Wood Deck SF'] + housing['Open Porch SF'] + housing['Enclosed Porch'] + housing['Screen Porch'] + housing['3Ssn Porch']

# Check new features
print(housing[['Price per SqFt', 'House Age', 'Since Remodel', 'Total House Area', 'Total Outdoor Area']].head())

# Select new features for standardization
new_features = ['Price per SqFt', 'House Age', 'Since Remodel', 'Total House Area', 'Total Outdoor Area']

# Standardize new features
housing[new_features] = scaler.fit_transform(housing[new_features])

# Check standardization of new features
print(housing[new_features].describe())

#--------------------------------------- Exploratory Data Analysis (EDA) ---------------------------------------#

# Create a correlation matrix for numerical data
numerical_cols = housing.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = housing[numerical_cols].corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, fmt=".5f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# Select variables with high correlation
high_corr_features = correlation_matrix['SalePrice'][correlation_matrix['SalePrice'].abs() > 0.5]
print("Variables with high correlation with SalePrice:")
print(high_corr_features)

# Variables with high correlation: Overall Qual, Year Built, Year Remod/Add, Mas Vnr Area, Total Bsmt SF, 1st Flr SF, Gr Liv Area, Full Bath, Garage Cars, Garage Area, House Age, Since Remodel, Total House Area
fig, axes = plt.subplots(3, 5, figsize=(20, 20))  # Larger figure size
axes = axes.flatten()
features = high_corr_features.index.drop("SalePrice")  # Select variables except SalePrice

# Plot scatter plots
for i, feature in enumerate(features):
    sns.scatterplot(data=housing, x=feature, y='SalePrice', alpha=0.6, ax=axes[i])
    axes[i].set_title(f"{feature} vs SalePrice", fontsize=14)
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel("SalePrice", fontsize=12)
    axes[i].tick_params(axis='both', which='major', labelsize=10)

# Remove extra subplots
for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Get the list of variable names
selected_features = high_corr_features.index

# Create a correlation matrix for the selected variables
hcf_correlation_matrix = housing[selected_features].corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(hcf_correlation_matrix, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix for Highly Correlated Variables")
plt.show()

# Remaining variables: Overall Qual, Year Built, Year Remod/Add, Mas Vnr Area, Gr Liv Area, Full Bath, Total House Area
# Removed variables: Total Bsmt SF, 1st Flr SF, Garage Cars, Garage Area, House Age, Since Remodel

#----------------------- Create Predictive Model ----------------------------#
# Implement a linear regression model using scikit-learn.

# Independent variables (features) and target variable
X = housing[['Overall Qual', 'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 'Gr Liv Area', 'Full Bath', 'Garage Cars', 'Total House Area']]
y = housing['SalePrice']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
lin_reg = LinearRegression()

# Train the model
lin_reg.fit(X_train, y_train)

# Predict for test data
y_pred = lin_reg.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {rmse}")
print(f"R-squared (R2): {r2}")
print(f"Mean Absolute Error (MAE): {mae}")

#------ Optimize the model using regularization techniques (Lasso, Ridge) ------#

# Search for optimal alpha for Ridge
ridge_params = {'alpha': [0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5)
ridge_grid.fit(X_train, y_train)
print("Best alpha for Ridge:", ridge_grid.best_params_)

# Search for optimal alpha for Lasso
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1]}
lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=5)
lasso_grid.fit(X_train, y_train)
print("Best alpha for Lasso:", lasso_grid.best_params_)

# Ridge (L2) Regularization
ridge = Ridge(alpha=100.0)
ridge.fit(X_train, y_train)

# Prediction for Ridge
y_pred_ridge = ridge.predict(X_test)

# Evaluate the Ridge model
print("Evaluation of the Ridge model")
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)
print(f"Mean Squared Error (MSE): {ridge_rmse}")
print(f"R-squared (R2): {ridge_r2}")
print(f"Mean Absolute Error (MAE): {ridge_mae}")

# Lasso (L1) Regularization
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)

# Prediction for Lasso
y_pred_lasso = lasso.predict(X_test)
print("Evaluation of the Lasso model")

# Evaluate the Lasso model
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_rmse = np.sqrt(lasso_mse)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)
print(f"Mean Squared Error (MSE): {lasso_rmse}")
print(f"R-squared (R2): {lasso_r2}")
print(f"Mean Absolute Error (MAE): {lasso_mae}")