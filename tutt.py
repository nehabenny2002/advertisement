
import pandas as pd  # Import pandas for data manipulation.
import statsmodels.formula.api as sm  # Import statsmodels for regression.
import numpy as np  # Import numpy for numerical operations.

# Load the data
data = pd.read_csv('setadv.csv')  # Read the CSV file into a DataFrame.

# Remove the index column
data = data.drop(columns=[data.columns[0]])  # Drop the first column (assumed index).

# Model with TV, Radio, and Newspaper
model = sm.ols(formula='sales ~ TV + radio + newspaper', data=data).fit()  # Create and fit the regression model.
print("Model with TV, Radio, and Newspaper:\n", model.summary())  # Print the model summary.

# Model with only TV and Radio
model_tv_radio = sm.ols(formula='sales ~ TV + radio', data=data).fit()  # Create and fit the regression model.
print("\nModel with TV and Radio:\n", model_tv_radio.summary())  # Print summary of this model.

# Model with only TV
model_tv = sm.ols(formula='sales ~ TV', data=data).fit()  # Create and fit the regression model.
print("\nModel with TV only:\n", model_tv.summary())  # Print summary of this model.

# Extract RSE, R-squared, and F-statistic for the full model
rse = np.sqrt(model.mse_resid)  # Calculate RSE.
r_squared = model.rsquared  # Extract R-squared.
f_statistic = model.fvalue  # Extract F-statistic.
f_pvalue = model.f_pvalue # Extract F-statistic p-value

print("\nFull Model Metrics:")  # Header.
print(f"  RSE: {rse:.3f}")  # Print RSE.
print(f"  R-squared: {r_squared:.3f}")  # Print R-squared.
print(f"  F-statistic: {f_statistic:.3f}")  # Print F-statistic.
print(f"  F-statistic p-value: {f_pvalue:.3f}") # Print F-statistic 



