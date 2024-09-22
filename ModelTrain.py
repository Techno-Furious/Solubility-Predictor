import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
# Load the dataset using pandas with the specified delimiter
df = pd.read_csv("All Drugs.csv", delimiter=';', on_bad_lines='skip') # Dataset Obtained from Kaggle

# Check if the file is loaded successfully
if not df.empty:
    print(f"Dataset loaded successfully.")
else:
    print(f"Failed to load the dataset.")
# Check the column names in the DataFrame
print(df.columns)
# Filter the DataFrame to include only rows where Type is "Small molecule"
small_molecules_df = df[df['Type'] == 'Small molecule']

# Check if any rows match the filter condition
if not small_molecules_df.empty:
    print(f"Filtered DataFrame contains {len(small_molecules_df)} rows with Type 'Small molecule'.")
else:
    print("No rows match the filter condition.")

# Save the filtered DataFrame as a new CSV file called "small_molecules.csv"
small_molecules_df.to_csv("small_molecules.csv", index=False)

print("Filtered DataFrame saved as 'small_molecules.csv'.")


# Read the small_molecules.csv file
small_molecules_df = pd.read_csv("small_molecules.csv")

# List of columns to keep
columns_to_keep = [
    "Molecular Weight",
    "AlogP",
    "Polar Surface Area",
    "HBD",
    "HBA",
    "#Rotatable Bonds",
    "Aromatic Rings",
    "Heavy Atoms",
]

# Filter the DataFrame to include only the selected columns
filtered_small_molecules_df = small_molecules_df[columns_to_keep]

# Check if any rows match the filter condition
if not filtered_small_molecules_df.empty:
    print(f"Filtered DataFrame contains {len(filtered_small_molecules_df)} rows with selected columns.")
else:
    print("No rows match the filter condition.")

# Save the filtered DataFrame as a new CSV file called "solubility_data.csv"
filtered_small_molecules_df.to_csv("solubility_data.csv", index=False)

print("Filtered DataFrame with selected columns saved as 'solubility_data.csv'.")


#Display the list of columns and the data
data = pd.read_csv('solubility_data.csv')
x = data[['Molecular Weight', 'Polar Surface Area', 'HBD', 'HBA', '#Rotatable Bonds', 'Aromatic Rings', 'Heavy Atoms']]
# Check the shape of the feature matrix (X)
print(x.shape)
print(x)

# Convert the specified columns to float64
columns_to_convert = ['AlogP', 'Polar Surface Area', 'HBD', 'HBA', '#Rotatable Bonds', 'Aromatic Rings', 'Heavy Atoms']

for column in columns_to_convert:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Calculate missing values
data.isnull().sum()

# To remove rows with missing values (NaN)
data.dropna(inplace=True)

# To impute missing values with the mean
data.fillna(data.mean(), inplace=True)


# Double check that no more missing values
data.isnull().sum()


x = data[['Molecular Weight', 'Polar Surface Area', 'HBD', 'HBA', '#Rotatable Bonds', 'Aromatic Rings', 'Heavy Atoms']]
y = data['AlogP']

N = len(x)
print(N)

# Define a numpy array containing ones and concatenate it with the feature matrix
ones = np.ones(N)
Xp = np.c_[ones, x]

# Reshape the target variable (y) to be a column vector
y = y.values.reshape(-1, 1)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(Xp, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the weights (coefficients) with random values
w = 2 * np.random.rand(8, 1) - 1  # Shape (8, 1) for 7 features + bias term


# Function to train the model and return the trained model
def train_model(X_train, y_train):
    # Initialize the weights (coefficients) with random values for training data
    w = 2 * np.random.rand(8, 1) - 1  # Shape (8, 1) for 7 features + bias term

    # Define the number of training epochs and the learning rate
    epochs = 100000
    learning_rate = 0.00001

    # Training loop
    for epoch in range(epochs):
        # Calculate the predicted values using the current weights for training data
        y_train_predicted = X_train @ w

        # Calculate the error for training data
        train_error = y_train - y_train_predicted

        # Calculate the gradient of the loss with respect to the weights for training data
        train_gradient = -(1/X_train.shape[0]) * X_train.T @ train_error

        # Update the weights using gradient descent for training data
        w = w - learning_rate * train_gradient

        # Calculate the mean squared error (L2 loss) for training data
        train_L2 = 0.5 * np.mean(train_error**2)

        # Print progress every 10% of the epochs
        if epoch % (epochs / 10) == 0:
            print(f"Epoch {epoch}: Training L2 Loss = {train_L2}")

    # Create a trained Linear Regression model
    reg = LinearRegression()
    reg.coef_ = w[1:].T  # Set the coefficients
    reg.intercept_ = w[0][0]  # Set the intercept

    return reg

# Train the model using your training data
trained_model = train_model(X_train, y_train)

# Save the trained model to use later
joblib.dump(trained_model, 'trained_model.pkl')


#-------------------------------------------------------------------------------------------------------------------------



# Function to calculate mean squared error (MSE)
def calculate_mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)

# Function to train the model and return the trained model and validation accuracy
def train_model(X_train, y_train, X_val, y_val, current_model=None):
    # If a model is provided, use its weights as initialization
    if current_model is not None:
        w = np.vstack((current_model.intercept_, current_model.coef_.flatten()))
    else:
        # Otherwise, initialize weights with random values
        w = 2 * np.random.rand(8, 1) - 1  # Shape (8, 1) for 7 features + bias term

    # Define the number of training epochs and the learning rate
    epochs = 100000
    learning_rate = 0.00001

    best_val_accuracy = float('-inf')
    best_model = None

    # Training loop
    for epoch in range(epochs):
        # Calculate the predicted values using the current weights for training data
        y_train_predicted = X_train @ w

        # Calculate the error for training data
        train_error = y_train - y_train_predicted

        # Calculate the gradient of the loss with respect to the weights for training data
        train_gradient = -(1/X_train.shape[0]) * X_train.T @ train_error

        # Update the weights using gradient descent for training data
        w = w - learning_rate * train_gradient

        # Calculate the mean squared error (L2 loss) for training data
        train_mse = calculate_mse(y_train_predicted, y_train)

        # Calculate the predicted values for validation data
        y_val_predicted = X_val @ w

        # Calculate the mean squared error (L2 loss) for validation data
        val_mse = calculate_mse(y_val_predicted, y_val)

        # Print progress every 10% of the epochs
        if epoch % (epochs / 10) == 0:
            print(f"Epoch {epoch}: Training MSE = {train_mse}, Validation MSE = {val_mse}")

        # Update the best model if the validation accuracy improves
        if val_mse > best_val_accuracy:
            best_val_accuracy = val_mse
            best_model = LinearRegression()
            best_model.coef_ = w[1:].T  # Set the coefficients
            best_model.intercept_ = w[0][0]  # Set the intercept

            # Save the best model to use later
            joblib.dump(best_model, 'best_model.pkl')

        # Check if the target accuracy is reached
        if best_val_accuracy >= 0.95:
            print(f"Target accuracy reached. Stopping training.")
            break

    return best_model

# Train the model using your training and validation data
current_model = None  # Set to existing model if available
trained_model = train_model(X_train, y_train, X_val, y_val, current_model)
