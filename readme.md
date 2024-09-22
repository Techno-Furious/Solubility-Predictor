A Log P Prediction Model and Visualization Web Application
This repository contains a web application for predicting A Log P values using a linear regression model trained on molecular properties. The application is built using Streamlit and allows users to visualize the relationship between actual and predicted values through scatter plots, histograms, and regression analysis.

Features
A Log P Prediction: Enter molecular properties such as molecular weight, polar surface area, H-bond donors/acceptors, rotatable bonds, aromatic rings, and heavy atoms to predict the A Log P value using a pre-trained linear regression model.
Data Visualization: Explore visualizations including scatter plots, histograms, and regression lines to understand the relationship between molecular properties and predicted A Log P.
Interactive Plotting: Select and plot various molecular properties to analyze their behavior and distribution.
Model Training: The project includes code for training the model using linear regression with gradient descent.
Getting Started
Prerequisites
Ensure you have the following installed on your machine:

Python 3.x
Required Python packages (can be installed using requirements.txt)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/alogp-prediction.git
cd alogp-prediction
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Make sure you have the necessary CSV files:

solubility_data.csv: Contains molecular property data.
answer.csv: Stores the predicted and actual A Log P values.
Running the Application
Start the Streamlit application:

bash
Copy code
streamlit run app.py
Use the Application:

Open the URL shown in your terminal (usually http://localhost:8501/) in your web browser.
Navigate through the different pages using the sidebar to predict A Log P values or visualize the dataset.
Directory Structure
bash
Copy code
├── app.py                       # Main Streamlit app file
├── All Drugs.csv                # Original dataset from Kaggle
├── solubility_data.csv           # Filtered dataset used for training
├── answers.csv                   # Predicted and actual A log P results
├── trained_model.pkl             # Pre-trained Linear Regression model
├── requirements.txt              # List of required Python packages
├── README.md                     # Project documentation
└── .gitignore                    # Ignore unnecessary files in Git
Dataset Information
All Drugs.csv: Contains raw molecular data. It's filtered based on the "Small molecule" type, and relevant columns are extracted to build the training dataset (solubility_data.csv).
solubility_data.csv: Contains features like molecular weight, polar surface area, H-bond donors (HBD), H-bond acceptors (HBA), rotatable bonds, aromatic rings, and heavy atoms for each molecule.
Model Training
The linear regression model is trained using a custom gradient descent algorithm. It takes into account the following molecular properties to predict A Log P:

Molecular Weight
Polar Surface Area
Number of H-bond Donors
Number of H-bond Acceptors
Number of Rotatable Bonds
Aromatic Rings
Heavy Atoms
Training is performed using the train_model function found in app.py, and the final model is saved as trained_model.pkl.

Model Evaluation
The model is evaluated using the following metrics:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R-squared (R²)
The regression line and R² values are displayed in the application when plotting the Actual vs Predicted A Log P values.

Usage
A Log P Prediction
Input molecular properties (e.g., molecular weight, polar surface area, number of hydrogen bond donors/acceptors, etc.).
Click on "Predict A Log P".
The predicted value will be shown on the screen.
Data Visualization
Scatter Plot & Histograms: Display a scatter plot of actual vs predicted A Log P values and view histograms to analyze the distribution of predicted and actual values.
Plot Selected Data: Select and visualize the relationships between different molecular properties.