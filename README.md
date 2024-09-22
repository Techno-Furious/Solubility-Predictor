# A log P Prediction Web Application

This repository contains a web application designed to predict the **A log P** value for small molecules based on various molecular features. The application also allows users to visualize data, analyze linear regression results, and explore correlations between different molecular properties. 
This was a project that I built during my 1st semester of Engineering.

The dataset used in this project is obtained from Kaggle and has been pre-processed to include only small molecules with a set of key features.

## Deployed Web Application

You can access the deployed web app [here](https://techno-furious-solubility-predictor-streamapp-pmmlbq.streamlit.app/).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview
This project predicts the **A log P** value, which is an important physicochemical property of small molecules. **A log P** measures the lipophilicity of a compound, indicating its ability to dissolve in fats, oils, and non-polar solvents. The app also includes visual tools to compare actual vs. predicted values and analyze relationships between molecular properties.

## Features
- **Predict A log P**: Input molecular features and predict the corresponding A log P value using a trained linear regression model.
- **Data Visualization**: Visualize scatter plots and histograms of predicted vs. actual A log P values.
- **Linear Regression Analysis**: Perform linear regression analysis on the dataset to determine relationships between the predicted and actual values.
- **Custom Data Plotting**: Choose molecular features from the dataset and create custom plots to explore how these features interact with each other.

## Dataset
You can find the original dataset [here](https://www.kaggle.com/code/ahmedelmaamounamin/solubility-prediction/input).
The dataset is sourced from **Kaggle** and contains information about small molecules, including the following features:
- Molecular Weight
- Polar Surface Area
- Number of Hydrogen Bond Donors (HBD)
- Number of Hydrogen Bond Acceptors (HBA)
- Number of Rotatable Bonds
- Number of Aromatic Rings
- Number of Heavy Atoms
- **A log P** (target variable)

## Model Training
The machine learning model used in this project is a simple linear regression model trained to predict A log P based on the features listed above. Gradient descent is used to update the model weights over multiple epochs.

### Steps:
1. **Data Preprocessing**: The dataset is cleaned, and missing values are handled.
2. **Model Training**: A linear regression model is trained using gradient descent, with periodic checks on validation accuracy.
3. **Model Evaluation**: After training, the model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics.

The trained model is saved using the `joblib` library for later use in the web application.

## Web Application
The web application is built using **Streamlit** and **Plotly** for interactive data visualization.

### Main Pages:
- **A log P Prediction**: Users can input molecular properties to predict the A log P value.
- **Scatter Plot and Histograms**: Visualize how well the predicted A log P values match the actual values, and explore the distribution of both.
- **Plot Selected Data**: Customize plots by selecting different molecular features to examine their relationships.

## Installation

### Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.x
- Streamlit
- Plotly
- scikit-learn
- numpy
- pandas

### Installation Steps
1. **Clone the repository**:
    ```bash
    git clone https://github.com/Techno-Furious/Solubility-Predictor.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd Solubility-Predictor
    ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**:
    ```bash
    streamlit run streamapp.py
    ```

5. **Open your browser** and navigate to `http://localhost:8501` to access the web application.

## Usage
### Predicting A log P
- Open the web app.
- Go to the **A log P Prediction** page.
- Input the molecular features (Molecular Weight, Polar Surface Area, HBD, HBA, etc.).
- Click "Predict A log P" to get the predicted value.

### Visualizing Data
- Go to the **Scatter Plot and Histograms** page.
- View the scatter plot of **Predicted vs Actual** A log P values.
- Explore the histograms of the **distribution** of both actual and predicted values.

### Custom Plotting
- Navigate to the **Plot Selected Data** page.
- Select any two features from the dataset.
- The app will display a plot showing the relationship between the chosen features.
