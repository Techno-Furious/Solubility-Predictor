import pandas as pd
import numpy as np
import joblib
import time

t1 = time.time()

# Load the trained model
model = joblib.load("trained_model.pkl")

def predict_A_log_P(model, molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms):
    # Create a feature vector
    feature_vector = np.array([[molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms]])

    # Predict A log P using the provided model
    predicted_A_log_P = model.predict(feature_vector)

    return predicted_A_log_P[0][0]

# Read solubility data from CSV file
solubility_data = pd.read_csv("solubility_data.csv")

# Create answer.csv and write header
answer_df = pd.DataFrame(columns=["Predicted AlogP", "Actual AlogP"])
answer_df.to_csv("answers.csv", index=False)

# Process each row in solubility_data
for index, row in solubility_data.iterrows():
    try:
        # Extract features from the current row
        molecular_weight, AlogP, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms = row

        # Predict AlogP
        predicted_AlogP = predict_A_log_P(model, molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds,
                                          aromatic_rings, heavy_atoms)

        # Append results to answer.csv
        result_row = pd.DataFrame([[predicted_AlogP, AlogP]], columns=["Predicted AlogP", "Actual AlogP"])
        result_row.to_csv("answers.csv", mode="a", header=False, index=False)

        # Print progress
        print(f"Row {index + 1} completed")

    except Exception as e:
        print(f"Error processing row {index + 1}: {str(e)}")

print("Prediction complete. Results saved in answer.csv")
t2 = time.time()

print("The time taken is: ",t2-t1)