import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load data from Excel sheet
excel_file = 'answer.csv'
df = pd.read_csv(excel_file)


# Load the trained model
model = joblib.load("trained_model.pkl")


def predict_A_log_P(molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms):
    try:
        feature_vector = np.array(
            [[molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms]])
        predicted_A_log_P = model.predict(feature_vector)
        return predicted_A_log_P[0][0]
    except ValueError:
        return None


def plot_linear_regression_graph(subset_size, new_point_x=None):
    df_subset = df.head(subset_size)
    predicted_log_p = df_subset['Predicted AlogP'].values.reshape(-1, 1)
    actual_log_p = df_subset['Actual AlogP'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(predicted_log_p, actual_log_p)
    predictions = model.predict(predicted_log_p)

    fig, ax = plt.subplots(figsize=(8, 6))

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    equation = f'Actual log P = {slope:.2f} * Predicted log P + {intercept:.2f}'
    r_squared = r2_score(actual_log_p, predictions)
    equation_text = f'Equation: {equation}\nR-squared: {r_squared:.2f}'
    ax.text(0.5, -0.2, equation_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.5))

    ax.scatter(predicted_log_p, actual_log_p, label='Actual vs Predicted', s=5)
    ax.plot(predicted_log_p, predictions, color='red', label='Linear Regression Line')

    if new_point_x is not None:
        new_point_y = model.predict(np.array([[new_point_x]]))
        ax.scatter(new_point_x, new_point_y, color='black', label='AlogP')
        a = f'Actual AlogP:{new_point_y}'
        ax.text(0.5, -0.25, a, transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xlabel('Predicted log P')
    ax.set_ylabel('Actual log P')
    ax.set_title(f'Linear Regression: Actual vs Predicted log P ({subset_size} Values)')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# Function to Read Data from a Different Excel File
def read_data_from_excel(file_path):
    try:
        df_new = pd.read_excel(file_path)
        return df_new
    except Exception as e:
        st.error(f"Error reading data from Excel file: {e}")
        return None


def scatter_plot_and_histograms():
    st.title("Scatter Plot and Histograms")
    st.write("Working with a large dataset.............. \nThe page may become unresponsive at times!")
    scatter_fig = px.scatter(x=df['Predicted AlogP'], y=df['Actual AlogP'], title='Actual vs Predicted A log P',
                             labels={'x': 'Predicted A log P', 'y': 'Actual A log P'}, template='plotly_dark')
    scatter_fig.update_layout(showlegend=True)
    st.plotly_chart(scatter_fig)

    hist_actual_fig = px.histogram(x=df['Actual AlogP'], nbins=20, color_discrete_sequence=['blue'],
                                   labels={'x': 'Actual A log P'}, template='plotly_dark')
    hist_actual_fig.update_layout(barmode='overlay', title='Distribution of Actual A log P')
    st.plotly_chart(hist_actual_fig)

    hist_pred_fig = px.histogram(x=df['Predicted AlogP'], nbins=20, color_discrete_sequence=['green'],
                                 labels={'x': 'Predicted A log P'}, template='plotly_dark')
    hist_pred_fig.update_layout(barmode='overlay', title='Distribution of Predicted A log P')
    st.plotly_chart(hist_pred_fig)



def plot_selected_data_page():
    st.title("Plot Selected Data")
    st.markdown("This page allows you to plot different properties against each other to analyse their variations with respect to each other.")

    # Read the Excel file into a pandas DataFrame
    file_path = 'solubility_data.csv'
    dataset = pd.read_csv(file_path)

    # Ask the user for the number of data points to be taken
    num_data_points = st.number_input("Enter the number of data points to plot:", min_value=1, max_value=len(dataset), value=len(dataset))

    # Limit the dataset to the specified number of data points
    dataset = dataset.head(num_data_points)

    # Display column names for the user to choose the x-axis column using selectbox (dropdown)
    x_axis_column = st.selectbox("Select the column for the X-axis:", dataset.columns, index=0)
    X = dataset[x_axis_column]

    # Display column names for the user to choose the y-axis column using selectbox (dropdown)
    y_axis_column = st.selectbox("Select the column for the Y-axis:", dataset.columns, index=1)
    Y = dataset[y_axis_column]

    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(X, Y, '.')
    ax.set_title(f'{y_axis_column} vs {x_axis_column}')
    ax.set_ylabel(y_axis_column)
    ax.set_xlabel(x_axis_column)
    ax.grid(True)

    # Show the plot using st.pyplot
    st.pyplot(fig)

def main():
    palp = 0
    st.title("A log P Prediction")

    st.markdown(
        "This web app predicts the A log P value for a molecule based on its features. Enter the details and click 'Predict A log P.'")

    # Section for A log P prediction
    st.header("")

    col1, col2 = st.columns(2)

    with col1:
        molecular_weight = st.number_input("Molecular Weight", min_value=0.0)
        polar_surface_area = st.number_input("Polar Surface Area", min_value=0.0)
        hbd = st.number_input("Number of H-Bond Donors", min_value=0, step=1)
        hba = st.number_input("Number of H-Bond Acceptors", min_value=0, step=1)

    with col2:
        rotatable_bonds = st.number_input("Number of Rotatable Bonds", min_value=0, step=1)
        aromatic_rings = st.number_input("Number of Aromatic Rings", min_value=0, step=1)
        heavy_atoms = st.number_input("Number of Heavy Atoms", min_value=0.0)

    if st.button("Predict A log P"):
        result = predict_A_log_P(molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings,
                                 heavy_atoms)

        if result is not None:
            st.success(f"Predicted A log P: {result:.2f}")
            palp = result
        else:
            st.error("Invalid input. Please enter valid numeric values for all features.")

    st.markdown("---")  # Add a horizontal line to separate sections
    #Section for linear regression graph
    st.header("Linear Regression Graph")
    subset_size = st.slider("Select the number of values for the linear regression graph", min_value=10,
                      max_value=len(df), value=100)

    plot_linear_regression_graph(subset_size, new_point_x=palp)


# Main Execution
if __name__ == "__main__":
    # Page options and descriptions
    pages = {
        'A log P Prediction': main,
        'Scatter Plot and Histograms': scatter_plot_and_histograms,
        'Plot Selected Data': plot_selected_data_page
    }

    # Descriptions for each page
    page_descriptions = {
        'A log P Prediction': "This page provides a predictive model for A log P values.",
        'Scatter Plot and Histograms': "Visualize data using scatter plots and histograms on this page.",
        'Plot Selected Data': "Select specific data points from your dataset and plot them here."
    }

    # Sidebar navigation through clickable buttons
    st.sidebar.write("Page Navigation:")

    # Variable to store the current selection
    selected_page = None

    # Create a button for each page
    for page in pages.keys():
        if st.sidebar.button(page):
            selected_page = page

    # If no page is selected, default to the first page
    if not selected_page:
        selected_page = list(pages.keys())[0]

    # Call the function corresponding to the selected page
    pages[selected_page]()


