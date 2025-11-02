import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Define the model class (same as in training)
class simpleNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(input_size, 1, dtype=torch.float64))
        self.bias = nn.Parameter(torch.rand(1, dtype=torch.float64))

    def forward(self, X):
        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred

# Load the breast cancer dataset to get feature names and scaler
@st.cache_resource
def load_model_and_scaler():
    # Load dataset for feature names and scaling
    data = load_breast_cancer(return_X_y=False, as_frame=True)
    feature_names = data.feature_names.tolist()
    
    # Fit scaler on the full dataset
    scaler = StandardScaler()
    scaler.fit(data.data)
    
    # Load the trained model
    model = simpleNN(input_size=30)
    model.load_state_dict(torch.load('simple_nn_model.pth'))
    model.eval()
    
    return model, scaler, feature_names, data

# Streamlit UI
st.title("🏥 Breast Cancer Classification")
st.write("Predict whether a breast tumor is **Malignant** or **Benign** using a Neural Network")

# Load model and data
model, scaler, feature_names, data = load_model_and_scaler()

# Sidebar for input method
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Input", "Use Sample Data"])

if input_method == "Manual Input":
    st.header("Enter Feature Values")
    st.write("Please enter the values for all 30 features:")
    
    # Create input fields for all features
    user_input = {}
    cols = st.columns(3)
    
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            # Get min and max from dataset for better default values
            min_val = float(data.data[feature].min())
            max_val = float(data.data[feature].max())
            mean_val = float(data.data[feature].mean())
            
            user_input[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                format="%.4f",
                key=feature
            )
    
    input_data = np.array([list(user_input.values())])

else:
    st.header("Sample Data Selection")
    sample_idx = st.selectbox("Select a sample from the dataset:", range(len(data.data)))
    
    # Get the sample
    input_data = data.data.iloc[sample_idx].values.reshape(1, -1)
    actual_label = data.target.iloc[sample_idx]
    
    st.write(f"**Actual Label:** {'Malignant (0)' if actual_label == 0 else 'Benign (1)'}")
    
    # Show feature values in an expander
    with st.expander("View Feature Values"):
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_data[0]
        })
        st.dataframe(feature_df, use_container_width=True)

# Predict button
if st.button("🔍 Predict", type="primary"):
    # Preprocess input
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.from_numpy(input_scaled)
    
    # Make prediction
    with torch.no_grad():
        prediction = model.forward(input_tensor)
        prediction_prob = prediction.item()
        prediction_class = 1 if prediction_prob > 0.5 else 0
    
    # Display results
    st.header("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prediction", "Benign (1)" if prediction_class == 1 else "Malignant (0)")
    
    with col2:
        st.metric("Confidence", f"{prediction_prob * 100:.2f}%")
    
    # Progress bar for probability
    st.write("### Probability Score")
    st.progress(prediction_prob)
    
    # Interpretation
    if prediction_class == 1:
        st.success("✅ The tumor is predicted to be **BENIGN** (non-cancerous)")
    else:
        st.error("⚠️ The tumor is predicted to be **MALIGNANT** (cancerous)")
    
    st.info(f"Raw probability: {prediction_prob:.4f} (threshold: 0.5)")

# Footer
st.markdown("---")
st.caption("This is a demonstration model. Always consult medical professionals for actual diagnosis.")
