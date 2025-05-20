import streamlit as st
import pickle
import os
import numpy as np
st.set_page_config(
    page_title="Cylinder Cost Prediction",
    layout="centered",
    initial_sidebar_state="expanded",
)
import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
        background-image: 
            url('https://raw.githubusercontent.com/Lasya03/CylCP/refs/heads/main/ALLCYL/img1.png'),
            url('https://raw.githubusercontent.com/Lasya03/CylCP/refs/heads/main/ALLCYL/img2.png');
        background-repeat: repeat;
        background-position: 0 0, 5px 5px; /* Offset the second image */
        background-size: 50px 50px; /* Adjust spacing */
        opacity: 1;
    }

    /* Apply opacity only to images using overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: 
            url('https://raw.githubusercontent.com/Lasya03/CylCP/main/ALLCYL/drawing1.png'),
            url('https://raw.githubusercontent.com/Lasya03/CylCP/main/ALLCYL/drawing2.png');
        background-repeat: repeat;
        background-position: 0 0, 75px 75px;
        background-size: 150px 150px;
        opacity: 0.07; /* Make them faint */
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Model-specific required features
model_features = {
    'HD': ['Bore','Stroke','RPC','Rod','R bearing','B bearing','Block','Val A'],
    'HDE': ['Bore','Stroke','RPC','Rod','R bearing','B bearing','Block','Val A'],
    'HDI': ['Bore','Stroke','RPC','Rod','R bearing','B bearing','Block','Val A'],
    'LD': ['Bore','Stroke','RPC','Rod','R bearing','B bearing','Block','Val A'],
    'LDH': ['Bore','Stroke','RPC','Rod','Block','Val A'],
    'MD': ['Bore','Stroke','RPC','Rod','R bearing','B bearing','Block','Val A'],
    'NR': ['Bore','Stroke','RPC','Rod','R bearing'],
    'H': ['Bore','Stroke','RPC','Rod','R bearing','B bearing','Block','Val A'],
    'L': ['Bore','Stroke','RPC','Rod','Block'],
    'M': ['Bore','Stroke','RPC','Rod','R bearing','B bearing','Block','Val A'],
    'N': ['Bore','Stroke','RPC','Rod','R bearing','B bearing','Block','Val A'],
}
# Load model
def load_model(model_key):
    model_path = os.path.join(os.path.dirname(__file__), f"{model_key}_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        st.error(f"Model file {model_path} not found!")
        return None
# Sidebar selection
st.sidebar.title("Model Selection")
model_key = st.sidebar.selectbox("Select Model Type", list(model_features.keys()))
model = load_model(model_key)
if model is None:
    st.stop()
st.title("Cylinder Cost Prediction - Columbus")
# Input helper with synced slider + input box
def synced_input(label, min_val, max_val, default):
    col_slider, col_input = st.columns([2, 1])
    with col_slider:
        slider_val = st.slider(label, min_value=min_val, max_value=max_val, value=default, key=label+"slider")
    with col_input:
        input_val = st.number_input(f"{label} value", min_value=min_val, max_value=max_val, value=slider_val, key=label+"input")
    return input_val if input_val != slider_val else slider_val
# Input layout
col1, col2 = st.columns(2)
inputs = {}
with col1:
    bore = synced_input("Bore", 0, 300, 100)
    stroke = synced_input("Stroke", 0, 300, 50)
    rpc = synced_input("RPC", 0, 100, 20)
    rod = synced_input("Rod", 0, bore, min(30, bore))  # Rod must be <= Bore
with col2:
    r_bearing = st.radio("R bearing", ["Yes", "No"])
    b_bearing = st.radio("B bearing", ["Yes", "No"])
    val_a = st.radio("Val A", ["Yes", "No"])
    block = st.radio("Block", ["Yes", "No"])
    # Optional, if used in the model
    val_b = st.radio("Val B", ["Yes", "No"]) if 'Val B' in model_features[model_key] else None
# Add numerical features
inputs.update({
    'Bore': bore,
    'Stroke': stroke,
    'RPC': rpc,
    'Rod': rod,
    'R bearing': 1 if r_bearing == 'Yes' else 0,
    'B bearing': 1 if b_bearing == 'Yes' else 0,
    'Val A': 1 if val_a == 'Yes' else 0,
    'Block': 1 if block == 'Yes' else 0,
})
if val_b is not None:
    inputs['Val B'] = 1 if val_b == 'Yes' else 0
# Feature engineering based on model
if model_key in ['HD', 'HDE', 'HDI']:
    inputs['Bore2'] = bore ** 2
    inputs['Bore_Rod'] = bore * rod
    inputs['RPC_Bore'] = rpc * bore
    if model_key == 'HDI':
        inputs['Bore_stroke'] = bore * stroke
elif model_key == 'LDH':
    inputs['Bore_stroke'] = bore * stroke
    inputs['Bore_Rod'] = bore * rod
    inputs['RPC_Bore'] = rpc * bore
    inputs['Stroke_Rod'] = stroke * rod
elif model_key == 'MD':
    inputs['Bore2'] = bore ** 2
    inputs['Bore_RPC'] = bore * rpc
    inputs['Bore_Stroke'] = rpc * stroke
    inputs['Bore_Rod'] = bore * rod
elif model_key == 'NR':
    inputs['RPC2'] = rpc ** 2
    inputs['Bore_RPC'] = bore * rpc
    inputs['RPC_Stroke'] = rpc * stroke
    inputs['Stroke2'] = stroke ** 2
    inputs['RPC_Rod'] = rpc * rod
elif model_key == 'H':
    inputs['RPC2'] = rpc ** 2
    inputs['Bore_Rod'] = bore * rod
    inputs['RPC_Bore'] = rpc * bore
    inputs['Bore2'] = bore ** 2
    inputs['RPC_Rod'] = rpc * rod
elif model_key == 'L':
    inputs['Bore_RPC'] = bore * rpc
    inputs['Bore_Stroke'] = bore * stroke
    inputs['Bore2'] = bore ** 2
    inputs['Stroke_Rod'] = stroke * rod
elif model_key == 'M':
    inputs['Bore_Stroke'] = bore * stroke
    inputs['Bore_Rod'] = bore * rod
    inputs['RPC_Bore'] = rpc * bore
    inputs['Bore2'] = bore ** 2
    inputs['RPC_Rod'] = rpc * rod
# Rename features to match trained model column names
name_mapping = {
    'R bearing': 'R bearing_Y',
    'B bearing': 'B bearing_Y',
    'Block': 'Block_Y',
    'Val A': 'Val A_Y',
    'Val B': 'Val B_Y'
}
remapped_inputs = {name_mapping.get(k, k): v for k, v in inputs.items()}
# Final feature list for model input
model_feature_list = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else model.feature_names_
# Fill in missing features with 0s
final_input = []
for feature in model_feature_list:
    final_input.append(remapped_inputs.get(feature, 0))
# Predict cost
predicted_log_cost = model.predict([final_input])[0]
predicted_cost = np.expm1(predicted_log_cost)  # Inverse of log1p
st.markdown(f"### Predicted Cost: **$ {predicted_cost:.2f}**")
