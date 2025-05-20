import streamlit as st
import pickle
import os
import numpy as np
st.set_page_config(
    page_title="Cylinder Cost Prediction",
    layout="centered",
    initial_sidebar_state="expanded",
)
# Feature configuration per model
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
st.markdown(
    """
    <div style='text-align:center; margin-top: -60px; margin-bottom: -30px; z-index: 9999;'>
        <img src='https://raw.githubusercontent.com/Lasya03/CylCP/main/ALLCYL/loading.gif' style='width:150px;' />
    </div>
    """,
    unsafe_allow_html=True
)
numerical_features = ['Bore','Stroke','RPC','Rod']
yesno_features = ['R bearing','B bearing','Block','Val A','Val B']
def load_model(model_key):
    filename = os.path.join(os.path.dirname(__file__), f"{model_key}_model.pkl")
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        st.error(f"Model file {filename} not found!")
        return None   
st.markdown(
    """
    <style>
    /* Sidebar container with background image */
    section[data-testid="stSidebar"] {
        position: relative;
        background-color: transparent;
        overflow: hidden;
    }

    section[data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url('https://raw.githubusercontent.com/Lasya03/CylCP/main/ALLCYL/img3.png');
        background-repeat: space;
        background-size: 100px 100px;
        pointer-events: none;
        z-index: 0;
        opacity: 0.75;  /* adjust visibility of image */
    }

    /* Sidebar content over background image */
    section[data-testid="stSidebar"] > div {
        position: relative;
        z-index: 1;
        background-color: transparent !important;
    }

    /* Make sidebar labels dark and readable */
    section[data-testid="stSidebar"] label {
        color: #111 !important;
        font-weight: bold;
        text-shadow: 0px 0px 2px white;  /* Optional: makes dark text stand out */
    }

    /* Style the dropdown box with dark border */
    section[data-testid="stSidebar"] .stSelectbox > div {
        border: 2px solid #333 !important;  /* Dark outline */
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar widgets
st.sidebar.title("Model Selection")
model_key = st.sidebar.selectbox("Select Model Type", list(model_features.keys()))

model = load_model(model_key)
if model is None:
    st.stop()
st.markdown(
    "<h1 style='white-space: nowrap;'>Cylinder Cost Prediction - Columbus</h1>",
    unsafe_allow_html=True
)
def synced_input(label, min_val, max_val, default):
    col_slider, col_input = st.columns([2, 1])
    with col_slider:
        slider_val = st.slider(label,min_value=float(min_val),max_value=float(max_val),value=float(default),step=0.001,key=label + "slider")
    with col_input:
        input_val = st.number_input(f"{label} value",min_value=float(min_val),max_value=float(max_val),value=slider_val,step=0.001,format="%.3f",key=label + "input")
    return input_val if input_val != slider_val else slider_val
col1, col2 = st.columns(2)
inputs = {}
with st.expander("🔢 Numerical Inputs", expanded=True):
    col1a, col1b = st.columns(2)
    with col1a:
        bore = synced_input("Bore", 0.0, 25.0, 5.0)
        stroke = synced_input("Stroke", 0.0, 400.0, 50.0)
    with col1b:
        rpc = synced_input("RPC", 0.0, 400.0, 20.0)
        rod = synced_input("Rod", 0.0, bore, min(5.0, bore))
with st.expander("⚙️ Categorical Options", expanded=True):
    col2a, col2b = st.columns(2)
    with col2a:
        r_bearing = st.radio("R bearing", ["No", "Yes"])
        b_bearing = st.radio("B bearing", ["No", "Yes"])
    with col2b:
        block = st.radio("Block", ["No", "Yes"])
        val_a = st.radio("Val A", ["No", "Yes"])
# Add inputs to dictionary
inputs['Bore'] = bore
inputs['Stroke'] = stroke
inputs['RPC'] = rpc
inputs['Rod'] = rod
inputs['R bearing'] = 1 if r_bearing == 'Yes' else 0
inputs['B bearing'] = 1 if b_bearing == 'Yes' else 0
inputs['Val A'] = 1 if val_a == 'Yes' else 0
inputs['Block'] = 1 if block == 'Yes' else 0
# Custom feature engineering
if model_key in ['HD', 'HDE', 'HDI']:
    inputs['Bore2'] = inputs['Bore'] ** 2
    inputs['Bore_Rod'] = inputs['Bore'] * inputs['Rod']
    inputs['RPC_Bore'] = inputs['RPC'] * inputs['Bore']
    if model_key == 'HDI':
        inputs['Bore_stroke'] = inputs['Bore'] * inputs['Stroke']
elif model_key == 'LDH':
    inputs['Bore_stroke'] = inputs['Bore'] * inputs['Stroke']
    inputs['Bore_Rod'] = inputs['Bore'] * inputs['Rod']
    inputs['RPC_Bore'] = inputs['RPC'] * inputs['Bore']
    inputs['Stroke_Rod'] = inputs['Stroke'] * inputs['Rod']
elif model_key == 'MD':
    inputs['Bore2'] = inputs['Bore'] ** 2
    inputs['Bore_RPC'] = inputs['Bore'] * inputs['RPC']
    inputs['Bore_Stroke'] = inputs['RPC'] * inputs['Stroke']
    inputs['Bore_Rod'] = inputs['Bore'] * inputs['Rod']
elif model_key == 'NR':
    inputs['RPC2'] = inputs['RPC'] ** 2
    inputs['Bore_RPC'] = inputs['Bore'] * inputs['RPC']
    inputs['RPC_Stroke'] = inputs['RPC'] * inputs['Stroke']
    inputs['Stroke2'] = inputs['Stroke'] ** 2
    inputs['RPC_Rod'] = inputs['RPC'] * inputs['Rod']
elif model_key == 'H':
    inputs['RPC2'] = inputs['RPC'] ** 2
    inputs['Bore_Rod'] = inputs['Bore'] * inputs['Rod']
    inputs['RPC_Bore'] = inputs['RPC'] * inputs['Bore']
    inputs['Bore2'] = inputs['Bore'] ** 2
    inputs['RPC_Rod'] = inputs['RPC'] * inputs['Rod']
elif model_key == 'L':
    inputs['Bore_RPC'] = inputs['Bore'] * inputs['RPC']
    inputs['Bore_Stroke'] = inputs['Bore'] * inputs['Stroke']
    inputs['Bore2'] = inputs['Bore'] ** 2
    inputs['Stroke_Rod'] = inputs['Stroke'] * inputs['Rod']
elif model_key == 'M':
    inputs['Bore_Stroke'] = inputs['Bore'] * inputs['Stroke']
    inputs['Bore_Rod'] = inputs['Bore'] * inputs['Rod']
    inputs['RPC_Bore'] = inputs['RPC'] * inputs['Bore']
    inputs['Bore2'] = inputs['Bore'] ** 2
    inputs['RPC_Rod'] = inputs['RPC'] * inputs['Rod']
# Remap yes/no feature names to match training column names
input_name_mapping = {
    'R bearing': 'R bearing_Y',
    'B bearing': 'B bearing_Y',
    'Block': 'Block_Y',
    'Val A': 'Val A_Y',
    'Val B': 'Val B_Y'
}
remapped_inputs = {}
for k, v in inputs.items():
    remapped_inputs[input_name_mapping.get(k, k)] = v
# Ensure all features expected by model are present
model_input = [remapped_inputs.get(f, 0) for f in model.feature_names_]
# Predict and show result
predicted_cost = np.expm1(model.predict([model_input])[0])
st.markdown(f"### Predicted Cost: **$ {predicted_cost:.2f}**")
