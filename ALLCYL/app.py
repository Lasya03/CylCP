import streamlit as st 
import pickle
import os
import numpy as np

st.set_page_config(
    page_title="Cylinder Cost Prediction-Columbus",
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

# Styling
st.markdown("""
<div style="text-align: center; margin-top: 0px; margin-bottom: 30px;">
    <img src="https://raw.githubusercontent.com/Lasya03/CylCP/main/ALLCYL/loading.gif" 
         width="200" alt="Logo">
</div>
<div style="overflow:hidden; white-space:nowrap; box-sizing:border-box;">
  <div style="display:inline-block; padding-left:100%; animation: scroll-left 10s linear infinite; color:red; font-weight:bold;">
    **All the numerical inputs are in inches**
  </div>
</div>
<style>
@keyframes scroll-left {
  0% { transform: translateX(0%); }
  100% { transform: translateX(-100%); }
}
</style>
""", unsafe_allow_html=True)

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

# Sidebar styling
st.markdown("""
<style>
section[data-testid="stSidebar"] .stSelectbox > div {
    border: 2px solid black !important;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar widgets
st.sidebar.title("Model Selection")
model_key = st.sidebar.selectbox("Select Model Type", list(model_features.keys()))
required_features = model_features.get(model_key, [])
optional_features = [f for f in yesno_features if f not in required_features]

st.sidebar.markdown(f"**Model Selected:** {model_key}")
if optional_features:
    st.sidebar.markdown(f"**Note:** While training the {model_key} model, it was observed that the following features had minimal impact on prediction. As a result, their costs are not included in the predicted value. However, you may choose to manually add their respective costs if needed.")
    for feat in optional_features:
        st.sidebar.markdown(f"- {feat}")
else:
    st.sidebar.markdown("*All yes/no features are required for this model.*")

model = load_model(model_key)
if model is None:
    st.stop()

st.markdown("<h1 style='white-space: nowrap;'>Cylinder Cost Prediction - Columbus</h1>", unsafe_allow_html=True)

# Additional styling for inputs
st.markdown("""
<style>
.stNumberInput input {
    border: 1.5px solid #222 !important;
    border-radius: 8px !important;
    padding: 8px !important;
    background-color: #fff !important;
    color: #000 !important;
}
</style>
""", unsafe_allow_html=True)

# Define input ranges
input_ranges = {
    'HD': {'Bore': (0.5, 6.5, 3.0), 'Stroke': (1.0, 130.0, 20.0), 'RPC': (1.0, 130.0, 20.0), 'Rod': (0.0, 4.0, 1.5)},
    'HDE': {'Bore': (1.0, 6.5, 3.0), 'Stroke': (1.0, 75.0, 20.0), 'RPC': (5.0, 90.0, 20.0), 'Rod': (0.0, 4.0, 1.5)},
    'HDI': {'Bore': (0.5, 5.0, 3.0), 'Stroke': (0.0, 65.0, 15.0), 'RPC': (0.0, 80.0, 20.0), 'Rod': (0.0, 4.0, 1.5)},
    'LD': {'Bore': (0.5, 4.0, 2.0), 'Stroke': (0.0, 50.0, 13.0), 'RPC': (1.0, 65.0, 21.0), 'Rod': (0.0, 2.0, 1.0)},
    'LDH': {'Bore': (1.0, 5.0, 2.5), 'Stroke': (0.0, 75.0, 15.0), 'RPC': (1.0, 85.0, 25.0), 'Rod': (0.0, 2.5, 1.0)},
    'MD': {'Bore': (0.0, 6.5, 2.5), 'Stroke': (0.0, 100.0, 13.0), 'RPC': (1.0, 110.0, 21.0), 'Rod': (0.0, 2.5, 1.5)},
    'NR': {'Bore': (0.0, 3.0, 2.0), 'Stroke': (2.5, 45.0, 11.0), 'RPC': (7.0, 50.0, 18.0), 'Rod': (0.0, 2.5, 1.0)},
    'H': {'Bore': (0.0, 6.5, 3.0), 'Stroke': (0.0, 115.0, 16.0), 'RPC': (0.0, 130.0, 25.0), 'Rod': (0.0, 3.5, 1.5)},
    'L': {'Bore': (0.5, 5.0, 2.5), 'Stroke': (0.0, 75.0, 15.0), 'RPC': (1.0, 85.0, 20.0), 'Rod': (0.0, 2.5, 1.0)},
    'M': {'Bore': (0.0, 6.5, 2.5), 'Stroke': (0.5, 100.0, 15.0), 'RPC': (2.0, 110.0, 21.0), 'Rod': (0.0, 2.5, 1.5)},
    'N': {'Bore': (0.0, 3.0, 2.0), 'Stroke': (2.5, 45.0, 11.0), 'RPC': (7.0, 50.0, 20.0), 'Rod': (0.0, 2.5, 1.0)}
}
default_range = {'Bore': (0.0, 10.0, 2.0), 'Stroke': (0.0, 300.0, 50.0), 'RPC': (0.0, 300.0, 20.0)}

# Synced input function
def synced_input(label, min_val, max_val, default):
    col_slider, col_input = st.columns([2, 1])
    with col_slider:
        slider_val = st.slider(label, min_value=float(min_val), max_value=float(max_val), value=float(default), step=0.001, key=label + "slider")
    with col_input:
        input_val = st.number_input(f"{label} value", min_value=float(min_val), max_value=float(max_val), value=slider_val, step=0.001, format="%.3f", key=label + "input")
    return input_val if input_val != slider_val else slider_val

# Numerical input section
st.markdown("## Input Parameters")
with st.expander("🔢 Numerical Inputs", expanded=True):
    ranges = input_ranges.get(model_key, default_range)
    bore_min, bore_max, bore_default = ranges['Bore']
    stroke_min, stroke_max, stroke_default = ranges['Stroke']
    rpc_min, rpc_max, rpc_default = ranges['RPC']
    rod_min, rod_max, rod_default = ranges.get('Rod', (0.0, 2.5, 1.0))

    col1, col2 = st.columns(2)
    with col1:
        bore = synced_input("Bore", bore_min, bore_max, bore_default)
        stroke = synced_input("Stroke", stroke_min, stroke_max, stroke_default)
    with col2:
        rpc = synced_input("RPC", rpc_min, rpc_max, rpc_default)
        rod = synced_input("Rod", rod_min, rod_max, rod_default)
# You can now use `bore`, `stroke`, `rpc`, and `rod` as inputs for model prediction logic.
with st.expander("⚙️ Categorical Options", expanded=True):
    col2a, col2b = st.columns(2)
    with col2a:
        r_bearing = st.radio("R bearing", ["No", "Yes"])
        b_bearing = st.radio("B bearing", ["No", "Yes"])
    with col2b:
        block = st.radio("Block", ["No", "Yes"])
        val_a = st.radio("Val A", ["No", "Yes"])
# Add inputs to dictionary
inputs['Bore'] = Bore
inputs['Stroke'] = Stroke
inputs['RPC'] = RPC
inputs['Rod'] = Rod
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
