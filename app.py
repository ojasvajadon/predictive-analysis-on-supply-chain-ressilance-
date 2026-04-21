import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Supply Chain Resilience Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR CYBERPUNK NEON ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0d0d12;
        color: #e0e0e0;
    }
    
    /* Neon Text & Headers */
    h1, h2, h3 {
        color: #ff007f !important;
        text-shadow: 0 0 5px #ff007f, 0 0 10px #ff007f;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #00f0ff !important;
        text-shadow: 0 0 8px #00f0ff;
        font-size: 2.5rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1a24;
        border-right: 1px solid #ff007f;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: transparent;
        border: 2px solid #00f0ff;
        color: #00f0ff;
        border-radius: 8px;
        transition: all 0.3s;
        box-shadow: 0 0 5px #00f0ff inset, 0 0 5px #00f0ff;
    }
    .stButton>button:hover {
        background-color: #00f0ff;
        color: #000;
        box-shadow: 0 0 15px #00f0ff inset, 0 0 15px #00f0ff;
    }
    
    /* Dataframes/Tables */
    .stDataFrame {
        border: 1px solid #ff007f;
        border-radius: 5px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #ff007f !important;
    }
</style>
""", unsafe_allow_html=True)

# --- PORT COORDINATES (Mock) ---
PORT_COORDS = {
    "Mumbai Port": {"lat": 18.9438, "lon": 72.8353},
    "Shanghai Port": {"lat": 31.2304, "lon": 121.4737},
    "Los Angeles Port": {"lat": 33.7420, "lon": -118.2755},
    "Rotterdam Port": {"lat": 51.9225, "lon": 4.4791},
    "Dubai Port": {"lat": 25.2769, "lon": 55.2962},
    "Singapore Port": {"lat": 1.2902, "lon": 103.8519}
}

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_models():
    try:
        with open('model_clf.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open('model_reg.pkl', 'rb') as f:
            reg = pickle.load(f)
        with open('shap_explainer.pkl', 'rb') as f:
            explainer = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return clf, reg, explainer, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please run ml_model.py first.")
        st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("training_data.csv")
        # Ensure correct column order
        return df
    except FileNotFoundError:
        st.error("training_data.csv not found.")
        st.stop()

clf, reg, explainer, feature_names = load_models()
data = load_data()

# Separate features from target
X = data[feature_names]

# Get predictions for all data to calculate metrics
probs = clf.predict_proba(X)[:, 1]
data['Risk_Probability'] = probs
data['Risk_Level'] = pd.cut(probs, bins=[-1, 0.3, 0.7, 1.1], labels=['Low', 'Medium', 'High'])
# Only predict duration if risk is high enough
data['Expected_Delay'] = np.where(probs > 0.3, np.round(reg.predict(X)), 0)

# --- SIDEBAR ---
st.sidebar.title("⚡ Control Panel")
st.sidebar.markdown("---")
view_mode = st.sidebar.radio("View Mode", ["Global Overview", "Shipment Deep Dive", "Scenario Simulator"])

# --- VIEW 1: GLOBAL OVERVIEW ---
if view_mode == "Global Overview":
    st.title("🌐 Global Supply Chain Network")
    st.markdown("Real-time visibility into active shipments and predictive risks.")
    
    # Top Metrics
    col1, col2, col3 = st.columns(3)
    
    high_risk_count = len(data[data['Risk_Level'] == 'High'])
    avg_delay = data[data['Expected_Delay'] > 0]['Expected_Delay'].mean() if 'Expected_Delay' in data else 0
    # Create fake current inventory reduction metric
    
    col1.metric("Active Shipments", f"{len(data):,}")
    col2.metric("High Risk Alerts", f"{high_risk_count}", delta=f"{int((high_risk_count/len(data))*100)}% of total", delta_color="inverse")
    col3.metric("Unified Risk Score", f"{int(probs.mean() * 100)}/100", delta="-5% from yesterday", delta_color="normal")
    
    st.markdown("---")
    
    # Map Visualization
    st.subheader("📍 Active Risk Map")
    
    # Map data creation
    map_data = []
    # Original data doesn't have raw origin back after one-hot encoding if drop_first=False
    # Let's reconstruct origin for the map
    origin_cols = [c for c in data.columns if c.startswith('origin_') and c not in ['origin_weather_severity', 'origin_nlp_risk']]
    dest_cols = [c for c in data.columns if c.startswith('destination_')]
    
    # Since we need raw origin/dest, let's load shipments.csv just for display
    try:
        raw_shipments = pd.read_csv("shipments.csv")
        # Merge risk prob back to raw shipments for display
        raw_shipments['Risk_Probability'] = data['Risk_Probability'].values
        raw_shipments['Expected_Delay'] = data['Expected_Delay'].values
        raw_shipments['Risk_Level'] = data['Risk_Level'].values
        
        # Add lat/lon
        raw_shipments['lat'] = raw_shipments['origin'].map(lambda x: PORT_COORDS.get(x, {}).get('lat'))
        raw_shipments['lon'] = raw_shipments['origin'].map(lambda x: PORT_COORDS.get(x, {}).get('lon'))
        
        fig = px.scatter_mapbox(
            raw_shipments, 
            lat="lat", lon="lon", 
            color="Risk_Probability",
            size="Expected_Delay",
            hover_name="origin",
            hover_data=["destination", "item", "Expected_Delay"],
            color_continuous_scale=px.colors.sequential.Sunsetdark,
            size_max=15, zoom=1.5,
            mapbox_style="carto-darkmatter",
            title="Global Origin Port Risks"
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, paper_bgcolor="#0d0d12", font_color="#e0e0e0")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("Could not load map visualization. Please check data formats.")

# --- VIEW 2: SHIPMENT DEEP DIVE ---
elif view_mode == "Shipment Deep Dive":
    st.title("🔍 Predictive Deep Dive")
    st.markdown("Select a high-risk shipment to view AI-generated SHAP explanations.")
    
    # Filter for interesting shipments
    at_risk = data[data['Risk_Level'].isin(['Medium', 'High'])].copy()
    
    if len(at_risk) == 0:
        st.info("No risky shipments detected right now.")
    else:
        # We need shipment IDs
        raw_shipments = pd.read_csv("shipments.csv")
        at_risk['shipment_id'] = raw_shipments.loc[at_risk.index, 'shipment_id']
        at_risk['origin'] = raw_shipments.loc[at_risk.index, 'origin']
        at_risk['destination'] = raw_shipments.loc[at_risk.index, 'destination']
        
        selected_id = st.selectbox("Select Shipment ID", at_risk['shipment_id'].tolist())
        
        # Get data for selected
        selected_idx = at_risk[at_risk['shipment_id'] == selected_id].index[0]
        shipment_data = at_risk.loc[selected_idx]
        feature_vector = X.iloc[[selected_idx]]
        
        # Two columns for details
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Shipment Details")
            st.markdown(f"**Origin:** {shipment_data['origin']}")
            st.markdown(f"**Destination:** {shipment_data['destination']}")
            
            # Gauge chart for risk
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = shipment_data['Risk_Probability'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Delay Probability %", 'font': {'color': '#00f0ff'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#ff007f"},
                    'bgcolor': "black",
                    'steps': [
                        {'range': [0, 30], 'color': "#1a1a24"},
                        {'range': [30, 70], 'color': "#4d1a3e"},
                        {'range': [70, 100], 'color': "#800040"}],
                }
            ))
            fig.update_layout(paper_bgcolor="#0d0d12", font={'color': "#e0e0e0"}, height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**Expected Duration of Delay:** <span style='color:#ff007f; font-size:1.5rem;'>{int(shipment_data['Expected_Delay'])} Days</span>", unsafe_allow_html=True)
            
        with col2:
            st.subheader("🧠 AI Reasoning (SHAP)")
            st.markdown("Why did the model predict this delay?")
            
            # Compute SHAP values for this instance
            shap_values = explainer.shap_values(feature_vector)
            
            # Since XGBoost classifier, shape depends on multiclass vs binary. Usually binary is (1, num_features)
            if isinstance(shap_values, list):
                shap_val_instance = shap_values[1][0] # class 1
            else:
                shap_val_instance = shap_values[0]
                
            # Create a dataframe for plotting
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_val_instance,
                'Value': feature_vector.iloc[0].values
            })
            # Filter zero/near zero impacts
            shap_df = shap_df[abs(shap_df['SHAP Value']) > 0.05]
            shap_df = shap_df.sort_values(by='SHAP Value', ascending=True)
            
            fig = px.bar(
                shap_df, 
                x='SHAP Value', 
                y='Feature', 
                orientation='h',
                color='SHAP Value',
                color_continuous_scale=px.colors.diverging.Picnic,
                title="Impact of Features on Delay Prediction"
            )
            fig.update_layout(paper_bgcolor="#0d0d12", plot_bgcolor="#0d0d12", font_color="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)

# --- VIEW 3: SCENARIO SIMULATOR ---
elif view_mode == "Scenario Simulator":
    st.title("⛈️ What-If Scenario Simulator")
    st.markdown("Inject external risks and immediately see the impact on shipment predictions.")
    
    st.info("Pick an average shipment and modify its external factors.")
    
    raw_shipments = pd.read_csv("shipments.csv")
    base_idx = st.number_input("Select Base Shipment Index (0-100)", min_value=0, max_value=len(data)-1, value=0)
    
    base_feature_vector = X.iloc[[base_idx]].copy()
    
    st.subheader(f"Simulating Shipment: {raw_shipments.loc[base_idx, 'origin']} ➔ {raw_shipments.loc[base_idx, 'destination']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Manipulate External Factors")
        new_weather = st.slider("Origin Weather Severity (0=Clear, 10=Cyclone)", 0.0, 10.0, float(base_feature_vector['origin_weather_severity'].iloc[0]))
        new_nlp = st.slider("NLP Risk Score (News Sentiment)", 0.0, 1.0, float(base_feature_vector['origin_nlp_risk'].iloc[0]))
        
        # Apply changes
        base_feature_vector['origin_weather_severity'] = new_weather
        base_feature_vector['origin_nlp_risk'] = new_nlp
        
        if st.button("Run Simulation 🚀"):
            new_prob = clf.predict_proba(base_feature_vector)[0][1]
            new_delay = reg.predict(base_feature_vector)[0]
            
            st.session_state['sim_prob'] = new_prob
            st.session_state['sim_delay'] = new_delay
            
    with col2:
        st.markdown("### Simulated Outcome")
        if 'sim_prob' in st.session_state:
            st.metric("New Delay Probability", f"{int(st.session_state['sim_prob']*100)}%", delta=f"{int((st.session_state['sim_prob'] - data.iloc[base_idx]['Risk_Probability'])*100)}% from baseline", delta_color="inverse")
            if st.session_state['sim_prob'] > 0.3:
                st.metric("New Estimated Delay Duration", f"{int(np.round(st.session_state['sim_delay']))} Days")
            else:
                st.metric("New Estimated Delay Duration", "0 Days (On Time)")
        else:
            st.markdown("*Adjust sliders and run simulation to see results.*")

