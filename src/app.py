import streamlit as st

# Page config
st.set_page_config(page_title="AgriDrought-AI", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predictions", "Analysis"])

if page == "Home":
    st.title("🌱 AgriDrought-AI Dashboard")
    st.write("Welcome to the MVP dashboard!")
    st.write("Here you’ll see metrics, maps, and drought trends.")

elif page == "Predictions":
    st.title("📈 Predictions")
    st.write("This page will show drought forecasts.")
    days = st.slider("Select forecast horizon (days)", 30, 60, 30)
    st.write(f"Showing forecast for {days} days ahead.")

elif page == "Analysis":
    st.title("🔍 Analysis")
    uploaded_file = st.file_uploader("Upload your CSV or GeoTIFF data", type=["csv", "tif"])
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
