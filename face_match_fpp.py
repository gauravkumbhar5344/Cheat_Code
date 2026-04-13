import streamlit as st
import requests
from PIL import Image
import io

# --- API CONFIGURATION ---
# Pro-tip: In a real app, use st.secrets to hide these!
API_KEY = ""
API_SECRET = ""
API_URL = "https://api-us.faceplusplus.com/facepp/v3/compare"

def compare_faces(img_bytes1, img_bytes2):
    files = {
        'image_file1': img_bytes1,
        'image_file2': img_bytes2
    }
    data = {
        'api_key': API_KEY,
        'api_secret': API_SECRET
    }
    
    response = requests.post(API_URL, files=files, data=data)
    return response.json()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Face Matcher", layout="wide")

st.title("👤 Face Recognition Comparison")
st.write("Upload two photos to check if they belong to the same person.")

# Create two columns for the uploads
col1, col2 = st.columns(2)

with col1:
    st.header("Image 1")
    file1 = st.file_uploader("Choose first image...", type=['jpg', 'jpeg', 'png'], key="1")
    if file1:
        st.image(file1, use_container_width=True)

with col2:
    st.header("Image 2")
    file2 = st.file_uploader("Choose second image...", type=['jpg', 'jpeg', 'png'], key="2")
    if file2:
        st.image(file2, use_container_width=True)

st.divider()

# Comparison Logic
if file1 and file2:
    if st.button("🔍 Compare Faces"):
        with st.spinner("Analyzing faces... please wait."):
            # Convert files to bytes for the API
            img_payload1 = file1.getvalue()
            img_payload2 = file2.getvalue()
            
            result = compare_faces(img_payload1, img_payload2)
            
            if "error_message" in result:
                st.error(f"API Error: {result['error_message']}")
            else:
                confidence = result.get('confidence', 0)
                thresholds = result.get('thresholds', {})
                # Using 1e-5 (one in 100,000 error rate) as the benchmark
                pass_threshold = thresholds.get('1e-5', 80)

                st.subheader(f"Confidence Score: {confidence:.2f}%")
                
                if confidence > pass_threshold:
                    st.success("✅ **MATCH!** These images appear to be of the same person.")
                    st.balloons()
                else:
                    st.error("❌ **NO MATCH!** These images appear to be of different people.")
                
                # Optional: Show raw data in an expander
                with st.expander("See technical details"):
                    st.json(result)
else:
    st.info("Please upload both images to enable comparison.")