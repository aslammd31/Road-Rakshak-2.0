import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================
load_dotenv()  # loads .env file

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

genai.configure(api_key=API_KEY)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Road Rakshak 2.0",
    page_icon="🚦",
    layout="wide"
)

# =====================================================
# ADVANCED UI STYLING
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
}
h1 {
    text-align:center;
    font-weight:800;
    color:white;
}
h2, h3 {
    color:#e8f1ff;
}
.stTabs [data-baseweb="tab-list"] {
    gap:12px;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.08);
    border-radius:10px;
    padding:10px 20px;
    color:white;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#ff416c,#ff4b2b);
    font-weight:bold;
}
.stButton>button {
    background: linear-gradient(90deg,#ff416c,#ff4b2b);
    color:white;
    border-radius:12px;
    border:none;
    padding:10px 20px;
    font-weight:bold;
}
.stButton>button:hover {
    transform:scale(1.05);
}
.result-card {
    background:white;
    color:black;
    padding:22px;
    border-radius:14px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.25);
    line-height:1.6;
    white-space:pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
for key in ["hazard_result", "top_location", "top_cause"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =====================================================
# TITLE
# =====================================================
st.title("🚦 Road Rakshak 2.0")
st.write("AI Powered Road Safety Reporting System")

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Visual Intelligence",
    "📷 Live Camera Detection",
    "📊 Data Analytics",
    "📜 Civic Reporting"
])

# =====================================================
# TAB 1 — IMAGE ANALYSIS
# =====================================================
with tab1:
    st.header("Upload Road Image")

    uploaded_image = st.file_uploader(
        "Choose a road image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_container_width=True)

        if st.button("Analyze Road Condition"):
            with st.spinner("Analyzing road condition..."):
                model = genai.GenerativeModel("gemini-1.5-flash")

                prompt = """
Detect ONE hazard:
- Pothole
- Waterlogging
- Broken streetlight
- Traffic sign obstruction

Assign severity: Low, Medium, High.

Reply STRICTLY in this format:

Hazard:
Severity:
Explanation:
"""

                response = model.generate_content(
                    [{"role": "user", "parts": [prompt, image]}]
                )

                st.session_state.hazard_result = response.text

                st.success("✅ Analysis Completed")
                st.markdown(
                    f"<div class='result-card'>{response.text}</div>",
                    unsafe_allow_html=True
                )

# =====================================================
# TAB 2 — LIVE CAMERA
# =====================================================
with tab2:
    st.header("📷 Live Camera Road Detection")

    camera_image = st.camera_input("Capture live road image")

    if camera_image:
        live_image = Image.open(camera_image)
        st.image(live_image, use_container_width=True)

        if st.button("Analyze Live Road Condition", key="live"):
            with st.spinner("Analyzing live image..."):
                model = genai.GenerativeModel("gemini-1.5-flash")

                prompt = """
Detect ONE hazard:
- Pothole
- Waterlogging
- Broken streetlight
- Traffic sign obstruction
- No Hazard

Assign severity: Low, Medium, High.

Reply STRICTLY in this format:

Hazard:
Severity:
Explanation:
"""

                response = model.generate_content(
                    [{"role": "user", "parts": [prompt, live_image]}]
                )

                st.success("✅ Live Analysis Completed")
                st.markdown(
                    f"<div class='result-card'>{response.text}</div>",
                    unsafe_allow_html=True
                )

# =====================================================
# TAB 3 — DATA ANALYTICS
# =====================================================
with tab3:
    st.header("Upload Accident Data (CSV)")

    csv_file = st.file_uploader("Upload accident dataset", type=["csv"])

    if csv_file:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip().str.lower()

        st.dataframe(df.head())

        if "location" in df.columns:
            st.session_state.top_location = df["location"].value_counts().idxmax()
            count = df["location"].value_counts().max()
            st.info(f"📍 Most Accident-Prone Location: **{st.session_state.top_location}** ({count} cases)")

            fig, ax = plt.subplots()
            df["location"].value_counts().head(5).plot(kind="bar", ax=ax)
            ax.set_xlabel("Location")
            ax.set_ylabel("Accidents")
            st.pyplot(fig)

        if "cause" in df.columns:
            st.session_state.top_cause = df["cause"].value_counts().idxmax()
            st.warning(f"⚠️ Most Common Cause: **{st.session_state.top_cause}**")

# =====================================================
# TAB 4 — CIVIC REPORTING
# =====================================================
with tab4:
    st.header("Generate Official Complaint Letter")

    if st.button("Generate Complaint Letter"):
        if st.session_state.hazard_result is None:
            st.warning("⚠️ Analyze an image first.")
        elif st.session_state.top_location is None or st.session_state.top_cause is None:
            st.warning("⚠️ Upload accident CSV first.")
        else:
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""
Write a professional complaint letter to the Municipal Commissioner.

Hazard Details:
{st.session_state.hazard_result}

Location:
{st.session_state.top_location}

Cause:
{st.session_state.top_cause}
"""

            letter = model.generate_content(prompt)

            st.subheader("📜 Generated Complaint Letter")
            st.markdown(
                f"<div class='result-card'>{letter.text}</div>",
                unsafe_allow_html=True
            )

            st.download_button(
                "⬇️ Download Complaint Letter",
                letter.text,
                "complaint_letter.txt"
            )

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr style="margin-top:40px; border:0.5px solid rgba(255,255,255,0.2);">
<p style='text-align:center;color:#cfd8dc;font-size:14px;'>
🚦 Road Rakshak 2.0 — Built with ❤️ to make roads safer and smarter
</p>
""", unsafe_allow_html=True)