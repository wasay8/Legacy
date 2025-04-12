import streamlit as st

# Set up the page configuration and layout
st.set_page_config(page_title="Mental Health Assistant", layout="wide")



# Home Page Content

st.markdown("""
<div style="text-align:center; padding: 20px;">
    <h1 style="color:#4B8BBE;">🧠 Welcome to the Mental Health Counseling Assistant</h1>
    <p style="font-size:18px; color:#444;">This tool aims to support mental health professionals by providing:</p>
    <ul style="font-size:16px; color:#444;">
        <li></strong> Database Search and LLM suggestions for Users/li>
        <li></strong> Predictions using a Machine Learning Model</li>
    </ul>
    <p style="font-size:16px; color:#444;">Please navigate through the sidebar to start working on the tasks. Select a task that fits your needs.</p>
</div>
""", unsafe_allow_html=True)

# Add a button to continue with more details or jump to specific tasks
st.markdown("---")
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_button = st.button("🚀 Start Working with Tasks", use_container_width=True)
    if start_button:
        st.write("Choose a task from the sidebar to begin.")

