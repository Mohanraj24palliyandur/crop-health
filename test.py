import streamlit as st

def main():
    st.set_page_config(
        page_title="Test App",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Crop Health Monitor")
    st.write("Welcome to the Crop Health Monitoring System")

if __name__ == "__main__":
    main()