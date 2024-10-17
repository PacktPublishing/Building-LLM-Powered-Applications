import streamlit as st

st.set_page_config(page_title="GlobeBotter", page_icon="")
st.header('🌐 Welcome to Globebotter, your travel assistant with Internet access. What are you planning for your next trip?')

if st.sidebar.button("Reset chat history"):
    st.session_state.messages = []
    
    