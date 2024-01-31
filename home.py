import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import lime
import matplotlib.pyplot as plt
import time
# import utils

# from utils import saveSession, getSession
from sklearn.model_selection import train_test_split
# from st_pages import show_pages_from_config

# show_pages_from_config()
st.set_option('deprecation.showPyplotGlobalUse', False)
# https://github.com/daniellewisDL/streamlit-cheat-sheet/blob/master/README.md
st.title("Home Page")
st.subheader("Upload Dataset File")
file_csv = st.file_uploader("Browse CSV file", type=["csv"], on_change=lambda: st.session_state.clear())

@st.cache_data(show_spinner=False)
def readData(file_csv):
    return pd.read_csv(file_csv)

if (file_csv):
    df = readData(file_csv)
    st.session_state["file_csv"] = df
    st.session_state["uploaded_file"] = file_csv.name

    with st.expander("Data Information"):
        st.write("Data Shape:", df.shape)

        col1, col2= st.columns(2)

        with col1:
            st.write("Data Columns:", df.columns)

        with col2:
            st.write("Data Types:", df.dtypes)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Describe")
            st.write(df.describe())

        with col2:
            st.subheader("Data Header")
            st.write(df.head())
        
        # st.subheader("Data Correlation")
        # use_annotation = st.toggle("Use Annotation", value=False, key="use_annotation")
        # st.pyplot(sns.heatmap(df.drop(df.select_dtypes(
        #     include=['object']).columns, axis=1).corr(), annot=use_annotation).figure)

        # st.markdown(''' <a target="_self" href="#data-information">
        #                 Back to Top
        #         </a>''', unsafe_allow_html=True)
        # st.write("")
            
    st.subheader("Feature Selection :", divider="grey")

    sbDropped = []
    dataColumns = [df.columns[idx] for idx in range(len(df.columns))]

    with st.form("form_dropped_features"):
        st.caption("Select features to be dropped in the dataset")
        sbDropped = st.multiselect('Dropped Features', dataColumns, key="dropped_features")
        st.form_submit_button("Proceed", type="primary", args=["confirmDropped"])
    if (len(sbDropped) > len(dataColumns)-2):
        st.caption(":red *Leave at least two (2) features to proceed!*")





# with st.spinner("Loading...", ):
#     time.sleep(5)







# st.markdown("aaaaa")




