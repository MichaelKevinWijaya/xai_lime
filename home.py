import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import lime
import matplotlib.pyplot as plt
import time

from session_utils import (saveSession, getSession)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from st_pages import show_pages_from_config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics

# show_pages_from_config()
# https://github.com/daniellewisDL/streamlit-cheat-sheet/blob/master/README.md
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Home Page")
st.subheader("Upload Dataset File")
file_csv = st.file_uploader("Browse CSV file", type=["csv"], on_change=lambda: st.session_state.clear())

# First Initialization when project run
if 'clicked' not in st.session_state:
    st.session_state["clicked"] = {"confirmTarget": False, "initData": False, "initModel": False, "confirmDropped": False,
                                   "click_generate_instance_tab0": True,
                                   "click_generate_instance_tab1": True,
                                   "click_generate_instance_tab2": True}

@st.cache_data(show_spinner=False)
def readData(file_csv):
    return pd.read_csv(file_csv)

def clicked(button):
    if (button == "initModel"):
        saveSession({"confirmProgressInit": True})
    st.session_state.clicked[button] = True


def getKNN_k(x_train, y_train, x_test, y_test, x_columns, min_iter=2, max_iter=25):

    k_neighbor_score = []
    error_rate = []

    for k in range(min_iter, max_iter):
        knn_model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn_model.fit(x_train, y_train)
        pred_i = knn_model.predict(x_test)
        error_rate.append({k: np.mean(pred_i != y_test)})
        knn_score = knn_model.score(x_test, y_test)
        k_neighbor_score.append({k: knn_score})
    max_dict = min(error_rate, key=lambda x: list(x.values())[0])
    k_neighbor = list(max_dict.keys())[0]
    return k_neighbor, k_neighbor_score

if (file_csv):
    df = readData(file_csv)
    file_name = file_csv.name
    if (file_name == "data_commuter.csv"):
        df.loc[df["Kepuasan Hidup"] == "Sangat Tidak Puas", "Kepuasan Hidup"] = 0
        df.loc[df["Kepuasan Hidup"] == "Biasa", "Kepuasan Hidup"] = 1
        df.loc[df["Kepuasan Hidup"] == "Sangat Puas", "Kepuasan Hidup"] = 2
    df = df.astype(float)
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

    st.subheader("Feature Selection", divider="grey")

    sbDropped = []
    dataColumns = [df.columns[idx] for idx in range(len(df.columns))]

    with st.form("form_dropped_features"):
        st.caption("Select features to be dropped in the dataset")
        sbDropped = st.multiselect('Dropped Features', dataColumns, key="dropped_features")
        st.form_submit_button("Proceed", type="primary", on_click=clicked, args=["confirmDropped"])
    if (len(sbDropped) > len(dataColumns)-2):
        st.caption(":red *Leave at least two (2) features to proceed!*")

    if getSession("clicked")["confirmDropped"] and len(sbDropped) <= len(dataColumns)-2:
        columnOptions = [col for col in df.columns if col not in sbDropped]

        with st.form(key="form_feature_selection"):
            # col1, col2 = st.columns(2)
            # with col1:
            st.caption(
                "Select target feature to be predicted in the dataset")

            sbTarget = st.selectbox(
                'Feature Name', columnOptions, index=len(columnOptions)-1, key="target_feature", )
            # BUTTON PROCEED
            st.form_submit_button("Proceed", type="primary", on_click=clicked, args=["confirmTarget"])
    
    # DATA INITIALIZATION
    if getSession("clicked")["confirmTarget"]:
        st.subheader("Data Initialization", divider="grey")
        with st.form(key="form_data_initialization"):

            col_init1, col_init2= st.columns(2)

            with col_init1:
                st.number_input(
                    'Test Size Ratio', key="test_size", min_value=0.1, max_value=0.9, value=0.2, step=0.1)
                st.caption(
                    "Test and Train Size Ratio to be Divided for Train and Testing")
                
            with col_init2:
                st.number_input(
                    'Random State', key='random_state', min_value=0, max_value=100, value=42, step=1)
                st.caption("State of Randomness for Reproducibility")

            st.form_submit_button("Proceed", type="primary", on_click=clicked, args=["initData"])

    # MODEL INITIALIZATION
    if getSession("clicked")["initData"]:
        # st.write(getSession("random_state"))
        st.subheader("Model Initialization", divider="grey")

        with st.form(key="form_model_initialization"):
            st.write("#### KNN")
            knn_auto_k = False
            k_input = st.number_input(
                'K Neighbor', key="knn_inp_k", min_value=2, value=5)
            st.caption("*or*")
            knn_auto_k = st.checkbox(
                "Auto K", value=True, key="knn_auto_k")
            st.caption(
                "Automatically find the most optimal N for KNN model using KNN Regressor")
            
            st.form_submit_button(
                    "Begin Initialization", type="primary", on_click=clicked, args=["initModel"])

    # PROCESS INITIALIZATION
    if getSession("clicked")["initModel"]:
        with st.spinner('Initializing the Data, please wait...'):
            random_state = getSession("random_state")
            target_feature = sbTarget
            test_size = getSession("test_size")
            dropped_features = sbDropped
            df = df.drop(dropped_features, axis=1)
            x = df.drop(sbTarget, axis=1)
            y = df[sbTarget]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=random_state)
            x_train.reset_index(drop=True, inplace=True)
            x_test.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)

            scaller = MinMaxScaler()
            x_train = scaller.fit_transform(x_train)
            x_test = scaller.transform(x_test)
            saveSession({"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test, "scaler": scaller})

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Linear Regression", divider='grey')
            logreg = LogisticRegression()
            logreg.fit(x_train, y_train)
            logreg_score = logreg.score(x_test, y_test)
            # sns.regplot(x=x, y=y, data=df, logistic=True, ci=None)
            y_pred=logreg.predict(x_test)

            with st.expander("Show Logistic Regression Information"):
                # plt.scatter(y_test, y_pred, color='steelblue',
                #     label='Actual vs. Predicted', alpha=0.4)
                # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
                #         linestyle='--', color='red', linewidth=2, label='Perfect Prediction')

                # plt.xlabel('Actual Values')
                # plt.ylabel('Predicted Values')
                # st.pyplot(plt.gcf())
                st.markdown("**Score :** *{:.3f}*".format(logreg_score))
                st.markdown("*Logistic Regression Model Initialization Complete*")

        with col2:
            st.subheader("KNN", divider='grey')
            # ==================
            # Plot error rate  
            # ==================
            error_rate = []
            list_score = []
            for i in range(1,40):
                knn = KNeighborsClassifier(n_neighbors=i, weights="distance")
                knn.fit(x_train,y_train)
                pred_i = knn.predict(x_test)
                error_rate.append(np.mean(pred_i != y_test))
                list_score.append(knn.score(x_test, y_test))
                knn_score = knn.score(x_test, y_test)
            
            plt.figure(figsize=(10,6))
            plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
                    markerfacecolor='red', markersize=10)
            plt.title('Error Rate vs. K Value')
            plt.xlabel('K')
            plt.ylabel('Error Rate')
            
            # Get the best K
            k_auto_neighbor, k_neighbor_score = getKNN_k(x_train, y_train, x_test, y_test, list(x.columns))
            
            if (knn_auto_k):
                load_knn = KNeighborsClassifier(n_neighbors=k_auto_neighbor, weights="distance")
            else:
                load_knn = KNeighborsClassifier(n_neighbors=k_input, weights="distance")
            load_knn.fit(x_train, y_train)
            y_pred_knn = load_knn.predict(x_test)
            load_knn_score = load_knn.score(x_test, y_test)
            saveSession({"knn_model": load_knn, "knn_score": load_knn_score, "k_neighbor": k_auto_neighbor, "k_neighbor_score": k_neighbor_score})



            with st.expander("Show KNN Information"):
                st.markdown("**Score :** *{:.3f}*".format(load_knn_score))
                if (knn_auto_k):
                    st.markdown("**Best K :** *{}*".format(k_auto_neighbor))
                    st.pyplot(plt.gcf())
                st.markdown("*KNN Model Initialization Complete*")
                plt.clf()
                plt.scatter(y_test, y_pred, color='steelblue',label='Actual vs. Predicted', alpha=0.4)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                st.pyplot(plt.gcf())








# with st.spinner("Loading...", ):
#     time.sleep(5)
# st.markdown("aaaaa")



















