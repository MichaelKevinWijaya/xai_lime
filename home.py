# pip install statsmodels

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import lime
import matplotlib.pyplot as plt
import time
import io

from session_utils import (saveSession, getSession)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from st_pages import show_pages_from_config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import statsmodels.api as sm

# show_pages_from_config()
# https://github.com/daniellewisDL/streamlit-cheat-sheet/blob/master/README.md
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Home Page")
st.subheader("Upload Dataset File")
file_csv = st.file_uploader("Browse CSV file", type=["csv"], on_change=lambda: st.session_state.clear())

# First Initialization when project run
if 'clicked' not in st.session_state:
    st.session_state["clicked"] = {"confirmTarget": False, "initData": False, "initModel": False, "confirmDropped": False, 
                                   "initLogregModel": False, "initKNNModel": False, "initSVMModel": False, 
                                   "confirmProgressInit": False,
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


def getKNN_k(x_train, y_train, x_test, y_test, x_columns, min_iter=2, max_iter=40):

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
    st.session_state["file_csv"] = file_csv
    st.session_state["uploaded_file_name"] = file_name
    if (file_name == "data_commuter.csv"):
        df.loc[df["Kepuasan Hidup"] == "Sangat Tidak Puas", "Kepuasan Hidup"] = 0
        df.loc[df["Kepuasan Hidup"] == "Biasa", "Kepuasan Hidup"] = 1
        df.loc[df["Kepuasan Hidup"] == "Sangat Puas", "Kepuasan Hidup"] = 2
        df = df.astype(float)
    if (file_name == "patient_treatment.csv"):
        df.loc[df["SOURCE"] == "out", "SOURCE"] = 0
        df.loc[df["SOURCE"] == "in", "SOURCE"] = 1
        df.loc[df["SEX"] == "M", "SEX"] = 0
        df.loc[df["SEX"] == "F", "SEX"] = 1
        df = df.astype(float)
    if (file_name == "jogja_air_quality.csv"):
        df.loc[df["Category"] == "Good", "Category"] = 0
        df.loc[df["Category"] == "Moderate", "Category"] = 1
        indexAge = df[(df['Category'] == 'Unhealthy')].index
        df.drop(indexAge, inplace=True)
        dummy = df['Critical Component'].str.get_dummies(sep=',').add_prefix('CC_')
        df = pd.concat([df.drop(columns=['Critical Component']), dummy], axis=1)
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
    sbLogregVisualized = []
    sbKNNVisualized = []
    dataColumns = [df.columns[idx] for idx in range(len(df.columns))]
    dataColumnsX = [df.columns[idx] for idx in range(len(df.columns)-1)]

    with st.form("form_dropped_features"):
        st.caption("Select features to be dropped in the dataset")
        sbDropped = st.multiselect('Dropped Features', dataColumns, key="dropped_features")
        st.form_submit_button("Proceed", type="primary", on_click=clicked, args=["confirmDropped"])
    if (len(sbDropped) > len(dataColumns)-2):
        st.caption(":red *Leave at least two (2) features to proceed!*")

    if getSession("clicked")["confirmDropped"] and len(sbDropped) <= len(dataColumns)-2:
        columnOptions = [col for col in df.columns if col not in sbDropped]
        with st.form(key="form_feature_selection"):
            st.caption("Select target feature to be predicted in the dataset")
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
                st.number_input('Test Size Ratio', key="test_size", min_value=0.1, max_value=0.9, value=0.2, step=0.1)
                st.caption("Test and Train Size Ratio to be Divided for Train and Testing")
            with col_init2:
                st.number_input('Random State', key='random_state', min_value=0, max_value=100, value=42, step=1)
                st.caption("State of Randomness for Reproducibility")

            st.form_submit_button("Proceed", type="primary", on_click=clicked, args=["initData"])

    # LOGISTIC MODEL INITIALIZATION
    if getSession("clicked")["initData"]:
        # st.write(getSession("random_state"))
        st.subheader("Logistic Regression Model Initialization", divider="grey")
        with st.form(key="form_logregmodel_initialization"):
            st.write("#### Feature Visualization")
            st.caption("Select features to be visualized in the dataset")
            columnOptions = [col for col in df.columns if col not in sbDropped]
            sbLogregVisualized = st.multiselect('Visualized Features', columnOptions, key="logreg_visualized_features")
            if (len(sbLogregVisualized) != 2):
                st.caption(":red[**WARNING!**] *Pick 2 features to be visualized!*")
            sbLogregVisualized_idx = [dataColumns.index(value) for value in sbLogregVisualized]
            st.form_submit_button("Proceed", type="primary", on_click=clicked, args=["initLogregModel"])
            st.session_state["clicked"]["initLogregModel"] = True

    # KNN MODEL INITIALIZATION
    if getSession("clicked")["initLogregModel"] and len(sbLogregVisualized) == 2:
        # st.write(getSession("random_state"))
        st.subheader("KNN Model Initialization", divider="grey")
        with st.form(key="form_knnmodel_initialization"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### KNN")
                knn_auto_k = False
                st.caption("Select K value for KNN model manually or automatically")
                k_input = st.number_input('K Neighbor', key="knn_inp_k", min_value=2, value=5)
                st.caption("*or*")
                knn_auto_k = st.checkbox("Auto K", value=True, key="knn_auto_k")
                st.caption("Automatically find the most optimal N for KNN model using KNN Regressor")  
            with col2:
                st.write("#### Feature Visualization")
                st.caption("Select features to be visualized in the dataset")
                columnOptions = [col for col in df.columns if col in dataColumnsX]
                sbKNNVisualized = st.multiselect('Visualized Features', columnOptions, key="knn_visualized_features")
                if (len(sbKNNVisualized) != 2):
                    st.caption(
                        ":red[**WARNING!**] *Pick 2 features to be visualized!*")
                sbKNNVisualized_idx = [columnOptions.index(value) for value in sbKNNVisualized]
            st.form_submit_button("Proceed", type="primary", on_click=clicked, args=["initKNNModel"])
            st.session_state["clicked"]["initKNNModel"] = True

    # SVM MODEL INITIALIZATION
    if getSession("clicked")["initKNNModel"] and len(sbKNNVisualized) == 2:
        # st.write(getSession("random_state"))
        st.subheader("SVM Model Initialization", divider="grey")
        with st.form(key="form_svmmodel_initialization"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("#### C value")
                st.caption("Select C value")
                c_input = st.selectbox('C value', [0.1, 1.0, 10.0, 100.0], key="svm_inp_c")
                # st.caption("*or*")
                # svm_auto_c = st.checkbox("Auto Select", value=True, key="svm_auto_c")
                # st.caption("Automatically find the most optimal C for SVM model")  
            with col2:
                st.write("#### Gamma value")
                st.caption("Select Gamma value")
                gamma_input = st.selectbox('Gamma value', [1, 0.1, 0.01, 0.001], key="svm_inp_gamma")
                # st.caption("*or*")
                # svm_auto_gmma = st.checkbox("Auto Select", value=True, key="svm_auto_gamma")
                # st.caption("Automatically find the most optimal Gamma for SVM model")  
            with col3:
                st.write("#### Gamma value")
                st.caption("Select Kernal")
                kernel_input = st.selectbox('Kernel', ['rbf', 'linear', 'poly', 'sigmoid'], key="svm_inp_kernel")
                # st.caption("*or*")
                # svm_auto_gmma = st.checkbox("Auto Select", value=True, key="svm_auto_kernel")
                # st.caption("Automatically find the most optimal kernel for SVM model")  
            st.form_submit_button("Begin Initialization", type="primary", on_click=clicked, args=["initSVMModel"])
            st.session_state["clicked"]["initSVMModel"] = True


    # PROCESS INITIALIZATION
    if getSession("clicked")["initSVMModel"]:
        with st.spinner('Initializing the Data, please wait...'):
            random_state = getSession("random_state")
            target_feature = sbTarget
            test_size = getSession("test_size")
            dropped_features = sbDropped
            df = df.drop(dropped_features, axis=1)
            x = df.drop(sbTarget, axis=1)
            y = df[sbTarget]
            y=y.astype('int')
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

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Logistic Regression", divider='grey')
            logreg = LogisticRegression()
            logreg.fit(x_train, y_train)
            logreg_score = logreg.score(x_test, y_test)
            y_pred_logreg=logreg.predict(x_test)
            saveSession({"logreg_model": logreg, "logreg_score": logreg_score})

            if (getSession("clicked")["initLogregModel"] and len(sbLogregVisualized) == 2):
                with st.expander("Show Logistic Regression Information"):
                    st.markdown("**Score :** *{:.3f}*".format(logreg_score))
                    # Logistic Regression Model
                    feat_x = df[sbLogregVisualized[0]]
                    feat_y = df[sbLogregVisualized[1]]
                    feat_y = feat_y.astype('int')
                    # Fit logistic regression model
                    X = sm.add_constant(feat_x)  # Add constant for intercept
                    model = sm.Logit(feat_y, X)
                    result = model.fit()
                    # Get the logistic regression line
                    x_values = np.linspace(feat_x.min(), feat_x.max(), 100)
                    y_values = result.predict(sm.add_constant(x_values))
                    # Plot using seaborn
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=feat_x, y=feat_y, ax=ax)
                    ax.plot(x_values, y_values)
                    ax.set_xlabel(sbLogregVisualized[0])
                    ax.set_ylabel(sbLogregVisualized[1])
                    ax.legend()
                    st.pyplot(fig)

                    # AUC/ROC Curve
                    probabilities = logreg.predict_proba(x_test)
                    # fpr, tpr, _ = roc_curve(y_test, y_pred_knn)
                    fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
                    roc_auc = auc(fpr, tpr)
                    # Plot ROC curve
                    plt.figure()
                    lw = 2
                    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random Guess')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc="lower right")
                    st.pyplot(plt.gcf())

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
            plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
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
        
            if (getSession("clicked")["initKNNModel"] and len(sbKNNVisualized) == 2):
                with st.expander("Show KNN Information"):
                    st.markdown("**Score :** *{:.3f}*".format(load_knn_score))
                    if (knn_auto_k):
                        st.markdown("**Best K :** *{}*".format(k_auto_neighbor))
                        st.pyplot(plt.gcf())
                    plt.clf()
                    test_point = x_test[0]
                    if (knn_auto_k):
                        distances, indices = knn.kneighbors([test_point], n_neighbors=k_auto_neighbor)
                    else:
                        distances, indices = knn.kneighbors([test_point], n_neighbors=k_input)
                    plt.scatter(x_train[:, sbKNNVisualized_idx[0]], x_train[:, sbKNNVisualized_idx[1]], c=[(0.5, 0.5, 0.5, 0.1)], s=30, edgecolors=(0, 0, 0, 0.5), label='Training Data')
                    plt.scatter(x_train[indices[0], sbKNNVisualized_idx[0]], x_train[indices[0], sbKNNVisualized_idx[1]], c='cyan', marker='o', edgecolors='k', label='Nearest Neighbors')
                    plt.scatter(test_point[sbKNNVisualized_idx[0]], test_point[sbKNNVisualized_idx[1]], c='red', marker='x', edgecolors='k', s=100, label='Test Point')
                    plt.xlabel('Feature 1')
                    plt.ylabel('Feature 2')
                    plt.title('Nearest Neighbors Visualization')
                    plt.legend()
                    st.pyplot(plt.gcf())

                    # AUC/ROC Curve
                    probabilities = knn.predict_proba(x_test)
                    # fpr, tpr, _ = roc_curve(y_test, y_pred_knn)
                    fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
                    roc_auc = auc(fpr, tpr)

                    # Plot ROC curve
                    plt.figure()
                    lw = 2
                    plt.plot(fpr, tpr, color='darkorange',
                            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random Guess')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc="lower right")
                    st.pyplot(plt.gcf())
                    st.markdown("*KNN Model Initialization Complete*")

        with col3:
            st.subheader("SVM", divider='grey')
            svm_model = SVC()
            # param_grid = {'C': c_input, 'gamma': gamma_input, 'kernel': kernel_input}
            # c_input = c_input[0] if c_input else 1.0
            # gamma_input = gamma_input[0] if gamma_input else 0.1
            # kernel_input = kernel_input[0] if kernel_input else 'rbf'
            svm_model = SVC(C=c_input, gamma=gamma_input, kernel=kernel_input, probability=True)
            svm_model.fit(x_train, y_train)
            y_pred = svm_model.predict(x_test)
            svm_accuracy = accuracy_score(y_test, y_pred)
            saveSession({"svm_model": svm_model, "svm_score": svm_accuracy})

            if (getSession("clicked")["initSVMModel"]):
                with st.expander("Show SVM Information"):
                    st.markdown("**Score :** *{:.3f}*".format(svm_accuracy))

                    probabilities = svm_model.predict_proba(x_test)
                    fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
                    roc_auc = auc(fpr, tpr)

                    # Plot ROC curve
                    plt.figure()
                    lw = 2
                    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random Guess')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc="lower right")
                    st.pyplot(plt.gcf())

                    st.markdown("*SVM Model Initialization Complete*")








# with st.spinner("Loading...", ):
#     time.sleep(5)
# st.markdown("aaaaa")



















