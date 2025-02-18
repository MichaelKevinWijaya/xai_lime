# pip install statsmodels
# pip install plotly


import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import lime
import matplotlib.pyplot as plt
import tempfile
import base64
import plotly.graph_objs as go

from session_utils import (saveSession, getSession)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
# from lime.html import explanation_html

import statsmodels.api as sm

# show_pages_from_config()
# https://github.com/daniellewisDL/streamlit-cheat-sheet/blob/master/README.md
st.set_page_config(layout="wide", page_title="Tabular Explanation", page_icon="🗄")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Tabular Data Explanation")
st.subheader("Upload Dataset File")
file_csv = st.file_uploader("Browse CSV file", type=["csv"], on_change=lambda: st.session_state.clear())

# First Initialization when project run
if 'clicked' not in st.session_state:
    st.session_state["clicked"] = {"confirmTarget": False, "initData": False, "initModel": False, "confirmDropped": False, 
                                   "initLogregModel": False, "initKNNModel": False, "initSVMModel": False, "initTreeModel": False, 
                                   "confirmProgressInit": False,
                                   "click_generate_instance_tab0": True,
                                   "click_generate_instance_tab1": True,
                                   "click_generate_instance_tab2": True,
                                   "model_visual" : False,
                                   "initLIME" : False}

@st.cache_data(show_spinner=False)
def readData(file_csv):
    return pd.read_csv(file_csv)

def clicked(button):
    print("================================================")
    print(st.session_state["clicked"])
    if (button == "initModel"):
        saveSession({"confirmProgressInit": True})
    st.session_state.clicked[button] = True
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print(st.session_state["clicked"])


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

@st.cache_data(show_spinner=False)
def calculate_error_rates_and_plot():
    error_rate = []
    list_score = []
    for i in range(1,40):
        knn = KNeighborsClassifier(n_neighbors=i, weights="distance")
        knn.fit(x_train,y_train)
        pred_i = knn.predict(x_test)
        error_rate.append(np.mean(pred_i != y_test))
        list_score.append(knn.score(x_test, y_test))
        knn_score = knn.score(x_test, y_test)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    ax.set_title('Error Rate vs. K Value')
    ax.set_xlabel('K')
    ax.set_ylabel('Error Rate')
    return fig

@st.cache_data
def fit_model(_train_x_vector, _train_y, model_type, k=5, c=1.0, gamma=1, kernel="linear"):
    if model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    elif model_type == "SVM":
        model = SVC(C=c, gamma=gamma, kernel=kernel, probability=True)
    model.fit(_train_x_vector, _train_y)
    accuracy_score = model.score(_train_x_vector, _train_y)
    return accuracy_score, model

if (file_csv):
    df = readData(file_csv)
    file_name = file_csv.name
    st.session_state["file_csv"] = file_csv
    st.session_state["uploaded_file_name"] = file_name
    st.session_state["df"] = df
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
    unique_values = []
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
            if (file_name == "data_commuter.csv"):
                unique_values = df[sbTarget].unique()
                saveSession({"unique_values": unique_values})
                df.loc[df["Kepuasan Hidup"] == "Sangat Tidak Puas", "Kepuasan Hidup"] = 0
                # df.loc[df["Kepuasan Hidup"] == "Biasa", "Kepuasan Hidup"] = 1
                df.loc[df["Kepuasan Hidup"] == "Sangat Puas", "Kepuasan Hidup"] = 1
                df = df.astype(float)
            if (file_name == "patient_treatment.csv"):
                unique_values = df[sbTarget].unique()
                saveSession({"unique_values": unique_values})
                df.loc[df["SOURCE"] == "out", "SOURCE"] = 0
                df.loc[df["SOURCE"] == "in", "SOURCE"] = 1
                df.loc[df["SEX"] == "M", "SEX"] = 0
                df.loc[df["SEX"] == "F", "SEX"] = 1
                # df.rename(columns = {'THROMBOCYTE':'THROM'}, inplace = True)
                # df.rename(columns = {'HAEMATOCRIT':'HATOC'}, inplace = True)
                # df.rename(columns = {'HAEMOGLOBINS':'HAGLO'}, inplace = True)
                # df.rename(columns = {'ERYTHROCYTE':'ERYTH'}, inplace = True)
                # df.rename(columns = {'LEUCOCYTE':'LEUCO'}, inplace = True)
                df = df.astype(float)
            if (file_name == "jogja_air_quality.csv"):
                unique_values = ['Moderate', 'Good']
                saveSession({"unique_values": unique_values})
                df.loc[df["Category"] == "Good", "Category"] = 0
                df.loc[df["Category"] == "Moderate", "Category"] = 1
                indexAge = df[(df['Category'] == 'Unhealthy')].index
                df.drop(indexAge, inplace=True)
                dummy = df['Critical Component'].str.get_dummies(sep=',').add_prefix('CC_')
                df = pd.concat([df.drop(columns=['Critical Component']), dummy], axis=1)
                # st.write(df.shape)
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
            # st.session_state["clicked"]["initLogregModel"] = True

    # KNN MODEL INITIALIZATION
    if getSession("clicked")["initLogregModel"] and len(sbLogregVisualized) == 2:
        st.subheader("KNN Model Initialization", divider="grey")
        with st.form(key="form_knnmodel_initialization"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### KNN")
                knn_auto_k = False
                st.caption("Select K value for KNN model manually or automatically")
                k_input = st.number_input('K Neighbor', key="knn_inp_k", min_value=2, max_value=40, value=5)
                st.caption("*or*")
                knn_auto_k = st.checkbox("Auto K", value=True, key="knn_auto_k")
                st.caption("Automatically find the most optimal N for KNN model using KNN")  
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
            # st.session_state["clicked"]["initKNNModel"] = True

    # SVM MODEL INITIALIZATION
    if getSession("clicked")["initKNNModel"] and len(sbKNNVisualized) == 2:
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
                st.write("#### Kernel value")
                st.caption("Select Kernel")
                kernel_input = st.selectbox('Kernel', ['linear', 'rbf', 'poly', 'sigmoid'], key="svm_inp_kernel")
                st.caption(":red[**WARNING!**] *Choose linear for further visualization!*")
                saveSession({"kernel_input": kernel_input})
                # st.caption("*or*")
                # svm_auto_gmma = st.checkbox("Auto Select", value=True, key="svm_auto_kernel")
                # st.caption("Automatically find the most optimal kernel for SVM model")  
            st.form_submit_button("Proceed", type="primary", on_click=clicked, args=["initSVMModel"])
            # st.session_state["clicked"]["initSVMModel"] = True

    if getSession("clicked")["initSVMModel"]:
        st.subheader("Decision Tree Model Initialization", divider="grey")
        with st.form(key="form_treemodel_initialization"):
            max_depth = st.number_input('Max Depth', key="max_depth", min_value=1, max_value=10, value=3)
            st.form_submit_button("Begin Initialization", type="primary", on_click=clicked, args=["initTreeModel"])
            # st.session_state["clicked"]["initTreeModel"] = True

    # PROCESS INITIALIZATION
    if getSession("clicked")["initTreeModel"]:
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

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Logistic Regression", divider='grey')
            logreg = LogisticRegression()
            logreg.fit(x_train, y_train)
            logreg_score = logreg.score(x_test, y_test)
            # logreg_score, logreg = fit_model(x_train, y_train, "Logistic Regression")
            y_pred_logreg=logreg.predict(x_test)
            saveSession({"logreg_model": logreg, "logreg_score": logreg_score})

            if (getSession("clicked")["initLogregModel"] and len(sbLogregVisualized) == 2):
                with st.spinner("Loading...", ):
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
            with st.spinner("Loading...", ):
                # ==================
                # Plot error rate  
                # ==================
                fig = calculate_error_rates_and_plot()
                # st.pyplot(fig)
                
                # plt.figure(figsize=(10,6))
                # plt.title('Error Rate vs. K Value')
                # plt.xlabel('K')
                # plt.ylabel('Error Rate')
                # plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
                
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
                            st.pyplot(fig)
                        else:
                            st.markdown("K Neighbors : {}".format(k_input))
                        plt.clf()
                        test_point = x_test[0]
                        if (knn_auto_k):
                            distances, indices = load_knn.kneighbors([test_point], n_neighbors=k_auto_neighbor)
                        else:
                            distances, indices = load_knn.kneighbors([test_point], n_neighbors=k_input)
                        plt.scatter(x_train[:, sbKNNVisualized_idx[0]], x_train[:, sbKNNVisualized_idx[1]], c=[(0.5, 0.5, 0.5, 0.1)], s=30, edgecolors=(0, 0, 0, 0.5), label='Other Data')
                        plt.scatter(x_train[indices[0], sbKNNVisualized_idx[0]], x_train[indices[0], sbKNNVisualized_idx[1]], c='cyan', marker='o', edgecolors='k', label='Nearest Neighbors')
                        plt.scatter(test_point[sbKNNVisualized_idx[0]], test_point[sbKNNVisualized_idx[1]], c='red', marker='x', edgecolors='k', s=100, label='Test Point')
                        plt.xlabel('Feature 1')
                        plt.ylabel('Feature 2')
                        plt.title('Nearest Neighbors Visualization')
                        plt.legend()
                        st.pyplot(plt.gcf())

                        # AUC/ROC Curve
                        probabilities = load_knn.predict_proba(x_test)
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
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SVM", divider='grey')
            # svm_model = SVC()
            # param_grid = {'C': c_input, 'gamma': gamma_input, 'kernel': kernel_input}
            # c_input = c_input[0] if c_input else 1.0
            # gamma_input = gamma_input[0] if gamma_input else 0.1
            # kernel_input = kernel_input[0] if kernel_input else 'rbf'
            with st.spinner("Loading...", ):
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
        with col2:
            class_names = getSession("unique_values")
            max_depth = getSession("max_depth")
            x_train_lime = pd.DataFrame(x_train, columns = x.columns)
            feature_names = list(x_train_lime.columns)
            st.subheader("Decision Tree", divider='grey')
            with st.spinner("Loading...", ):
                if (getSession("clicked")["initSVMModel"]):
                    with st.expander("Show Decision Tree Information"):
                        decisiontree = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
                        decisiontree.fit(x_train, y_train)
                        y_pred = decisiontree.predict(x_test)
                        tree_accuracy = accuracy_score(y_test, y_pred)
                        st.markdown("**Score :** *{:.3f}*".format(tree_accuracy))
                        saveSession({"decisiontree": decisiontree, "svm_score": tree_accuracy})
                        plt.figure(figsize=(13,5))
                        plot_tree(decisiontree, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
                        st.pyplot()

                        # AUC/ROC Curve
                        probabilities = decisiontree.predict_proba(x_test)
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
                        st.markdown("*Decision Tree Model Initialization Complete*")

        st.session_state["clicked"]["model_visual"] = True

    if st.session_state.get("clicked", {}).get("model_visual", True):
        st.subheader("Global Explainer", divider='grey')
        knn_model = getSession("knn_model")
        logreg_model = getSession("logreg_model")
        svm_model = getSession("svm_model")
        kernel_input = getSession("kernel_input")
        with st.spinner("Loading...", ):
            # LOGREG
            coefficients = logreg_model.coef_.ravel()
            feature_names = list(x.columns)
            sorted_indices = np.argsort(np.abs(coefficients))[::-1] # Reverse order for descending sort
            sorted_coefficients = coefficients[sorted_indices]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]
            sorted_data = pd.DataFrame({'feature': sorted_feature_names, 'coefficient': sorted_coefficients})
            colors = ['royalblue' if coef < 0 else 'darkorange' for coef in sorted_coefficients]

            # Create the horizontal bar plot using plotly
            trace = go.Bar(
                x=sorted_data['coefficient'],
                y=sorted_data['feature'],
                orientation='h',
                marker=dict(color=colors),  # Use emerald-like color
            )

            # Customize layout
            layout = go.Layout(
                title='Logistic Regression Global Explanation: Coefficient Magnitudes',
                titlefont=dict(color='black'),
                xaxis=dict(
                    title='Coefficient Magnitude',
                    titlefont=dict(color='black'),  # Set x-axis title font color
                    tickfont=dict(color='black')  # Set x-axis tick labels color
                ),
                yaxis=dict(
                    title='Feature Name',
                    titlefont=dict(color='black'),  # Set y-axis title font color
                    tickfont=dict(color='black'),  # Set y-axis tick labels color
                    autorange="reversed"
                ),
                hovermode='closest',
                plot_bgcolor='rgba(255, 255, 255, 1)',
                paper_bgcolor='rgba(255, 255, 255, 1)',
                title_x=0.15,  # Invert y-axis for top-down display
            )
            # Create a Figure object
            fig = go.Figure(data=[trace], layout=layout)
            # Display the plot in Streamlit
            st.plotly_chart(fig)

            # SVM
            if kernel_input == "linear" :
                coefficients = svm_model.coef_.ravel()
                # coefficients = -coefficients
                feature_names = list(x.columns)
                sorted_indices = np.argsort(np.abs(coefficients))[::-1] # Reverse order for descending sort
                sorted_coefficients = coefficients[sorted_indices]
                sorted_feature_names = [feature_names[i] for i in sorted_indices]
                sorted_data = pd.DataFrame({'feature': sorted_feature_names, 'coefficient': sorted_coefficients})
                colors = ['royalblue' if coef < 0 else 'darkorange' for coef in sorted_coefficients]

                trace = go.Bar(
                    x=sorted_data['coefficient'],
                    y=sorted_data['feature'],
                    orientation='h',
                    marker=dict(color=colors),
                    hoverinfo='x+y'  # Display both values on hover
                )

                # Customize layout
                layout = go.Layout(
                    title='SVM Global Explanation: Coefficient Magnitudes',
                    titlefont=dict(color='black'),
                    xaxis=dict(
                        title='Coefficient Magnitude',
                        titlefont=dict(color='black'),  # Set x-axis title font color
                        tickfont=dict(color='black')  # Set x-axis tick labels color
                    ),
                    yaxis=dict(
                        title='Feature Name',
                        titlefont=dict(color='black'),  # Set y-axis title font color
                        tickfont=dict(color='black'),  # Set y-axis tick labels color
                        autorange="reversed"
                    ),
                    hovermode='closest',
                    plot_bgcolor='rgba(255, 255, 255, 1)',
                    paper_bgcolor='rgba(255, 255, 255, 1)',
                    title_x=0.25,  # Invert y-axis for top-down display
                )
                # Create a Figure object
                fig = go.Figure(data=[trace], layout=layout)
                # Display the plot in Streamlit
                st.plotly_chart(fig)

    if st.session_state.get("clicked", {}).get("model_visual", True):
        st.subheader("LIME Initialization", divider="grey")
        with st.form(key="form_lime_initialization"):
            col1, col2 = st.columns(2)
            with col1:
                max_idx = len(pd.DataFrame(x_train, columns=x.columns))
                lime_instance = st.number_input('Instance to be explained', key="lime_target", min_value=0, max_value=max_idx-1, step=1)
            with col2:
                max_idx = df.shape[1] - len(sbDropped)
                column_num = st.number_input('Number of columns to be displayed', key="max_columns", min_value=1, value=max_idx-1, step=1)
            st.form_submit_button("Begin Initialization", type="primary", on_click=clicked, args=["initLIME"])
        st.session_state["lime_instance"] = lime_instance
        st.session_state["column_num"] = column_num
        # st.session_state["clicked"]["initLIME"] = True


    if (getSession("clicked")["initLIME"]):
        x_train_lime = pd.DataFrame(x_train, columns = x.columns)
        x_test_lime = pd.DataFrame(x_test, columns = x.columns)
        
        class_names = getSession("unique_values")
        feature_names = list(x_train_lime.columns)
        knn_model = getSession("knn_model")
        logreg_model = getSession("logreg_model")
        svm_model = getSession("svm_model")
        decisiontree = getSession("decisiontree")
        lime_instance = getSession("lime_instance")
        column_num = getSession("column_num")
        kernel_input = getSession("kernel_input")

        st.subheader("LIME Explainer", divider='grey')
        true_pred = y_test.iloc[lime_instance]
        if(true_pred == 0):
            if(file_name == "patient_treatment.csv"):
                st.markdown("## True Prediction : " + "Out care patient")
            if(file_name == "jogja_air_quality.csv"):
                st.markdown("## True Prediction : " + "Moderate")
        else:
            if(file_name == "patient_treatment.csv"):
                st.markdown("## True Prediction : " + "In care patient")
            if(file_name == "jogja_air_quality.csv"):
                st.markdown("## True Prediction : " + "Good")
        with st.spinner("Loading...", ):
            # LIME for KNN =================================================================
            explainer = LimeTabularExplainer(x_test_lime.values, feature_names =
                                            feature_names,
                                            class_names = class_names,
                                            mode = 'classification')
            exp = explainer.explain_instance(data_row=x_test_lime.iloc[lime_instance], 
                                            predict_fn=logreg_model.predict_proba,
                                            num_features=column_num)
            html_explanation = exp.as_html()
            temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            temp_file_path.write(html_explanation.encode())
            temp_file_path.close()
            # Display the HTML explanation using an iframe
            st.markdown("#### Logistic Regression Model", unsafe_allow_html=True)
            st.markdown(f'<iframe src="data:text/html;base64,{base64.b64encode(open(temp_file_path.name, "rb").read()).decode()}" height="340" width="1000"></iframe>', unsafe_allow_html=True)

            # LIME for LOGREG =================================================================
            explainer = LimeTabularExplainer(x_test_lime.values, feature_names =
                                            feature_names,
                                            class_names = class_names,
                                            mode = 'classification')
            exp = explainer.explain_instance(data_row=x_test_lime.iloc[lime_instance], 
                                            predict_fn=knn_model.predict_proba,
                                            num_features=column_num)
            html_explanation = exp.as_html()
            temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            temp_file_path.write(html_explanation.encode())
            temp_file_path.close()
            # Display the HTML explanation using an iframe
            st.markdown("#### KNN Model", unsafe_allow_html=True)
            st.markdown(f'<iframe src="data:text/html;base64,{base64.b64encode(open(temp_file_path.name, "rb").read()).decode()}" height="340" width="1000"></iframe>', unsafe_allow_html=True)

            # LIME for SVM =================================================================
            explainer = LimeTabularExplainer(x_test_lime.values, feature_names =
                                            feature_names,
                                            class_names = class_names,
                                            mode = 'classification')
            exp = explainer.explain_instance(data_row=x_test_lime.iloc[lime_instance], 
                                            predict_fn=svm_model.predict_proba,
                                            num_features=column_num)
            html_explanation = exp.as_html()
            temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            temp_file_path.write(html_explanation.encode())
            temp_file_path.close()
            # Display the HTML explanation using an iframe
            st.markdown("#### SVM Model", unsafe_allow_html=True)
            st.markdown(f'<iframe src="data:text/html;base64,{base64.b64encode(open(temp_file_path.name, "rb").read()).decode()}" height="340" width="1000"></iframe>', unsafe_allow_html=True)

            # LIME for Decision Tree =================================================================
            exp = explainer.explain_instance(data_row=x_test_lime.iloc[lime_instance], 
                                             predict_fn=decisiontree.predict_proba, 
                                             num_features=column_num)
            html_explanation = exp.as_html()
            temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            temp_file_path.write(html_explanation.encode())
            temp_file_path.close()
            # Display the HTML explanation using an iframe
            st.markdown("#### Decision Tree Model", unsafe_allow_html=True)
            st.markdown(f'<iframe src="data:text/html;base64,{base64.b64encode(open(temp_file_path.name, "rb").read()).decode()}" height="340" width="1000"></iframe>', unsafe_allow_html=True)
