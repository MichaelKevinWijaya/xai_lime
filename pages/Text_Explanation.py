import streamlit as st
import pandas as pd
import numpy as np
import lime
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from numpy.random import default_rng
# from utils import getSession

#Load the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
import re
import re,string,unicodedata
import plotly.graph_objects as go
import tempfile
import base64

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import roc_curve, auc
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob, Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from session_utils import (saveSession, getSession)
from lime.lime_text import LimeTextExplainer
# from IPython.display import HTML

import os
import warnings
warnings.filterwarnings('ignore')

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# file_csv_session = getSession("file_csv")
# knn_model = getSession("knn_model")
st.set_page_config(layout="wide", page_title="Text Explanation", page_icon="📜")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("LIME Text Explanainer")
st.subheader("Upload Dataset File")
file_csv = st.file_uploader("Browse CSV file", type=["csv"], on_change=lambda: st.session_state.clear())

# First Initialization when project run
if 'clicked' not in st.session_state:
    st.session_state["clicked"] = {"confirmTarget": False, "initData": False, "initModel": False, "confirmDropped": False, 
                                   "initLogregModel": False, "initKNNModel": False, "initSVMModel": False, 
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
    if (button == "initModel"):
        saveSession({"confirmProgressInit": True})
    st.session_state.clicked[button] = True

def remove_html_tags(text):
  clean = re.sub(r'<[^>]*>', '', text)
  return clean

@st.cache_data(show_spinner=False)  # Cache the word cloud generation
def create_word_cloud(text):
    # Code to generate the Word Cloud image using WordCloud library
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    fig.canvas.draw()
    return fig

@st.cache_data(show_spinner=False)
def calculate_error_rates_and_plot(_train_x_vector, train_y, _test_x_vector, test_y):
    train_x_vector = _train_x_vector.toarray()
    test_x_vector = _test_x_vector.toarray()
    error_rate = []
    k_neighbor_score = []
    for i in range(1, 40):
        knn_patient = KNeighborsClassifier(n_neighbors=i, weights='distance')
        knn_patient.fit(train_x_vector, train_y)
        pred_i = knn_patient.predict(test_x_vector)
        error_rate.append(np.mean(pred_i != test_y))
    min_index = np.argmin(error_rate)
    k_neighbor = min_index + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
            markerfacecolor='red', markersize=10)
    ax.set_title('Error Rate vs. K Value')
    ax.set_xlabel('K')
    ax.set_ylabel('Error Rate')
    return fig, k_neighbor

@st.cache_data
def plot_roc_curve(test_y, _test_x_vector, _model):
    """
    Plots the ROC curve for a given model and data.

    Args:
        test_y (list): True labels for the test data.
        test_x_vector (array): Features for the test data.
        model (object): The model to use for prediction.
        label_binarizer (object, optional): Label binarizer (if needed). Defaults to None.

    Returns:
        None
    """
    label_binarizer = LabelBinarizer()
    test_y_binary = label_binarizer.fit_transform(test_y)
    probabilities = _model.predict_proba(_test_x_vector)[:, 1]

    fpr, tpr, thresholds = roc_curve(test_y_binary, probabilities)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt.gcf())

@st.cache_data
def lime_explanation(model, instance, column_num):
    def predict_proba_wrapper(texts):
        return knn_model.predict_proba(tfidf.transform(texts))
    exp = explainer.explain_instance(instance[0], predict_proba_wrapper, num_features=column_num)
    # exp.show_in_notebook(text=True)
    html_explanation = exp.as_html()
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    temp_file_path.write(html_explanation.encode())
    temp_file_path.close()
    st.markdown("#### KNN Model", unsafe_allow_html=True)
    st.markdown(f'<iframe src="data:text/html;base64,{base64.b64encode(open(temp_file_path.name, "rb").read()).decode()}" height="340" width="1000"></iframe>', unsafe_allow_html=True)
    
if (file_csv):
    df = readData(file_csv)
    file_name = file_csv.name
    st.session_state["file_csv"] = file_csv
    st.session_state["uploaded_file_name"] = file_name
    st.session_state["df"] = df
    knn_auto_k = False

    with st.expander("Data Information"):
        df["text"] = df["text"].apply(remove_html_tags)
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
        
        col1, col2 = st.columns(2)
        with col1:
            # Negative Word Cloud (using cached create_word_cloud)
            st.subheader("Negative Word Cloud")
            negative_text = df[df['sentiment'] == "negative"]['text']
            negative_text_combined = ' '.join(negative_text)
            if st.session_state.get("neg_word_cloud", None) is None:
                st.session_state["neg_word_cloud"] = create_word_cloud(negative_text_combined)
            st.pyplot(st.session_state["neg_word_cloud"])  # Display cached negative word cloud
        with col2:
            # Positive Word Cloud (using cached create_word_cloud)
            st.subheader("Positive Word Cloud")
            positive_text = df[df['sentiment'] == "positive"]['text']
            positive_text_combined = ' '.join(positive_text)
            if st.session_state.get("pos_word_cloud", None) is None:
                st.session_state["pos_word_cloud"] = create_word_cloud(positive_text_combined)
            st.pyplot(st.session_state["pos_word_cloud"])  # Display cached positive word cloud
    
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
    
    # KNN MODEL INITIALIZATION
    if (getSession("clicked")["initData"]) :
        # st.write(getSession("random_state"))
        st.subheader("KNN Model Initialization", divider="grey")
        with st.form(key="form_knnmodel_initialization"):
            st.write("#### KNN")
            knn_auto_k = False
            st.caption("Select K value for KNN model manually or automatically")
            k_input = st.number_input('K Neighbor', key="knn_inp_k", min_value=2, max_value=40, value=5)
            st.caption("*or*")
            knn_auto_k = st.checkbox("Auto K", value=True, key="knn_auto_k")
            st.caption("Automatically find the most optimal N for KNN model using KNN")  
            st.form_submit_button("Proceed", type="primary", on_click=clicked, args=["initKNNModel"])
            # st.session_state["clicked"]["initKNNModel"] = True
    
        # SVM MODEL INITIALIZATION
    if (getSession("clicked")["initKNNModel"]) :
        # st.write(getSession("random_state"))
        st.subheader("SVM Model Initialization", divider="grey")
        with st.form(key="form_svmmodel_initialization"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("#### C value")
                st.caption("Select C value")
                c_input = st.selectbox('C value', [0.1, 1.0, 10.0, 100.0], key="svm_inp_c")
            with col2:
                st.write("#### Gamma value")
                st.caption("Select Gamma value")
                gamma_input = st.selectbox('Gamma value', [1, 0.1, 0.01, 0.001], key="svm_inp_gamma")
            with col3:
                st.write("#### Gamma value")
                st.caption("Select Kernel")
                kernel_input = st.selectbox('Kernel', ['linear', 'rbf', 'poly', 'sigmoid'], key="svm_inp_kernel")
                st.caption(":red[**WARNING!**] *Choose linear for further visualization!*")
                saveSession({"kernel_input": kernel_input})
            st.form_submit_button("Begin Initialization", type="primary", on_click=clicked, args=["initSVMModel"])
            # st.session_state["clicked"]["initSVMModel"] = True
    
    if (getSession("clicked")["initSVMModel"]):
        random_state = getSession("random_state")
        test_size = getSession("test_size")
        x = df["text"]
        y = df["sentiment"]
        train,test = train_test_split(df, test_size=test_size, random_state=random_state)
        train_x, train_y = train['text'], train['sentiment']
        test_x, test_y = test['text'], test['sentiment']
        tfidf = TfidfVectorizer(stop_words='english')
        train_x_vector = tfidf.fit_transform(train_x)
        test_x_vector = tfidf.transform(test_x)

        col1, col2, col3 = st.columns(3)
        with col1:
            # Logreg
            st.subheader("Logistic Regression", divider='grey')
            with st.spinner("Loading...", ):
                with st.expander("Show Logistic Regression Information"):
                    logreg_model = LogisticRegression()
                    logreg_model.fit(train_x_vector,train_y)
                    logreg_accuracy = logreg_model.score(test_x_vector, test_y)
                    st.markdown("**Score :** *{:.3f}*".format(logreg_accuracy))
                    # AUC/ROC
                    plot_roc_curve(test_y, test_x_vector, logreg_model)
                    saveSession({"logreg_model": logreg_model, "logreg_accuracy": logreg_accuracy})
        with col2:
            # KNN
            st.subheader("KNN", divider='grey')
            with st.spinner("Loading...", ):
                with st.expander("Show KNN Information"):
                    if (knn_auto_k):
                        fig, k_auto_neighbor = calculate_error_rates_and_plot(train_x_vector, train_y, test_x_vector, test_y)
                        knn_model = KNeighborsClassifier(n_neighbors=k_auto_neighbor, weights="distance")
                        knn_model.fit(train_x_vector, train_y)
                        knn_accuracy = knn_model.score(test_x_vector, test_y)
                        st.markdown("**Score :** *{:.3f}*".format(knn_accuracy))
                        st.markdown("**Best K :** *{}*".format(k_auto_neighbor))
                        st.pyplot(fig)
                    else:
                        knn_model = KNeighborsClassifier(n_neighbors=k_input, weights="distance")
                        knn_model.fit(train_x_vector, train_y)
                        knn_accuracy = knn_model.score(test_x_vector, test_y)
                        st.markdown("**Score :** *{:.3f}*".format(knn_accuracy))
                        st.markdown("K Neighbors : {}".format(k_input))
                    # AUC/ROC
                    plot_roc_curve(test_y, test_x_vector, knn_model)
                    saveSession({"knn_model": knn_model, "knn_accuracy": knn_accuracy})  
        with col3:
            # SVM
            st.subheader("SVM", divider='grey')
            with st.spinner("Loading...", ):
                with st.expander("Show SVM Information"):
                    svm_model = SVC(C=c_input, gamma=gamma_input, kernel=kernel_input, probability=True)
                    svm_model.fit(train_x_vector, train_y)
                    svm_accuracy = svm_model.score(test_x_vector, test_y)
                    st.markdown("**Score :** *{:.3f}*".format(svm_accuracy))
                    # AUC/ROC
                    plot_roc_curve(test_y, test_x_vector, svm_model)
                    saveSession({"svm_model": svm_model, "svm_score": svm_accuracy})
        st.session_state["clicked"]["model_visual"] = True
    
    if (getSession("clicked")["model_visual"]):
        st.subheader("LIME Initialization", divider="grey")
        with st.form(key="form_lime_initialization"):
            col1, col2 = st.columns(2)
            with col1:
                x = df["text"]
                max_idx = len(df["text"])
                lime_instance = st.number_input('Instance to be explained', key="lime_target", min_value=0, max_value=max_idx-1, step=1)
            with col2:
                column_num = st.number_input('Number of columns to be displayed', key="max_columns", min_value=1, value=5, max_value=10, step=1)
            st.form_submit_button("Begin Initialization", type="primary", on_click=clicked, args=["initLIME"])
        st.session_state["lime_instance"] = lime_instance
        st.session_state["column_num"] = column_num
        st.session_state["clicked"]["initLIME"] = True

    if (getSession("clicked")["initLIME"]):
        class_names = ["negative", "positive"]
        knn_model = getSession("knn_model")
        logreg_model = getSession("logreg_model")
        svm_model = getSession("svm_model")
        lime_instance = getSession("lime_instance")
        kernel_input = getSession("kernel_input")
        column_num = getSession("column_num")
        explainer = LimeTextExplainer(class_names=class_names)
        st.subheader("Global Explainer", divider='grey')
        # with st.spinner("Loading...", ):
        #     # LOGREG
        #     feature_names = tfidf.get_feature_names_out()
        #     feature_importances = logreg_model.coef_[0]
        #     sorted_indices = feature_importances.argsort()
        #     sorted_feature_names = [feature_names[idx] for idx in sorted_indices]
        #     sorted_feature_importances = feature_importances[sorted_indices]
        #     top_n = 15  # Number of top features to visualize
        #     top_feature_names = sorted_feature_names[-top_n:]
        #     top_feature_importances = sorted_feature_importances[-top_n:]

            # trace = go.Bar(
            #     x=top_feature_importances,
            #     y=top_feature_names,
            #     orientation='h',
            #     marker=dict(color='#2ecc71'),  # Use emerald-like color
            #     hoverinfo='x+y'  # Display both values on hover
            # )
            # layout = go.Layout(
            #     title='Logistic Regression Global Explanation: Coefficient Magnitudes',
            #     titlefont=dict(color='black'),
            #     xaxis=dict(
            #         title='Coefficient Magnitude',
            #         titlefont=dict(color='black'),  # Set x-axis title font color
            #         tickfont=dict(color='black')  # Set x-axis tick labels color
            #     ),
            #     yaxis=dict(
            #         title='Word',
            #         titlefont=dict(color='black'),  # Set y-axis title font color
            #         tickfont=dict(color='black'),  # Set y-axis tick labels color
            #         # autorange="reversed"
            #     ),
            #     hovermode='closest',
            #     plot_bgcolor='rgba(255, 255, 255, 1)',
            #     paper_bgcolor='rgba(255, 255, 255, 1)',
            #     title_x=0.15,  # Invert y-axis for top-down display
            # )
            # fig = go.Figure(data=[trace], layout=layout)
            # # Display the plot in Streamlit
            # st.plotly_chart(fig)

            # # KNN
            # feature_names = tfidf.get_feature_names_out()
            # neighbors = knn_model.kneighbors(train_x_vector, return_distance=False)
            # feature_occurrences = np.zeros(train_x_vector.shape[1])  # Initialize an array to store feature occurrences
            # for i, neighbor_indices in enumerate(neighbors):
            #     labels = train_y.values[neighbor_indices]
            #     for label in labels:
            #         feature_occurrences[train_x_vector[i].indices] += 1
            # sorted_indices = np.argsort(feature_occurrences)
            # sorted_feature_names = [feature_names[idx] for idx in sorted_indices]
            # sorted_feature_occurrences = feature_occurrences[sorted_indices]
            # top_n = 15  # Number of top features to visualize
            # top_feature_names = sorted_feature_names[-top_n:]
            # top_feature_occurrences = sorted_feature_occurrences[-top_n:]

            # trace = go.Bar(
            #     x=top_feature_occurrences,
            #     y=top_feature_names,
            #     orientation='h',
            #     marker=dict(color='#2ecc71'),  # Use hex code for emerald-like color
            #     hoverinfo='x+y',  # Display both x (mean distance) and y (feature name) in hover info
            # )
            # layout = go.Layout(
            #     title='KNN Global Explanation: Feature Occurrences',
            #     titlefont=dict(color='black'),
            #     xaxis=dict(
            #         title='Feature Occurrences',
            #         titlefont=dict(color='black'),  # Set x-axis title font color
            #         tickfont=dict(color='black')  # Set x-axis tick labels color
            #     ),
            #     yaxis=dict(
            #         title='Word',
            #         titlefont=dict(color='black'),  # Set y-axis title font color
            #         tickfont=dict(color='black'),  # Set y-axis tick labels color
            #         # autorange="reversed"
            #     ),
            #     hovermode='closest',
            #     plot_bgcolor='rgba(255, 255, 255, 1)',
            #     paper_bgcolor='rgba(255, 255, 255, 1)',
            #     title_x=0.27,
            # )
            # fig = go.Figure(data=[trace], layout=layout)
            # # Show plot
            # st.plotly_chart(fig)

            # # SVM
            # if kernel_input == "linear" :
            #     feature_names = tfidf.get_feature_names_out()
            #     coefficients = svm_model.coef_.toarray()[0]  # Convert the coefficients to a dense array
            #     sorted_indices = np.argsort(coefficients)
            #     sorted_feature_names = [feature_names[idx] for idx in sorted_indices]
            #     sorted_coefficients = coefficients[sorted_indices]
            #     top_n = 15  # Number of top features to visualize
            #     top_feature_names = sorted_feature_names[-top_n:]
            #     top_coefficients = sorted_coefficients[-top_n:]

            #     trace = go.Bar(
            #         x=top_coefficients,
            #         y=top_feature_names,
            #         orientation='h',
            #         marker=dict(color='#2ecc71'),  # Use hex code for emerald-like color
            #         hoverinfo='x+y',  # Display both x (mean distance) and y (feature name) in hover info
            #     )
            #     layout = go.Layout(
            #         title='SVM Global Explanation: Coefficient Magnitudes',
            #         titlefont=dict(color='black'),
            #         xaxis=dict(
            #             title='Coefficient Magnitudes',
            #             titlefont=dict(color='black'),  # Set x-axis title font color
            #             tickfont=dict(color='black')  # Set x-axis tick labels color
            #         ),
            #         yaxis=dict(
            #             title='Word',
            #             titlefont=dict(color='black'),  # Set y-axis title font color
            #             tickfont=dict(color='black'),  # Set y-axis tick labels color
            #             # autorange="reversed"
            #         ),
            #         hovermode='closest',
            #         plot_bgcolor='rgba(255, 255, 255, 1)',
            #         paper_bgcolor='rgba(255, 255, 255, 1)',
            #         title_x=0.25,
            #     )
            #     fig = go.Figure(data=[trace], layout=layout)
            #     # Show plot
            #     st.plotly_chart(fig)
        with st.spinner("Loading...", ):
            with st.expander("Show Global Explanation"):
                # Function to highlight words based on their contribution to sentiment
                def highlight_text_with_sentiment(text, tfidf_vectorizer, coefficients):
                    words = text.split()
                    highlighted_text = ""
                    for word in words:
                        if word in tfidf_vectorizer.vocabulary_:
                            index = tfidf_vectorizer.vocabulary_[word]
                            tfidf_weight = train_x_vector[0, index]  # TF-IDF weight of the word in the example text
                            coef = coefficients[index]  # Coefficient of the word in the SVM model
                            opacity = abs(coef)
                            color = f"rgba(31,119,180, {opacity})" if coef < 0 else f"rgba(255,127,14, {opacity})"  # Blue for negative sentiment, orange for positive sentiment
                            highlighted_text += f"<span style='background-color:{color}'>{word}</span> "
                        else:
                            highlighted_text += word + " "
                    return highlighted_text

                # LOGREG
                st.markdown("#### Logistic Regression Model")
                coefficients = logreg_model.coef_[0]
                idx = lime_instance
                instance = test_x.iloc[idx : idx + 1].values[0]
                highlighted_example = highlight_text_with_sentiment(instance, tfidf, coefficients)
                # HTML(highlighted_example)
                st.markdown(highlighted_example, unsafe_allow_html=True)
                st.markdown("---")
                # SVM
                st.markdown("#### SVM Model")
                coefficients = svm_model.coef_.toarray()[0]
                idx = lime_instance
                instance = test_x.iloc[idx : idx + 1].values[0]
                highlighted_example = highlight_text_with_sentiment(instance, tfidf, coefficients)
                # HTML(highlighted_example)
                # HTML(highlighted_example)
                st.markdown(highlighted_example, unsafe_allow_html=True)


        st.subheader("LIME Explainer", divider='grey')
        with st.spinner("Loading...", ):
            instance = test_x.iloc[lime_instance : lime_instance + 1].values[0]
            true_label = test_y.iloc[lime_instance]
            instance = np.array([instance])
            st.markdown("## True Prediction : " + true_label)

            # LIME for Logreg =================================================================
            def predict_proba_wrapper(texts):
                return logreg_model.predict_proba(tfidf.transform(texts))
            exp = explainer.explain_instance(instance[0], predict_proba_wrapper, num_features=column_num)
            # exp.show_in_notebook(text=True)
            html_explanation = exp.as_html()
            temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            temp_file_path.write(html_explanation.encode())
            temp_file_path.close()
            st.markdown("#### Logistic Regression Model", unsafe_allow_html=True)
            st.markdown(f'<iframe src="data:text/html;base64,{base64.b64encode(open(temp_file_path.name, "rb").read()).decode()}" height="340" width="1000"></iframe>', unsafe_allow_html=True)

            # LIME for KNN =================================================================
            def predict_proba_wrapper(texts):
                return knn_model.predict_proba(tfidf.transform(texts))
            exp = explainer.explain_instance(instance[0], predict_proba_wrapper, num_features=column_num)
            # exp.show_in_notebook(text=True)
            html_explanation = exp.as_html()
            temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            temp_file_path.write(html_explanation.encode())
            temp_file_path.close()
            st.markdown("#### KNN Model", unsafe_allow_html=True)
            st.markdown(f'<iframe src="data:text/html;base64,{base64.b64encode(open(temp_file_path.name, "rb").read()).decode()}" height="340" width="1000"></iframe>', unsafe_allow_html=True)

            # LIME for SVM =================================================================
            def predict_proba_wrapper(texts):
                return svm_model.predict_proba(tfidf.transform(texts))
            exp = explainer.explain_instance(instance[0], predict_proba_wrapper, num_features=column_num)
            # exp.show_in_notebook(text=True)
            html_explanation = exp.as_html()
            temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            temp_file_path.write(html_explanation.encode())
            temp_file_path.close()
            st.markdown("#### SVM Model", unsafe_allow_html=True)
            st.markdown(f'<iframe src="data:text/html;base64,{base64.b64encode(open(temp_file_path.name, "rb").read()).decode()}" height="340" width="1000"></iframe>', unsafe_allow_html=True)













    # train,test = train_test_split(df,test_size = 0.33,random_state = 42)
    # train_x, train_y = train['text'], train['sentiment']
    # test_x, test_y = test['text'], test['sentiment']
    # train_y.value_counts()
    # st.subheader("Data Preprocessing")
