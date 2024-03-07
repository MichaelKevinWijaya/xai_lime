import streamlit as st
import pandas as pd
import numpy as np
import lime
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from numpy.random import default_rng
# from utils import getSession

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import lime
import matplotlib.pyplot as plt
import time
import io

from session_utils import saveSession, getSession
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

# st.caption("**LIME Explanation:**  *{}*".format(shap.__version__))
st.title("LIME Explanainer")

file_csv_session = getSession("file_csv")
knn_model = getSession("knn_model")

if not file_csv_session.empty:
    st.markdown("**Dataset:** *{}*".format(getSession("uploaded_file_name")))



# if getSession("file_csv") != False:
#     st.caption("**Dataset:** *{}*".format(getSession("uploaded_file_name")))