import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="IBRD Dashboard", layout="wide")
st.title("📊 Dashboard de Préstamos Históricos del IBRD")

# ============================================================
# CARGA AUTOMÁTICA DE TU DATASET FINAL
# ============================================================

DATA_URL = "IBRD_clean.parquet"   # TU ARCHIVO FINAL

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_URL)
    return df

df = load_data()
st.success("Dataset cargado exitosamente.")


# ==============
