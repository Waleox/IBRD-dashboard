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

DATA_URL = "IBRD_Clean.parquet"   # TU ARCHIVO FINAL

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_URL)
    return df

df = load_data()
st.success("Dataset cargado exitosamente.")


# ==============
# ================================
# COLUMNAS NUMÉRICAS
# ================================
num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# ================================
# EDA
# ================================
st.header("📊 Exploración de Datos")

col_to_plot = st.selectbox("Selecciona una variable numérica:", num_cols)

# Histograma
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df_clean[col_to_plot], kde=True, ax=ax)
st.pyplot(fig)

# Boxplot
fig2, ax2 = plt.subplots(figsize=(6, 3))
sns.boxplot(x=df_clean[col_to_plot], ax=ax2)
st.pyplot(fig2)

# Heatmap
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(df_clean[num_cols].corr(), cmap="coolwarm", annot=False, ax=ax3)
st.pyplot(fig3)

# ================================
# PCA
# ================================
st.header("🔻 PCA (2 Componentes)")

imputer = SimpleImputer(strategy="median")
scaled = StandardScaler().fit_transform(imputer.fit_transform(df_clean[num_cols]))

pca = PCA(n_components=2)
pca_res = pca.fit_transform(scaled)

df_clean["PCA1"] = pca_res[:, 0]
df_clean["PCA2"] = pca_res[:, 1]

fig4, ax4 = plt.subplots(figsize=(7, 5))
sns.scatterplot(x="PCA1", y="PCA2", data=df_clean, s=8)
st.pyplot(fig4)

# ================================
# KMEANS
# ================================
st.header("🎯 Clustering (KMeans)")

k = st.slider("Número de clusters", 2, 10, 4)

kmeans = KMeans(n_clusters=k, n_init=10)
df_clean["Cluster"] = kmeans.fit_predict(scaled)

fig5, ax5 = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue=df_clean["Cluster"].astype(str),
    palette="tab10",
    data=df_clean,
    s=10
)
st.pyplot(fig5)

# ================================
# REGRESIÓN
# ================================
st.header("📈 Regresión — Interest Rate")

df_reg = df_clean.dropna(subset=["Interest Rate"])
Xr = df_reg.drop(columns=["Interest Rate"])
yr = df_reg["Interest Rate"]

num_feats = Xr.select_dtypes(include=[np.number]).columns.tolist()
cat_feats = Xr.select_dtypes(include=["object"]).columns.tolist()

preprocess_reg = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_feats),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_feats),
])

model_reg = Pipeline([
    ("prep", preprocess_reg),
    ("lin", LinearRegression())
])

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    Xr, yr, test_size=0.25, random_state=42
)

model_reg.fit(X_train_r, y_train_r)
pred_r = model_reg.predict(X_test_r)

st.write("**R²:**", r2_score(y_test_r, pred_r))
st.write("**RMSE:**", mean_squared_error(y_test_r, pred_r, squared=False))

# ================================
# CLASIFICACIÓN
# ================================
st.header("🔵 Clasificación — Fully Repaid vs Others")

df_clean["LoanStatus_bin"] = df_clean["Loan Status"].apply(
    lambda x: 1 if x == "Fully Repaid" else 0
)

features_c = [
    "Original Principal Amount (US$)", "Disbursed Amount (US$)",
    "Borrower's Obligation (US$)", "Cancelled Amount (US$)",
    "Due to IBRD (US$)", "Repaid to IBRD (US$)", "Loans Held (US$)",
    "Region", "Country", "Loan Type"
]

Xc = df_clean[features_c]
yc = df_clean["LoanStatus_bin"]

num_feats_c = Xc.select_dtypes(include=[np.number]).columns.tolist()
cat_feats_c = Xc.select_dtypes(include=["object"]).columns.tolist()

preprocess_clf = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_feats_c),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_feats_c),
])

clf = Pipeline([
    ("prep", preprocess_clf),
    ("log", LogisticRegression(max_iter=400))
])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    Xc, yc, test_size=0.25, random_state=42
)

clf.fit(X_train_c, y_train_c)
pred_c = clf.predict(X_test_c)

st.write("**Accuracy:**", accuracy_score(y_test_c, pred_c))
st.text(classification_report(y_test_c, pred_c))
