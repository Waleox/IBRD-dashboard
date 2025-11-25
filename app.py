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


# ============================================
# CONFIGURACIÓN GENERAL
# ============================================
st.set_page_config(page_title="IBRD Dashboard", layout="wide")
st.title("📊 Dashboard de Préstamos Históricos del IBRD")


# ============================================
# CARGA DE ARCHIVO
# ============================================

uploaded_file = st.sidebar.file_uploader("Sube tu archivo JSON o Parquet", type=["json", "parquet"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)

    st.success("Dataset cargado exitosamente.")

    # =============================
    # MAPEO DE VARIABLES
    # =============================

    column_map = {
        "original_principal_amount": "Original Principal Amount (US$)",
        "disbursed_amount": "Disbursed Amount (US$)",
        "borrowers_obligation": "Borrower's Obligation (US$)",
        "interest_rate": "Interest Rate",
        "cancelled_amount": "Cancelled Amount (US$)",
        "due_to_ibrd": "Due to IBRD (US$)",
        "repaid_to_ibrd": "Repaid to IBRD (US$)",
        "loans_held": "Loans Held (US$)",
        "country": "Country",
        "region": "Region",
        "loan_type": "Loan Type",
        "loan_status": "Loan Status",
        "project_name": "Project Name",
        "board_approval_date": "Board Approval Date",
        "last_repayment_date": "Last Repayment Date",
        "end_of_period": "End of Period"
    }

    df_clean = pd.DataFrame()

    for raw_col, new_col in column_map.items():
        if raw_col in df.columns:
            df_clean[new_col] = df[raw_col]

    # Conversiones
    date_cols = ["Board Approval Date", "Last Repayment Date", "End of Period"]
    for col in date_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    st.subheader("📄 Vista previa del dataset")
    st.dataframe(df_clean.head())


    # ============================================
    # EDA
    # ============================================

    st.header("📊 Exploración de Datos (EDA)")

    if len(num_cols) > 0:
        col_to_plot = st.selectbox("Selecciona una variable numérica:", num_cols)

        st.subheader("Histograma")
        fig, ax = plt.subplots()
        sns.histplot(df_clean[col_to_plot], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Boxplot")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df_clean[col_to_plot], ax=ax2)
        st.pyplot(fig2)

        st.subheader("Mapa de correlación")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_clean[num_cols].corr(), cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)


    # ============================================
    # PCA
    # ============================================

    st.header("🔻 PCA – Reducción de Dimensionalidad")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df_clean[num_cols])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)

    df_clean["PCA1"] = pca_res[:, 0]
    df_clean["PCA2"] = pca_res[:, 1]

    fig4, ax4 = plt.subplots()
    sns.scatterplot(x="PCA1", y="PCA2", data=df_clean, s=10)
    st.pyplot(fig4)


    # ============================================
    # CLUSTERING KMEANS
    # ============================================

    st.header("🎯 Clustering – KMeans")

    k = st.slider("Número de clusters", 2, 10, 4)

    kmeans = KMeans(n_clusters=k, n_init=10)
    df_clean["Cluster"] = kmeans.fit_predict(X_scaled)

    fig5, ax5 = plt.subplots()
    sns.scatterplot(x="PCA1", y="PCA2", hue=df_clean["Cluster"].astype(str), palette="tab10", data=df_clean)
    st.pyplot(fig5)


    # ============================================
    # REGRESIÓN
    # ============================================

    st.header("📈 Regresión — Predicción del Interest Rate")

    df_reg = df_clean.dropna(subset=["Interest Rate"])

    if len(df_reg) > 0:
        Xr = df_reg.drop(columns=["Interest Rate"])
        yr = df_reg["Interest Rate"]

        num_feats = Xr.select_dtypes(include=[np.number]).columns.tolist()
        cat_feats = Xr.select_dtypes(include=["object"]).columns.tolist()

        preprocess_reg = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_feats),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats)
        ])

        model_reg = Pipeline([
            ("prep", preprocess_reg),
            ("lin", LinearRegression())
        ])

        X_train, X_test, y_train, y_test = train_test_split(Xr, yr, test_size=0.25, random_state=42)

        model_reg.fit(X_train, y_train)
        pred = model_reg.predict(X_test)

        st.write("**R²:**", r2_score(y_test, pred))
        st.write("**RMSE:**", mean_squared_error(y_test, pred, squared=False))


    # ============================================
    # CLASIFICACIÓN
    # ============================================

    st.header("🟦 Clasificación – Fully Repaid vs Others")

    df_clean["LoanStatus_bin"] = df_clean["Loan Status"].apply(lambda x: 1 if x == "Fully Repaid" else 0)

    features_clf = [
        "Original Principal Amount (US$)", "Disbursed Amount (US$)",
        "Borrower's Obligation (US$)", "Cancelled Amount (US$)",
        "Due to IBRD (US$)", "Repaid to IBRD (US$)", "Loans Held (US$)",
        "Country", "Region", "Loan Type"
    ]

    df_clf = df_clean.dropna(subset=["LoanStatus_bin"])

    Xc = df_clf[features_clf]
    yc = df_clf["LoanStatus_bin"]

    num_feats_c = Xc.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats_c = Xc.select_dtypes(include=["object"]).columns.tolist()

    preprocess_clf = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_feats_c),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats_c)
    ])

    clf = Pipeline([
        ("prep", preprocess_clf),
        ("log", LogisticRegression(max_iter=300))
    ])

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(Xc, yc, test_size=0.25, random_state=42)

    clf.fit(X_train_c, y_train_c)
    pred_c = clf.predict(X_test_c)

    st.write("**Accuracy:**", accuracy_score(y_test_c, pred_c))
    st.text(classification_report(y_test_c, pred_c))

else:
    st.info("Por favor sube un archivo para comenzar.")
