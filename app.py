import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="IBRD Dashboard", layout="wide")

st.title("📊 Dashboard de Préstamos Históricos del IBRD")

# ---------------------------------------
# 1. Cargar archivo
# ---------------------------------------
st.sidebar.header("Sube tu archivo JSON o Parquet")
uploaded_file = st.sidebar.file_uploader("Carga un archivo", type=["json", "parquet"])

@st.cache_data
def load_data(file):
    if file.name.endswith(".json"):
        return pd.read_json(file)
    elif file.name.endswith(".parquet"):
        return pd.read_parquet(file)

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("📁 Dataset cargado exitosamente.")
else:
    st.stop()

# ---------------------------------------
# 2. Limpieza mínima (fechas)
# ---------------------------------------
date_columns = [
    "Board Approval Date",
    "Last Repayment Date",
    "End of Period",
]

for c in date_columns:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# ---------------------------------------
# 3. KPIs
# ---------------------------------------
st.subheader("📌 Indicadores Clave")

col1, col2, col3 = st.columns(3)
col1.metric("Total Desembolsado", f"${df['Disbursed Amount (US$)'].sum():,.0f}")
col2.metric("Principal Original", f"${df['Original Principal Amount (US$)'].sum():,.0f}")

if "Interest Rate" in df.columns:
    col3.metric("Tasa Promedio (%)", f"{df['Interest Rate'].mean():.2f}")
else:
    col3.metric("Tasa Promedio (%)", "N/A")

# ---------------------------------------
# 4. Gráfico: Desembolsos por Región
# ---------------------------------------
if "Region" in df.columns:
    st.subheader("🌍 Desembolsos por Región")
    fig = px.bar(
        df.groupby("Region")["Disbursed Amount (US$)"].sum().reset_index(),
        x="Region",
        y="Disbursed Amount (US$)",
        title="Total Desembolsado por Región",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------
# 5. Heatmap de Correlación
# ---------------------------------------
st.subheader("📈 Matriz de Correlación")

num_cols = df.select_dtypes(include=[np.number])

if num_cols.shape[1] > 0:
    fig = px.imshow(num_cols.corr(), color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No hay columnas numéricas suficientes para mostrar la correlación.")

# ---------------------------------------
# 6. Tabla final
# ---------------------------------------
st.subheader("📄 Vista del Dataset")
st.dataframe(df)
