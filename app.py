import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# --- Page setup ---
st.set_page_config(page_title="Energy Optimization Dashboard", layout="wide")

st.markdown("""
    <style>
    body {background-color: #0e1117; color: #FAFAFA;}
    div[data-testid="stMetricValue"] {color: #00FFAA; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("example_scada_energy_data.csv", parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["efficiency"] = np.where(df["production_units"] > 0,
                                df["energy_kWh"] / df["production_units"], np.nan)
    return df

df = load_data()

# --- Header ---
st.title("⚙️ Enerji Optimizasyonu Dashboard")
st.caption("SCADA verisine dayalı ML analizi, tahmin ve tasarruf önerileri")

# --- KPIs ---
total_energy = df["energy_kWh"].sum()
total_production = df["production_units"].sum()
efficiency_mean = df["efficiency"].mean()
working_hours = df[df["machine_state"] == 1].shape[0] / 4

col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Enerji (kWh)", f"{total_energy:,.0f}")
col2.metric("Toplam Üretim (birim)", f"{total_production:,.0f}")
col3.metric("Ortalama Verimlilik (kWh/ürün)", f"{efficiency_mean:.2f}")
col4.metric("Toplam Çalışma Süresi (saat)", f"{working_hours:.0f}")

# --- Energy trend ---
st.subheader("🔌 Günlük Enerji Tüketimi")
daily = df.groupby("date")["energy_kWh"].sum().reset_index()
fig_energy = px.line(daily, x="date", y="energy_kWh", title="Günlük Enerji Kullanımı (kWh)",
                     markers=True, template="plotly_dark", line_shape="spline")
st.plotly_chart(fig_energy, use_container_width=True)

# --- ML Prediction (basit Linear Regression) ---
st.subheader("🤖 Enerji Tüketimi Tahmini (ML Modeli)")
X = df[["hour"]]
y = df["energy_kWh"]
model = LinearRegression().fit(X, y)
df["predicted"] = model.predict(X)

fig_pred = px.line(df.iloc[:96], x="timestamp", y=["energy_kWh", "predicted"],
                   labels={"value": "kWh"}, title="Gerçek vs Tahmin (ilk gün örnek)",
                   template="plotly_dark")
st.plotly_chart(fig_pred, use_container_width=True)

# --- Anomaly detection (Z-score) ---
st.subheader("⚠️ Anomali Tespiti")
df["zscore"] = (df["energy_kWh"] - df["energy_kWh"].mean()) / df["energy_kWh"].std()
df["anomaly"] = np.where(abs(df["zscore"]) > 2.5, 1, 0)
anom_count = df["anomaly"].sum()
st.metric("Tespit Edilen Anomali Sayısı", anom_count)

fig_anom = px.scatter(df, x="timestamp", y="energy_kWh", color=df["anomaly"].astype(str),
                      title="Enerji Kullanımı ve Anomaliler", template="plotly_dark")
st.plotly_chart(fig_anom, use_container_width=True)

# --- Efficiency vs Temperature ---
st.subheader("🌡️ Verimlilik - Sıcaklık İlişkisi")
fig_eff = px.scatter(df, x="temperature", y="efficiency",
                     title="Sıcaklık ile Enerji Verimliliği", trendline="ols",
                     template="plotly_dark")
st.plotly_chart(fig_eff, use_container_width=True)

# --- Recommendations ---
st.subheader("💡 Optimizasyon Önerileri")
recs = []
if efficiency_mean > 6:
    recs.append("⚙️ Üretim hatlarında yüksek enerji tüketimi var — bakım önerilir.")
if anom_count > 10:
    recs.append("🚨 Çok sayıda anomali tespit edildi — makine duruş saatleri kontrol edilmeli.")
if df["machine_state"].mean() < 0.6:
    recs.append("📉 Hat verimliliği düşük — gereksiz duruşları azaltın.")
if len(recs) == 0:
    recs.append("✅ Sistem verimli çalışıyor, belirgin bir tasarruf fırsatı yok.")

for r in recs:
    st.write(r)

st.caption("© 2025 Enerji Optimizasyon Demo — SCADA + ML + Dashboard")



# --- Tasarruf Hesaplama Modülü ---
st.subheader("💶 Yıllık Enerji Maliyeti ve Tasarruf Hesaplama")

col_in1, col_in2 = st.columns(2)
with col_in1:
    energy_price = st.number_input("Elektrik Birim Fiyatı (€ / kWh)", min_value=0.1, max_value=1.0, value=0.30, step=0.01)
with col_in2:
    saving_target = st.slider("Hedef Tasarruf (%)", min_value=1, max_value=30, value=10, step=1)

# Günlük ortalama tüketim
daily_avg_energy = df.groupby("date")["energy_kWh"].sum().mean()
yearly_energy = daily_avg_energy * 365
yearly_cost = yearly_energy * energy_price
saving_euro = yearly_cost * (saving_target / 100)
new_cost = yearly_cost - saving_euro

col1, col2, col3 = st.columns(3)
col1.metric("📆 Yıllık Enerji Maliyeti", f"{yearly_cost:,.0f} €")
col2.metric("💰 Yıllık Tasarruf Potansiyeli", f"{saving_euro:,.0f} €", f"-{saving_target}%")
col3.metric("🎯 Yeni Hedef Maliyet", f"{new_cost:,.0f} €")

st.caption("💡 İpucu: %10 tasarruf tipik bir fabrikanın yıllık maliyetinde 30–50 bin € avantaj sağlar.")
