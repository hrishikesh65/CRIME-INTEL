# ==========================================
# 🚔 Crime Intelligence System (FINAL FIXED)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import plotly.express as px

st.set_page_config(page_title="Crime Intelligence", layout="wide")
st.title("🚔 Crime Intelligence Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crime_data.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna()

    df['Hour'] = df['Date'].dt.hour
    df['Day'] = df['Date'].dt.day_name()

    df['Crime_Code'] = df['Crime_Type'].astype('category').cat.codes
    df['Area_Code'] = df['Area'].astype('category').cat.codes

    return df

df = load_data()

# Fix day order
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
df['Day'] = pd.Categorical(df['Day'], categories=day_order, ordered=True)

# -----------------------------
# 🌍 HOTSPOT MAP
# -----------------------------
st.subheader("🌍 Crime Hotspots")

sample = df.sample(min(3000, len(df)))
m = folium.Map(location=[sample['Latitude'].mean(), sample['Longitude'].mean()], zoom_start=11)
HeatMap(sample[['Latitude','Longitude']].values.tolist()).add_to(m)

folium_static(m)

# -----------------------------
# 📊 STACKED TREND
# -----------------------------
st.subheader("📊 Crime Trend by Day (%)")

trend_df = df.groupby(['Day','Crime_Type']).size().reset_index(name='Count')
trend_df['Percentage'] = trend_df.groupby('Day')['Count'].transform(lambda x: x/x.sum()*100)
trend_df = trend_df.sort_values('Day')

st.plotly_chart(px.bar(trend_df, x='Day', y='Percentage', color='Crime_Type'), use_container_width=True)

# -----------------------------
# 📊 AREA DISTRIBUTION
# -----------------------------
st.subheader("📊 Crime Distribution by Area (%)")

area_df = df.groupby(['Area','Crime_Type']).size().reset_index(name='Count')
area_df['Percentage'] = area_df.groupby('Area')['Count'].transform(lambda x: x/x.sum()*100)

st.plotly_chart(px.bar(area_df, x='Area', y='Percentage', color='Crime_Type'), use_container_width=True)

# =====================================================
# 🔍 CRIME PATTERN ANALYSIS
# =====================================================
st.header("🔍 Crime Pattern Analysis")

# 🕒 Hour pattern
hour_df = df.groupby('Hour').size().reset_index(name='Count')
st.plotly_chart(px.line(hour_df, x='Hour', y='Count', markers=True), use_container_width=True)

# 📅 Day pattern
day_pattern = df.groupby('Day').size().reset_index(name='Count').sort_values('Day')
st.plotly_chart(px.bar(day_pattern, x='Day', y='Count', color='Count',
                       color_continuous_scale='Turbo'), use_container_width=True)

# 🏙️ Area pattern
area_pattern = df.groupby('Area').size().reset_index(name='Count')
st.plotly_chart(px.bar(area_pattern, x='Area', y='Count', color='Count',
                       color_continuous_scale='Viridis'), use_container_width=True)

# Peak hour
peak_hour = hour_df.sort_values('Count', ascending=False).iloc[0]
st.success(f"🔥 Peak Crime Hour: {peak_hour['Hour']}")

# =====================================================
# 🛡️ URBAN SAFETY (FIXED)
# =====================================================
st.header("🛡️ Urban Safety Planning")

safety_df = df.groupby(['Area','Hour']).size().reset_index(name='Crime_Count')

# ✅ Balanced classification
avg = safety_df['Crime_Count'].mean()

def classify(x):
    if x > 1.2 * avg:
        return "High"
    elif x < 0.8 * avg:
        return "Low"
    else:
        return "Medium"

safety_df['Risk_Level'] = safety_df['Crime_Count'].apply(classify)

# Show all types
st.subheader("🚨 High Risk")
st.dataframe(safety_df[safety_df['Risk_Level']=="High"].head(5))

st.subheader("⚠️ Medium Risk")
st.dataframe(safety_df[safety_df['Risk_Level']=="Medium"].head(5))

st.subheader("✅ Low Risk")
st.dataframe(safety_df[safety_df['Risk_Level']=="Low"].head(5))

# -----------------------------
# 🔍 CHECK SAFETY
# -----------------------------
st.subheader("🔍 Check Area Safety")

col1, col2 = st.columns(2)

with col1:
    area_check = st.selectbox("Area", df['Area'].unique())

with col2:
    hour_check = st.slider("Hour", 0, 23, 20)

res = safety_df[
    (safety_df['Area']==area_check) &
    (safety_df['Hour']==hour_check)
]

if not res.empty:
    r = res['Risk_Level'].values[0]

    if r == "High":
        st.error(f"🚨 Avoid {area_check}")
    elif r == "Medium":
        st.warning(f"⚠️ Be cautious in {area_check}")
    else:
        st.success(f"✅ Safe in {area_check}")

# =====================================================
# 🤖 ML MODEL
# =====================================================
@st.cache_resource
def train_model(df):
    df = df.copy()
    df['Risk'] = 0

    df.loc[(df['Hour'] >= 20) | (df['Hour'] <= 2), 'Risk'] = 1
    df.loc[df['Crime_Type'].isin(['Auto Theft','Ganja Smuggling','Robbery']), 'Risk'] = 1
    df.loc[df['Area'].isin(['Chintal','Kukatpally']), 'Risk'] = 1

    X = df[['Area_Code','Hour','Crime_Code']]
    y = df['Risk']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model

model = train_model(df)

st.subheader("🤖 Predict Crime Risk")

a = st.selectbox("Area ", df['Area'].unique())
c = st.selectbox("Crime ", df['Crime_Type'].unique())
h = st.slider("Hour ", 0, 23, 22)

ac = df[df['Area']==a]['Area_Code'].iloc[0]
cc = df[df['Crime_Type']==c]['Crime_Code'].iloc[0]

if st.button("Predict"):
    pred = model.predict(pd.DataFrame([[ac,h,cc]],
              columns=['Area_Code','Hour','Crime_Code']))[0]

    if pred == 1:
        st.error("🚨 High Risk")
    else:
        st.success("✅ Low Risk")

st.markdown("---")
st.write("🚀 Crime Intelligence System")