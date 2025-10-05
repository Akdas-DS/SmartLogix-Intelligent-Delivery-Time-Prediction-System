# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="centered")
st.title("Amazon Delivery Time Predictor 🚚")

# --------------------------
# Helper functions
# --------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def eval_reg(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def encode_category(x):
    return abs(hash(x)) % 100

def init_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        order_id TEXT,
        distance_km REAL,
        order_hour INTEGER,
        order_dayofweek INTEGER,
        agent_age INTEGER,
        agent_rating REAL,
        weather TEXT,
        traffic TEXT,
        vehicle TEXT,
        area TEXT,
        category TEXT,
        predicted_delivery_time REAL
    );""")
    conn.commit()

def store_prediction(conn, row):
    conn.execute("""
    INSERT INTO predictions (
        timestamp, order_id, distance_km, order_hour, order_dayofweek,
        agent_age, agent_rating, weather, traffic, vehicle, area, category, predicted_delivery_time
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
    conn.commit()

# --------------------------
# Load and preprocess dataset
# --------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "amazon_delivery_1.csv")  # CSV must be in same folder as app.py
df = pd.read_csv(DATA_PATH)

if 'Order_ID' in df.columns:
    df = df.drop_duplicates(subset=['Order_ID'])

num_cols = df.select_dtypes(include=['int','float']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode().iloc[0])

if {'Store_Latitude','Store_Longitude','Drop_Latitude','Drop_Longitude'}.issubset(df.columns):
    df['distance_km'] = haversine(
        df['Store_Latitude'],
        df['Store_Longitude'],
        df['Drop_Latitude'],
        df['Drop_Longitude']
    )

if 'Order_Date' in df.columns and 'Order_Time' in df.columns:
    df['order_datetime'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Order_Time'].astype(str),
                                          errors='coerce')
elif 'Order_Date' in df.columns:
    df['order_datetime'] = pd.to_datetime(df['Order_Date'], errors='coerce')
else:
    df['order_datetime'] = None

df['order_hour'] = df['order_datetime'].dt.hour.fillna(0).astype(int)
df['order_dayofweek'] = df['order_datetime'].dt.dayofweek.fillna(0).astype(int)
df['order_weekend'] = df['order_dayofweek'].isin([5,6]).astype(int)

cat_features = ['Weather','Traffic','Vehicle','Area','Category']
for c in cat_features:
    if c in df.columns:
        df[c+'_catcode'] = df[c].astype(str).apply(encode_category)

feature_cols = ['distance_km','order_hour','order_dayofweek','order_weekend','Agent_Age','Agent_Rating'] + \
               [c+'_catcode' for c in cat_features if c+'_catcode' in df.columns]
target = 'Delivery_Time'
df_model = df.dropna(subset=feature_cols + [target])
X = df_model[feature_cols]
y = df_model[target].astype(float)

# --------------------------
# Train model
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
model.fit(X_train, y_train)

st.subheader("Model Performance on Test Data")
y_pred = model.predict(X_test)
metrics = eval_reg(y_test, y_pred)
st.write(metrics)

# --------------------------
# Visualizations (all original)
# --------------------------
st.subheader("Visualizations 📊")

# 1) Distribution of Delivery Time
st.markdown("**Distribution of Delivery Time**")
fig1, ax1 = plt.subplots()
sns.histplot(df['Delivery_Time'], bins=30, kde=True, ax=ax1)
ax1.set_xlabel("Delivery Time (hours)")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

# 2) Boxplot Delivery Time vs Traffic
if 'Traffic' in df.columns:
    st.markdown("**Delivery Time vs Traffic Level**")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Traffic', y='Delivery_Time', data=df, order=df['Traffic'].value_counts().index, ax=ax2)
    ax2.set_xlabel("Traffic")
    ax2.set_ylabel("Delivery Time (hours)")
    st.pyplot(fig2)

# 3) Scatter Distance vs Delivery Time
if 'distance_km' in df.columns:
    st.markdown("**Distance vs Delivery Time**")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x='distance_km', y='Delivery_Time', hue='Traffic' if 'Traffic' in df.columns else None,
                    data=df, alpha=0.6, ax=ax3)
    ax3.set_xlabel("Distance (km)")
    ax3.set_ylabel("Delivery Time (hours)")
    st.pyplot(fig3)

# 4) Average Delivery Time by Day of Week
if 'order_dayofweek' in df.columns:
    st.markdown("**Average Delivery Time by Day of Week**")
    avg_by_day = df.groupby('order_dayofweek')['Delivery_Time'].mean()
    fig4, ax4 = plt.subplots()
    sns.barplot(x=avg_by_day.index, y=avg_by_day.values, palette="viridis", ax=ax4)
    ax4.set_xlabel("Day of Week (0=Mon, 6=Sun)")
    ax4.set_ylabel("Avg Delivery Time (hours)")
    st.pyplot(fig4)

# --------------------------
# User input form
# --------------------------
st.subheader("Predict Delivery Time 🚀")
with st.form(key='predict_form'):
    order_id = st.text_input("Order ID", "")
    col1, col2 = st.columns(2)
    with col1:
        store_lat = st.number_input("Store Latitude", format="%.6f")
        store_lon = st.number_input("Store Longitude", format="%.6f")
        agent_age = st.number_input("Agent Age", min_value=16, max_value=80, value=30)
        agent_rating = st.number_input("Agent Rating (0-5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
    with col2:
        drop_lat = st.number_input("Drop Latitude", format="%.6f")
        drop_lon = st.number_input("Drop Longitude", format="%.6f")
        weather = st.selectbox("Weather", ["Clear", "Rain", "Storm", "Cloudy", "Fog"])
        traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
        vehicle = st.selectbox("Vehicle", ["Bike", "Scooter", "Car", "Van"])

    area = st.selectbox("Area", ["Urban", "Metropolitan", "Rural"])
    category = st.text_input("Category (e.g., Food, Grocery, Electronics)", "Food")
    submitted = st.form_submit_button("Predict")

if submitted:
    distance_km = haversine(store_lat, store_lon, drop_lat, drop_lon)
    now = datetime.now()
    order_hour = now.hour
    order_dayofweek = now.weekday()
    order_weekend = 1 if order_dayofweek in [5,6] else 0

    weather_code = encode_category(weather)
    traffic_code = encode_category(traffic)
    vehicle_code = encode_category(vehicle)
    area_code = encode_category(area)
    category_code = encode_category(category)

    X_input = pd.DataFrame([{
        'distance_km': distance_km,
        'order_hour': order_hour,
        'order_dayofweek': order_dayofweek,
        'order_weekend': order_weekend,
        'Agent_Age': agent_age,
        'Agent_Rating': agent_rating,
        'Weather_catcode': weather_code,
        'Traffic_catcode': traffic_code,
        'Vehicle_catcode': vehicle_code,
        'Area_catcode': area_code,
        'Category_catcode': category_code
    }])

    X_input_aligned = X_input.reindex(columns=model.feature_names_in_, fill_value=0)
    predicted = model.predict(X_input_aligned)[0]
    st.success(f"Estimated delivery time: {predicted:.2f} hours")

    conn = sqlite3.connect("predictions.db")
    init_db(conn)
    row = (datetime.now().isoformat(), order_id, distance_km, order_hour, order_dayofweek,
           agent_age, agent_rating, weather, traffic, vehicle, area, category, float(predicted))
    store_prediction(conn, row)
    conn.close()

# --------------------------
# Show recent predictions
# --------------------------
if st.checkbox("Show recent predictions"):
    conn = sqlite3.connect("predictions.db")
    df_predictions = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 50", conn)
    st.dataframe(df_predictions)
    conn.close()
