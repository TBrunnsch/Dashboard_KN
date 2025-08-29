import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet



# --- Streamlit page config ---
st.set_page_config(
    page_title="Knecht Reisen Dashboard",
    page_icon="",
    layout="wide"
)

st.title("Knecht Reisen Dashboard")
st.markdown("Buchungszahlen, Umsatz, Destinationen.")

# --- Load CSV ---
@st.cache_data
def load_data(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["BookingDate", "TravelDate"])
    df['BookingLeadTime'] = (df['TravelDate'] - df['BookingDate']).dt.days
    return df

data = load_data("tour_operator_bookings.csv")

# --- Sidebar filters ---
st.sidebar.header("Filter Bookings")
destinations = data['Destination'].unique().tolist()
status_options = data['Status'].unique().tolist()

selected_destinations = st.sidebar.multiselect("Destination", destinations, default=destinations)
selected_status = st.sidebar.multiselect("Buchungsstatus", status_options, default=status_options)
date_range = st.sidebar.date_input(
    "Buchungsdaten",
    [data['BookingDate'].min(), data['BookingDate'].max()]
)

# Filter data
filtered_data = data[
    (data['Destination'].isin(selected_destinations)) &
    (data['Status'].isin(selected_status)) &
    (data['BookingDate'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# --- Key Metrics ---
st.subheader("📊 KPIs")
col1, col2, col3, col4 = st.columns(4)

total_bookings = len(filtered_data)
total_passengers = filtered_data['NumGuests'].sum()
total_PriceEUR = filtered_data['PriceEUR'].sum()
cancelled_bookings = len(filtered_data[filtered_data['Status']=="Cancelled"])

col1.metric("Anzahl Buchungen", total_bookings)
col2.metric("Anzahl Passagiere", total_passengers)
col3.metric("Gesamtumsatz (€)", f"{total_PriceEUR:,.2f}")
col4.metric("Stornierte Buchungen", cancelled_bookings)

# --- Destination color map (consistent colors across charts) ---
destination_colors = px.colors.qualitative.Set2
color_map = {dest: destination_colors[i % len(destination_colors)] for i, dest in enumerate(destinations)}

# --- Charts ---
st.subheader("📈 Zeitanalyse")
booking_over_time = filtered_data.groupby("BookingDate").size().reset_index(name="Bookings")
fig1 = px.line(booking_over_time, x="BookingDate", y="Bookings", title="Buchungen über die Zeit")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🌴 Destinationsanalyse")
dest_chart = filtered_data.groupby("Destination")['NumGuests'].sum().reset_index()
fig2 = px.bar(
    dest_chart, x="Destination", y="NumGuests",
    title="Gäste pro Destination",
    color="Destination", color_discrete_map=color_map
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("💰 Umsatz nach Destination")
PriceEUR_chart = filtered_data.groupby("Destination")['PriceEUR'].sum().reset_index()
fig3 = px.pie(
    PriceEUR_chart, names="Destination", values="PriceEUR",
    title="Umsatzanteil pro Destination",
    color="Destination", color_discrete_map=color_map
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("📏 Durchschnittlicher Umsatz pro Buchung (Destination)")
avg_rev_chart = (
    filtered_data.groupby("Destination")['PriceEUR']
    .mean()
    .reset_index()
    .rename(columns={"PriceEUR": "AvgRevenue"})
)
fig4 = px.bar(
    avg_rev_chart, x="Destination", y="AvgRevenue",
    title="Durchschnittlicher Umsatz pro Buchung",
    color="Destination", color_discrete_map=color_map
)
fig4.update_yaxes(title="Ø Umsatz (€)")
st.plotly_chart(fig4, use_container_width=True)

st.subheader("📆 Umsatzentwicklung nach Destination (monatliches Aggregat)")

# Aggregate revenue per month & destination
monthly_rev = (
    filtered_data
    .groupby([pd.Grouper(key="BookingDate", freq="M"), "Destination"])["PriceEUR"]
    .sum()
    .reset_index()
)

# Build stacked area chart
fig5 = px.area(
    monthly_rev,
    x="BookingDate", y="PriceEUR",
    color="Destination", color_discrete_map=color_map,
    title="Monatlicher Umsatz nach Destination"
)

fig5.update_layout(
    yaxis_title="Umsatz (€)",
    xaxis_title="Monat",
    legend_title="Destination",
    hovermode="x unified"
)

st.plotly_chart(fig5, use_container_width=True)

# --- Detailed Table ---
st.subheader("📋 Buchungsdetails")
st.dataframe(filtered_data.sort_values("BookingDate", ascending=False))

# --- ML Prediction ---
st.subheader("🤖 Umsatzvorhersage für neue Buchung")
from sklearn.preprocessing import OneHotEncoder

# Prepare training data
X = data[['Destination','NumGuests','BookingLeadTime']]
y = data['PriceEUR']
X = pd.get_dummies(X, columns=['Destination'], drop_first=True)
feature_columns = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with st.form("prediction_form"):
    new_dest = st.selectbox("Destination", destinations)
    new_guests = st.number_input("Anzahl Passagiere", min_value=1, max_value=20, value=2)
    new_leadtime = st.number_input("Vorlaufzeit (Tage)", min_value=0, max_value=365, value=30)
    submitted = st.form_submit_button("Vorhersage generieren")

    if submitted:
        # Build new input
        new_X = pd.DataFrame({'NumGuests':[new_guests],'BookingLeadTime':[new_leadtime]})
        # Add missing one-hot columns
        for col in feature_columns:
            if col.startswith('Destination_'):
                dest_name = col.split('_',1)[1]
                new_X[col] = 1 if new_dest == dest_name else 0
        # Reorder columns
        new_X = new_X[feature_columns]
        # Predict
        predicted_price = model.predict(new_X)[0]
        st.success(f"Vorhergesagter Umsatz: € {predicted_price:,.2f}")



st.subheader("📆 Prognose Gesamtumsatz über Zeit (Prophet)")

# Prepare data for Prophet
daily_data = filtered_data.groupby("BookingDate")['PriceEUR'].sum().reset_index()
daily_data = daily_data.rename(columns={"BookingDate": "ds", "PriceEUR": "y"})

# Train Prophet model
model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
model.fit(daily_data)

# Forecast next 30 days
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Plot
fig = px.line(forecast, x='ds', y='yhat', title="Historischer & prognostizierter Umsatz")
fig.add_scatter(x=daily_data['ds'], y=daily_data['y'], mode='lines', name='Historisch')
fig.update_layout(
    yaxis_title="Umsatz (€)",
    xaxis_title="Datum",
    legend=dict(title="Legende")
)
st.plotly_chart(fig, use_container_width=True)