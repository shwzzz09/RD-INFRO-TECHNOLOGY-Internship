import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv("creditcard_cleaned.csv")

# split
X = df.drop("Class", axis=1)
y = df["Class"]

# train model
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("Fraud Detection App")

amount = st.number_input("Enter Transaction Amount")

if st.button("Predict"):
    sample = X[0].reshape(1, -1)
    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("Fraud Transaction Detected")
    else:
        st.success("Normal Transaction")

