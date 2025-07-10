import streamlit as st
import joblib

vectorization = joblib.load('vectorizer.jb')
model = joblib.load('RFC_model.jb')

st.title("Fake News Detection App")
st.write("Enter the news text below to check if it's real or fake:")

news_input = st.text_area("News Text","")

if st.button("Check News"):
  if news_input.strip():
    transform_input = vectorization.transform([news_input])
    prediction = model.predict(transform_input)

    if prediction[0] == 1:
      st.success("The news is REAL!")
    else:
      st.error("The news is FAKE!")
  else:
    st.warning("Please enter some text to check.")

