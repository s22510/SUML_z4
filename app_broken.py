# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)
import os

import streamlit as st
import pickle
from datetime import datetime
import pandas as pd

startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = 'model.sv'
model = pickle.load(open(filename, 'rb'))
# otwieramy wcześniej wytrenowany model

embarked_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}

sex_d = {0: "Kobieta", 1: "Mezczyzna"}
pclass_d = {0: "Klasa Pierwsza", 1: "Klasa Druga", 2: "Klasa Trzecia"}
filename2 = 'DSP_1.csv'
base_data = pd.read_csv(filename2)

def main():
    st.set_page_config(page_title="Zadanie 4 - Piotr Trzos")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://www.gov.pl/photo/format/0443d82a-dd3a-4cd7-affc-c2a4f9dedbe8/resolution/1920x810")

    with overview:
        st.title("Zadanie 4 - Piotr Trzos")

    with left:
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        embarked_radio = st.radio("Port zaokrętowania", list(embarked_d.keys()), index=2,
                                  format_func=lambda x: embarked_d[x])
        pclass_radio = st.radio("Klasa", list(pclass_d.keys()), index=0,
                                  format_func=lambda x: pclass_d[x])

    with right:
        age_slider = st.slider("Wiek",
                               value=1,
                               min_value=int(base_data['Age'].min()),
                               max_value=int(base_data['Age'].max()))

        sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera",
                                 min_value=base_data['SibSp'].min(),
                                 max_value=base_data['SibSp'].max())

        parch_slider = st.slider("Liczba rodziców i/lub dzieci",
                                 min_value=base_data['Parch'].min(),
                                 max_value=base_data['Parch'].max())

        fare_slider = st.slider("Cena biletu",
                                min_value=int(base_data['Fare'].min()),
                                max_value=int(base_data['Fare'].max()))

    data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba przeżyłaby katastrofę?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()
