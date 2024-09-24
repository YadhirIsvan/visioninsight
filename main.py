import streamlit as st
import zipfile
import pandas as pd
import altair as alt
import joblib

image_path = "https://firebasestorage.googleapis.com/v0/b/habacuc-javascript.appspot.com/o/images%2FHS.png?alt=media&token=4e9389a4-52a1-4acc-b558-49a8763b2206"

st.image(image_path, width=50)

st.markdown("""
    # VisionInsight
    The objective of this project was to develop a machine learning model capable of predicting whether a person has diabetes based on a set of medical variables, using a dataset from the National Institute of Diabetes and Digestive and Kidney Diseases
""")

st.divider()

st.markdown("""
    ## Dataframe
""")

# Path
zip_file_path = "archive2.zip"

# FIlename
csv_file_name = "diabetes.csv"

# Read csv
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    with zip_ref.open(csv_file_name) as csv_file:
        df = pd.read_csv(csv_file)

st.dataframe(df)

st.divider()

st.markdown("""
    ## Datacard
    | Column | Description |
    |--------|------------ |
    | Pregnancies | Number of pregnancies |
    | Glucose | Glucose level in blood |
    | BloodPressure | Blood pressure measurement |
    | SkinThickness | Thickness skin |
    | Insulin | Insulin level in blood |
    | BMI | Body mass index |
    | DiabetesPedigreeFunction | Diabetes percentage |
    | Age | Age |
    | Outcome | Final result, where 1 = Yes and 0 = No |
""")

st.divider()

st.markdown("""
    ## Overview
""")

col1, col2, col3 = st.columns(3)
col1.metric(label='Records', value='768', delta='Patients', delta_color='off')
col2.metric(label='Patients', value='34.9%', delta='Diabetics', delta_color='inverse')
col3.metric(label='Patients', value='65.1%', delta='No diabetics', delta_color='normal')

st.divider()

col4, col5 = st.columns(2)
with col4:
    st.header("Select a chart")
    chart_type = st.selectbox(
        "Available charts",
        ("Glucose distribution", "Heatmap correlations", "People with higher glucose", "People with diabetes and high/low glucose", "Outliers overview")
    )
with col5:
    st.header("Data visualization")
    
    if chart_type == "Glucose distribution":
        glucose_distribution = alt.Chart(df).transform_density(
            'Glucose',
            as_=['Glucose', 'density']
        ).mark_area().encode(
            x='Glucose:Q',
            y='density:Q'
        ).properties(
            title='Glucose distribution'
        ).interactive()

        st.altair_chart(glucose_distribution)
    
    elif chart_type == "Heatmap correlations":
        corr = df.corr().reset_index().melt('index')
        heatmap = alt.Chart(corr).mark_rect().encode(
            x='index:O',
            y='variable:O',
            color='value:Q'
        ).properties(
            title='Heatmap Correlations'
        )

        st.altair_chart(heatmap)
    
    elif chart_type == "People with higher glucose":
        quantile = df[df['Glucose'] > df['Glucose'].quantile(0.75)]
        higher_glucose = alt.Chart(quantile).mark_boxplot().encode(
            x='Glucose:Q',
            y='Outcome:O'
        ).properties(
            title='People with higher glucose'
        ).interactive()

        st.altair_chart(higher_glucose)

    elif chart_type == "People with diabetes and high/low glucose":
        diabetes_glucose = alt.Chart(df).transform_density(
            'Glucose',
            groupby=['Outcome'],
            as_=['Glucose', 'density']
        ).mark_area().encode(
            x='Glucose:Q',
            y='density:Q',
            color='Outcome:N'
        ).properties(
            title="People with diabetes and high/low glucose"
        )

        st.altair_chart(diabetes_glucose)

    elif chart_type == "Outliers overview":
        columns = ['Age','BMI','DiabetesPedigreeFunction','Glucose','Insulin','BloodPressure','SkinThickness']

        plots = [alt.Chart(df).mark_boxplot().encode(
            x=alt.X(col, title=col)
        ).properties(
            title=f"Boxplot for {col}"
        ) for col in columns]
        
        concat_plots = alt.vconcat(*plots)
        st.altair_chart(concat_plots, use_container_width=True)

st.divider()

st.markdown("""
    ## Model results
    During the model evaluation phase, **Random Forest** emerged as the best-performing model, achieving an F1-Score of 0.7234 on the test set, indicating a balanced performance in predicting both positive and negative cases of diabetes.
""")

st.divider()

st.markdown("""
    ## Random Forest - Confusion Matrix
""")

image_path2 = "https://firebasestorage.googleapis.com/v0/b/habacuc-javascript.appspot.com/o/images%2FScreenshot%20from%202024-09-23%2017-29-08.png?alt=media&token=2e2dc6bd-fd00-4c00-94c2-650a2fdc2da8"
st.image(image_path2, width=400)

st.divider()

st.markdown("""
    ## Diabetes Prediction Model
""")

model = joblib.load('trained_model.pkl')

pregnancies = st.number_input('Pregnancies', 0, 17, 3)
glucose = st.slider('Glucose', 44, 199, 121)
blood_pressure = st.slider('Blood Pressure', 24, 122, 72)
skin_thickness = st.slider('Skin Thickness', 7, 99, 29)
insulin = st.slider('Insulin', 14, 846, 146)
bmi = st.slider('Body Mass Index (BMI)', 18, 67, 32)
diabetes_pedigree = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.47)
age = st.slider('Age', 21, 81, 33)
binarized = 0
if pregnancies > 0:
    binarized = 1

if st.button('Predict'):
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age, binarized]]
    prediction = model.predict(input_data)
    st.write(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'No diabetic'}")

st.divider()

st.markdown("""
    ## Conclusions
    The results of the project demonstrate that it is indeed possible to predict whether a patient has diabetes based on diagnostic variables. While the model achieved a decent F1-Score, further improvement could be pursued by experimenting with advanced techniques such as hyperparameter tuning, feature engineering, or exploring ensemble methods to combine the strengths of multiple algorithms.
""")

st.divider()

st.markdown("""
    ## Author
    * José Habacuc Soto Hernández - SWE Student
        - GitHub: https://github.com/habacucsoto
        - Portfolio: https://habacuc.dev
""")

st.divider()

st.markdown("""
    ## References
    - UCI Machine Learning & Collaborator. (n.d.). Pima Indians Diabetes Database. Kaggle. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    - Dhaliwal, S. K. (2024, 28 de febrero). Bajo nivel de azúcar en la sangre. MedlinePlus. https://medlineplus.gov/spanish/ency/article/000386.htm#:~:text=El%20bajo%20nivel%20de%20az%C3%BAcar%20en%20la%20sangre%20grave%20es,denomina%20shock%20insul%C3%ADnico%20o%20hipogluc%C3%A9mico.
    - National Heart, Lung, and Blood Institute. (2022, 24 de junio). Presión arterial baja. https://www.nhlbi.nih.gov/es/salud/presion-arterial-baja
    - Zhou, X. (2023, 28 de enero). Skewness. Rankia. https://www.rankia.com/diccionario/fondos-inversion/skewness
    - Kenton, W. (2024, 31 de julio). Kurtosis: Definition, Types, and Importance. Investopedia. https://www.investopedia.com/terms/k/kurtosis.asp
    - Vega-Altair Developers. (2016–2024). Vega-Altair: Declarative Visualization in Python. https://altair-viz.github.io/
    - Scikit-learn Developers. (2007–2024). confusion_matrix. Scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    - Feregrino. (2019, 3 de junio). Machine learning: Las métricas de la clasificación [Video]. YouTube. https://www.youtube.com/watch?v=E-zICBXTqzs&t=382s
""")
