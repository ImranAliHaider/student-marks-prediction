import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Marks Predictor", page_icon="📘")
st.title("Student Marks Prediction")
st.write("Enter daily study hours to get an estimated mark.")
st.warning("Learning demo only: this uses synthetic sample data, not real student records.")

@st.cache_resource
def train_model():
    data = pd.read_csv("student_marks.csv")
    model = LinearRegression()
    model.fit(data[["study_hours"]], data["marks"])
    return model

model = train_model()
hours = st.number_input(
    "Daily study hours",
    min_value=0.0,
    max_value=16.0,
    value=4.0,
    step=0.5
)

if st.button("Predict Marks"):
    sample = pd.DataFrame({"study_hours": [hours]})
    prediction = float(model.predict(sample)[0])
    prediction = max(0.0, min(100.0, prediction))
    st.metric("Estimated marks out of 100", f"{prediction:.1f}")
    st.caption("This is an estimate, not a guarantee of actual exam marks.")

st.subheader("Illustrative dataset")
st.dataframe(pd.read_csv("student_marks.csv"), use_container_width=True)
# Linear Regression Graph
data = pd.read_csv("student_marks.csv")

fig, ax = plt.subplots(figsize=(8, 5))

# Actual data points
ax.scatter(
    data["study_hours"],
    data["marks"],
    color="blue",
    label="Actual Marks"
)

# Linear Regression line
x_values = pd.DataFrame({
    "study_hours": sorted(data["study_hours"])
})

y_values = model.predict(x_values)

ax.plot(
    x_values["study_hours"],
    y_values,
    color="red",
    linewidth=2,
    label="Linear Regression Line"
)

ax.set_xlabel("Daily Study Hours")
ax.set_ylabel("Marks")
ax.set_title("Study Hours vs Marks - Linear Regression")
ax.legend()
ax.grid(True)

st.subheader("Linear Regression Graph")
st.pyplot(fig)