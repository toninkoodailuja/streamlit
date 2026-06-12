import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.title("Student Gaming and Academic Performance Analysis")

try:
    url = "https://raw.githubusercontent.com/toninkoodailuja/streamlit/refs/heads/main/Gaming_Academic_Performance.csv"

    df = pd.read_csv(url)

    st.header("2. Minimal Exploratory Data Analysis")
    st.write("First 5 rows:")
    st.dataframe(df.head())

    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

    missing_values = df.isnull().sum().sum()
    st.write(f"**Total Missing Values:** {missing_values}")

    st.write("**Missing Values by Column:**")
    st.dataframe(df.isnull().sum())

    st.header("Statistics")

    numeric_df = df.select_dtypes(include="number")

    stats = numeric_df.agg(["mean", "std", "min", "max"]).T
    st.dataframe(stats)

    st.header("3. One Basic Correlation Analysis")

    x = df["gaming_hours"]
    y = df["grades"]

    correlation = x.corr(y)

    st.subheader("Pearson Correlation")
    st.write(f"Correlation between Gaming Hours and Grades: **{correlation:.3f}**")

    fig, ax = plt.subplots()

    ax.scatter(x, y, alpha=0.6, label="Data Points")

    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    ax.plot(x, regression_line, linewidth=2, label="Regression Line")

    ax.set_title("Gaming Hours vs Grades with Regression Line")
    ax.set_xlabel("Gaming Hours")
    ax.set_ylabel("Grades")
    ax.legend()

    st.pyplot(fig)

    if abs(correlation) < 0.3:
        strength = "weak"
    elif abs(correlation) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if correlation > 0 else "negative"

    interpretation = f"""
    The Pearson correlation coefficient between gaming hours and grades is {correlation:.3f}.
    This indicates a {strength} {direction} relationship between the two variables.
    A {direction} correlation means that as gaming hours increase, grades tend to {'increase' if correlation > 0 else 'decrease'}.
    The relationship is not necessarily causal, but it shows a measurable association in the dataset.
    This result {'makes sense because excessive gaming may reduce study time.' if correlation < 0 else 'may suggest that moderate gaming does not negatively affect academic performance.'}
    """

    st.write(interpretation)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    col1 = st.selectbox(
    "Select First Numerical Variable",
    numeric_columns
    )

    col2 = st.selectbox(
    "Select Second Numerical Variable",
    numeric_columns,
    index=min(1, len(numeric_columns)-1)
    )

    corr_value = df[col1].corr(df[col2])

    st.write(f"### Pearson Correlation: {corr_value:.3f}")

    fig, ax = plt.subplots()

    x = df[col1]
    y = df[col2]

    ax.scatter(x, y, alpha=0.6)

    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(
    x,
    slope * x + intercept,
    linewidth=2
    )

    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"{col1} vs {col2}")

    st.pyplot(fig)

    st.caption(
    f"The Pearson correlation between {col1} and {col2} "
    f"is {corr_value:.3f}. Values closer to 1 or -1 indicate "
    f"stronger relationships, while values near 0 indicate weak relationships."
    )

    st.header("3. One Simple Supervised Learning Model")

    st.subheader("Linear Regression: Predicting Grades")

    target = "grades"

    features = ["gaming_hours", "addiction_score"]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("### Model Information")
    st.write(f"**Target Variable:** {target}")
    st.write(f"**Predictor Variables:** {features}")

    st.write("### Model Performance")
    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.3f}")

    st.subheader("Interpretation")

    interpretation = f"""
    The target variable for this model is grades, while gaming_hours and addiction_score
    are used as predictor variables. A Linear Regression model was trained using a 70/30
    train-test split. The model achieved an R² score of {r2:.3f}, indicating how much of
    the variation in grades can be explained by the selected predictors. The MAE of
    {mae:.3f} means that predictions differ from actual grades by approximately this amount
    on average. The results suggest that gaming behavior has some predictive relationship
    with academic performance, although other factors may also influence grades.
    """

    st.write(interpretation)

    st.header("C. Prediction Model")

    target = "grades"

    features = [
    col for col in numeric_columns
    if col != target
    ]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Information")

    st.write("Model Type: Linear Regression")
    st.write(f"Target Variable: {target}")
    st.write(f"R² Score: {r2:.3f}")

    fig, ax = plt.subplots()

    ax.scatter(y_test, y_pred, alpha=0.7)

    ax.set_xlabel("Actual Grades")
    ax.set_ylabel("Predicted Grades")
    ax.set_title("Predicted vs Actual Grades")

    st.pyplot(fig)

    st.caption(
    "Points closer to a diagonal trend indicate better model performance. "
    "The R² score measures how well the predictor variables explain variation in grades."
    )

except FileNotFoundError:
    st.error("Gaming_Academic_Performance.csv not found. Make sure the file is in the same folder as app.py.")
