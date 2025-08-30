import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# ✅ Page Config
st.set_page_config(page_title="🎀 Breast Cancer Prediction App", layout="centered")

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("breast_cancer_data.csv")  # <-- replace with your dataset filename
    return data

data = load_data()

# ------------------------------
# Preprocess Dataset
# ------------------------------
if "diagnosis" in data.columns:
    y = data["diagnosis"].map({"M": 1, "B": 0})
    X = data.drop(["id", "diagnosis"], axis=1, errors="ignore")
elif "class" in data.columns:
    y = data["class"]
    X = data.drop(["id", "class"], axis=1, errors="ignore")
else:
    st.error("⚠️ Could not find target column. Please make sure it's named 'diagnosis' or 'class'.")
    st.stop()

X = X.dropna(axis=1, how="all")
X = X.select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ------------------------------
# Feature Friendly Names + Meanings
# ------------------------------
feature_info = {
    "mean_radius": ("Average Cell Size", "Average distance from cell nucleus center to its boundary"),
    "mean_texture": ("Tissue Texture", "Variation in gray-scale values in the image"),
    "mean_symmetry": ("Cell Symmetry", "Symmetry of nucleus shape"),
    "mean_concavity": ("Cell Shape Irregularity", "Severity of concave (indented) parts of nucleus"),
}

# Rename for display only
X_display = X.rename(columns={k: v[0] for k, v in feature_info.items() if k in X.columns})

# ------------------------------
# Train-Test Split & Model
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# ------------------------------
# Fun Facts
# ------------------------------
fun_facts = [
    "💡 Early detection of breast cancer increases survival rates by 95%.",
    "🌸 Regular check-ups and screenings are super important!",
    "🥦 A healthy lifestyle lowers breast cancer risk.",
    "👩‍⚕️ 1 in 8 women will be diagnosed with breast cancer in their lifetime.",
    "🧪 AI helps doctors analyze thousands of cases quickly!"
]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎀 Breast Cancer Prediction App")
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose a page:", ["Prediction", "Insights & Awareness", "Feature Info & Medical Context"])

# ------------------------------
# Prediction Page
# ------------------------------
if choice == "Prediction":
    st.header("📋 Feature Values Across Age Groups")

    # ✅ Steady table (simulated averages for education)
    age_groups = pd.DataFrame({
        "Age Group": ["20-40", "40-60", "60-80"],
        "Average Cell Size": [14, 17, 19],
        "Tissue Texture": [18, 21, 24],
        "Cell Symmetry": [0.15, 0.18, 0.22],
        "Cell Shape Irregularity": [0.20, 0.25, 0.30],
    })

    st.table(age_groups)

    # ------------------------------
    # Prediction Inputs
    # ------------------------------
    st.header("Enter Patient Details 👩‍⚕️")

    user_input = {}
    user_input["Age"] = st.slider("Age of Patient", 20, 90, 40)

    for col in X_display.columns:
        user_input[col] = st.number_input(
            f"{col}",
            float(X_display[col].min()),
            float(X_display[col].max()),
            float(X_display[col].mean())
        )

    input_df = pd.DataFrame([user_input])
    input_df_model = input_df.drop("Age", axis=1, errors="ignore").rename(
        columns={v[0]: k for k, v in feature_info.items() if k in X.columns}
    )

    if st.button("🔍 Predict Now"):
        prediction = model.predict(input_df_model)[0]
        proba = model.predict_proba(input_df_model)[0][prediction] * 100

        if prediction == 1:
            st.error(f"🚨 High Risk Zone: Possible breast cancer.\n\n Model Confidence: {proba:.2f}%")
        else:
            st.success(f"✅ Safe Zone: No cancer risk detected.\n\n Model Confidence: {proba:.2f}%")
            st.balloons()

    st.info(f"🤖 Model Accuracy on test data: {acc*100:.2f}%")

# ------------------------------
# Insights & Awareness Page
# ------------------------------
elif choice == "Insights & Awareness":
    st.header("🌍 Breast Cancer Awareness & Insights")

    region_data = pd.DataFrame({
        "Region": ["North America", "Europe", "Asia", "Africa", "South America"],
        "Cases (per 100k women)": [90, 85, 60, 45, 70]
    })
    st.subheader("📊 Breast Cancer Incidence by Region")
    st.bar_chart(region_data.set_index("Region"))

    year_data = pd.DataFrame({
        "Year": list(range(2000, 2021, 5)),
        "Global Cases (in millions)": [1.0, 1.3, 1.6, 2.0, 2.3]
    })
    st.subheader("📈 Global Breast Cancer Cases Over Time")
    st.line_chart(year_data.set_index("Year"))

    st.subheader("💡 Did You Know?")
    st.success(random.choice(fun_facts))

# ------------------------------
# Feature Info Page
# ------------------------------
elif choice == "Feature Info & Medical Context":
    st.header("📖 Understanding Breast Cancer Features")

    feature_table = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in feature_info.items()],
        columns=["Dataset Feature", "Friendly Name", "Meaning in Medical Context"]
    )
    st.table(feature_table)

    st.info("ℹ️ These features are extracted from **Fine Needle Aspiration (FNA) biopsy images** of breast tissue using digital image analysis.")
