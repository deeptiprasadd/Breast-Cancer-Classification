# 🎀 Breast Cancer Classification App - Full Guide

## 🔗 Quick Access
👉 [Click here to use the app directly](https://breast-cancer-classificationn.streamlit.app/)

---

## Part 1: About the App
This app uses **Machine Learning (Random Forest Classifier)** to predict whether a breast tumor is **Benign (B)** or **Malignant (M)** based on 30 real-valued features computed from digitized images of a **Fine Needle Aspirate (FNA) biopsy**.

It includes:
- 📋 **Feature Reference Table** (Age groups vs feature values) on the home page.  
- 🧪 **Prediction Section** where users can input patient details.  
- 🌍 **Insights Page** with global breast cancer statistics and awareness facts.  
- 📖 **Feature Info Page** explaining what each dataset feature means in medical context.  

---

## Part 2: How to Use the Breast Cancer Classification App

### Step 1: Open the App
👉 [Click here to use the app directly](https://breast-cancer-classificationn.streamlit.app/)

*(Alternative: Run the Jupyter Notebook or the Streamlit app locally – see Part 4 below)*

### Step 2: Enter Patient Details
- Adjust the **Age slider** (20–90).  
- Input values for features like **Average Cell Size, Tissue Texture, Cell Symmetry**, etc.  

### Step 3: Prediction Result
- ✅ **Safe Zone (Benign)** → No cancer risk detected.  
- 🚨 **High Risk Zone (Malignant)** → Possible breast cancer.  

The app also shows **model confidence (%)**.  

---

## Part 3: Workflow Diagram (Code Flow)
[Breast Cancer Dataset]
↓ Load & Preprocess
↓ Train-Test Split
↓ Train RandomForest Classifier
↓ Prediction Inputs (Age + 30 features)
↓ Predict Diagnosis (Benign / Malignant)
↓ Show Result + Confidence Score

---

## Part 4: Run Locally

### Step 1: Install Dependencies
Make sure the following Python libraries are installed:
streamlit
pandas
numpy
scikit-learn

### Step 2: Run the App
streamlit run app.py

---

## Part 5: About the Dataset
📌 Source
Features are computed from a digitized image of a breast mass’s Fine Needle Aspirate (FNA). They describe the characteristics of the cell nuclei present in the image.

📌 Attribute Information
ID number
Diagnosis: M = malignant, B = benign
30 Real-Valued Features are computed for each cell nucleus:
Radius: mean of distances from the center to points on the perimeter
Texture: standard deviation of gray-scale values
Perimeter
Area
Smoothness: local variation in radius lengths
Compactness: (perimeter² / area - 1.0)
Concavity: severity of concave portions of the contour
Concave points: number of concave portions of the contour
Symmetry
Fractal dimension: “coastline approximation” - 1

### For each of these features, the dataset provides:
Mean value
Standard error (SE)
“Worst” (mean of the 3 largest values)
Total Features = 30.

📌 Dataset Properties
Instances: 569
Features: 30
Class Distribution:
-357 Benign
-212 Malignant
-Missing Values: None
Significant Digits: 4

---

## Part 6: Key Components and Files

app.py → Streamlit application
breast_cancer_data.csv → Dataset

Libraries Used:
pandas
numpy
scikit-learn
streamlit

### 🌸 Awareness Note

Breast cancer is one of the most common cancers worldwide.
This app is intended for educational purposes only and not for medical diagnosis.
Always consult a medical professional for clinical decisions.
