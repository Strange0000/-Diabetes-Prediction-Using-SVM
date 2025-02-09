# 📊 Diabetes Prediction Using SVM

## 🚀 Overview
This project implements a **Support Vector Machine (SVM)** classifier to predict whether a person is diabetic based on medical data. The model is trained using **scikit-learn**, with **StandardScaler** for feature scaling to improve accuracy.

## 📌 Features
- Uses **Support Vector Machine (SVM)** for classification
- Standardizes input features with **StandardScaler**
- Takes user input for real-time diabetes prediction
- Outputs whether the person is **diabetic** or **not diabetic**

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Strange0000/diabetes-prediction-svm.git
cd diabetes-prediction-svm
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Dataset
This model is trained using the **Pima Indians Diabetes Dataset**, which consists of the following features:
- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

## 🚀 Usage
### Run the Prediction Script
```python
python predict.py
```
### Example Input & Output
```python
input_data = (5,166,72,19,175,25.8,0.587,51)
# Output: The person is diabetic
```

## 📌 Model Workflow
1. Load the dataset and preprocess it.
2. Train an **SVM Classifier** with a **linear kernel**.
3. Standardize input features using **StandardScaler**.
4. Predict diabetes based on new user input.

## 📈 Example Code
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Load dataset (ensure you have trained the model)
df = pd.read_csv('diabetes.csv')
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('Outcome', axis=1))

# Train SVM Model
clf = svm.SVC(kernel='linear')
clf.fit(X, df['Outcome'])

# Sample Prediction
input_data = np.array([[5,166,72,19,175,25.8,0.587,51]])
scaled_data = scaler.transform(input_data)
result = clf.predict(scaled_data)
print('Diabetic' if result[0] == 1 else 'Not Diabetic')
```

## 🛠️ Technologies Used
- Python 🐍
- Scikit-Learn 🤖
- NumPy 🔢
- Pandas 📊
- Matplotlib 📈

## 🤝 Contributing
Feel free to fork this project and submit **pull requests**!

## 📜 License
This project is licensed under the **MIT License**.

## ⭐ Show Your Support
If you like this project, give it a ⭐ on GitHub!

