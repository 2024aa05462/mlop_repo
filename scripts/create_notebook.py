import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# Heart Disease EDA and Modeling
**MLOps Assignment**

## 1. Load Data
"""

code_load = """import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=columns)

# Handle missing values ('?')
import numpy as np
df = df.replace('?', np.nan)
df = df.dropna()
df = df.astype(float)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

df.head()
"""

text_eda = """## 2. Exploratory Data Analysis"""

code_eda = """# 1. Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title('Class Distribution')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 3. Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# 4. Pairplot (Top 3 features)
sns.pairplot(df[['age', 'chol', 'thalach', 'target']], hue='target')
plt.show()
"""

text_model = """## 3. Modeling"""

code_model = """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell(text_eda),
    nbf.v4.new_code_cell(code_eda),
    nbf.v4.new_markdown_cell(text_model),
    nbf.v4.new_code_cell(code_model)
]

with open('notebooks/eda_modeling.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created notebooks/eda_modeling.ipynb")
