import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from xgboost import XGBClassifier

# Import the TextPreprocessor class
from DataPreprocessing import TextPreprocessor

# Load dataset
df = pd.read_csv(r"C:\Data Extraction\Total_Dataset.csv", encoding="ISO-8859-1")

# Text Preprocessing
preprocessor = TextPreprocessor(use_tfidf=True)
df["processed_text"] = df["Text"].apply(preprocessor.preprocess)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["processed_text"])
y = df["Label"]  # Assuming "Label" is the target column

# Split original dataset (before SMOTE) into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(probability=True, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
pa_model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)

# Train and evaluate models
models = {
    "Naive Bayes": nb_model,
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "SVM": svm_model,
    "Decision Tree": dt_model,
    "Passive Aggressive": pa_model,
    "XGBoost": xgb_model
}

results = {}
roc_curves = {}

for name, model in models.items():
    print(f"\n----- {name} -----")
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["AI", "Human"]))

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_curves[name] = (fpr, tpr, roc_auc)

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=["AI", "Human"], yticklabels=["AI", "Human"])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Plot ROC Curves
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.show()

# Print Accuracy Scores
print("\n=== Final Accuracy Scores ===")
for name, accuracy in results.items():
    print(f'{name}: {accuracy:.4f}')
