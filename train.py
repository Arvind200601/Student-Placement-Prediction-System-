# ==========================================================
# Student Placement Prediction System using Machine Learning
# Model Comparison:
# 1. Random Forest Classifier
# 2. Linear SVM
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("student_placement_dataset_100000 (1).csv")

print("\nDataset Loaded Successfully.")
print("Shape :", df.shape)

# ==========================================================
# Display Dataset Information
# ==========================================================

print("\nDataset Information")
print(df.info())

print("\nFirst Five Records")
print(df.head())

print("\nStatistical Summary")
print(df.describe())

# ==========================================================
# Data Preprocessing
# ==========================================================

print("\nChecking Missing Values")

print(df.isnull().sum())
# Fill missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)
print("\nChecking Duplicate Records")

duplicates = df.duplicated().sum()

print("Duplicate Records :", duplicates)

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate Records Removed.")

print("\nDataset Shape After Cleaning :", df.shape)

# ==========================================================
# Features and Target
# ==========================================================

X = df.drop("Placement", axis=1)

y = df["Placement"]

print("\nFeatures")

print(X.columns.tolist())

print("\nTarget Column : Placement")
# ==========================================================
# Train-Test Split (80 : 20)
# ==========================================================

print("\n" + "=" * 60)
print("Splitting Dataset (80% Training | 20% Testing)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

print("\nTrain-Test Split Completed Successfully.")

# ==========================================================
# Feature Scaling
# ==========================================================

print("\n" + "=" * 60)
print("Applying Feature Scaling")
print("=" * 60)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Feature Scaling Completed.")

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

print("Scaler Saved Successfully (scaler.pkl)")

# ==========================================================
# Dataset Summary
# ==========================================================

print("\n" + "=" * 60)
print("Dataset Summary")
print("=" * 60)

print(f"Total Records      : {len(df)}")
print(f"Training Records   : {len(X_train)}")
print(f"Testing Records    : {len(X_test)}")
print(f"Number of Features : {X.shape[1]}")

print("\nFeature Names")

for feature in X.columns:
    print("✔", feature)

print("\nData Preprocessing Completed Successfully.")
# ==========================================================
# Train Random Forest Model
# ==========================================================

print("\n" + "=" * 60)
print("Training Random Forest Model")
print("=" * 60)

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest.fit(X_train_scaled, y_train)

rf_prediction = random_forest.predict(X_test_scaled)

rf_accuracy = accuracy_score(y_test, rf_prediction)

rf_precision = precision_score(y_test, rf_prediction)

rf_recall = recall_score(y_test, rf_prediction)

rf_f1 = f1_score(y_test, rf_prediction)

print("\nRandom Forest Completed Successfully.")

# ==========================================================
# Train Linear SVM Model
# ==========================================================

print("\n" + "=" * 60)
print("Training Linear SVM Model")
print("=" * 60)

linear_svm = LinearSVC(
    random_state=42,
    max_iter=10000
)

linear_svm.fit(X_train_scaled, y_train)

svm_prediction = linear_svm.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, svm_prediction)

svm_precision = precision_score(y_test, svm_prediction)

svm_recall = recall_score(y_test, svm_prediction)

svm_f1 = f1_score(y_test, svm_prediction)

print("\nLinear SVM Completed Successfully.")

# ==========================================================
# Model Comparison
# ==========================================================

print("\n")
print("=" * 75)
print("          MACHINE LEARNING MODEL COMPARISON")
print("=" * 75)

print(f"{'Metric':<20}{'Random Forest':<20}{'Linear SVM'}")
print("-"*75)

print(f"{'Accuracy':<20}{rf_accuracy*100:.2f}%{'':<10}{svm_accuracy*100:.2f}%")

print(f"{'Precision':<20}{rf_precision*100:.2f}%{'':<10}{svm_precision*100:.2f}%")

print(f"{'Recall':<20}{rf_recall*100:.2f}%{'':<10}{svm_recall*100:.2f}%")

print(f"{'F1 Score':<20}{rf_f1*100:.2f}%{'':<10}{svm_f1*100:.2f}%")

print("=" * 75)
# ==========================================================
# Select Best Model
# ==========================================================

print("\n" + "=" * 60)
print("Selecting Best Machine Learning Model")
print("=" * 60)

if rf_accuracy >= svm_accuracy:

    best_model = random_forest
    best_model_name = "Random Forest"
    best_accuracy = rf_accuracy

    print("\n✅ Best Model Selected : Random Forest")

else:

    best_model = linear_svm
    best_model_name = "Linear SVM"
    best_accuracy = svm_accuracy

    print("\n✅ Best Model Selected : Linear SVM")

# ==========================================================
# Save Best Model
# ==========================================================

joblib.dump(best_model, "model.pkl")

print("\nModel Saved Successfully.")
print("File Name : model.pkl")

# ==========================================================
# Confusion Matrix
# ==========================================================

print("\n" + "=" * 60)
print("Confusion Matrix")
print("=" * 60)

if best_model_name == "Random Forest":
    final_prediction = rf_prediction
else:
    final_prediction = svm_prediction

cm = confusion_matrix(y_test, final_prediction)

print(cm)

# ==========================================================
# Classification Report
# ==========================================================

print("\n" + "=" * 60)
print("Classification Report")
print("=" * 60)

print(classification_report(y_test, final_prediction))

# ==========================================================
# Feature Importance (Random Forest Only)
# ==========================================================

if best_model_name == "Random Forest":

    print("\n" + "=" * 60)
    print("Feature Importance")
    print("=" * 60)

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": random_forest.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(importance)

print("\nEvaluation Completed Successfully.")
# ==========================================================
# Final Training Summary
# ==========================================================

print("\n" + "=" * 70)
print("         STUDENT PLACEMENT PREDICTION SYSTEM")
print("         MACHINE LEARNING TRAINING SUMMARY")
print("=" * 70)

print(f"\nDataset Name          : student_placement_dataset_100000.csv")
print(f"Total Records         : {len(df)}")
print(f"Training Records      : {len(X_train)}")
print(f"Testing Records       : {len(X_test)}")

print("\nTrain-Test Split      : 80% Training | 20% Testing")

print("\nMachine Learning Models Used")
print("-------------------------------------")
print("1. Random Forest Classifier")
print("2. Linear SVM")

print("\nModel Comparison")
print("-------------------------------------")
print(f"Random Forest Accuracy : {rf_accuracy*100:.2f}%")
print(f"Linear SVM Accuracy    : {svm_accuracy*100:.2f}%")

print("\nSelected Model")
print("-------------------------------------")
print(best_model_name)

print("\nSaved Files")
print("-------------------------------------")
print("✔ model.pkl")
print("✔ scaler.pkl")

print("\nData Preprocessing Completed")
print("-------------------------------------")
print("✔ Missing Values Checked")
print("✔ Duplicate Records Removed")
print("✔ Features & Target Separated")
print("✔ Train-Test Split Applied (80:20)")
print("✔ Feature Scaling Applied")

print("\nEvaluation Metrics")
print("-------------------------------------")
print(f"Accuracy  : {best_accuracy*100:.2f}%")

if best_model_name == "Random Forest":
    print(f"Precision : {rf_precision*100:.2f}%")
    print(f"Recall    : {rf_recall*100:.2f}%")
    print(f"F1 Score  : {rf_f1*100:.2f}%")
else:
    print(f"Precision : {svm_precision*100:.2f}%")
    print(f"Recall    : {svm_recall*100:.2f}%")
    print(f"F1 Score  : {svm_f1*100:.2f}%")

print("\nProject Status")
print("-------------------------------------")
print("✔ Dataset Loaded")
print("✔ Data Preprocessed")
print("✔ Dataset Split into 80:20")
print("✔ Random Forest Trained")
print("✔ Linear SVM Trained")
print("✔ Model Comparison Completed")
print("✔ Best Model Selected")
print("✔ Model Saved Successfully")

print("\n" + "=" * 70)
print(" TRAINING COMPLETED SUCCESSFULLY ")
print("=" * 70)

print("\nYou can now run your Flask application using:")

print("\npython app.py")

print("\nBest Model :", best_model_name)
print("Accuracy   : {:.2f}%".format(best_accuracy * 100))

print("\nThank You.")