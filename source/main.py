import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data():
    url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
    heart_disease = pd.read_csv(url)
    return heart_disease

def train_model():
    heart_disease = load_data()
    X = heart_disease.drop("target", axis=1)
    y = heart_disease["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    print(f"Train accuracy: {clf.score(X_train, y_train):.2f}")
    print(f"Test accuracy: {clf.score(X_test, y_test):.2f}")

if __name__ == "__main__":
    train_model()
