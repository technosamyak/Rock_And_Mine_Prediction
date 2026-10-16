from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data(csv_path):
    return pd.read_csv(sonardata.csv, header=None)


def explore(df):
    print("Shape (rows, columns):", df.shape)
    print("\nClass counts (M = Mine, R = Rock):")
    print(df[60].value_counts())
    print("\nAverage of each feature per class:")
    print(df.groupby(60).mean())


def main():
    csv_path = Path(__file__).parent / "sonar_data.csv"
    sonar_data = load_data(csv_path)
    explore(sonar_data)

    # 2. Separate features (X) and labels (Y)
    X = sonar_data.drop(columns=60, axis=1)
    Y = sonar_data[60]

    # 3. Train/test split (90% / 10%), keeping the Rock/Mine ratio the same
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, stratify=Y, random_state=1
    )
    print("\nData shapes ->", X.shape, X_train.shape, X_test.shape)

    # 4. Train the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # 5. Evaluate
    train_acc = accuracy_score(model.predict(X_train), Y_train)
    test_acc = accuracy_score(model.predict(X_test), Y_test)
    print(f"\nAccuracy on training data: {train_acc:.4f}")
    print(f"Accuracy on test data:     {test_acc:.4f}")

    # 6. Predict on one sample the model has never seen (taken from the test set)
    sample = X_test.iloc[[0]]          # keeps it as a 1-row table
    actual = Y_test.iloc[0]
    prediction = model.predict(sample)[0]

    print("\nPrediction for one unseen sample:")
    print("Actual label:   ", actual)
    print("Predicted label:", prediction)
    print("The object is a Rock" if prediction == "R" else "The object is a Mine")


if __name__ == "__main__":
    main()
