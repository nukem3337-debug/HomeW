# 5-LABORATORIYA ISHI
# Mavzu: Logistic Regression yordamida imtihondan o'tishni prognoz qilish

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)

# 1. Datasetni yuklash
df = pd.read_csv("exam.csv")
print("Dataset:")
print(df.head())

# 2. X va y ajratish
X = df[["hours_studied", "attendance"]]
y = df["passed"]

# 3. Train-test ajratish
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Modelni yaratish va o'qitish
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Bashorat qilish
pred = model.predict(X_test)
print("\nBashorat:", pred)
print("Haqiqiy:", y_test.values)

# 6. Baholash
cm = confusion_matrix(y_test, pred)
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, zero_division=0)
rec = recall_score(y_test, pred, zero_division=0)

print("\nConfusion Matrix:")
print(cm)
print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")

# 7. Klassifikatsiya grafigi
plt.figure(figsize=(8, 5))
plt.scatter(df["hours_studied"], df["attendance"], c=df["passed"],
            cmap="coolwarm", s=80, edgecolor="black")
plt.xlabel("Hours Studied")
plt.ylabel("Attendance (%)")
plt.title("Imtihondan o'tish klassifikatsiyasi")
plt.colorbar(label="0 = O'tmagan, 1 = O'tgan")
plt.grid(True)
plt.savefig("classification.png")
plt.show()
