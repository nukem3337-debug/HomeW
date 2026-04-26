# 7-LABORATORIYA ISHI
# Mavzu: Decision Tree yordamida qarz berish qarorini aniqlash

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Datasetni yuklash
df = pd.read_csv("loan.csv")
print("Datasetning dastlabki 5 qatori:")
print(df.head())

# 2. X va y ajratish
X = df[["income", "age", "credit_score"]]
y = df["approved"]

# 3. Train-test ajratish
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Decision Tree modelini yaratish (Gini, max_depth=3)
model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 5. Bashorat qilish
pred = model.predict(X_test)
print("\nTest uchun bashoratlar:", pred)
print("Haqiqiy qiymatlar:    ", y_test.values)

# 6. Baholash
acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
print(f"\nAccuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)

# 7. Feature importance
print("\nBelgilar muhimligi:")
for col, imp in zip(X.columns, model.feature_importances_):
    print(f"  {col}: {imp:.4f}")

# 8. Qaror daraxtini chizish
plt.figure(figsize=(12, 7))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
)
plt.title("Decision Tree — Qarz berish qarori")
plt.savefig("decision_tree.png", dpi=120, bbox_inches="tight")
plt.show()
