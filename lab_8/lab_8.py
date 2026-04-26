# 8-LABORATORIYA ISHI
# Mavzu: Random Forest yordamida bank riskini aniqlash

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# 1. Datasetni yuklash
df = pd.read_csv("bank.csv")
print("Dataset:")
print(df.head())

# 2. X va y ajratish
X = df[["income", "age", "loan_amount", "credit_score"]]
y = df["risk"]

# 3. Train-test ajratish
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Random Forest modelini yaratish va o'qitish
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42,
)
model.fit(X_train, y_train)

# 5. Bashorat qilish
pred = model.predict(X_test)
print("\nBashorat:", pred)
print("Haqiqiy: ", y_test.values)

# 6. Baholash
acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
print(f"\nAccuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, pred, zero_division=0))

# 7. Feature importance grafigi
importance = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.bar(X.columns, importance, color="teal")
plt.title("Feature Importance — Bank risk")
plt.xticks(rotation=30)
plt.ylabel("Muhimlik darajasi")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nFeature importance:")
for col, imp in zip(X.columns, importance):
    print(f"  {col}: {imp:.4f}")
