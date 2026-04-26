# 6-LABORATORIYA ISHI
# Mavzu: K-Nearest Neighbors (KNN) algoritmi yordamida kasallikni aniqlash

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Datasetni yuklash
df = pd.read_csv("disease.csv")
print("Dataset:")
print(df.head())

# 2. X va y ajratish
X = df[["temperature", "heart_rate"]]
y = df["disease"]

# 3. Train-test ajratish
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. KNN modelini yaratish va o'qitish (K=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 5. Bashorat qilish
pred = model.predict(X_test)
print("\nBashorat:", pred)
print("Haqiqiy:", y_test.values)

# 6. Baholash
acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
print(f"\nAccuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)

# 7. Turli K qiymatlarini sinab ko'rish
print("\nTurli K qiymatlari uchun accuracy:")
for k in range(1, 6):
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    print(f"  K = {k}: Accuracy = {accuracy_score(y_test, p):.4f}")
