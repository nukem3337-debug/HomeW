# 10-LABORATORIYA ISHI
# Mavzu: Neural Network (MLP) yordamida talabaning natijasini prognoz qilish

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# 1. Datasetni yuklash
df = pd.read_csv("student_nn.csv")
print("Datasetning birinchi 5 qatori:")
print(df.head())

# 2. X (kiruvchi belgilar) va y (target) ajratish
X = df[["math", "english", "attendance"]]
y = df["result"]

# 3. Train va test guruhlarga ajratish
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("\nTrain o'lchami:", X_train.shape)
print("Test o'lchami:", X_test.shape)

# 4. Ma'lumotlarni normallashtirish (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. MLP modelini yaratish
model = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation="relu",
    max_iter=1000,
    random_state=42,
)

# 6. Modelni o'qitish
model.fit(X_train_scaled, y_train)

# 7. Bashorat olish
pred = model.predict(X_test_scaled)
print("\nBashorat:", pred)
print("Haqiqiy: ", y_test.values)

# 8. Natijani baholash
acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
print(f"\nAccuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, pred, zero_division=0))

# 9. Training Loss grafigi
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("Training Loss — MLP")
plt.xlabel("Iteratsiya")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

# 10. Yangi talaba uchun bashorat
new_student = [[78, 74, 82]]
new_student_scaled = scaler.transform(new_student)
new_pred = model.predict(new_student_scaled)
print(f"\nYangi talaba (math=78, english=74, attendance=82) uchun bashorat: {new_pred[0]}")
if new_pred[0] == 1:
    print("Natija: talaba muvaffaqiyatli bo'lishi kutilmoqda.")
else:
    print("Natija: talaba xavf guruhida bo'lishi mumkin.")
