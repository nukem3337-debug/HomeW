# 3-LABORATORIYA ISHI
# Mavzu: Ma'lumotlarni vizualizatsiya qilish (Matplotlib va Seaborn)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datasetni yuklash
df = pd.read_csv("students.csv")

# 1. Scatter plot — Math vs Physics bog'liqligi
plt.figure(figsize=(8, 5))
plt.scatter(df["math"], df["physics"])
plt.xlabel("Math")
plt.ylabel("Physics")
plt.title("Math va Physics bog'liqligi")
plt.savefig("scatter_math_physics.png")
plt.show()

# 2. Histogram — Math ball taqsimoti
plt.figure(figsize=(8, 5))
plt.hist(df["math"], bins=5, color="skyblue", edgecolor="black")
plt.xlabel("Math ball")
plt.ylabel("Talabalar soni")
plt.title("Math ball taqsimoti")
plt.savefig("histogram_math.png")
plt.show()

# 3. Bar chart — Gender bo'yicha o'rtacha Math ball
df["gender"] = df["gender"].map({"M": 0, "F": 1})
group = df.groupby("gender")["math"].mean()
plt.figure(figsize=(8, 5))
group.plot(kind="bar", color=["#4c72b0", "#dd8452"])
plt.xticks([0, 1], ["Male", "Female"], rotation=0)
plt.ylabel("O'rtacha Math ball")
plt.title("Gender bo'yicha Math o'rtacha ball")
plt.savefig("bar_gender_math.png")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelyatsiya matritsasi")
plt.savefig("heatmap_correlation.png")
plt.show()

print("4 ta grafik chizildi va PNG fayllarga saqlandi.")
