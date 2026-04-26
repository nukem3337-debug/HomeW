# 9-LABORATORIYA ISHI
# Mavzu: K-Means algoritmi yordamida talabalarni klasterlash

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Datasetni yuklash
df = pd.read_csv("students_cluster.csv")
print("Dataset:")
print(df)

# 2. Elbow Method orqali optimal K ni topish
inertia = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), inertia, marker="o")
plt.title("Elbow Method — Optimal K ni tanlash")
plt.xlabel("K (klasterlar soni)")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig("elbow_method.png")
plt.show()

# 3. Optimal K=3 bilan klasterlash
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df)
df["Cluster"] = clusters

print("\nKlasterlash natijasi:")
print(df)

# Har bir klasterdagi talabalar soni
print("\nHar bir klasterdagi talabalar soni:")
print(df["Cluster"].value_counts().sort_index())

# 4. Klaster grafigi va markazlar
centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 5))
plt.scatter(df["math"], df["english"], c=df["Cluster"], cmap="viridis",
            s=80, edgecolor="black", label="Talabalar")
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=250,
            marker="X", label="Markazlar")
plt.xlabel("Math")
plt.ylabel("English")
plt.title("Talabalar klasterlari va markazlari (K=3)")
plt.legend()
plt.grid(True)
plt.savefig("clusters.png")
plt.show()

print("\nKlaster markazlari:")
for i, c in enumerate(centers):
    print(f"  Cluster {i}: math={c[0]:.1f}, english={c[1]:.1f}")
