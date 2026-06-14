import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read Excel file
df = pd.read_excel("customer_data.xlsx")

# Select columns for clustering
X = df[['Income', 'SpendingScore']]

# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)

# Assign cluster to each customer
df['Cluster'] = kmeans.fit_predict(X)

# Create graph
plt.scatter(df['Income'], df['SpendingScore'], c=df['Cluster'])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()

# Save output
df.to_excel("customer_segmented.xlsx", index=False)

print("Project Completed Successfully!")