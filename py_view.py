import pickle

# Load and view product_data.pkl
with open("model/product_data.pkl", "rb") as f:
    df = pickle.load(f)
print("Product Data Sample:")
print(df.head())  # Displays the first few rows of the DataFrame

# Load and view cosine_sim.pkl
with open("model/cosine_sim.pkl", "rb") as f:
    cos_sim = pickle.load(f)
print("\nCosine Similarity Matrix Shape:", cos_sim.shape)  # Shows matrix dimensions
print(cos_sim[:5, :5])  # Displays a small portion of the matrix

# Load and view indices.pkl
with open("model/indices.pkl", "rb") as f:
    indices = pickle.load(f)
print("\nIndices Sample:")
print(indices.head())  # Displays the first few mappings