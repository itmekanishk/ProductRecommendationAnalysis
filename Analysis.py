
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


df = pd.read_csv("amazon.csv")


df['about_product'] = df['about_product'].fillna('')
df['category'] = df['category'].fillna('')
df['product_name'] = df['product_name'].fillna('')


df['combined'] = df['product_name'] + ' ' + df['category'] + ' ' + df['about_product']


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])


cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


indices = pd.Series(df.index, index=df['product_name']).drop_duplicates()

os.makedirs("model", exist_ok=True)


with open("model/product_data.pkl", "wb") as f:
    pickle.dump(df, f)

with open("model/cosine_sim.pkl", "wb") as f:
    pickle.dump(cos_sim, f)

with open("model/indices.pkl", "wb") as f:
    pickle.dump(indices, f)

print("Processing complete! Pickle files saved in 'model' directory.")