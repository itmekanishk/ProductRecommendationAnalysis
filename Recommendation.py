import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("amazon.csv")


df['about_product'] = df['about_product'].fillna('')
df['combined'] = df['product_name'] + " " + df['category'] + " " + df['about_product']


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend(product_name):
    index = df[df['product_name'] == product_name].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_names = []
    recommended_images = []
    for i in distances[1:6]:  
        recommended_names.append(df.iloc[i[0]].product_name)
        recommended_images.append(df.iloc[i[0]].img_link)
    return recommended_names, recommended_images


st.header('🛒 Product Recommender System')

product_list = df['product_name'].values
selected_product = st.selectbox(
    "Type or select a product",
    product_list
)

if st.button('Show Recommendation'):
    names, images = recommend(selected_product)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(images[i], use_column_width=True)
