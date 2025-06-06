# ProductRecommendationAnalysis

├── amazon.csv                # Product dataset
├── Analysis.ipynb           # Jupyter notebook for initial data exploration
├── Product.ipynb            # Recommender logic demonstration (notebook form)
├── Analysis.py              # Data processing & pickle file generation
├── py_view.py               # Code to view content of saved pickle files
├── Recommendation.py        # Streamlit web app for recommendations
├── model/
│   ├── product_data.pkl     # Pickled product DataFrame
│   ├── cosine_sim.pkl       # Pickled cosine similarity matrix
│   └── indices.pkl          # Pickled mapping of product names to indices



⚙️ How It Works
1. Data Preprocessing
Missing values are handled.

product_name, category, and about_product columns are combined into a single text field.

TF-IDF vectorization is applied to this text to represent each product numerically.






2. Similarity Calculation
Cosine similarity is computed between all product vectors.

The most similar products are retrieved using this similarity matrix.






3. Streamlit Interface
A simple UI lets users select a product.

On clicking "Show Recommendation", it displays 5 visually similar products (with images and names).





🚀 Getting Started
🔧 Prerequisites
Ensure you have Python installed. Then install dependencies:

->pip install pandas scikit-learn streamlit


💻 Run the Application



1.Generate Pickle Files (Optional):

->python Analysis.py


2.View Data (Optional):


->python py_view.py



3.Start the Streamlit App:


->streamlit run Recommendation.py








📊 Dataset
The dataset amazon.csv includes fields like:

product_name

category

about_product

img_link (for displaying images)

🎯 Features
Fast content-based filtering using cosine similarity.

Lightweight and easy to deploy with Streamlit.

Visual recommendations with product names and images.

🧠 Algorithms Used
TF-IDF Vectorization

Cosine Similarity

Content-Based Filtering

📌 Notes
Ensure that amazon.csv is in the root directory.

The image links should be valid URLs to be displayed properly in Streamlit.

📜 License
This project is for educational and research purposes.


