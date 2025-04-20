from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv('dataset.csv')
print(df)

# Combine features for vectorization
df['features'] = df['description'] + " " + df['genre']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['features'])

# Recommendation function
def recommend(user_input, top_n=5):
    # Filter rows where genre contains the user input (case-insensitive)
    filtered_df = df[df['genre'].str.contains(user_input, case=False, na=False)]

    if filtered_df.empty:
        return []  # No matches found

    # Use only the filtered data for similarity comparison
    filtered_df['features'] = filtered_df['description'] + " " + filtered_df['genre']
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(filtered_df['features'])
    input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vec, X).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    return filtered_df.iloc[top_indices][['title', 'platform', 'genre']].to_dict(orient='records')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    user_input = request.form['user_input']
    results = recommend(user_input)
    return render_template('index.html', results=results, query=user_input)

if __name__ == '__main__':
    app.run(debug=True)
