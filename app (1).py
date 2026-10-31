import streamlit as st
st.set_page_config(
    page_title="SmartCart AI",
    page_icon="🛒",
    layout="wide"
)


import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PRODUCT DATA (loaded from products.csv)
# -----------------------------

df_products = pd.read_csv("products.csv")

# -----------------------------
# USER INTERACTION DATA (loaded from interactions.csv)
# -----------------------------

df_interactions = pd.read_csv("interactions.csv")

interaction_weights = {
    "viewed": 1,
    "liked": 2,
    "purchased": 3
}

df_interactions["weight"] = df_interactions["interaction"].map(
    interaction_weights
)

df_user_products = pd.merge(
    df_interactions,
    df_products,
    on="product_id"
)

# -----------------------------
# ML RECOMMENDATION ENGINE
# -----------------------------

# Convert product features into TF-IDF vectors
tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(
    df_products["features"]
)

# Calculate similarity between all products
similarity_matrix = cosine_similarity(
    tfidf_matrix
)


def recommend_for_user(user_id, top_n=5):

    # Get the user's previous interactions
    user_data = df_user_products[
        df_user_products["user_id"] == user_id
    ]

    # Get products and interaction weights
    user_products = user_data["product_id"].tolist()
    user_weights = user_data["weight"].values

    # Convert product IDs to dataframe indexes
    user_indices = [
        df_products.index[
            df_products["product_id"] == product_id
        ][0]
        for product_id in user_products
    ]

    # Get similarity scores
    user_similarity_scores = similarity_matrix[user_indices]

    # Give more importance to stronger interactions
    weighted_scores = (
        user_similarity_scores * user_weights[:, None]
    )

    # Combine all influence scores
    final_scores = weighted_scores.sum(axis=0)

    # Rank products
    ranked_indices = final_scores.argsort()[::-1]

    # Remove products the user already interacted with
    filtered_indices = [
        index for index in ranked_indices
        if df_products.iloc[index]["product_id"]
        not in user_products
    ]

    # Select top recommendations
    top_indices = filtered_indices[:top_n]

    recommendations = df_products.iloc[
        top_indices
    ].copy()

    # Add recommendation score
    recommendations["recommendation_score"] = [
        round(final_scores[index], 2)
        for index in top_indices
    ]

    return recommendations[
        [
            "product_id",
            "product_name",
            "category",
            "price",
            "recommendation_score"
        ]
    ]

# -----------------------------
# RECOMMENDATION EXPLANATION
# -----------------------------

def explain_recommendation(user_id, recommended_product_id):

    user_data = df_user_products[
        df_user_products["user_id"] == user_id
    ]

    user_products = user_data["product_id"].tolist()
    user_weights = user_data["weight"].values

    user_indices = [
        df_products.index[
            df_products["product_id"] == product_id
        ][0]
        for product_id in user_products
    ]

    recommended_index = df_products.index[
        df_products["product_id"] == recommended_product_id
    ][0]

    influence_scores = (
        similarity_matrix[user_indices, recommended_index]
        * user_weights
    )

    max_index = influence_scores.argmax()

    influencing_product = user_data.iloc[
        max_index
    ]["product_name"]

    recommended_product = df_products.iloc[
        recommended_index
    ]["product_name"]

    return (
        f"Recommended because you interacted with "
        f"{influencing_product}, which is similar to "
        f"{recommended_product}."
    )

# -----------------------------

# STREAMLIT WEBSITE INTERFACE

st.title("🛒 SmartCart AI")
st.subheader("AI-Powered Personalized Shopping Assistant")

st.write(
    "SmartCart AI analyzes your previous interactions "
    "and recommends products that match your interests."
)

st.divider()

user_id = st.selectbox(
    "👤 Select User",
    ["U001", "U002", "U003", "U004", "U005"]
)

category = st.selectbox(
    "📂 Select Category",
    ["All", "Laptop", "Smartphone", "Headphones",
     "Keyboard", "Mouse", "Monitor"]
)

budget = st.slider(
    "💰 Maximum Budget (₹)",
    min_value=1000,
    max_value=60000,
    value=60000,
    step=1000
)

top_n = st.slider(
    "🔢 Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("🤖 Get Recommendations"):

    recommendations = recommend_for_user(user_id, 30)

    if category != "All":
        recommendations = recommendations[
            recommendations["category"] == category
        ]

    recommendations = recommendations[
        recommendations["price"] <= budget
    ]

    recommendations = recommendations.head(top_n)

    st.subheader("✨ AI Recommendations")

    if len(recommendations) == 0:

        st.warning(
            "No products match your selected category and budget."
        )

    else:

        for _, product in recommendations.iterrows():

            st.markdown(
                f"### 🛍️ {product['product_name']}"
            )

            st.write(
                f"**Category:** {product['category']}"
            )

            st.write(
                f"**Price:** ₹{product['price']:,}"
            )

            st.write(
                f"**Recommendation Score:** "
                f"{product['recommendation_score']}"
            )

            explanation = explain_recommendation(
                user_id,
                product["product_id"]
            )

            st.info(
                f"💡 {explanation}"
            )

            st.divider()
