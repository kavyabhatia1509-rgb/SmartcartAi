import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartCart AI",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# PRODUCT DATA
# ============================================================

products = {
    "product_id": [
        "P001", "P002", "P003", "P004", "P005",
        "P006", "P007", "P008", "P009", "P010",
        "P011", "P012", "P013", "P014", "P015",
        "P016", "P017", "P018", "P019", "P020",
        "P021", "P022", "P023", "P024", "P025",
        "P026", "P027", "P028", "P029", "P030"
    ],

    "product_name": [
        "Acer Aspire 5",
        "Lenovo IdeaPad Slim 5",
        "HP Pavilion 14",
        "ASUS VivoBook 15",
        "Dell Inspiron 15",
        "Samsung Galaxy A55",
        "OnePlus Nord CE",
        "Google Pixel 8a",
        "Redmi Note 14",
        "Samsung Galaxy S24 FE",
        "Sony WH-CH720N",
        "JBL Tune 770NC",
        "Boat Rockerz 550",
        "Anker Soundcore Q20i",
        "Realme Buds Wireless",
        "Redragon K552 Keyboard",
        "Logitech K380 Keyboard",
        "HP Wireless Keyboard",
        "Cosmic Byte Mechanical Keyboard",
        "Dell Multimedia Keyboard",
        "Logitech M331 Mouse",
        "HP Wireless Mouse",
        "Dell MS116 Mouse",
        "Redragon Gaming Mouse",
        "Lenovo Wireless Mouse",
        "LG 24-inch Monitor",
        "Samsung 24-inch Monitor",
        "Acer Nitro Monitor",
        "Lenovo 27-inch Monitor",
        "Dell 27-inch Monitor"
    ],

    "category": (
        ["Laptop"] * 5
        + ["Smartphone"] * 5
        + ["Headphones"] * 5
        + ["Keyboard"] * 5
        + ["Mouse"] * 5
        + ["Monitor"] * 5
    ),

    "brand": [
        "Acer", "Lenovo", "HP", "ASUS", "Dell",
        "Samsung", "OnePlus", "Google", "Redmi", "Samsung",
        "Sony", "JBL", "Boat", "Anker", "Realme",
        "Redragon", "Logitech", "HP", "Cosmic Byte", "Dell",
        "Logitech", "HP", "Dell", "Redragon", "Lenovo",
        "LG", "Samsung", "Acer", "Lenovo", "Dell"
    ],

    "price": [
        55000, 58000, 62000, 52000, 60000,
        40000, 25000, 45000, 18000, 55000,
        8500, 6500, 3500, 4500, 2500,
        2500, 3500, 1800, 3000, 1500,
        1200, 1000, 800, 1800, 900,
        12000, 14000, 18000, 16000, 20000
    ],

    "features": [
        "laptop programming student performance",
        "laptop programming student productivity",
        "laptop programming office productivity",
        "laptop student office lightweight",
        "laptop programming business performance",
        "smartphone camera display performance",
        "smartphone 5g performance battery",
        "smartphone camera ai performance",
        "smartphone budget battery display",
        "smartphone camera performance display",
        "headphones wireless noise cancellation music",
        "headphones wireless noise cancellation bass",
        "headphones wireless gaming music bass",
        "headphones wireless noise cancellation music",
        "headphones wireless music lightweight",
        "keyboard mechanical gaming programming",
        "keyboard wireless compact productivity",
        "keyboard wireless office productivity",
        "keyboard mechanical gaming rgb",
        "keyboard office multimedia productivity",
        "mouse wireless office productivity",
        "mouse wireless office laptop",
        "mouse wired office productivity",
        "mouse gaming performance rgb",
        "mouse wireless laptop productivity",
        "monitor display office productivity",
        "monitor display office entertainment",
        "monitor gaming display performance",
        "monitor large display productivity",
        "monitor large display office productivity"
    ]
}

df_products = pd.DataFrame(products)


# ============================================================
# USER INTERACTION DATA
# ============================================================

interactions = {
    "user_id": [
        "U001", "U001", "U001",
        "U002", "U002", "U002",
        "U003", "U003", "U003",
        "U004", "U004", "U004",
        "U005", "U005", "U005"
    ],

    "product_id": [
        "P001", "P003", "P016",
        "P006", "P011", "P012",
        "P021", "P022", "P025",
        "P026", "P027", "P029",
        "P002", "P004", "P018"
    ],

    "interaction": [
        "purchased", "liked", "liked",
        "purchased", "liked", "viewed",
        "purchased", "liked", "liked",
        "purchased", "liked", "viewed",
        "liked", "viewed", "liked"
    ]
}

df_interactions = pd.DataFrame(interactions)


# ============================================================
# INTERACTION WEIGHTS
# ============================================================

interaction_weights = {
    "viewed": 1,
    "liked": 2,
    "purchased": 3
}

df_interactions["weight"] = (
    df_interactions["interaction"].map(interaction_weights)
)


df_user_products = pd.merge(
    df_interactions,
    df_products,
    on="product_id"
)


# ============================================================
# ML RECOMMENDATION ENGINE
# ============================================================

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(
    df_products["features"]
)

similarity_matrix = cosine_similarity(
    tfidf_matrix
)


def recommend_for_user(user_id, top_n=5):

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

    user_similarity_scores = (
        similarity_matrix[user_indices]
    )

    weighted_scores = (
        user_similarity_scores
        * user_weights[:, None]
    )

    final_scores = weighted_scores.sum(axis=0)

    ranked_indices = (
        final_scores.argsort()[::-1]
    )

    filtered_indices = [
        index
        for index in ranked_indices
        if df_products.iloc[index]["product_id"]
        not in user_products
    ]

    top_indices = filtered_indices[:top_n]

    recommendations = df_products.iloc[
        top_indices
    ].copy()

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


# ============================================================
# RECOMMENDATION EXPLANATION
# ============================================================

def explain_recommendation(
    user_id,
    recommended_product_id
):

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
        df_products["product_id"]
        == recommended_product_id
    ][0]

    influence_scores = (
        similarity_matrix[
            user_indices,
            recommended_index
        ]
        * user_weights
    )

    max_index = influence_scores.argmax()

    influencing_product = (
        user_data.iloc[max_index]["product_name"]
    )

    recommended_product = (
        df_products.iloc[
            recommended_index
        ]["product_name"]
    )

    return (
        f"Recommended because you interacted with "
        f"{influencing_product}, which is similar to "
        f"{recommended_product}."
    )


# ============================================================
# SMART SHOPPING AGENT
# ============================================================

def shopping_agent(
    user_id,
    category="All",
    budget=60000,
    top_n=5
):

    recommendations = recommend_for_user(
        user_id,
        30
    )

    if category != "All":

        recommendations = recommendations[
            recommendations["category"]
            == category
        ]

    recommendations = recommendations[
        recommendations["price"] <= budget
    ]

    recommendations = recommendations.head(
        top_n
    )

    if len(recommendations) == 0:

        return (
            recommendations,
            f"I couldn't find suitable "
            f"{category.lower()} products "
            f"within your budget of "
            f"₹{budget:,}."
        )

    response = (
        f"I found {len(recommendations)} "
        f"personalized recommendation(s) "
        f"based on your previous shopping "
        f"interactions, selected category, "
        f"and budget."
    )

    return recommendations, response


# ============================================================
# NAVIGATION TABS
# ============================================================

home_tab, learn_tab, activity_tab, help_tab = st.tabs(
    [
        "🏠 Home",
        "📚 Learn More",
        "📊 My Activity",
        "❓ Help & Support"
    ]
)


# ============================================================
# HOME TAB
# ============================================================

with home_tab:

    st.title("🛒 SmartCart AI")

    st.subheader(
        "AI-Powered Personalized Shopping Assistant"
    )

    st.write(
        "SmartCart AI analyzes your previous "
        "shopping interactions and recommends "
        "products that match your interests."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        user_id = st.selectbox(
            "👤 Select User",
            [
                "U001",
                "U002",
                "U003",
                "U004",
                "U005"
            ]
        )

    with col2:

        category = st.selectbox(
            "📂 Select Category",
            [
                "All",
                "Laptop",
                "Smartphone",
                "Headphones",
                "Keyboard",
                "Mouse",
                "Monitor"
            ]
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

    st.write("")

    if st.button(
        "🤖 Get Recommendations",
        use_container_width=True
    ):

        recommendations, agent_message = (
            shopping_agent(
                user_id,
                category,
                budget,
                top_n
            )
        )

        st.info(
            "🤖 " + agent_message
        )

        st.subheader(
            "✨ AI Recommendations"
        )

        if len(recommendations) == 0:

            st.warning(
                "No products match your "
                "selected category and budget."
            )

        else:

            for _, product in (
                recommendations.iterrows()
            ):

                st.markdown(
                    f"### 🛍️ "
                    f"{product['product_name']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write(
                        f"**Category:** "
                        f"{product['category']}"
                    )

                with col2:

                    st.write(
                        f"**Price:** "
                        f"₹{product['price']:,}"
                    )

                with col3:

                    st.write(
                        f"**AI Score:** "
                        f"{product['recommendation_score']}"
                    )

                explanation = (
                    explain_recommendation(
                        user_id,
                        product["product_id"]
                    )
                )

                st.info(
                    f"💡 {explanation}"
                )

                st.divider()


# ============================================================
# LEARN MORE TAB
# ============================================================

with learn_tab:

    st.title("📚 How SmartCart AI Works")

    st.write(
        "SmartCart AI combines machine learning "
        "with an intelligent decision-making layer "
        "to create personalized recommendations."
    )

    st.subheader("1️⃣ User Interaction Data")

    st.write(
        "The system records whether a user viewed, "
        "liked, or purchased a product."
    )

    st.subheader("2️⃣ Interaction Weighting")

    st.write(
        "Different actions have different importance:"
    )

    st.write(
        "👀 Viewed → Weight 1\n\n"
        "❤️ Liked → Weight 2\n\n"
        "🛒 Purchased → Weight 3"
    )

    st.subheader("3️⃣ TF-IDF")

    st.write(
        "TF-IDF converts product features into "
        "numerical vectors so that the system "
        "can compare products."
    )

    st.subheader("4️⃣ Cosine Similarity")

    st.write(
        "Cosine similarity measures how similar "
        "two products are based on their features."
    )

    st.subheader("5️⃣ Personalized Recommendation")

    st.write(
        "The system combines product similarity "
        "with the user's interaction weights to "
        "calculate a personalized recommendation score."
    )

    st.subheader("6️⃣ Intelligent Shopping Agent")

    st.write(
        "The shopping agent acts as a decision-making "
        "layer. It takes the user's selected category "
        "and budget, gets personalized recommendations "
        "from the ML model, filters unsuitable products, "
        "and returns the best available choices."
    )

    st.success(
        "SmartCart AI = Machine Learning "
        "+ Personalization + Intelligent Decision Making"
    )


# ============================================================
# MY ACTIVITY TAB
# ============================================================

with activity_tab:

    st.title("📊 My Shopping Activity")

    activity_user = st.selectbox(
        "Select User to View Activity",
        [
            "U001",
            "U002",
            "U003",
            "U004",
            "U005"
        ],
        key="activity_user"
    )

    user_activity = df_user_products[
        df_user_products["user_id"]
        == activity_user
    ]

    st.write(
        "Previous interactions for "
        f"**{activity_user}**:"
    )

    activity_display = user_activity[
        [
            "product_name",
            "category",
            "price",
            "interaction",
            "weight"
        ]
    ].copy()

    activity_display.columns = [
        "Product",
        "Category",
        "Price",
        "Interaction",
        "Weight"
    ]

    st.dataframe(
        activity_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# HELP & SUPPORT TAB
# ============================================================

with help_tab:

    st.title("❓ Help & Support")

    st.subheader(
        "How do I get recommendations?"
    )

    st.write(
        "1. Select a user.\n"
        "2. Choose a product category.\n"
        "3. Set your maximum budget.\n"
        "4. Choose how many recommendations "
        "you want.\n"
        "5. Click **Get Recommendations**."
    )

    st.subheader(
        "What does the AI Score mean?"
    )

    st.write(
        "The AI Score represents the calculated "
        "relevance of a product based on similarity "
        "to products the user previously interacted with."
    )

    st.subheader(
        "Why did I receive a particular recommendation?"
    )

    st.write(
        "SmartCart AI provides an explanation showing "
        "which previous interaction influenced the "
        "recommendation."
    )

    st.subheader(
        "Does the system use my real shopping account?"
    )

    st.write(
        "No. This academic prototype uses a "
        "sample dataset containing simulated "
        "users, products, and interactions."
    )

    st.success(
        "🛒 SmartCart AI is a prototype developed "
        "for demonstrating personalized "
        "e-commerce recommendations."
    )
