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
    top_n=5,
    purpose="General Use",
    priority="Overall Quality"
):

    recommendations = recommend_for_user(
        user_id,
        30
    )

    # --------------------------------------------------------
    # CATEGORY FILTER
    # --------------------------------------------------------

    if category != "All":

        recommendations = recommendations[
            recommendations["category"] == category
        ]

    # --------------------------------------------------------
    # BUDGET FILTER
    # --------------------------------------------------------

    recommendations = recommendations[
        recommendations["price"] <= budget
    ]

    # --------------------------------------------------------
    # PURPOSE BOOST
    # --------------------------------------------------------

    purpose_keywords = {

        "Study / College": [
            "student",
            "programming",
            "productivity",
            "office",
            "lightweight"
        ],

        "Work / Productivity": [
            "productivity",
            "office",
            "business"
        ],

        "Gaming": [
            "gaming",
            "performance",
            "mechanical",
            "rgb"
        ],

        "Entertainment": [
            "music",
            "camera",
            "display",
            "entertainment",
            "bass"
        ],

        "General Use": []
    }

    selected_keywords = purpose_keywords.get(
        purpose,
        []
    )

    if selected_keywords:

        def calculate_purpose_score(row):

            text = (
                str(row["product_name"])
                + " "
                + str(row["category"])
            ).lower()

            product_index = df_products.index[
                df_products["product_id"]
                == row["product_id"]
            ][0]

            feature_text = (
                df_products.iloc[
                    product_index
                ]["features"]
            ).lower()

            combined_text = text + " " + feature_text

            matches = sum(
                1
                for keyword in selected_keywords
                if keyword in combined_text
            )

            return matches

        recommendations["purpose_score"] = (
            recommendations.apply(
                calculate_purpose_score,
                axis=1
            )
        )

    else:

        recommendations["purpose_score"] = 0

    # --------------------------------------------------------
    # PRIORITY BOOST
    # --------------------------------------------------------

    priority_keywords = {

        "Price": [
            "budget",
            "wireless",
            "office"
        ],

        "Performance": [
            "performance",
            "gaming",
            "5g",
            "mechanical"
        ],

        "Portability": [
            "lightweight",
            "compact",
            "wireless"
        ],

        "Battery": [
            "battery",
            "wireless"
        ],

        "Features": [
            "camera",
            "display",
            "noise cancellation",
            "rgb",
            "ai"
        ],

        "Overall Quality": [
            "performance",
            "productivity",
            "quality"
        ]
    }

    selected_priority_keywords = priority_keywords.get(
        priority,
        []
    )

    def calculate_priority_score(row):

        product_index = df_products.index[
            df_products["product_id"]
            == row["product_id"]
        ][0]

        feature_text = (
            df_products.iloc[
                product_index
            ]["features"]
        ).lower()

        matches = sum(
            1
            for keyword in selected_priority_keywords
            if keyword in feature_text
        )

        return matches

    recommendations["priority_score"] = (
        recommendations.apply(
            calculate_priority_score,
            axis=1
        )
    )

    # --------------------------------------------------------
    # FINAL SMART SCORE
    # --------------------------------------------------------

    recommendations["smart_score"] = (
        recommendations["recommendation_score"]
        + recommendations["purpose_score"] * 0.5
        + recommendations["priority_score"] * 0.5
    )

    recommendations = recommendations.sort_values(
        by="smart_score",
        ascending=False
    )

    recommendations = recommendations.head(
        top_n
    )

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

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
        f"based on your shopping history, "
        f"shopping purpose, priority and budget."
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
        "Your Personalized AI Shopping Assistant"
    )

    st.write(
        "Tell SmartCart AI what you are looking for, "
        "what matters most to you, and your budget. "
        "The AI will find the most suitable products."
    )

    st.divider()

    # --------------------------------------------------------
    # USER AND CATEGORY
    # --------------------------------------------------------

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
            "📂 What are you shopping for?",
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

    # --------------------------------------------------------
    # PURPOSE AND PRIORITY
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        purpose = st.selectbox(
            "🎯 What is your shopping purpose?",
            [
                "Study / College",
                "Work / Productivity",
                "Gaming",
                "Entertainment",
                "General Use"
            ]
        )

    with col2:

        priority = st.selectbox(
            "⭐ What matters most to you?",
            [
                "Overall Quality",
                "Price",
                "Performance",
                "Portability",
                "Battery",
                "Features"
            ]
        )

    # --------------------------------------------------------
    # BUDGET AND NUMBER OF RECOMMENDATIONS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        budget = st.slider(
            "💰 Maximum Budget (₹)",
            min_value=1000,
            max_value=60000,
            value=60000,
            step=1000
        )

    with col2:

        top_n = st.slider(
            "🔢 Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5
        )

    st.write("")

    # --------------------------------------------------------
    # GET RECOMMENDATIONS
    # --------------------------------------------------------

    if st.button(
        "🤖 Get Smart Recommendations",
        use_container_width=True
    ):

        recommendations, agent_message = (
            shopping_agent(
                user_id,
                category,
                budget,
                top_n,
                purpose,
                priority
            )
        )

        st.session_state["recommendations"] = recommendations
        st.session_state["recommendation_user"] = user_id

        st.success(
            "🤖 " + agent_message
        )

    # --------------------------------------------------------
    # DISPLAY RECOMMENDATIONS
    # --------------------------------------------------------

    if "recommendations" in st.session_state:

        recommendations = st.session_state[
            "recommendations"
        ]

        recommendation_user = st.session_state[
            "recommendation_user"
        ]

        st.divider()

        st.subheader(
            "✨ AI Recommended Products"
        )

        if len(recommendations) == 0:

            st.warning(
                "No products match your requirements. "
                "Try increasing your budget or changing "
                "your category."
            )

        else:

            for number, (_, product) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                st.markdown(
                    f"### {number}. 🛍️ "
                    f"{product['product_name']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "📂 Category",
                        product["category"]
                    )

                with col2:

                    st.metric(
                        "💰 Price",
                        f"₹{product['price']:,}"
                    )

                with col3:

                    st.metric(
                        "⭐ SmartCart Score",
                        round(
                            product["smart_score"],
                            2
                        )
                    )

                with st.expander(
                    "💡 Why was this recommended?"
                ):

                    explanation = (
                        explain_recommendation(
                            recommendation_user,
                            product["product_id"]
                        )
                    )

                    st.write(explanation)

                    st.write(
                        f"🎯 **Purpose:** {purpose}"
                    )

                    st.write(
                        f"⭐ **Priority:** {priority}"
                    )

                st.divider()

        # ----------------------------------------------------
        # COMPARISON SECTION
        # ----------------------------------------------------

        if len(recommendations) >= 2:

            st.subheader(
                "⚖️ Compare Recommended Products"
            )

            st.write(
                "Select products from the AI recommendations "
                "to compare them before making your decision."
            )

            product_options = (
                recommendations["product_name"]
                .tolist()
            )

            selected_products = st.multiselect(
                "🛍️ Select 2 or 3 products to compare",
                product_options,
                default=product_options[:2]
            )

            if len(selected_products) >= 2:

                comparison = recommendations[
                    recommendations["product_name"].isin(
                        selected_products
                    )
                ].copy()

                comparison = comparison[
                    [
                        "product_name",
                        "category",
                        "price",
                        "recommendation_score",
                        "purpose_score",
                        "priority_score",
                        "smart_score"
                    ]
                ]

                comparison.columns = [
                    "Product",
                    "Category",
                    "Price",
                    "ML Recommendation Score",
                    "Purpose Match",
                    "Priority Match",
                    "SmartCart Score"
                ]

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # AI VERDICT
                # ------------------------------------------------

                best_product = comparison.loc[
                    comparison["SmartCart Score"].idxmax()
                ]

                st.success(
                    f"🤖 **SmartCart AI Verdict:** "
                    f"{best_product['Product']} "
                    f"is the strongest match based on "
                    f"your selected purpose, priority, "
                    f"shopping history and budget."
                )

            else:

                st.info(
                    "Please select at least 2 products "
                    "to compare."
                )


# ============================================================
# LEARN MORE TAB
# ============================================================

with learn_tab:

    st.title("📚 How SmartCart AI Works")

    st.write(
        "SmartCart AI combines machine learning, "
        "personalization and intelligent decision-making "
        "to recommend suitable products."
    )

    with st.expander(
        "1️⃣ User Interaction Data",
        expanded=True
    ):

        st.write(
            "The system records whether a user viewed, "
            "liked, or purchased a product."
        )

    with st.expander(
        "2️⃣ Interaction Weighting"
    ):

        st.write(
            "Different actions have different importance."
        )

        st.write(
            "👀 Viewed → Weight 1"
        )

        st.write(
            "❤️ Liked → Weight 2"
        )

        st.write(
            "🛒 Purchased → Weight 3"
        )

    with st.expander(
        "3️⃣ TF-IDF"
    ):

        st.write(
            "TF-IDF converts product feature descriptions "
            "into numerical vectors."
        )

    with st.expander(
        "4️⃣ Cosine Similarity"
    ):

        st.write(
            "Cosine similarity measures how similar "
            "different products are based on their features."
        )

    with st.expander(
        "5️⃣ Personalization"
    ):

        st.write(
            "The system uses the user's previous "
            "interactions to understand their interests."
        )

    with st.expander(
        "6️⃣ Shopping Purpose & Priority"
    ):

        st.write(
            "The user can specify why they are shopping "
            "and what matters most to them. These preferences "
            "help the system adjust the recommendation ranking."
        )

    with st.expander(
        "7️⃣ Intelligent Shopping Agent"
    ):

        st.write(
            "The shopping agent combines the recommendation "
            "model with category, budget, purpose and priority "
            "to select suitable products."
        )

    with st.expander(
        "8️⃣ Product Comparison"
    ):

        st.write(
            "Users can compare recommended products using "
            "their price, recommendation score, purpose match, "
            "priority match and overall SmartCart score."
        )

    st.divider()

    st.success(
        "🛒 SmartCart AI = Machine Learning "
        "+ Personalization "
        "+ Intelligent Decision Making "
        "+ Product Comparison"
    )


# ============================================================
# MY ACTIVITY TAB
# ============================================================

with activity_tab:

    st.title("📊 My Shopping Activity")

    st.caption(
        "Previous interactions used by SmartCart AI "
        "to understand user preferences."
    )

    activity_user = st.selectbox(
        "👤 Select User",
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
        df_user_products["user_id"] == activity_user
    ]

    total_interactions = len(user_activity)

    purchases = len(
        user_activity[
            user_activity["interaction"] == "purchased"
        ]
    )

    likes = len(
        user_activity[
            user_activity["interaction"] == "liked"
        ]
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📊 Total Interactions",
            total_interactions
        )

    with col2:

        st.metric(
            "❤️ Liked",
            likes
        )

    with col3:

        st.metric(
            "🛒 Purchased",
            purchases
        )

    st.divider()

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

    st.write(
        "Find answers to common questions about SmartCart AI."
    )

    with st.expander(
        "How do I get recommendations?"
    ):

        st.write(
            "Select a user, category, shopping purpose, "
            "priority and maximum budget. Then choose how "
            "many recommendations you want and click "
            "**Get Smart Recommendations**."
        )

    with st.expander(
        "What is Shopping Purpose?"
    ):

        st.write(
            "Shopping Purpose tells the AI why you need "
            "the product, such as study, work, gaming "
            "or entertainment."
        )

    with st.expander(
        "What is Priority?"
    ):

        st.write(
            "Priority tells the AI what matters most to "
            "you, such as price, performance, portability, "
            "battery or features."
        )

    with st.expander(
        "What is the SmartCart Score?"
    ):

        st.write(
            "The SmartCart Score combines the original "
            "ML recommendation score with purpose and "
            "priority matching."
        )

    with st.expander(
        "Can I compare products?"
    ):

        st.write(
            "Yes. After receiving recommendations, select "
            "two or three products and SmartCart AI will "
            "create a comparison table and provide an AI verdict."
        )

    with st.expander(
        "Does the system use a real shopping account?"
    ):

        st.write(
            "No. This academic prototype uses simulated "
            "users, products and shopping interactions."
        )

    st.divider()

    st.success(
        "🛒 SmartCart AI is an academic prototype "
        "for personalized e-commerce recommendations."
    )
