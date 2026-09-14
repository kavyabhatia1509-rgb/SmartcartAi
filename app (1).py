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

    ranked_indices = final_scores.argsort()[::-1]

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
    category,
    budget,
    top_n,
    purpose,
    priority
):

    recommendations = recommend_for_user(
        user_id,
        30
    )

    # Category filter
    if category != "All":

        recommendations = recommendations[
            recommendations["category"] == category
        ]

    # Budget filter
    recommendations = recommendations[
        recommendations["price"] <= budget
    ]

    # Purpose keywords
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

    selected_purpose_keywords = purpose_keywords.get(
        purpose,
        []
    )

    def purpose_score(row):

        product_index = df_products.index[
            df_products["product_id"]
            == row["product_id"]
        ][0]

        feature_text = (
            df_products.iloc[
                product_index
            ]["features"]
        ).lower()

        return sum(
            1
            for keyword in selected_purpose_keywords
            if keyword in feature_text
        )

    recommendations["purpose_score"] = (
        recommendations.apply(
            purpose_score,
            axis=1
        )
    )

    # Priority keywords
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
            "business"
        ]
    }

    selected_priority_keywords = priority_keywords.get(
        priority,
        []
    )

    def priority_score(row):

        product_index = df_products.index[
            df_products["product_id"]
            == row["product_id"]
        ][0]

        feature_text = (
            df_products.iloc[
                product_index
            ]["features"]
        ).lower()

        return sum(
            1
            for keyword in selected_priority_keywords
            if keyword in feature_text
        )

    recommendations["priority_score"] = (
        recommendations.apply(
            priority_score,
            axis=1
        )
    )

    # Final SmartCart score
    recommendations["smart_score"] = (
        recommendations["recommendation_score"]
        + recommendations["purpose_score"] * 0.5
        + recommendations["priority_score"] * 0.5
    )

    recommendations = recommendations.sort_values(
        by="smart_score",
        ascending=False
    )

    recommendations = recommendations.head(top_n)

    if len(recommendations) == 0:

        return (
            recommendations,
            "No products matched your current "
            "category and budget. Try increasing "
            "your budget or choosing another category."
        )

    response = (
        f"I found {len(recommendations)} "
        f"personalized recommendation(s) based on "
        f"your shopping history, purpose, priority "
        f"and budget."
    )

    return recommendations, response


# ============================================================
# NAVIGATION
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
# HOME
# ============================================================

with home_tab:

    # --------------------------------------------------------
    # HERO SECTION
    # --------------------------------------------------------

    st.title("🛒 SmartCart AI")

    st.subheader(
        "Your Personalized AI Shopping Assistant"
    )

    st.write(
        "Find products that fit your needs, preferences "
        "and budget using personalized machine learning "
        "recommendations."
    )

    st.divider()

    # --------------------------------------------------------
    # SECTION 1
    # --------------------------------------------------------

    st.header("👤 Tell us about your shopping needs")

    st.write(
        "SmartCart AI will use these preferences to "
        "personalize your recommendations."
    )

    st.write("")

    # User + Category

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

    st.write("")

    # Purpose + Priority

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

    st.write("")

    # Budget + Number

    col1, col2 = st.columns(2)

    with col1:

        budget = st.slider(
            "💰 Maximum Budget (₹)",
            min_value=1000,
            max_value=60000,
            value=60000,
            step=1000
        )

        st.caption(
            f"Selected budget: ₹{budget:,}"
        )

    with col2:

        top_n = st.slider(
            "🔢 Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5
        )

        st.caption(
            f"SmartCart AI will show {top_n} recommendation(s)."
        )

    st.write("")
    st.write("")

    # --------------------------------------------------------
    # BIG ACTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🤖 Get My Smart Recommendations",
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

        st.session_state["recommendations"] = (
            recommendations
        )

        st.session_state["recommendation_user"] = (
            user_id
        )

        st.session_state["purpose"] = purpose

        st.session_state["priority"] = priority

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    if "recommendations" in st.session_state:

        recommendations = st.session_state[
            "recommendations"
        ]

        recommendation_user = st.session_state[
            "recommendation_user"
        ]

        selected_purpose = st.session_state[
            "purpose"
        ]

        selected_priority = st.session_state[
            "priority"
        ]

        st.divider()

        st.header("✨ Your AI Recommendations")

        st.info(
            f"🤖 SmartCart AI considered your "
            f"shopping purpose (**{selected_purpose}**), "
            f"priority (**{selected_priority}**), "
            f"shopping history and budget."
        )

        if len(recommendations) == 0:

            st.warning(
                "No products matched your current "
                "requirements. Try increasing your "
                "budget or changing the category."
            )

        else:

            # ------------------------------------------------
            # PRODUCT RESULTS
            # ------------------------------------------------

            for number, (_, product) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                st.subheader(
                    f"{number}. 🛍️ "
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
                        f"🎯 **Shopping Purpose:** "
                        f"{selected_purpose}"
                    )

                    st.write(
                        f"⭐ **Priority:** "
                        f"{selected_priority}"
                    )

                    st.write(
                        f"🎯 **Purpose Match Score:** "
                        f"{product['purpose_score']}"
                    )

                    st.write(
                        f"⭐ **Priority Match Score:** "
                        f"{product['priority_score']}"
                    )

                st.divider()

            # ------------------------------------------------
            # COMPARE PRODUCTS
            # ------------------------------------------------

            if len(recommendations) >= 2:

                st.header(
                    "⚖️ Compare Your Recommendations"
                )

                st.write(
                    "Not sure which product to choose? "
                    "Select two or three recommendations "
                    "and SmartCart AI will compare them."
                )

                product_options = (
                    recommendations[
                        "product_name"
                    ].tolist()
                )

                selected_products = st.multiselect(
                    "🛍️ Choose products to compare",
                    product_options,
                    default=product_options[:2]
                )

                if len(selected_products) >= 2:

                    comparison = recommendations[
                        recommendations[
                            "product_name"
                        ].isin(selected_products)
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
                        "ML Score",
                        "Purpose Match",
                        "Priority Match",
                        "SmartCart Score"
                    ]

                    st.dataframe(
                        comparison,
                        use_container_width=True,
                        hide_index=True
                    )

                    best_product = comparison.loc[
                        comparison[
                            "SmartCart Score"
                        ].idxmax()
                    ]

                    st.success(
                        f"🤖 **SmartCart AI Verdict:** "
                        f"{best_product['Product']} "
                        f"is the strongest match for your "
                        f"selected requirements."
                    )

                else:

                    st.info(
                        "Select at least two products "
                        "to compare."
                    )


# ============================================================
# LEARN MORE
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
            "liked or purchased a product."
        )

    with st.expander(
        "2️⃣ Interaction Weighting"
    ):

        st.write(
            "Different actions have different importance:"
        )

        st.write("👀 Viewed → Weight 1")
        st.write("❤️ Liked → Weight 2")
        st.write("🛒 Purchased → Weight 3")

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
            "products are based on their features."
        )

    with st.expander(
        "5️⃣ Personalized Recommendation"
    ):

        st.write(
            "The recommendation engine uses previous "
            "user interactions to identify products "
            "that may be relevant to the user."
        )

    with st.expander(
        "6️⃣ Shopping Purpose & Priority"
    ):

        st.write(
            "The user can tell SmartCart AI why they "
            "are shopping and what matters most to them. "
            "These preferences influence the final ranking."
        )

    with st.expander(
        "7️⃣ Intelligent Shopping Agent"
    ):

        st.write(
            "The shopping agent combines the ML "
            "recommendations with category, budget, "
            "shopping purpose and priority."
        )

    with st.expander(
        "8️⃣ Product Comparison"
    ):

        st.write(
            "Users can compare recommended products "
            "using price, ML score, purpose match, "
            "priority match and SmartCart Score."
        )

    st.divider()

    st.success(
        "🛒 SmartCart AI = Machine Learning "
        "+ Personalization "
        "+ Intelligent Decision Making "
        "+ Product Comparison"
    )


# ============================================================
# MY ACTIVITY
# ============================================================

with activity_tab:

    st.title("📊 My Shopping Activity")

    st.write(
        "View the previous interactions used by "
        "SmartCart AI to understand user preferences."
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
# HELP & SUPPORT
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
            "priority and budget. Then choose the number "
            "of recommendations and click "
            "**Get My Smart Recommendations**."
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
            "Priority tells the AI what matters most "
            "to you, such as price, performance, "
            "portability, battery or features."
        )

    with st.expander(
        "What is SmartCart Score?"
    ):

        st.write(
            "SmartCart Score combines the original "
            "machine learning recommendation score "
            "with purpose and priority matching."
        )

    with st.expander(
        "Can I compare products?"
    ):

        st.write(
            "Yes. Select two or three recommended products "
            "and SmartCart AI will create a comparison "
            "table and provide an AI verdict."
        )

    with st.expander(
        "Does SmartCart AI use a real shopping account?"
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
