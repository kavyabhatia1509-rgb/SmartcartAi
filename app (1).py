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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

    /* ==============================
       GLOBAL
       ============================== */

    .stApp {
        background: #f6f7fb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ==============================
       HERO SECTION
       ============================== */

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 55%,
            #374151 100%
        );

        padding: 42px 45px;
        border-radius: 28px;
        margin-bottom: 28px;
        color: white;
        box-shadow: 0 12px 35px rgba(0,0,0,0.12);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        padding: 7px 14px;
        border-radius: 30px;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 18px;
        margin-top: 12px;
        opacity: 0.82;
        max-width: 700px;
        line-height: 1.6;
    }

    .hero-small {
        margin-top: 22px;
        font-size: 14px;
        opacity: 0.7;
    }

    /* ==============================
       SECTION HEADINGS
       ============================== */

    .section-title {
        font-size: 27px;
        font-weight: 750;
        color: #111827;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 20px;
    }

    /* ==============================
       STAT CARDS
       ============================== */

    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
    }

    .stat-icon {
        font-size: 25px;
    }

    .stat-number {
        font-size: 25px;
        font-weight: 750;
        color: #111827;
        margin-top: 4px;
    }

    .stat-label {
        color: #6b7280;
        font-size: 13px;
    }

    /* ==============================
       FILTER PANEL
       ============================== */

    .filter-panel {
        background: white;
        padding: 24px;
        border-radius: 22px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(0,0,0,0.04);
        margin-top: 20px;
        margin-bottom: 28px;
    }

    .filter-title {
        font-size: 19px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 3px;
    }

    .filter-description {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 15px;
    }

    /* ==============================
       PRODUCT CARDS
       ============================== */

    .product-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 22px;
        min-height: 285px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.045);
        transition: 0.2s ease;
        margin-bottom: 15px;
    }

    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(0,0,0,0.08);
    }

    .product-icon {
        font-size: 36px;
        margin-bottom: 12px;
    }

    .product-category {
        color: #6b7280;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
    }

    .product-name {
        font-size: 19px;
        font-weight: 750;
        color: #111827;
        margin-top: 5px;
        margin-bottom: 12px;
    }

    .product-price {
        font-size: 22px;
        font-weight: 800;
        color: #111827;
    }

    .score-pill {
        display: inline-block;
        background: #f0fdf4;
        color: #15803d;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-top: 12px;
    }

    .product-id {
        color: #9ca3af;
        font-size: 11px;
        margin-top: 10px;
    }

    /* ==============================
       AI MESSAGE
       ============================== */

    .ai-message {
        background: white;
        border-left: 4px solid #111827;
        padding: 18px 20px;
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        color: #374151;
    }

    .ai-label {
        font-weight: 750;
        color: #111827;
        margin-bottom: 5px;
    }

    /* ==============================
       EXPLANATION BOX
       ============================== */

    .explanation {
        background: #f9fafb;
        border-radius: 12px;
        padding: 11px 13px;
        margin-top: 13px;
        color: #6b7280;
        font-size: 12px;
        line-height: 1.5;
    }

    /* ==============================
       INFO CARDS
       ============================== */

    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 15px;
        min-height: 155px;
    }

    .info-icon {
        font-size: 28px;
    }

    .info-title {
        font-size: 17px;
        font-weight: 700;
        margin-top: 8px;
        color: #111827;
    }

    .info-text {
        font-size: 13px;
        color: #6b7280;
        line-height: 1.6;
        margin-top: 7px;
    }

    /* ==============================
       FOOTER
       ============================== */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding-top: 30px;
        padding-bottom: 10px;
    }

</style>
""", unsafe_allow_html=True)


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
    df_interactions["interaction"].map(
        interaction_weights
    )
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
# HELPER FUNCTIONS FOR UI
# ============================================================

def get_category_icon(category):

    icons = {
        "Laptop": "💻",
        "Smartphone": "📱",
        "Headphones": "🎧",
        "Keyboard": "⌨️",
        "Mouse": "🖱️",
        "Monitor": "🖥️"
    }

    return icons.get(category, "🛍️")


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-badge">
        ✦ AI-Powered Shopping Intelligence
    </div>

    <div class="hero-title">
        SmartCart AI
    </div>

    <div class="hero-subtitle">
        Your intelligent shopping companion.
        Discover personalized products based on
        your interests, shopping behavior and budget.
    </div>

    <div class="hero-small">
        Machine Learning • Personalization • Intelligent Decision Making
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# QUICK STATS
# ============================================================

stat1, stat2, stat3, stat4 = st.columns(4)

with stat1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">🛍️</div>
        <div class="stat-number">30</div>
        <div class="stat-label">Products</div>
    </div>
    """, unsafe_allow_html=True)

with stat2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">👥</div>
        <div class="stat-number">5</div>
        <div class="stat-label">Sample Users</div>
    </div>
    """, unsafe_allow_html=True)

with stat3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">🧠</div>
        <div class="stat-number">ML</div>
        <div class="stat-label">Recommendation Engine</div>
    </div>
    """, unsafe_allow_html=True)

with stat4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon">⚡</div>
        <div class="stat-number">AI</div>
        <div class="stat-label">Shopping Agent</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# ============================================================
# NAVIGATION
# ============================================================

home_tab, learn_tab, activity_tab, help_tab = st.tabs(
    [
        "🛒 Shop",
        "🧠 How It Works",
        "📊 My Activity",
        "❓ Help"
    ]
)


# ============================================================
# SHOP TAB
# ============================================================

with home_tab:

    st.markdown(
        '<div class="section-title">Find your next favorite product</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Tell SmartCart AI what you are looking for and let the recommendation engine do the work.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FILTER PANEL
    # --------------------------------------------------------

    st.markdown("""
    <div class="filter-panel">

        <div class="filter-title">
            🎯 Personalize your search
        </div>

        <div class="filter-description">
            Your selections help the shopping agent narrow down the most suitable products.
        </div>

    </div>
    """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    with col1:

        user_id = st.selectbox(
            "👤 Shopping Profile",
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
            "📂 Product Category",
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


    col3, col4 = st.columns(2)

    with col3:

        budget = st.slider(
            "💰 Maximum Budget",
            min_value=1000,
            max_value=60000,
            value=60000,
            step=1000,
            format="₹%d"
        )

    with col4:

        top_n = st.slider(
            "🔢 Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5
        )


    st.write("")


    if st.button(
        "✨ Find My Personalized Products",
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


        # ----------------------------------------------------
        # AI RESPONSE
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="ai-message">

                <div class="ai-label">
                    🤖 SmartCart AI
                </div>

                {agent_message}

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="section-title">✨ Recommended for you</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'These products were selected using your interaction history, product similarity, category and budget.'
            '</div>',
            unsafe_allow_html=True
        )


        if len(recommendations) == 0:

            st.warning(
                "No products match your selected category and budget. "
                "Try increasing your budget or selecting another category."
            )

        else:

            # ------------------------------------------------
            # PRODUCT CARDS
            # ------------------------------------------------

            for start in range(
                0,
                len(recommendations),
                3
            ):

                row = recommendations.iloc[
                    start:start + 3
                ]

                columns = st.columns(3)


                for col, (_, product) in zip(
                    columns,
                    row.iterrows()
                ):

                    with col:

                        icon = get_category_icon(
                            product["category"]
                        )

                        st.markdown(
                            f"""
                            <div class="product-card">

                                <div class="product-icon">
                                    {icon}
                                </div>

                                <div class="product-category">
                                    {product["category"]}
                                </div>

                                <div class="product-name">
                                    {product["product_name"]}
                                </div>

                                <div class="product-price">
                                    ₹{product["price"]:,}
                                </div>

                                <div class="score-pill">
                                    ✦ AI Score: {product["recommendation_score"]}
                                </div>

                                <div class="product-id">
                                    Product ID: {product["product_id"]}
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        explanation = (
                            explain_recommendation(
                                user_id,
                                product["product_id"]
                            )
                        )

                        with st.expander(
                            "💡 Why this product?"
                        ):

                            st.write(
                                explanation
                            )


# ============================================================
# HOW IT WORKS TAB
# ============================================================

with learn_tab:

    st.markdown(
        '<div class="section-title">🧠 How SmartCart AI Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A simple look at the technology behind your personalized recommendations.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">

            <div class="info-icon">👤</div>

            <div class="info-title">
                1. User Interaction
            </div>

            <div class="info-text">
                The system records simulated shopping actions
                such as viewing, liking and purchasing products.
            </div>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div class="info-card">

            <div class="info-icon">⚖️</div>

            <div class="info-title">
                2. Interaction Weighting
            </div>

            <div class="info-text">
                Different actions receive different importance.
                Viewed = 1, Liked = 2 and Purchased = 3.
            </div>

        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">

            <div class="info-icon">🔢</div>

            <div class="info-title">
                3. TF-IDF
            </div>

            <div class="info-text">
                TF-IDF converts product feature descriptions
                into numerical representations that the model
                can compare.
            </div>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div class="info-card">

            <div class="info-icon">📐</div>

            <div class="info-title">
                4. Cosine Similarity
            </div>

            <div class="info-text">
                Cosine similarity measures how closely related
                two products are based on their features.
            </div>

        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">

            <div class="info-icon">🎯</div>

            <div class="info-title">
                5. Personalized Score
            </div>

            <div class="info-text">
                Product similarity is combined with the user's
                interaction weights to calculate a personalized
                recommendation score.
            </div>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div class="info-card">

            <div class="info-icon">🤖</div>

            <div class="info-title">
                6. Shopping Agent
            </div>

            <div class="info-text">
                The shopping agent considers the user's category
                and budget, filters unsuitable products and
                presents the best available recommendations.
            </div>

        </div>
        """, unsafe_allow_html=True)


    st.success(
        "SmartCart AI = Machine Learning + Personalization + Intelligent Decision Making"
    )


# ============================================================
# MY ACTIVITY TAB
# ============================================================

with activity_tab:

    st.markdown(
        '<div class="section-title">📊 My Shopping Activity</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore the shopping interactions used by SmartCart AI to understand user preferences.'
        '</div>',
        unsafe_allow_html=True
    )


    activity_user = st.selectbox(
        "👤 Select Shopping Profile",
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


    # Activity summary

    total_interactions = len(
        user_activity
    )

    purchases = len(
        user_activity[
            user_activity["interaction"]
            == "purchased"
        ]
    )

    likes = len(
        user_activity[
            user_activity["interaction"]
            == "liked"
        ]
    )


    stat1, stat2, stat3 = st.columns(3)


    with stat1:

        st.metric(
            "Total Interactions",
            total_interactions
        )


    with stat2:

        st.metric(
            "Products Liked",
            likes
        )


    with stat3:

        st.metric(
            "Purchases",
            purchases
        )


    st.write("")


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
# HELP TAB
# ============================================================

with help_tab:

    st.markdown(
        '<div class="section-title">❓ Help & Support</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Everything you need to understand and use SmartCart AI.'
        '</div>',
        unsafe_allow_html=True
    )


    questions = [

        (
            "🛒",
            "How do I get recommendations?",
            "Select a shopping profile, choose a product category, "
            "set your maximum budget, select the number of recommendations "
            "and click the recommendation button."
        ),

        (
            "⭐",
            "What does the AI Score mean?",
            "The AI Score represents the calculated relevance of a product "
            "based on its similarity to products previously interacted with "
            "by the selected user."
        ),

        (
            "💡",
            "Why did I receive a particular recommendation?",
            "SmartCart AI identifies previous products that influenced "
            "the recommendation and displays an explanation for each result."
        ),

        (
            "🔒",
            "Does this use my real shopping account?",
            "No. This academic prototype uses a simulated dataset containing "
            "sample users, products and shopping interactions."
        )

    ]


    for icon, question, answer in questions:

        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-icon">
                    {icon}
                </div>

                <div class="info-title">
                    {question}
                </div>

                <div class="info-text">
                    {answer}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.success(
        "🛒 SmartCart AI is an academic prototype demonstrating "
        "personalized e-commerce recommendations using machine learning."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    SmartCart AI • Personalized Shopping Assistant
    <br>
    Built as an AI/ML academic project

</div>
""", unsafe_allow_html=True)
