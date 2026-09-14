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
# PRODUCT DATA - 100 PRODUCTS
# ============================================================

products = {

    "product_id": [

        # Laptops 001-020
        "P001", "P002", "P003", "P004", "P005",
        "P006", "P007", "P008", "P009", "P010",
        "P011", "P012", "P013", "P014", "P015",
        "P016", "P017", "P018", "P019", "P020",

        # Smartphones 021-040
        "P021", "P022", "P023", "P024", "P025",
        "P026", "P027", "P028", "P029", "P030",
        "P031", "P032", "P033", "P034", "P035",
        "P036", "P037", "P038", "P039", "P040",

        # Headphones 041-055
        "P041", "P042", "P043", "P044", "P045",
        "P046", "P047", "P048", "P049", "P050",
        "P051", "P052", "P053", "P054", "P055",

        # Keyboards 056-070
        "P056", "P057", "P058", "P059", "P060",
        "P061", "P062", "P063", "P064", "P065",
        "P066", "P067", "P068", "P069", "P070",

        # Mice 071-085
        "P071", "P072", "P073", "P074", "P075",
        "P076", "P077", "P078", "P079", "P080",
        "P081", "P082", "P083", "P084", "P085",

        # Monitors 086-100
        "P086", "P087", "P088", "P089", "P090",
        "P091", "P092", "P093", "P094", "P095",
        "P096", "P097", "P098", "P099", "P100"
    ],

    "product_name": [

        # ---------------- LAPTOPS ----------------

        "Acer Aspire 5",
        "Lenovo IdeaPad Slim 5",
        "HP Pavilion 14",
        "ASUS VivoBook 15",
        "Dell Inspiron 15",
        "Acer Aspire 7",
        "Lenovo IdeaPad 3",
        "HP 15s",
        "ASUS TUF Gaming F15",
        "Dell Vostro 15",
        "Lenovo ThinkBook 15",
        "HP Victus 15",
        "Acer Swift Go 14",
        "ASUS ZenBook 14",
        "Dell Inspiron 14",
        "Lenovo LOQ Gaming",
        "HP Envy x360",
        "Acer Nitro V",
        "ASUS ROG Strix G16",
        "Dell G15 Gaming",

        # ---------------- SMARTPHONES ----------------

        "Samsung Galaxy A55",
        "OnePlus Nord CE",
        "Google Pixel 8a",
        "Redmi Note 14",
        "Samsung Galaxy S24 FE",
        "OnePlus 12R",
        "iPhone 15",
        "Google Pixel 9",
        "Samsung Galaxy S23",
        "Nothing Phone 2a",
        "Realme 13 Pro",
        "Vivo V40",
        "Oppo Reno 12",
        "Motorola Edge 50",
        "Redmi Note 13 Pro",
        "OnePlus Nord 4",
        "Samsung Galaxy A35",
        "iQOO Neo 9",
        "Realme GT 6",
        "Nothing Phone 2",

        # ---------------- HEADPHONES ----------------

        "Sony WH-CH720N",
        "JBL Tune 770NC",
        "Boat Rockerz 550",
        "Anker Soundcore Q20i",
        "Realme Buds Wireless",
        "Sony WH-1000XM5",
        "JBL Live 660NC",
        "Boat Nirvana 751",
        "Sennheiser HD 450BT",
        "Anker Soundcore Q30",
        "OnePlus Bullets Z2",
        "Noise Air Buds Pro",
        "JBL Tune 760NC",
        "Skullcandy Crusher Evo",
        "Sony WH-XB910N",

        # ---------------- KEYBOARDS ----------------

        "Redragon K552 Keyboard",
        "Logitech K380 Keyboard",
        "HP Wireless Keyboard",
        "Cosmic Byte Mechanical Keyboard",
        "Dell Multimedia Keyboard",
        "Logitech K480 Keyboard",
        "HP K1500 Keyboard",
        "Redragon K617 Keyboard",
        "Logitech MX Keys Mini",
        "Keychron K2 Keyboard",
        "Portronics Key5",
        "Dell KB216 Keyboard",
        "HP 330 Wireless Keyboard",
        "Zebronics Zeb-Max Pro",
        "Ant Esports MK3400",

        # ---------------- MICE ----------------

        "Logitech M331 Mouse",
        "HP Wireless Mouse",
        "Dell MS116 Mouse",
        "Redragon Gaming Mouse",
        "Lenovo Wireless Mouse",
        "Logitech M185 Mouse",
        "HP X1000 Mouse",
        "Dell WM126 Mouse",
        "Logitech G102 Gaming Mouse",
        "Redragon M711 Mouse",
        "Razer DeathAdder Essential",
        "Logitech MX Master 3S",
        "HP 150 Wireless Mouse",
        "Portronics Toad Mouse",
        "Zebronics Zeb Transformer",

        # ---------------- MONITORS ----------------

        "LG 24-inch Monitor",
        "Samsung 24-inch Monitor",
        "Acer Nitro Monitor",
        "Lenovo 27-inch Monitor",
        "Dell 27-inch Monitor",
        "LG UltraGear Gaming Monitor",
        "Samsung Odyssey G3",
        "Acer 24-inch IPS Monitor",
        "BenQ 24-inch Monitor",
        "ASUS TUF Gaming Monitor",
        "Dell 24-inch Monitor",
        "HP 24-inch Monitor",
        "Lenovo 24-inch Monitor",
        "LG 27-inch IPS Monitor",
        "Samsung 27-inch Monitor"
    ],

    "category": (
        ["Laptop"] * 20
        + ["Smartphone"] * 20
        + ["Headphones"] * 15
        + ["Keyboard"] * 15
        + ["Mouse"] * 15
        + ["Monitor"] * 15
    ),

    "brand": [

        # Laptops
        "Acer", "Lenovo", "HP", "ASUS", "Dell",
        "Acer", "Lenovo", "HP", "ASUS", "Dell",
        "Lenovo", "HP", "Acer", "ASUS", "Dell",
        "Lenovo", "HP", "Acer", "ASUS", "Dell",

        # Smartphones
        "Samsung", "OnePlus", "Google", "Redmi", "Samsung",
        "OnePlus", "Apple", "Google", "Samsung", "Nothing",
        "Realme", "Vivo", "Oppo", "Motorola", "Redmi",
        "OnePlus", "Samsung", "iQOO", "Realme", "Nothing",

        # Headphones
        "Sony", "JBL", "Boat", "Anker", "Realme",
        "Sony", "JBL", "Boat", "Sennheiser", "Anker",
        "OnePlus", "Noise", "JBL", "Skullcandy", "Sony",

        # Keyboards
        "Redragon", "Logitech", "HP", "Cosmic Byte", "Dell",
        "Logitech", "HP", "Redragon", "Logitech", "Keychron",
        "Portronics", "Dell", "HP", "Zebronics", "Ant Esports",

        # Mice
        "Logitech", "HP", "Dell", "Redragon", "Lenovo",
        "Logitech", "HP", "Dell", "Logitech", "Redragon",
        "Razer", "Logitech", "HP", "Portronics", "Zebronics",

        # Monitors
        "LG", "Samsung", "Acer", "Lenovo", "Dell",
        "LG", "Samsung", "Acer", "BenQ", "ASUS",
        "Dell", "HP", "Lenovo", "LG", "Samsung"
    ],

    "price": [

        # Laptops
        55000, 58000, 62000, 52000, 60000,
        65000, 48000, 45000, 72000, 58000,
        62000, 68000, 70000, 75000, 57000,
        78000, 82000, 75000, 125000, 85000,

        # Smartphones
        40000, 25000, 45000, 18000, 55000,
        42000, 65000, 70000, 52000, 24000,
        28000, 35000, 32000, 30000, 22000,
        30000, 27000, 36000, 42000, 40000,

        # Headphones
        8500, 6500, 3500, 4500, 2500,
        28000, 12000, 5500, 11000, 8000,
        2000, 3500, 7500, 14000, 18000,

        # Keyboards
        2500, 3500, 1800, 3000, 1500,
        3000, 900, 2200, 8500, 7500,
        1800, 1200, 2500, 4000, 2000,

        # Mice
        1200, 1000, 800, 1800, 900,
        1000, 700, 900, 1800, 2200,
        2500, 8500, 1300, 800, 1500,

        # Monitors
        12000, 14000, 18000, 16000, 20000,
        28000, 22000, 14000, 15000, 25000,
        15000, 13000, 12500, 19000, 18000
    ],

    "features": [

        # ---------------- LAPTOP FEATURES ----------------

        "laptop programming student performance productivity",
        "laptop programming student productivity lightweight",
        "laptop programming office productivity student",
        "laptop student office lightweight productivity",
        "laptop programming business performance productivity",
        "laptop programming performance gaming student",
        "laptop student office productivity budget",
        "laptop student office productivity lightweight",
        "laptop gaming performance graphics mechanical",
        "laptop business office productivity performance",
        "laptop business programming productivity performance",
        "laptop gaming performance graphics student",
        "laptop student programming lightweight productivity",
        "laptop premium student office lightweight productivity",
        "laptop programming office performance lightweight",
        "laptop gaming programming performance graphics",
        "laptop student office convertible productivity",
        "laptop gaming performance graphics programming",
        "laptop gaming performance graphics high performance",
        "laptop gaming performance graphics productivity",

        # ---------------- SMARTPHONE FEATURES ----------------

        "smartphone camera display performance battery",
        "smartphone 5g performance battery productivity",
        "smartphone camera ai performance display",
        "smartphone budget battery display performance",
        "smartphone camera performance display premium",
        "smartphone performance gaming battery 5g",
        "smartphone camera performance premium ios",
        "smartphone camera ai performance display premium",
        "smartphone camera performance display premium",
        "smartphone budget camera display battery",
        "smartphone camera portrait battery performance",
        "smartphone camera display battery performance",
        "smartphone camera portrait performance battery",
        "smartphone camera performance battery 5g",
        "smartphone camera display performance budget",
        "smartphone performance battery 5g productivity",
        "smartphone camera battery display 5g",
        "smartphone gaming performance battery 5g",
        "smartphone gaming performance display battery",
        "smartphone camera performance display premium",

        # ---------------- HEADPHONE FEATURES ----------------

        "headphones wireless noise cancellation music",
        "headphones wireless noise cancellation bass",
        "headphones wireless gaming music bass",
        "headphones wireless noise cancellation music",
        "headphones wireless music lightweight",
        "headphones premium wireless noise cancellation music",
        "headphones wireless noise cancellation music premium",
        "headphones wireless noise cancellation bass music",
        "headphones wireless music audio premium",
        "headphones wireless noise cancellation bass",
        "headphones wireless music lightweight battery",
        "headphones wireless noise cancellation music",
        "headphones wireless noise cancellation bass music",
        "headphones wireless bass music gaming",
        "headphones wireless noise cancellation bass premium",

        # ---------------- KEYBOARD FEATURES ----------------

        "keyboard mechanical gaming programming rgb",
        "keyboard wireless compact productivity",
        "keyboard wireless office productivity",
        "keyboard mechanical gaming rgb programming",
        "keyboard office multimedia productivity",
        "keyboard wireless compact productivity office",
        "keyboard office wired productivity typing",
        "keyboard mechanical compact gaming rgb",
        "keyboard wireless premium productivity typing",
        "keyboard mechanical wireless gaming programming",
        "keyboard wireless office productivity compact",
        "keyboard office multimedia typing productivity",
        "keyboard wireless office productivity typing",
        "keyboard mechanical gaming rgb performance",
        "keyboard mechanical gaming programming rgb",

        # ---------------- MOUSE FEATURES ----------------

        "mouse wireless office productivity",
        "mouse wireless office laptop",
        "mouse wired office productivity",
        "mouse gaming performance rgb",
        "mouse wireless laptop productivity",
        "mouse wireless office laptop productivity",
        "mouse wireless office productivity budget",
        "mouse wireless office laptop productivity",
        "mouse gaming performance rgb programming",
        "mouse gaming performance rgb precision",
        "mouse gaming performance ergonomic",
        "mouse wireless premium productivity office",
        "mouse wireless office productivity laptop",
        "mouse wireless office productivity compact",
        "mouse gaming performance rgb precision",

        # ---------------- MONITOR FEATURES ----------------

        "monitor display office productivity",
        "monitor display office entertainment",
        "monitor gaming display performance",
        "monitor large display productivity",
        "monitor large display office productivity",
        "monitor gaming display performance high refresh",
        "monitor gaming display performance high refresh",
        "monitor display office productivity ips",
        "monitor display office productivity color",
        "monitor gaming display performance high refresh",
        "monitor display office productivity",
        "monitor display office productivity",
        "monitor display office productivity",
        "monitor large display productivity ips",
        "monitor large display office entertainment"
    ]
}

df_products = pd.DataFrame(products)


# ============================================================
# USER INTERACTION DATA
# ============================================================

interactions = {

    "user_id": [

        # U001
        "U001", "U001", "U001", "U001", "U001",

        # U002
        "U002", "U002", "U002", "U002", "U002",

        # U003
        "U003", "U003", "U003", "U003", "U003",

        # U004
        "U004", "U004", "U004", "U004", "U004",

        # U005
        "U005", "U005", "U005", "U005", "U005"
    ],

    "product_id": [

        # U001 - laptops / keyboards
        "P001", "P003", "P016", "P057", "P061",

        # U002 - smartphones / headphones
        "P021", "P041", "P042", "P026", "P033",

        # U003 - mice / keyboards
        "P071", "P072", "P075", "P056", "P069",

        # U004 - monitors
        "P086", "P087", "P089", "P090", "P098",

        # U005 - laptops / productivity
        "P002", "P004", "P011", "P058", "P082"
    ],

    "interaction": [

        # U001
        "purchased", "liked", "liked", "viewed", "liked",

        # U002
        "purchased", "liked", "viewed", "liked", "liked",

        # U003
        "purchased", "liked", "liked", "viewed", "liked",

        # U004
        "purchased", "liked", "liked", "viewed", "liked",

        # U005
        "liked", "viewed", "liked", "liked", "purchased"
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
    df_interactions["interaction"]
    .map(interaction_weights)
)


# ============================================================
# MERGE DATA
# ============================================================

df_user_products = pd.merge(
    df_interactions,
    df_products,
    on="product_id"
)


# ============================================================
# TF-IDF MODEL
# ============================================================

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(
    df_products["features"]
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_for_user(user_id, top_n=5):

    user_data = df_user_products[
        df_user_products["user_id"] == user_id
    ]

    user_products = user_data[
        "product_id"
    ].tolist()

    user_weights = user_data[
        "weight"
    ].values

    user_indices = [

        df_products.index[
            df_products["product_id"]
            == product_id
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

    final_scores = weighted_scores.sum(
        axis=0
    )

    ranked_indices = (
        final_scores.argsort()[::-1]
    )

    filtered_indices = [

        index

        for index in ranked_indices

        if df_products.iloc[index][
            "product_id"
        ] not in user_products
    ]

    top_indices = filtered_indices[:top_n]

    recommendations = (
        df_products.iloc[
            top_indices
        ].copy()
    )

    recommendations[
        "recommendation_score"
    ] = [

        round(
            final_scores[index],
            2
        )

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
# EXPLANATION FUNCTION
# ============================================================

def explain_recommendation(
    user_id,
    recommended_product_id
):

    user_data = df_user_products[
        df_user_products["user_id"]
        == user_id
    ]

    user_products = user_data[
        "product_id"
    ].tolist()

    user_weights = user_data[
        "weight"
    ].values

    user_indices = [

        df_products.index[
            df_products["product_id"]
            == product_id
        ][0]

        for product_id in user_products
    ]

    recommended_index = (
        df_products.index[
            df_products["product_id"]
            == recommended_product_id
        ][0]
    )

    influence_scores = (
        similarity_matrix[
            user_indices,
            recommended_index
        ]
        * user_weights
    )

    max_index = influence_scores.argmax()

    influencing_product = (
        user_data.iloc[
            max_index
        ]["product_name"]
    )

    recommended_product = (
        df_products.iloc[
            recommended_index
        ]["product_name"]
    )

    return (
        f"Recommended because you interacted "
        f"with {influencing_product}, which is "
        f"similar to {recommended_product}."
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
        100
    )

    # --------------------------------------------------------
    # CATEGORY FILTER
    # --------------------------------------------------------

    if category != "All":

        recommendations = recommendations[
            recommendations["category"]
            == category
        ]

    # --------------------------------------------------------
    # BUDGET FILTER
    # --------------------------------------------------------

    recommendations = recommendations[
        recommendations["price"]
        <= budget
    ]

    # --------------------------------------------------------
    # PURPOSE KEYWORDS
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

    selected_purpose_keywords = (
        purpose_keywords.get(
            purpose,
            []
        )
    )

    def purpose_score(row):

        product_index = (
            df_products.index[
                df_products["product_id"]
                == row["product_id"]
            ][0]
        )

        feature_text = (
            df_products.iloc[
                product_index
            ]["features"]
        ).lower()

        return sum(
            1
            for keyword
            in selected_purpose_keywords
            if keyword in feature_text
        )

    recommendations[
        "purpose_score"
    ] = recommendations.apply(
        purpose_score,
        axis=1
    )

    # --------------------------------------------------------
    # PRIORITY KEYWORDS
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
            "business"
        ]
    }

    selected_priority_keywords = (
        priority_keywords.get(
            priority,
            []
        )
    )

    def priority_score(row):

        product_index = (
            df_products.index[
                df_products["product_id"]
                == row["product_id"]
            ][0]
        )

        feature_text = (
            df_products.iloc[
                product_index
            ]["features"]
        ).lower()

        return sum(
            1
            for keyword
            in selected_priority_keywords
            if keyword in feature_text
        )

    recommendations[
        "priority_score"
    ] = recommendations.apply(
        priority_score,
        axis=1
    )

    # --------------------------------------------------------
    # FINAL SMART SCORE
    # --------------------------------------------------------

    recommendations[
        "smart_score"
    ] = (

        recommendations[
            "recommendation_score"
        ]

        + recommendations[
            "purpose_score"
        ] * 0.5

        + recommendations[
            "priority_score"
        ] * 0.5
    )

    recommendations = (
        recommendations.sort_values(
            by="smart_score",
            ascending=False
        )
    )

    recommendations = (
        recommendations.head(top_n)
    )

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

    return (
        recommendations,
        response
    )


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

    st.title("🛒 SmartCart AI")

    st.subheader(
        "Your Personalized AI Shopping Assistant"
    )

    st.write(
        "Find products that fit your needs, "
        "preferences and budget using personalized "
        "machine learning recommendations."
    )

    st.divider()

    st.header(
        "👤 Tell us about your shopping needs"
    )

    st.write(
        "SmartCart AI will use these preferences "
        "to personalize your recommendations."
    )

    st.write("")

    # --------------------------------------------------------
    # USER + CATEGORY
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

    st.write("")

    # --------------------------------------------------------
    # PURPOSE + PRIORITY
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

    st.write("")

    # --------------------------------------------------------
    # BUDGET + NUMBER OF RECOMMENDATIONS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        budget = st.slider(
            "💰 Maximum Budget (₹)",
            min_value=1000,
            max_value=125000,
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
            f"SmartCart AI will show "
            f"{top_n} recommendation(s)."
        )

    st.write("")
    st.write("")

    # --------------------------------------------------------
    # RECOMMENDATION BUTTON
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

        st.session_state[
            "recommendations"
        ] = recommendations

        st.session_state[
            "recommendation_user"
        ] = user_id

        st.session_state[
            "purpose"
        ] = purpose

        st.session_state[
            "priority"
        ] = priority

    # --------------------------------------------------------
    # SHOW RESULTS
    # --------------------------------------------------------

    if "recommendations" in st.session_state:

        recommendations = (
            st.session_state[
                "recommendations"
            ]
        )

        recommendation_user = (
            st.session_state[
                "recommendation_user"
            ]
        )

        selected_purpose = (
            st.session_state[
                "purpose"
            ]
        )

        selected_priority = (
            st.session_state[
                "priority"
            ]
        )

        st.divider()

        st.header(
            "✨ Your AI Recommendations"
        )

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
            # PRODUCT CARDS
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

                    st.write(
                        explanation
                    )

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
            # PRODUCT COMPARISON
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

                selected_products = (
                    st.multiselect(
                        "🛍️ Choose products to compare",
                        product_options,
                        default=product_options[:2]
                    )
                )

                if len(selected_products) >= 2:

                    comparison = (
                        recommendations[
                            recommendations[
                                "product_name"
                            ].isin(
                                selected_products
                            )
                        ].copy()
                    )

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

                    best_product = (
                        comparison.loc[
                            comparison[
                                "SmartCart Score"
                            ].idxmax()
                        ]
                    )

                    st.success(
                        f"🤖 **SmartCart AI Verdict:** "
                        f"{best_product['Product']} "
                        f"is the strongest match for "
                        f"your selected requirements."
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

    st.title(
        "📚 How SmartCart AI Works"
    )

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

    st.title(
        "📊 My Shopping Activity"
    )

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
        df_user_products["user_id"]
        == activity_user
    ]

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

    st.title(
        "❓ Help & Support"
    )

    st.write(
        "Find answers to common questions "
        "about SmartCart AI."
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
            "Yes. Select two or three recommended "
            "products and SmartCart AI will create "
            "a comparison table and provide an "
            "AI verdict."
        )

    with st.expander(
        "Does SmartCart AI use a real shopping account?"
    ):

        st.write(
            "No. This academic prototype uses "
            "simulated users, products and "
            "shopping interactions."
        )

    st.divider()

    st.success(
        "🛒 SmartCart AI is an academic prototype "
        "for personalized e-commerce recommendations."
    )
