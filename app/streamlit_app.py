import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import os
import math
import re
from rapidfuzz import process, fuzz
import random


# =====================================================
# BENGALURU LOCATION → STATIC CENTER COORDINATES
# (Representative, not exact addresses)
# =====================================================
# LOCATION_COORDS = {
#     "Electronic City": (12.8452, 77.6602),
#     "Whitefield": (12.9698, 77.7499),
#     "Yelahanka": (13.1007, 77.5963),
#     "Yeshwanthpur": (13.0285, 77.5400),
#     "Hebbal": (13.0358, 77.5970),
#     "Marathahalli": (12.9562, 77.7019),
#     "KR Puram": (13.0099, 77.6950),
#     "Banashankari": (12.9255, 77.5468),
#     "JP Nagar": (12.9077, 77.5855),
#     "BTM Layout": (12.9166, 77.6101),
#     "Indiranagar": (12.9784, 77.6408),
#     "Koramangala": (12.9352, 77.6245),
#     "HSR Layout": (12.9116, 77.6474),
#     "Rajajinagar": (12.9916, 77.5554),
#     "Malleshwaram": (13.0035, 77.5646),
# }


# =====================================================
# LOCATION CLEANING & STANDARDIZATION
# =====================================================

CANONICAL_AREAS = [
    "whitefield", "electronic city", "koramangala", "btm layout",
    "hsr layout", "jp nagar", "indiranagar", "yelahanka",
    "hebbal", "rajajinagar", "malleshwaram", "bannerghatta",
    "sarjapur", "bellandur", "marathahalli", "mahadevpura",
    "rr nagar", "basavanagudi", "vijayanagar", "yemlur",
    "kengeri", "hosur road", "tumkur road", "nagavara",
    "ramamurthy nagar", "kr puram", "yeshwanthpur"
]

def normalize_location(loc):
    loc = str(loc).lower().strip()
    loc = re.sub(r'\d+', ' ', loc)
    loc = re.sub(r'[^\w\s]', ' ', loc)
    loc = re.sub(r'\s+', ' ', loc)
    return loc.strip()

def extract_parent_area(loc):
    for area in CANONICAL_AREAS:
        if area in loc:
            return area
    return loc

def fuzzy_fix(loc):
    match, score, _ = process.extractOne(
        loc, CANONICAL_AREAS, scorer=fuzz.token_sort_ratio
    )
    return match if score >= 90 else loc

def clean_location_pipeline(loc):
    loc = normalize_location(loc)
    loc = extract_parent_area(loc)
    loc = fuzzy_fix(loc)
    return loc

def get_unique_image(listing_id):
    """
    Assigns ONE unique image per listing.
    Image never changes until new search.
    """

    # Already assigned → reuse
    if listing_id in st.session_state.listing_images:
        return st.session_state.listing_images[listing_id]

    # Exhausted image pool → fallback
    # if st.session_state.image_pointer >= len(st.session_state.image_pool):
    #     return "images/default.jpg"

    if st.session_state.image_pointer >= len(st.session_state.image_pool):
        random.shuffle(st.session_state.image_pool)
        st.session_state.image_pointer = 0


    img = st.session_state.image_pool[st.session_state.image_pointer]
    st.session_state.image_pointer += 1

    st.session_state.listing_images[listing_id] = img
    return img


@st.cache_data
def apply_location_cleaning(df):
    df = df.copy()
    df["location_clean"] = df["location"].apply(clean_location_pipeline)
    return df

@st.cache_data
def load_location_coords():
    return pd.read_csv("outputs/bengaluru_location_coordinates.csv")



# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Housing Affordability Explorer",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
# if "img_index" not in st.session_state:
#     st.session_state.img_index = {}

if "limit" not in st.session_state:
    st.session_state.limit = 5
if "submitted" not in st.session_state:
    st.session_state.submitted = False


# =====================================================
# GLOBAL IMAGE POOL (NO REPEAT, RANDOM)
# =====================================================
def load_all_images():
    folder = "images"
    if not os.path.exists(folder):
        return []
    imgs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and f.lower() != "default.jpg"
    ]
    return imgs


if "image_pool" not in st.session_state:
    st.session_state.image_pool = load_all_images()

if "image_pointer" not in st.session_state:
    st.session_state.image_pointer = 0

if "listing_images" not in st.session_state:
    st.session_state.listing_images = {}



# filtered_df = pd.DataFrame()
# =====================================================
# IMAGE HANDLING (SEQUENTIAL, NO REPEAT)
# =====================================================
# def get_rotating_image(location):
#     folder = location.lower().replace(" ", "_").replace("&", "and").replace("/", "")
#     base_path = os.path.join("images", folder)

#     if not os.path.exists(base_path):
#         return "images/default.jpg"

#     images = sorted([
#         f for f in os.listdir(base_path)
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     ])

#     if not images:
#         return "images/default.jpg"

#     if location not in st.session_state.img_index:
#         st.session_state.img_index[location] = 0

#     idx = st.session_state.img_index[location]
#     st.session_state.img_index[location] += 1

#     return os.path.join(base_path, images[idx % len(images)])

# =====================================================
# LOAD MODEL & DATA
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("models/kmeans_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("outputs/clustered_bengaluru.csv")

model = load_model()
df = load_data()

# df["location_clean"] = df["location"].apply(clean_location_pipeline)
df = apply_location_cleaning(df)

coords_df = load_location_coords()

df = df.merge(
    coords_df,
    on="location_clean",
    how="left"
)

missing = df[df["latitude"].isna()]["location_clean"].nunique()
# st.info(f"📍 Locations without coordinates: {missing}")


df["cluster_label"] = (
    df["cluster"]
    .map(model["cluster_label_map"])
    .astype(str)
    .str.title()
)

# =====================================================
# TITLE
# =====================================================
st.title("🏙 Smart Housing Affordability Explorer")
st.markdown(
    "Unsupervised ML-based **housing affordability clustering** for Bengaluru "
    "using **price, location & infrastructure indicators**."
)

# =====================================================
# USER INPUT
# =====================================================
with st.form("user_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        housing_type = st.selectbox(
            "Housing Category",
            ["Any", "Affordable", "Moderate", "Luxury"]
        )

    with c2:
        size_filter = st.selectbox(
            "House Size",
            ["Any", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5+ BHK"]
        )

    with c3:
        location_filter = st.selectbox(
            "Location",
            ["Any"] + sorted(df["location_clean"].dropna().str.title().unique().tolist())
        )
        

    c4, c5 = st.columns(2)

    with c4:
        income = st.number_input(
            "Your Monthly Income (₹)",
            min_value=10000,
            value=50000,
            step=5000
        )

    with c5:
        savings_amount = st.number_input(
            "Your Current Savings (₹)",
            min_value=0,
            value=0,
            step=50000
        )

    submit = st.form_submit_button("🔍 Explore Housing")

# =====================================================
# RESULTS
# =====================================================
if submit:
    st.session_state.submitted = True


    st.session_state.limit = 5

    random.shuffle(st.session_state.image_pool)
    st.session_state.image_pointer = 0
    st.session_state.listing_images = {}

    # filtered_df = df.copy()

    # if housing_type != "Any":
    #     filtered_df = filtered_df[filtered_df["cluster_label"] == housing_type]

    # if size_filter != "Any":
    #     filtered_df = filtered_df[filtered_df["size"] == size_filter]

    # if location_filter != "Any":
    #     filtered_df = filtered_df[filtered_df["location"] == location_filter]

    # if filtered_df.empty:
    #     st.warning("No housing data found for selected filters.")
    #     st.stop()

# =====================================================
# RESULTS
# =====================================================
if st.session_state.submitted:

    filtered_df = df.copy()

    if housing_type != "Any":
        filtered_df = filtered_df[filtered_df["cluster_label"] == housing_type]

    if size_filter != "Any":
        # filtered_df = filtered_df[filtered_df["size"] == size_filter]
        if size_filter == "5+ BHK":
            filtered_df = filtered_df[
                filtered_df["size"]
                .str.extract(r'(\d+)')[0]
                .astype(float) >= 5
            ]
        else:
            filtered_df = filtered_df[filtered_df["size"] == size_filter]


    if location_filter != "Any":
        filtered_df = filtered_df[filtered_df["location_clean"] == location_filter.lower()]

    if filtered_df.empty:
        st.warning("No housing data found for selected filters.")
        st.stop()

    # =================================================
    # MAP — ONE PIN PER LOCATION
    # =================================================
    st.markdown("---")
    st.subheader("📍 Housing Locations in Bengaluru")
    st.caption("Only locations with verified latitude & longitude are shown on the map.")

    # Legend
    st.markdown(
        "🔵 **Affordable** &nbsp;&nbsp; "
        "🟠 **Moderate** &nbsp;&nbsp; "
        "🔴 **Luxury**"
    )

    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11)

    color_map = {
        "Affordable": "blue",
        "Moderate": "orange",
        "Luxury": "red"
    }

    location_summary = (
        filtered_df
        .groupby(["location_clean", "cluster_label", "latitude", "longitude"])
        .size()
        .reset_index(name="count")
    )



    # for _, row in location_summary.iterrows():
    #     loc = row["location_clean"]

    #     # # if loc not in LOCATION_COORDS:
    #     # #     continue

    #     # # lat, lon = LOCATION_COORDS[loc]
        
    #     # if loc in LOCATION_COORDS:
    #     #     lat, lon = LOCATION_COORDS[loc]
    #     # else:
    #     #     # fallback near Bengaluru center (stable)
    #     #     lat = 12.9716 + (hash(loc) % 20) * 0.002
    #     #     lon = 77.5946 + (hash(loc) % 20) * 0.002

    #     loc_title = loc.title()

    #     if loc_title in LOCATION_COORDS:
    #         lat, lon = LOCATION_COORDS[loc_title]
    #     else:
    #         idx = abs(hash(loc)) % 360
    #         angle = math.radians(idx)
    #         radius = 0.02  # spread radius

    #         lat = 12.9716 + radius * math.cos(angle)
    #         lon = 77.5946 + radius * math.sin(angle)

    #     # else:
    #     #     lat = 12.9716 + (hash(loc) % 20) * 0.002
    #     #     lon = 77.5946 + (hash(loc) % 20) * 0.002


    #     folium.CircleMarker(
    #         location=[lat, lon],
    #         radius=10,
    #         color=color_map.get(row["cluster_label"], "gray"),
    #         fill=True,
    #         fill_opacity=0.85,
    #         popup=(
    #             f"<b>{loc_title}</b><br>"
    #             f"Category: {row['cluster_label']}<br>"
    #             f"Matching Homes: {row['count']}"
    #         )
    #     ).add_to(m)

    for _, row in location_summary.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]

        if pd.isna(lat) or pd.isna(lon):
            continue  # skip unknown

        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color=color_map.get(row["cluster_label"], "gray"),
            fill=True,
            fill_opacity=0.85,
            popup=f"""
            <b>{row['location_clean'].title()}</b><br>
            Category: {row['cluster_label']}<br>
            Homes: {row['count']}
            """
        ).add_to(m)


    st_folium(m, height=550, use_container_width=True)

    # =================================================
    # HOUSING OPTIONS
    # =================================================
    st.markdown("---")
    st.subheader("🏘 Available Housing Options")

    monthly_saving = max(income * 0.40, 1)

    show_df = filtered_df.head(st.session_state.limit)

    for _, area in show_df.iterrows():

        col_img, space, col_info, col_calc = st.columns([3, 0.2, 2, 3])

        # IMAGE
        with col_img:
            listing_id = area.name   # unique & stable per row

            st.image(
                # get_rotating_image(area["location"]),
                get_unique_image(listing_id),
                use_container_width=True
            )

        # DETAILS
        with col_info:
            price = area["price_inr"]
            price_lakhs = price / 100_000

            st.markdown(f"### 📍 {area['location']}")
            st.caption(f"Standardized Area: {area['location_clean'].title()}")
            st.write(f"💰 **Price:** ₹{price_lakhs:.2f} Lakhs (₹{price:,.0f})")
            st.write(f"💰 **Rent:** ₹{price * 0.003:,.0f}")
            st.write(f"🏠 **Type:** {area['size']}")
            st.write(f"📐 **Area:** {area['total_sqft']} sq.ft")
            st.write(f"🛁 **Bath:** {area['bath']} | 🪟 **Balcony:** {area['balcony']}")
            st.write(f"🏢 **Society:** {area['society'] if pd.notna(area['society']) else 'Independent'}")
            st.write(f"📅 **Availability:** {area['availability']}")

            st.markdown("**📍 Connectivity**")
            st.write(f"✈ Airport: {area['airport_distance_km']} km")
            st.write(f"🚆 Railway: {area['railway_station_distance_km']} km")
            st.write(f"🏥 Hospital: {area['nearest_hospital_distance_km']} km")
            st.write(f"🏫 Schools: {area['schools_count']}")
            st.write(f"🏥 Hospitals: {area['hospitals_count']}")

        # AFFORDABILITY
        with col_calc:
            down_payment = price * 0.20
            used_savings = min(savings_amount, down_payment)
            remaining_price = price - down_payment
            months_total = remaining_price / monthly_saving
            amount_for_payment = remaining_price
            years = int(months_total // 12)
            months = int(months_total % 12)
            years_decimal = round(months_total / 12, 1)

            st.markdown("### 💰 Affordability Analysis")

            st.write(f"• **Total Property Price:** ₹{price:,.0f}")
            st.write(f"• **Your Savings:** ₹{savings_amount:,.0f}")
            st.write(f"• **Down Payment (20%)**: ₹{down_payment:,.0f}")
            if savings_amount < down_payment:
                gap = down_payment - savings_amount
                st.warning(
                    f"⚠ Your savings (₹{savings_amount:,.0f}) are **less than** the required down payment.\n\n"
                    f"You need to arrange an additional **₹{gap:,.0f}** "
                    f"(loan or other sources) to register the property in your name."
                )
            else:
                st.success(
                    "✅ Your savings are sufficient to cover the down payment. "
                    "No additional loan required for registration."
                )

            # st.write(f"• **Savings Used for Down Payment:** ₹{used_savings:,.0f}")
            st.write(f"• **Remaining Amount After Down Payment:** ₹{remaining_price:,.0f}")
            # 👉 If savings are MORE than down payment
            if savings_amount > down_payment:
                remaining_saving = savings_amount - down_payment
                adjusted_amount = remaining_price - remaining_saving

                if adjusted_amount < 0:
                    adjusted_amount = 0

                st.write(
                    f"• **Remaining Amount to Pay after using remaining savings:** "
                    f"₹{adjusted_amount:,.0f}"
                )

                # Use adjusted amount for calculation
                amount_for_payment = adjusted_amount
                months_total = amount_for_payment / monthly_saving
                years = int(months_total // 12)
                months = int(months_total % 12)
                years_decimal = round(months_total / 12, 1)

            st.write(f"• **Your Monthly Income:** ₹{income:,.0f}")
            st.write(
                f"• **40% of Monthly Income Used for Payment:** ₹{monthly_saving:,.0f}"
            )


            st.success(
                f"⏱ You will complete the payment in **{int(months_total)} months** "
                f"(≈ **{years_decimal} years**, i.e. {years} years {months} months)"
            )

            # ============================
            # RENT VS BUY ANALYSIS
            # ============================

            monthly_rent = price * 0.003
            rent_percentage = (monthly_rent / income) * 100

            st.markdown("### 🏠 Rent vs Buy Insight")

            st.write(f"• **Estimated Monthly Rent:** ₹{monthly_rent:,.0f}")
            st.write(f"• **Rent as % of Income:** {rent_percentage:.1f}%")

            if rent_percentage <= 30:
                st.success(
                    "✅ Renting this property is **financially comfortable**.\n\n"
                    "Your rent consumes a healthy portion of your income."
                )
            elif rent_percentage <= 40:
                st.warning(
                    "⚠ Renting this property is **manageable but risky**.\n\n"
                    "You may feel financial pressure during emergencies."
                )
            else:
                st.error(
                    "❌ Renting this property is **not advisable**.\n\n"
                    "More than 40% of your income would go towards rent."
                )


            # months_total = remaining_price / monthly_saving
            # years = int(months_total // 12)
            # months = int(months_total % 12)

            # st.markdown("### 💰 Affordability Analysis")
            # st.write(f"• Total Price: ₹{price:,.0f}")
            # st.write(f"• Your Savings: ₹{savings_amount:,.0f}")
            # st.write(f"• Down Payment (20%): ₹{down_payment:,.0f}")
            # st.write(f"• Savings Used: ₹{used_savings:,.0f}")
            # st.write(f"• Remaining Amount: ₹{remaining_price:,.0f}")
            # st.write(f"• Monthly Saving (40%): ₹{monthly_saving:,.0f}")

            # st.success(
            #     f"⏱ Remaining amount payable in **{years} years {months} months**"
            # )

        st.markdown("---")

    # =================================================
    # SHOW MORE
    # =================================================
    if st.session_state.limit < len(filtered_df):
        if st.button("➕ Show More Options"):
            st.session_state.limit += 5

