import pandas as pd
import time
import re
from geopy.geocoders import Nominatim
from rapidfuzz import process, fuzz

# =====================================================
# LOCATION CLEANING LOGIC
# (MUST MATCH STREAMLIT APP)
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
        loc, CANONICAL_AREAS,
        scorer=fuzz.token_sort_ratio
    )
    return match if score >= 90 else loc

def clean_location_pipeline(loc):
    loc = normalize_location(loc)
    loc = extract_parent_area(loc)
    loc = fuzzy_fix(loc)
    return loc

# =====================================================
# LOAD DATA
# =====================================================

print("📂 Loading dataset...")
df = pd.read_csv("outputs/clustered_bengaluru.csv")

# CREATE location_clean (THIS IS CRITICAL)
df["location_clean"] = df["location"].apply(clean_location_pipeline)

# UNIQUE LOCATIONS
locations = sorted(df["location_clean"].dropna().unique())
total = len(locations)

print(f"📍 Unique cleaned locations found: {total}")

# =====================================================
# GEOCODING SETUP
# =====================================================

geolocator = Nominatim(
    user_agent="bengaluru-housing-affordability-project"
)

rows = []

# =====================================================
# GEOCODING LOOP (SLOW BUT CORRECT)
# =====================================================

for i, loc in enumerate(locations, start=1):
    print(f"🔎 ({i}/{total}) Geocoding: {loc}")

    try:
        query = f"{loc}, Bengaluru, Karnataka, India"
        geo = geolocator.geocode(query, timeout=5)

        if geo:
            rows.append({
                "location_clean": loc,
                "latitude": geo.latitude,
                "longitude": geo.longitude
            })
        else:
            rows.append({
                "location_clean": loc,
                "latitude": None,
                "longitude": None
            })

        # REQUIRED BY NOMINATIM
        time.sleep(1)

    except Exception as e:
        print(f"⚠ Error for {loc}: {e}")
        rows.append({
            "location_clean": loc,
            "latitude": None,
            "longitude": None
        })

# =====================================================
# SAVE OUTPUT
# =====================================================

coords_df = pd.DataFrame(rows)
coords_df.to_csv(
    "outputs/bengaluru_location_coordinates.csv",
    index=False
)

# =====================================================
# FINAL REPORT
# =====================================================

missing = coords_df["latitude"].isna().sum()

print("\n✅ Geocoding completed successfully")
print(f"📍 Total locations processed: {len(coords_df)}")
print(f"❌ Locations without coordinates: {missing}")
print("📁 Saved to: outputs/bengaluru_location_coordinates.csv")
