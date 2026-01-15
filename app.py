import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# ==========================================
# 0. SESSION STATE
# ==========================================
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

# ==========================================
# 1. DATA & MODEL
# ==========================================
@st.cache_data
def load_data():
    # We don't need nrows anymore because the file is already small
    df = pd.read_csv('recipes_small.csv')
    df = df.dropna(subset=['name', 'ingredients', 'steps'])
    df['name'] = df['name'].astype(str)
    df['ingredients_list'] = df['ingredients'].apply(ast.literal_eval)
    df['nutrition_list'] = df['nutrition'].apply(ast.literal_eval)
    df['steps_list'] = df['steps'].apply(ast.literal_eval)
    df['search_text'] = df['name'] + " " + df['ingredients_list'].apply(lambda x: " ".join(x))
    return df

@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    recipe_matrix = vectorizer.fit_transform(df['search_text'])
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(recipe_matrix)
    df['cluster'] = kmeans.labels_
    return vectorizer, recipe_matrix, df

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_cuisine_type(ingredients):
    ing_string = " ".join(ingredients).lower()
    if 'soy sauce' in ing_string or 'ginger' in ing_string: return 'Asian'
    elif 'pasta' in ing_string or 'basil' in ing_string: return 'Italian'
    elif 'taco' in ing_string or 'beans' in ing_string: return 'Mexican'
    elif 'curry' in ing_string or 'masala' in ing_string: return 'Indian'
    else: return 'General'

def calculate_missing(user_items, recipe_items):
    user_set = set(x.strip().lower() for x in user_items)
    missing = []
    for r_item in recipe_items:
        r_item_clean = r_item.lower()
        if not any(u_item in r_item_clean for u_item in user_set):
            missing.append(r_item)
    return missing

def get_cluster_name(cluster_id, df):
    cluster_data = df[df['cluster'] == cluster_id]
    all_ingredients = [ing for sublist in cluster_data['ingredients_list'] for ing in sublist]
    if all_ingredients:
        from collections import Counter
        most_common = Counter(all_ingredients).most_common(2)
        return f"{most_common[0][0].title()} & {most_common[1][0].title()} Style"
    return "General Style"

def load_specific_recipe(recipe_series):
    st.session_state.search_results = [recipe_series]
    st.session_state.search_performed = True

# ==========================================
# 3. DEEP FOREST UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI Chef", page_icon="🍳", layout="wide")

# DEEP FOREST PALETTE
COLOR_BG = "#0d1110"         # Very Dark Green/Black (Main BG)
COLOR_CARD = "#16201a"       # Dark Jungle Green (Cards)
COLOR_PRIMARY = "#2E7D32"    # Primary Green (Buttons)
COLOR_ACCENT = "#43A047"     # Brighter Green (Borders/Text Pop)
COLOR_TEXT_MAIN = "#E8F5E9"  # Off-White/Mint (Main Text)
COLOR_TEXT_SEC = "#A5D6A7"   # Muted Green (Secondary Text)

st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background-color: {COLOR_BG};
    }}
    
    /* Text Colors */
    h1, h2, h3, p, li, .stMarkdown, label {{
        color: {COLOR_TEXT_MAIN} !important;
    }}
    
    /* === SIDEBAR INPUTS (Dark Glass) === */
    .stTextInput input {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: {COLOR_TEXT_MAIN} !important;
        border: 1px solid {COLOR_PRIMARY} !important;
        border-radius: 8px;
    }}
    .stTextInput input:focus {{
        border: 1px solid {COLOR_ACCENT} !important;
        box-shadow: 0 0 8px {COLOR_PRIMARY};
    }}
    
    /* === TITLE STYLING === */
    .recipe-title {{
        font-size: 28px;
        font-weight: 800;
        color: {COLOR_ACCENT} !important;
        margin-bottom: 5px;
    }}
    
    /* === CARD STYLING === */
    .recipe-card {{
        background-color: {COLOR_CARD};
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        border: 1px solid #1e3326;
        margin-bottom: 20px;
    }}
    
    /* === BADGES === */
    .badge {{
        background-color: {COLOR_PRIMARY};
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 5px;
        display: inline-block;
        margin-bottom: 5px;
    }}
    .badge-macro {{
        background-color: rgba(255,255,255,0.05);
        border: 1px solid {COLOR_PRIMARY};
        color: {COLOR_TEXT_SEC};
    }}
    
    /* === MISSING ALERT (Dark Red for Dark Mode) === */
    .missing-alert {{
        background-color: rgba(60, 20, 20, 0.6); 
        border: 1px solid #8B0000;
        color: #ffcccc !important;
        padding: 10px;
        border-radius: 8px;
        font-size: 14px;
        margin: 15px 0;
    }}
    
    /* === INGREDIENT CHIPS === */
    .ing-chip {{
        display: inline-block;
        background-color: rgba(46, 125, 50, 0.1); /* Subtle Green Tint */
        border: 1px solid #1e3326;
        color: {COLOR_TEXT_SEC};
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13px;
        margin: 3px;
    }}
    
    /* === BUTTONS === */
    div.stButton > button {{
        background-color: {COLOR_PRIMARY};
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    div.stButton > button:hover {{
        background-color: {COLOR_ACCENT};
        color: white;
        transform: scale(1.02);
    }}
    
    /* === NEON DIVIDER (Subtle Green) === */
    .neon-divider {{
        height: 1px;
        background: linear-gradient(90deg, {COLOR_BG}, {COLOR_PRIMARY}, {COLOR_BG});
        margin: 40px 0;
        opacity: 0.6;
        border: none;
    }}
    
    /* Expander Styling */
    .streamlit-expanderHeader {{
        background-color: {COLOR_CARD};
        color: {COLOR_TEXT_MAIN} !important;
        border-radius: 5px;
        border: 1px solid #1e3326;
    }}
    .streamlit-expanderContent {{
        background-color: rgba(0,0,0,0.2);
        border-radius: 0 0 5px 5px;
        border: 1px solid #1e3326;
        border-top: none;
        color: {COLOR_TEXT_SEC};
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. APP LOGIC
# ==========================================

# Sidebar
with st.sidebar:
    st.markdown(f"<h1 style='color:{COLOR_ACCENT} !important;'>🍳 AI Chef</h1>", unsafe_allow_html=True)
    st.markdown("Enter your ingredients to get started.")
    
    try:
        with st.spinner("Waking up the AI..."):
            data = load_data()
            vectorizer, matrix, data = train_model(data)
    except FileNotFoundError:
        st.error("RAW_recipes.csv not found!")
        st.stop()

    st.markdown(f"<hr style='border-color: {COLOR_PRIMARY}; opacity: 0.3;'>", unsafe_allow_html=True)
    
    user_input = st.text_input("Ingredients", placeholder="e.g. chicken, rice...")
    allergy_input = st.text_input("Allergies", placeholder="optional")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Find Recipes 🔍", use_container_width=True):
        if not user_input.strip():
            st.warning("Enter ingredients first!")
        else:
            user_vec = vectorizer.transform([user_input])
            scores = cosine_similarity(user_vec, matrix).flatten()
            if scores.max() == 0:
                st.error("No matches found.")
            else:
                top_indices = scores.argsort()[-5:][::-1]
                st.session_state.search_results = []
                for idx in top_indices:
                    st.session_state.search_results.append(data.iloc[idx])
                st.session_state.search_performed = True

# Main Content
if st.session_state.search_performed and st.session_state.search_results:
    st.markdown(f"<h2 style='color:{COLOR_TEXT_MAIN} !important;'>Top Recommendations</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    for i, recipe in enumerate(st.session_state.search_results):
        ingredients = recipe['ingredients_list']
        steps = recipe['steps_list']
        nutrition = recipe['nutrition_list'] 
        cluster_name = get_cluster_name(recipe['cluster'], data)
        missing = calculate_missing(user_input.split(','), ingredients) if user_input else []
        
        # MACROS
        protein = nutrition[4]
        carbs = nutrition[6]
        
        # --- CARD START ---
        col_left, col_right = st.columns([1, 1.5], gap="large")
        
        # --- LEFT: INFO ---
        with col_left:
            st.markdown(f'<div class="recipe-title">{recipe["name"].title()}</div>', unsafe_allow_html=True)
            
            # BADGES
            st.markdown(f"""
            <div style="margin-bottom:15px;">
                <span class="badge">🌍 {get_cuisine_type(ingredients)}</span>
                <span class="badge">🔥 {nutrition[0]} cal</span>
                <span class="badge badge-macro">💪 {protein}g Prot</span>
                <span class="badge badge-macro">🍞 {carbs}g Carb</span>
            </div>
            """, unsafe_allow_html=True)
            
            # AI PROFILE BADGE
            st.markdown(f"""
            <div style="margin-bottom:15px;">
                <span class="badge" style="background-color: {COLOR_PRIMARY}; color: white;">🔮 {cluster_name}</span>
            </div>
            """, unsafe_allow_html=True)
            
            if missing:
                m_txt = ", ".join(missing[:4]) + ("..." if len(missing)>4 else "")
                st.markdown(f'<div class="missing-alert">⚠️ <b>Missing:</b> {m_txt}</div>', unsafe_allow_html=True)
            
            st.markdown("**Ingredients:**")
            ing_html = "".join([f"<span class='ing-chip'>{ing}</span>" for ing in ingredients[:12]])
            if len(ingredients) > 12: ing_html += f"<span class='ing-chip'>+{len(ingredients)-12} more</span>"
            st.markdown(ing_html, unsafe_allow_html=True)

        # --- RIGHT: DETAILS ---
        with col_right:
            with st.expander("📝 Cooking Instructions", expanded=True):
                for idx, step in enumerate(steps):
                    st.markdown(f"**{idx+1}.** {step.strip().capitalize()}")
            
            st.markdown(f"**✨ More from '{cluster_name}'**")
            cluster_mates = data[(data['cluster'] == recipe['cluster']) & (data['name'] != recipe['name'])]
            if not cluster_mates.empty:
                recs = cluster_mates.sample(min(3, len(cluster_mates)), random_state=42)
                cols = st.columns(3) 
                for idx, (j, rec) in enumerate(recs.iterrows()):
                    with cols[idx]:
                        if st.button(f"🔹 {rec['name'][:15]}..", key=f"rec_{i}_{j}", help=rec['name']):
                            load_specific_recipe(rec)
                            st.rerun()

        # --- SEPARATOR ---
        st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

else:
    # Empty State
    st.markdown(f"""
    <div style='text-align: center; padding: 100px; color: {COLOR_TEXT_SEC};'>
        <h1 style='color: {COLOR_ACCENT} !important;'>Welcome to AI Chef</h1>
        <p style='font-size: 18px;'>Enter your ingredients in the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)
# ==========================================
# 5. COPYRIGHT FOOTER
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
    <div style='text-align: center; color: {COLOR_TEXT_SEC}; padding: 20px; font-size: 12px; border-top: 1px solid #1e3326;'>
        <p><b>Created by:   </b> Aum Adhikari and Gaurav Gupta</p>
        <p>© 2026 AI Chef Project. All Rights Reserved.</p>
    </div>
""", unsafe_allow_html=True)
