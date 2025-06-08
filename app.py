# app.py (Version 4 - The Complete Story)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai # Coming soon

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Grad School Advisor",
    page_icon="🎓",
    layout="wide"
)

# --- INITIALIZE SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = "Overview"

# --- NAVIGATION FUNCTION ---
def go_to_page(page_name):
    st.session_state.page = page_name

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    with open("app_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

artifacts = load_artifacts()

# Unpack all assets
user_univ_matrix = artifacts["user_univ_matrix"]
user_profiles_advanced = artifacts["user_profiles_advanced"]
scaled_user_profiles_advanced = artifacts["scaled_user_profiles_advanced"]
scaler_advanced = artifacts["scaler_advanced"]
scaled_avg_profile_df = artifacts["scaled_avg_profile_df"]
advanced_profile_features_rec = artifacts["advanced_profile_features_rec"]
unique_majors_rec = artifacts["unique_majors_rec"]
unique_programs_rec = artifacts["unique_programs_rec"]
predictor_model = artifacts["predictor_model"]
predictor_training_columns = artifacts["predictor_training_columns"]
all_universities_pred = artifacts["all_universities_pred"]
unique_majors_pred = artifacts["unique_majors_pred"]
unique_programs_pred = artifacts["unique_programs_pred"]
unique_ug_colleges_pred = artifacts["unique_ug_colleges_pred"]
correlation_matrix = artifacts["correlation_matrix"]
feature_importance_df = artifacts["feature_importance_df"]
imputation_medians = artifacts["imputation_medians"]

# --- HELPER FUNCTIONS ---
def classify_school(user_score, school_score):
    if user_score > school_score * 1.05: return "Safe"
    elif user_score < school_score * 0.95: return "Ambitious"
    else: return "Moderate"

def recommend_universities(new_profile_dict, weights, num_recommendations=10):
    new_profile_df = pd.DataFrame([new_profile_dict])
    new_profile_encoded = pd.get_dummies(new_profile_df)
    new_profile_aligned = new_profile_encoded.reindex(columns=advanced_profile_features_rec, fill_value=0)
    weight_vector = np.array([weights.get(f, 1.0) for f in advanced_profile_features_rec])
    scaled_new_profile = scaler_advanced.transform(new_profile_aligned)
    scaled_new_profile_weighted = scaled_new_profile * weight_vector
    scaled_user_profiles_weighted = scaled_user_profiles_advanced * weight_vector
    profile_similarity_scores = cosine_similarity(scaled_new_profile_weighted, scaled_user_profiles_weighted)
    similar_user_indices = np.argsort(profile_similarity_scores[0])[::-1][:50]
    recommendation_scores, uni_contributors = {}, {}
    for user_idx in similar_user_indices:
        username, similarity_score = user_profiles_advanced.index[user_idx], profile_similarity_scores[0][user_idx]
        try:
            admitted_unis = user_univ_matrix.loc[username]
            admitted_unis = admitted_unis[admitted_unis > 0].index
            for uni in admitted_unis:
                recommendation_scores[uni] = recommendation_scores.get(uni, 0) + similarity_score
                if uni not in uni_contributors: uni_contributors[uni] = []
                uni_contributors[uni].append((username, similarity_score))
        except KeyError: continue
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)
    final_output = []
    user_strength_score = scaled_new_profile.sum()
    for uni, score in sorted_recommendations[:num_recommendations]:
        try:
            school_strength_score = scaled_avg_profile_df.loc[uni].sum()
            category, top_contributors = classify_school(user_strength_score, school_strength_score), sorted(uni_contributors[uni], key=lambda x: x[1], reverse=True)[:3]
            final_output.append({"University": uni, "Recommendation_Score": score, "Category": category, "Why": [item[0] for item in top_contributors]})
        except KeyError: continue
    return final_output

def get_admission_predictions(student_profile):
    """Calculates admission probability for all universities and returns a sorted DataFrame."""
    recommendations = []
    for university in all_universities_pred:
        profile = student_profile.copy()
        profile['univName'] = university
        profile_df = pd.DataFrame([profile])
        profile_encoded = pd.get_dummies(profile_df)
        profile_aligned = profile_encoded.reindex(columns=predictor_training_columns, fill_value=0)
        admission_prob = predictor_model.predict_proba(profile_aligned)[0][1]
        recommendations.append({'University': university, 'Admission Probability': admission_prob})
    
    # Return the full, sorted list of all university predictions
    return pd.DataFrame(recommendations).sort_values(by='Admission Probability', ascending=False)


# --- UI NAVIGATION ---
st.sidebar.title("🎓 Grad School Advisor")
st.sidebar.radio(
    "Navigate", 
    ["Overview", "University Recommender", "Admission Predictor", "AI Advisor Chatbot"],
    key='page' 
)

# --- PAGE 1: OVERVIEW (The Pitch) ---
if st.session_state.page == "Overview":
    st.markdown("<h1 style='text-align: center;'>Overview of Grad School Advisor</h1>", unsafe_allow_html=True)
    st.markdown("""
    Navigating the graduate school application process is a journey filled with uncertainty, challenges, and a lots of learning. 
    This product is an attempt to make the process easier by building a data-driven advisory tool that provides **personalized, strategic, and explainable** university recommendations.
    """)

    # --- DATA SECTION ---
    st.markdown("<h5>The Data: Fueling the Engine</h5>", unsafe_allow_html=True)
    with st.expander("Glimpse into the Raw Data"):
        st.write("`original_data.csv` - The core student profiles and admission results.")
        st.dataframe(pd.read_csv('original_data.csv').head(3))
        st.write("`score.csv` - A conversion table to standardize old GRE scores.")
        st.dataframe(pd.read_csv('score.csv').head(3))

    # --- NEW AND IMPROVED CODE ---
    st.markdown("Key insights can be drawn from feature correlations:")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", ax=ax)
        plt.title('Correlation Matrix of Key Academic Features')
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # --- RECOMMENDER SECTION ---
    st.markdown("<h3>1. The University Recommender</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    University Recommender is a hybrid recommendation tool helps with **discovery**, answering the question: *\"Based on students like me, which universities should I consider? It combines two main techniques:
    1. **Content-Based Filtering:** Analyzes the student's academic profile (GRE, CGPA, experience) and matches it with the profiles of students who have been admitted to various universities.
    2. **Collaborative Filtering:** Identifies students with similar profiles to yours and recommends universities that they were successfully admitted to.
    """)

    with st.expander("Challenges & Learnings", icon="💡"):
        st.markdown("""
    - **The 'Silo' Effect:** A key challenge was that highly similar users often get into the exact same set of universities. 
        - *Solution:* We explicitly filter out universities where the user has a known admission, forcing the model to provide novel suggestions.
    - **Data Sparsity & Cleaning:** The raw data contained inconsistent GRE scores, missing values, and unstandardized CGPA scales.
        - *Solution:* A robust preprocessing pipeline was built to standardize scores, normalize CGPA to a 4.0 scale, and intelligently impute missing data using medians.
    """)

    with st.expander("What Makes Us Unique?", icon="💡"):
        st.markdown("""
    - **✅ Weighted Similarity:** You can't capture a student's ambition in a single score. Our sliders allow users to tell the model what's most important to them—be it research, GPA, or industry experience—for truly personalized results.
    - **✅ Strategic Recommendations:** We don't just provide a list; we categorize it. By classifying schools as **Ambitious, Moderate, or Safe**, we help users build a balanced and strategic application portfolio.
    - **✅ Trust Through Transparency:** We explain *why* a university is recommended by showing profiles of similar past students who were admitted there. This explainability builds immense user trust.
    """)

    st.markdown("<h5>Model Performance & Reliability</h5>", unsafe_allow_html=True)
    st.markdown("To ensure the Recommender is trustworthy, we evaluated its ability to predict a user's actual admissions from a held-out test set.")
    
    st.markdown("""<style>.custom-table{width:100%;border-collapse:collapse;margin-bottom:20px}.custom-table th,.custom-table td{border:1px solid #4F4F4F;padding:12px;text-align:center;vertical-align:middle}.custom-table th{background-color:#262730;font-weight:700;font-size:18px}.custom-table td .metric-value{font-size:28px;font-weight:600;line-height:1.2}.custom-table .tooltip{position:relative;display:inline-block;cursor:help}.custom-table .tooltip .tooltiptext{visibility:hidden;width:220px;background-color:#555;color:#fff;text-align:center;border-radius:6px;padding:5px;position:absolute;z-index:1;bottom:125%;left:50%;margin-left:-110px;opacity:0;transition:opacity .3s}.custom-table .tooltip:hover .tooltiptext{visibility:visible;opacity:1}.circle-question-mark{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;border:1px solid #A0A0A0;font-size:11px;font-weight:700;color:#A0A0A0;margin-left:8px}</style><table class=custom-table><thead><tr><th>Metric</th><th>Top 5 Recommendations (k=5)</th><th>Top 10 Recommendations (k=10)</th></tr></thead><tbody><tr><td>Average Precision<div class=tooltip><span class=circle-question-mark>?</span><span class=tooltiptext>Of the top schools we recommend, this is the percentage that the student was actually admitted to. Higher is better.</span></div></td><td class=metric-value>25.4%</td><td class=metric-value>16.8%</td></tr><tr><td>Average Recall<div class=tooltip><span class=circle-question-mark>?</span><span class=tooltiptext>Of all the schools a student was admitted to, this is the percentage we successfully included in our recommendations. Higher is better.</span></div></td><td class=metric-value>68.2%</td><td class=metric-value>76.7%</td></tr><tr><td>F1-Score<div class=tooltip><span class=circle-question-mark>?</span><span class=tooltiptext>A balanced measure of Precision and Recall. It's useful for comparing overall performance. Higher is better.</span></div></td><td class=metric-value>37.0%</td><td class=metric-value>27.6%</td></tr></tbody></table>""", unsafe_allow_html=True)
    

    st.markdown("---")

    # --- PREDICTOR SECTION ---
    st.markdown("<h3>2. Admission Predictor</h3>", unsafe_allow_html=True)
    st.markdown("""
    The Admission Predictor is a tool that provides **evaluation**, answering two key questions:  
    1. *"For a specific university, what are my chances of getting in?"*  
    2. *"Of all universities, where are my chances the highest?"*

    It is built upon a highly-tuned **XGBoost (Extreme Gradient Boosting)** model, an advanced machine learning algorithm renowned for its accuracy and performance.""")    
    
    st.markdown("<h5>The Journey to Peak Performance</h5>", unsafe_allow_html=True)
    st.markdown("Building a reliable predictor required methodical experimentation. We progressed through several models to maximize accuracy:")

    # --- This is the NEW, UPDATED block ---

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Baseline: Logistic Regression", "Improvement: Random Forest", "Better Algorithm: XGBoost", "Adding Features", "Final Model: Tuned XGBoost"])

    with tab1:
        st.write("Our first attempt used a simple, interpretable linear model as a baseline.")
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            st.metric("Model Accuracy", "68.2%")
        with col2:
            with st.expander("View Full Classification Report"):
                st.code("""
              precision    recall  f1-score   support
           0       0.67      0.65      0.66      5118
           1       0.69      0.71      0.70      5547
            """)


    with tab2:
        st.write("Next, we tried an ensemble method, Random Forest, to see if combining multiple decision trees would improve performance.")
        col1, col2 = st.columns(2, vertical_alignment="center"      )
        with col1:
            st.metric("Model Accuracy", "69.3%")
        with col2:
            with st.expander("View Full Classification Report"):
                st.code("""
              precision    recall  f1-score   support
           0       0.68      0.68      0.68      5118
           1       0.71      0.70      0.70      5547
            """)

    with tab3:
        st.write("We then moved to a more powerful gradient boosting algorithm, XGBoost, which is known for its performance.")
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            st.metric("Model Accuracy", "70.2%")
        with col2:
            with st.expander("View Full Classification Report"):
                st.code("""
              precision    recall  f1-score   support
           0       0.70      0.66      0.68      5118
           1       0.70      0.74      0.72      5547
            """)

    with tab4:
        st.write("A major leap came from feature engineering. We added categorical features like `major` and `ugCollege_grouped` for the model to learn from.")
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            st.metric("Model Accuracy", "72.5%")
        with col2:
            with st.expander("View Full Classification Report"):
                st.code("""
              precision    recall  f1-score   support
           0       0.72      0.70      0.71      5118
           1       0.73      0.75      0.74      5547
            """)

    with tab5:
        st.write("The final step was hyperparameter tuning. Using cross-validation, we found the optimal parameters for the XGBoost model, squeezing out the best possible performance.")
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            st.metric("Model Accuracy", "73.9%")
        with col2:
            with st.expander("View Full Classification Report"):
                st.code("""
              precision    recall  f1-score   support
           0       0.73      0.72      0.73      5118
           1       0.74      0.76      0.75      5547
            """)
            
    st.markdown("---")

    # --- FUTURE WORK ---
    st.markdown("<h4>What's Next?</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **Integrating AI:** Integrate an "AI Chatbot" to further enhance the user experience.
    - **Advanced `ugCollege` Feature:** Implement a tier-based grouping of undergraduate colleges (e.g., IITs, NITs, Top US Public) to create a powerful "prestige" feature for both models.
    - **User Feedback Loop:** Allow users to 'like' or 'dislike' recommendations to fine-tune the recommender model for them in real-time.
    - **Expanding the Dataset:** Incorporate more recent data and a wider variety of programs to keep the recommendations fresh and relevant.
    - **Cover other Exams too (including IELTS, PTE, etc):** 
    """)


# --- PAGE 2: UNIVERSITY RECOMMENDER ---
elif st.session_state.page == "University Recommender":
    st.title("University Recommender")
    st.write("Discover universities that are a good fit for your profile based on data from thousands of past applicants")

    def map_to_model_weight(user_value):
        return ((user_value / 10.0) * 2.0) + 0.5

    # --- CHECKBOXES DECLARED OUTSIDE THE FORM ---
    st.subheader("Your Academic Profile")
    st.write("First, indicate if you have scores for the following standardized tests:")
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        col_cb1, col_cb2 = st.columns(2)
        with col_cb1:
            no_gre = st.checkbox("I do not have GRE scores")
        with col_cb2:
            no_toefl = st.checkbox("I do not have a TOEFL score")
    
    # --- THIS IS THE NEW NOTIFICATION BLOCK ---
    if no_gre or no_toefl:
        st.info(
            "For any test you do not have a score for, the model will use a neutral, "
            "dataset-average value for its calculations.",
            icon="ℹ️"
        )

    st.markdown("---")

    # The form now contains only the inputs
    with st.form("recommender_form"):
        # The subheader is no longer needed here
        col1, col2, col3 = st.columns(3)
        with col1:
            # The disabled parameter now works instantly
            gre_v = st.number_input("GRE Verbal Score (130-170)", 130, 170, 155, disabled=no_gre)
            gre_q = st.number_input("GRE Quant Score (130-170)", 130, 170, 160, disabled=no_gre)
            gre_a = st.number_input("GRE AWA Score (0.0-6.0)", 0.0, 6.0, 4.0, 0.5, disabled=no_gre)
        with col2:
            cgpa_10 = st.number_input("CGPA (on a 10-point scale)", 0.0, 10.0, 8.5, 0.1)
            toefl = st.number_input("TOEFL Score (0-120)", 0, 120, 100, disabled=no_toefl)
            research_exp = st.number_input("Research Experience (months)", 0, 120, 6)
        with col3:
            industry_exp = st.number_input("Industry Experience (months)", 0, 120, 12)
            major = st.selectbox("Select Your Major", sorted(unique_majors_rec))
            program = st.selectbox("Select Your Program", sorted(unique_programs_rec))

        st.subheader("What Matters Most to You? (Scale 0-10)")
        weights_col1, weights_col2 = st.columns(2)
        with weights_col1:
            w_cgpa_user = st.slider("Importance of CGPA", 0, 10, 7)
            w_gre_user = st.slider("Importance of GRE Scores", 0, 10, 6, disabled=no_gre)
        with weights_col2:
            w_research_user = st.slider("Importance of Research Experience", 0, 10, 9)
            w_industry_user = st.slider("Importance of Industry Experience", 0, 10, 2)
        
        submit_button_rec = st.form_submit_button(label="🚀 Recommend Universities")

    if submit_button_rec:
        # --- Median Imputation Logic ---
        final_gre_v = gre_v if not no_gre else imputation_medians['greV_new']
        final_gre_q = gre_q if not no_gre else imputation_medians['greQ_new']
        final_gre_a = gre_a if not no_gre else imputation_medians['greA']
        final_toefl = toefl if not no_toefl else imputation_medians['toeflScore']

        profile = {
            'greQ_new': final_gre_q, 'greV_new': final_gre_v, 'greA': final_gre_a, 
            'cgpa_normalized': cgpa_10, 'researchExp': research_exp, 
            'industryExp': industry_exp, 'toeflScore': final_toefl, 
            'major': major, 'program': program
        }
        weights = {
            'cgpa_normalized': map_to_model_weight(w_cgpa_user),
            'greQ_new': map_to_model_weight(w_gre_user), 'greV_new': map_to_model_weight(w_gre_user),
            'researchExp': map_to_model_weight(w_research_user),
            'industryExp': map_to_model_weight(w_industry_user),
            f'major_{major}': 1.5 
        }
        
        with st.spinner("Finding best-fit universities..."):
            recs = recommend_universities(profile, weights)
        # ... (rest of the display logic is unchanged) ...
        st.subheader("Your Personalized Recommendations")
        if recs:
            rec_df = pd.DataFrame(recs)
            rec_df['Recommendation_Score'] = rec_df['Recommendation_Score'].map('{:.2f}'.format)
            rec_df['Why'] = rec_df['Why'].apply(lambda x: ', '.join(set(x)))
            rec_df.rename(columns={'Why': 'Justified By Similar Profiles'}, inplace=True)
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
            with st.expander("How to Interpret the Recommendations:", icon="💡"):
                st.markdown("""
                - **Recommendation Score:** Tells you what schools are a good fit based on a "wisdom of the crowd" approach (similarity to many past students).
                - **Category (Ambitious/Moderate/Safe):** Tells you how your individual profile stacks up against the average admitted student for that specific school.
                """)
        else:
            st.warning("Could not generate recommendations.")

# --- PAGE 3: ADMISSION PREDICTOR ---
elif st.session_state.page == "Admission Predictor":
    st.title("Admission Chance Predictor")
    st.write("Select a university and enter your profile to get a data-driven prediction of your admission chances.")

    # --- CHECKBOXES DECLARED OUTSIDE THE FORM ---
    st.subheader("1. Enter Your Complete Profile")
    st.write("First, indicate if you have scores for the following standardized tests:")
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        col_cb1, col_cb2 = st.columns(2)
        with col_cb1:
            no_gre_pred = st.checkbox("I do not have GRE scores", key="pred_no_gre")
        with col_cb2:
            no_toefl_pred = st.checkbox("I do not have a TOEFL score", key="pred_no_toefl")
    
    # --- THIS IS THE NEW NOTIFICATION BLOCK ---
    if no_gre_pred or no_toefl_pred:
        st.info(
            "For any test you do not have a score for, the model will use a neutral, "
            "dataset-average value for its calculations.",
            icon="ℹ️"
        )
    
    st.markdown("---")

    with st.form("predictor_form"):
        # The university selector is now part of the form
        col1, col2 = st.columns([0.4, 0.6], vertical_alignment="center")
        with col1:
            st.markdown("#### Select a University:")
        with col2:
            selected_university = st.selectbox("Select a university", options=sorted(all_universities_pred), label_visibility="collapsed")
        
        # Profile inputs
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            p_gre_v = st.number_input("GRE Verbal Score (130-170)", 130, 170, 165, disabled=no_gre_pred)
            p_gre_q = st.number_input("GRE Quant Score (130-170)", 130, 170, 170, disabled=no_gre_pred)
            p_gre_a = st.number_input("GRE AWA Score (0.0-6.0)", 0.0, 6.0, 5.0, 0.5, disabled=no_gre_pred)
        with pcol2:
            p_cgpa_4 = st.number_input("CGPA (on a 4.0 scale)", 0.0, 4.0, 3.9, 0.01)
            p_toefl = st.number_input("TOEFL Score (0-120)", 0, 120, 118, disabled=no_toefl_pred)
            p_research_exp = st.number_input("Research Experience (months)", 0, 120, 24)
        with pcol3:
            p_industry_exp = st.number_input("Industry Experience (months)", 0, 120, 6)
            p_major = st.selectbox("Select Your Major", sorted(unique_majors_pred))
            p_program = st.selectbox("Select Your Program", sorted(unique_programs_pred))
        p_ug_college = st.selectbox("Your Undergraduate College (or 'Other')", sorted(unique_ug_colleges_pred))
        
        submit_button_pred = st.form_submit_button(label="🔮 Predict My Chances")

    if submit_button_pred:
        # --- Median Imputation Logic ---
        final_gre_v_pred = p_gre_v if not no_gre_pred else imputation_medians['greV_new']
        final_gre_q_pred = p_gre_q if not no_gre_pred else imputation_medians['greQ_new']
        final_gre_a_pred = p_gre_a if not no_gre_pred else imputation_medians['greA']
        final_toefl_pred = p_toefl if not no_toefl_pred else imputation_medians['toeflScore']
        
        profile = {
            'greQ_new': final_gre_q_pred, 'greV_new': final_gre_v_pred, 'greA': final_gre_a_pred, 
            'cgpa_norm': p_cgpa_4, 'researchExp': p_research_exp, 
            'industryExp': p_industry_exp, 'toeflScore': final_toefl_pred,
            'program': p_program, 'ugCollege_grouped': p_ug_college, 'major': p_major
        }
        
        with st.spinner("Calculating admission probabilities..."):
            all_predictions_df = get_admission_predictions(profile)
        # ... (rest of the display logic is unchanged) ...
        st.markdown("---")
        col_res, col_comp = st.columns([0.4, 0.6])
        with col_res:
            st.subheader(f"Result for:")
            st.markdown(f"### {selected_university}")
            specific_prediction = all_predictions_df[all_predictions_df['University'] == selected_university]
            if not specific_prediction.empty:
                admission_prob = specific_prediction['Admission Probability'].iloc[0]
                if admission_prob > 0.80: verdict = "Very Likely (Safe)"; st.success(f"**Verdict:** {verdict}")
                elif admission_prob > 0.60: verdict = "Likely (Moderate)"; st.info(f"**Verdict:** {verdict}")
                else: verdict = "Less Likely (Ambitious)"; st.warning(f"**Verdict:** {verdict}")
                st.metric(label="Predicted Admission Probability", value=f"{admission_prob:.2%}")
                st.progress(float(admission_prob))
            else: st.warning("Could not find a prediction for the selected university.")
        with col_comp:
            st.subheader("For Comparison:")
            st.markdown("##### Your Top 10 Highest Probability Schools")
            top_10_df = all_predictions_df.head(10).copy()
            top_10_df['Admission Probability'] = top_10_df['Admission Probability'].map('{:.2%}'.format)
            def highlight_row(row):
                return ['background-color: #4A4A5A'] * len(row) if row.University == selected_university else [''] * len(row)
            st.dataframe(top_10_df.style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)

# --- PAGE 4: AI ADVISOR CHATBOT ---
elif st.session_state.page == "AI Advisor Chatbot":
    st.title("🤖 AI Advisor Chatbot")
    st.info("Ask me anything about the grad school application process, how to interpret your results, or for advice on your profile!", icon="💡")

    # --- GEMINI API CONFIGURATION ---
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error("Error configuring the Gemini API. Please make sure your GOOGLE_API_KEY is set in Streamlit's secrets.", icon="🚨")
        st.stop()

    # --- SYSTEM INSTRUCTION (The Chatbot's Persona) ---
    # This is the new, more robust way to set the persona.
    system_instruction = """
    You are a friendly, knowledgeable, and encouraging AI Grad School Advisor.
    Your name is 'Advisor Ally'. You are an expert on the US graduate school admissions process.
    Your goal is to help the user understand their results from this application and to give them general advice.
    The application has two main tools:
    1. A "University Recommender" that suggests schools based on similarity to past successful students.
    2. An "Admission Predictor" that uses a powerful XGBoost model to give a percentage chance of admission for a specific school.

    When a user asks a question, be supportive and provide clear, actionable advice.
    If you don't know an answer, say so honestly. Do not make up information about specific university programs.
    Keep your answers concise and easy to read, using bullet points or bold text where helpful.
    """

    # --- CHATBOT INITIALIZATION ---
    # We use the new system_instruction parameter here.
    # "gemini-1.5-flash-latest" is a great, fast, and powerful model for chat.
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = genai.GenerativeModel(
            "gemini-1.5-flash-latest",
            system_instruction=system_instruction
        )
    
    # Initialize the chat session, this time with an empty history
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=[])

    # --- DISPLAY CHAT HISTORY ---
    # We will display the full history from the chat session object
    for message in st.session_state.chat_session.history:
        # For the role, we map 'model' to 'assistant' for the chat bubble icon
        role = "assistant" if message.role == "model" else message.role
        with st.chat_message(role):
            st.markdown(message.parts[0].text)
            
    # Add a friendly initial greeting if the history is empty
    if not st.session_state.chat_session.history:
        st.chat_message("assistant").write("Hello! I'm Advisor Ally. How can I help you with your grad school journey today?")


    # --- CHAT INPUT AND RESPONSE ---
    user_input = st.chat_input("Ask for advice on your profile or results...")

    if user_input:
        # Send the user's message to the Gemini API
        with st.spinner("Advisor Ally is thinking..."):
            try:
                response = st.session_state.chat_session.send_message(user_input)
                # The history is automatically updated by the send_message method
                # We just need to rerun the script to display the new messages
                st.rerun() 
            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")