# filename: app.py

import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import warnings
import os
import streamlit.components.v1 as components  # For localStorage integration

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Dunklytics Predictor",
    page_icon="🏀",
    layout="centered"
)

# --- Load Models and Data ---
@st.cache_resource
def load_models_and_data():
    """Load all necessary models, encoders, and data files only once."""
    try:
        df = pd.read_csv("data/basketball_matches_with_opponents.csv")
        rf_model = joblib.load('models/random_forest_model.pkl')
        lstm_model = tf.keras.models.load_model('models/lstm_stat_predictor.h5')
        team_encoder = joblib.load('models/team_encoder.pkl')
        scaler = joblib.load('models/scaler.pkl')

        df['FG2_PCT'] = (df['FGM_2'] / df['FGA_2']).replace([np.inf, -np.inf], 0).fillna(0)
        df['FG3_PCT'] = (df['FGM_3'] / df['FGA_3']).replace([np.inf, -np.inf], 0).fillna(0)
        df['FT_PCT'] = (df['FTM'] / df['FTA']).replace([np.inf, -np.inf], 0).fillna(0)
        df['AST_TO_RATIO'] = (df['AST'] / df['TOV']).replace([np.inf, -np.inf], 0).fillna(0)
        total_rebounds = df['DREB'] + df['OREB']
        df['DREB_RATE'] = (df['DREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
        df['OREB_RATE'] = (df['OREB'] / total_rebounds).replace([np.inf, -np.inf], 0).fillna(0)
        df['TURNOVER_RATE'] = (df['TOV'] / (df.get('FGA', 0) + 0.44 * df.get('FTA', 0) + df.get('TOV', 1))).replace([np.inf, -np.inf], 0).fillna(0)
        df['MARGIN_VICTORY'] = df['team_score'] - df['opponent_team_score']
        
        rf_features = rf_model.feature_names_in_

        df['team_encoded'] = team_encoder.transform(df['team'])
        df['opponent_team_encoded'] = team_encoder.transform(df['opponent_team'])

        return df, rf_model, lstm_model, team_encoder, scaler, rf_features

    except FileNotFoundError as e:
        st.error(f"ERROR: A required file is missing -> {e}. Please check your repository's file structure.")
        return None, None, None, None, None, None
    except KeyError as e:
        st.error(f"ERROR: A required column was not found in your CSV file: {e}. Please check that the data file contains all necessary base columns.")
        return None, None, None, None, None, None

# Load all assets
df, rf_model, lstm_model, team_encoder, scaler, rf_features = load_models_and_data()
stat_features = ['FG2_PCT', 'FG3_PCT', 'FT_PCT', 'AST_TO_RATIO', 'DREB_RATE', 'OREB_RATE', 'TURNOVER_RATE', 'MARGIN_VICTORY']

# --- Helper Functions ---
def get_matchup_mean_stats(team_name, opponent_name, data_df, encoder):
    team_encoded = encoder.transform([team_name])[0]
    opponent_encoded = encoder.transform([opponent_name])[0]
    matchup_data = data_df[(data_df['team_encoded'] == team_encoded) & (data_df['opponent_team_encoded'] == opponent_encoded)]
    if matchup_data.empty: return None
    return matchup_data[stat_features].mean().to_dict()

def predict_future_stats(team_name, data_df, encoder, scaler_model, lstm_model, sequence_length=5):
    team_encoded = encoder.transform([team_name])[0]
    team_history = data_df[data_df['team_encoded'] == team_encoded].tail(sequence_length)
    if len(team_history) < sequence_length: return None
    team_history_scaled = scaler_model.transform(team_history[stat_features])
    team_history_input = np.array([team_history_scaled])
    predicted_stats_scaled = lstm_model.predict(team_history_input)
    predicted_stats = scaler_model.inverse_transform(predicted_stats_scaled)
    return dict(zip(stat_features, predicted_stats[0]))

# --- Streamlit User Interface ---
st.title("🏀 Dunklytics: Basketball Match Predictor")
st.write("This app predicts basketball match outcomes using a hybrid model that blends historical head-to-head data with LSTM-forecasted performance.")

if df is not None:
    TEAMS = sorted(team_encoder.classes_)

    col1, col2 = st.columns(2)
    with col1:
        default_home = "oklahoma_sooners"
        home_index = TEAMS.index(default_home) if default_home in TEAMS else 0
        home_team = st.selectbox("Select Home Team:", TEAMS, index=home_index)
    with col2:
        default_away = "baylor_bears"
        away_index = TEAMS.index(default_away) if default_away in TEAMS else 1
        away_team = st.selectbox("Select Away Team:", TEAMS, index=away_index)

    if home_team == away_team:
        st.error("Please select two different teams.")
    else:
        if st.button("Predict Win Probability", type="primary"):
            with st.spinner("Analyzing matchup and running models..."):
                team_h2h_stats = get_matchup_mean_stats(home_team, away_team, df, team_encoder)
                opponent_h2h_stats = get_matchup_mean_stats(away_team, home_team, df, team_encoder)
                team_predicted_stats = predict_future_stats(home_team, df, team_encoder, scaler, lstm_model)
                opponent_predicted_stats = predict_future_stats(away_team, df, team_encoder, scaler, lstm_model)
                team_overall_stats = df[df['team'] == home_team][stat_features].mean().to_dict()
                opponent_overall_stats = df[df['team'] == away_team][stat_features].mean().to_dict()

                prediction_made = False
                input_features = {}

                if all([team_h2h_stats, opponent_h2h_stats, team_predicted_stats, opponent_predicted_stats]):
                    st.info("Using Tier 1 Prediction: Head-to-Head History + LSTM Forecast")
                    for col in stat_features:
                        input_features[col] = (0.8 * team_h2h_stats.get(col, 0)) + (0.2 * team_predicted_stats.get(col, 0))
                        input_features[f"opponent_{col}"] = (0.8 * opponent_h2h_stats.get(col, 0)) + (0.2 * opponent_predicted_stats.get(col, 0))
                    prediction_made = True
                elif all([team_predicted_stats, opponent_predicted_stats]):
                    st.warning("Using Tier 2 Prediction: Overall Season Averages + LSTM Forecast (No head-to-head history found).")
                    for col in stat_features:
                        input_features[col] = (0.6 * team_overall_stats.get(col, 0)) + (0.4 * team_predicted_stats.get(col, 0))
                        input_features[f"opponent_{col}"] = (0.6 * opponent_overall_stats.get(col, 0)) + (0.4 * opponent_predicted_stats.get(col, 0))
                    prediction_made = True
                elif all([team_h2h_stats, opponent_h2h_stats]):
                    st.warning("Using Tier 3 Prediction: Head-to-Head History Only (Not enough recent games for LSTM).")
                    for col in stat_features:
                        input_features[col] = team_h2h_stats.get(col, 0)
                        input_features[f"opponent_{col}"] = opponent_h2h_stats.get(col, 0)
                    prediction_made = True
                else:
                    st.warning("Using Tier 4 Prediction: Overall Season Averages Only (Limited data available).")
                    for col in stat_features:
                        input_features[col] = team_overall_stats.get(col, 0)
                        input_features[f"opponent_{col}"] = opponent_overall_stats.get(col, 0)
                    prediction_made = True

                if prediction_made:
                    input_features["team_encoded"] = team_encoder.transform([home_team])[0]
                    input_features["opponent_team_encoded"] = team_encoder.transform([away_team])[0]
                    input_df = pd.DataFrame([input_features])[rf_features]
                    win_proba = rf_model.predict_proba(input_df)[:, 1][0]

                    st.success("Prediction Complete!")
                    st.subheader(f"Predicted Win Probability for {home_team}")
                    st.progress(win_proba, text=f"{win_proba:.0%}")
                    st.write(f"The model predicts that the **{home_team}** have a **{win_proba:.0%}** chance of winning against the **{away_team}**.")
                    
                  
                    predicted_winner = home_team if win_proba >= 0.5 else away_team
                    win_percentage = win_proba if predicted_winner == home_team else 1 - win_proba
                    history_entry = f"{home_team} vs {away_team} – {win_percentage:.0%} probability of {predicted_winner} winning"

                    components.html(f"""
                    <script>
                        let newEntry = "{history_entry}";
                        let history = JSON.parse(localStorage.getItem("matchHistory") || "[]");
                        history.push(newEntry);
                        localStorage.setItem("matchHistory", JSON.stringify(history));
                    </script>
                    """, height=0)
                else:
                    st.error("Could not generate a prediction. Not enough data even for the most basic model.")

  
    st.markdown("---")
    st.subheader("📜 Your Prediction History")
    
    components.html("""
    <div id="historyBox" style="padding: 10px; color: white;"></div>
    <script>
    const historyData = JSON.parse(localStorage.getItem("matchHistory") || "[]");
    const box = document.getElementById("historyBox");

    if (historyData.length === 0) {
        box.innerHTML = "<i style='color: #ccc;'>No match history yet.</i>";
    } else {
        box.innerHTML = "<h4 style='margin-bottom: 10px; color: white;'>Recent Predictions</h4>" +
        historyData.reverse().map(item => `<div style='margin-bottom: 6px;'>${item}</div>`).join("");
    }
    </script>
    """, height=300)


   
    components.html("""
    <script>
    function clearHistory() {
        localStorage.removeItem("matchHistory");
        window.parent.location.reload();
    }
    </script>

    <div style="margin-top: 10px;">
    <button onclick="clearHistory()"
        style="
        background-color: #0e1117;
        color: white;
        padding: 10px 18px;
        border: 1px solid #444;
        border-radius: 8px;
        cursor: pointer;
        font-size: 15px;
        transition: background-color 0.3s;
        "
        onmouseover="this.style.backgroundColor='#1f2937';"
        onmouseout="this.style.backgroundColor='#0e1117';"
    >
        Clear Prediction History
    </button>
    </div>
    """, height=80)

