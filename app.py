
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Define model and preprocessing pipeline paths
MODEL_PATH = 'models/model_totals_final.pkl'
PREPROCESSOR_PATH = 'models/preprocessing_pipeline_final.pkl'

# Load the trained model and preprocessor
try:
    model_pipeline = joblib.load(MODEL_PATH)
    # The preprocessor is part of the model_pipeline in our case
    # If it was separate, we'd load it like: preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    app.logger.error(f"Error loading model or preprocessor: {e}")
    model_pipeline = None # Handle case where model loading fails

# Define relevant features (must match training features)
relevant_features = ['HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']

# Define EV threshold
EV_THRESHOLD = 0.05

# Define class indices (0 for Under 2.5, 1 for Over 2.5)
UNDER_IDX = 0
OVER_IDX = 1

# Team mapping (The Odds API names to model names, consistent with training data)
team_mapping = {
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "Leverkusen",
    "Union Berlin": "Union Berlin",
    "SC Freiburg": "Freiburg",
    "Eintracht Frankfurt": "Ein Frankfurt", # Consistent with original training data
    "VfL Wolfsburg": "Wolfsburg",
    "Borussia Monchengladbach": "M'gladbach",
    "FSV Mainz 05": "Mainz",
    "FC Augsburg": "Augsburg",
    "VfB Stuttgart": "Stuttgart",
    "Werder Bremen": "Werder Bremen",
    "TSG Hoffenheim": "Hoffenheim",
    "FC Heidenheim": "Heidenheim",
    "VfL Bochum": "Bochum",
    "FC St. Pauli": "St Pauli",
    "Holstein Kiel": "Holstein Kiel"
}

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "Bundesliga ML API"})

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({"error": "Model not loaded. Server configuration issue."}), 500

    data = request.get_json(force=True)

    try:
        home_team_api = data.get('home_team')
        away_team_api = data.get('away_team')
        
        # Проверяем что все поля присутствуют
        required_fields = ['avg_h_odds', 'avg_d_odds', 'avg_a_odds', 'avg_over_odds', 'avg_under_odds']
        for field in required_fields:
            if data.get(field) is None:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        avg_h = float(str(data.get('avg_h_odds')).replace(',', '.'))
        avg_d = float(str(data.get('avg_d_odds')).replace(',', '.'))
        avg_a = float(str(data.get('avg_a_odds')).replace(',', '.'))
        avg_over_2_5 = float(str(data.get('avg_over_odds')).replace(',', '.'))
        avg_under_2_5 = float(str(data.get('avg_under_odds')).replace(',', '.'))

        # Map team names
        home_team_model = team_mapping.get(home_team_api, home_team_api)
        away_team_model = team_mapping.get(away_team_api, away_team_api)

        input_data = pd.DataFrame({
            'HomeTeam': [home_team_model],
            'AwayTeam': [away_team_model],
            'AvgH': [avg_h],
            'AvgD': [avg_d],
            'AvgA': [avg_a],
            'Avg>2.5': [avg_over_2_5],
            'Avg<2.5': [avg_under_2_5]
        })

        # Predict probabilities
        probabilities = model_pipeline.predict_proba(input_data[relevant_features])[0]
        prob_under = probabilities[UNDER_IDX]
        prob_over = probabilities[OVER_IDX]

        # Calculate EV
        ev_over = (prob_over * (avg_over_2_5 - 1)) - (prob_under * 1) if avg_over_2_5 and avg_over_2_5 > 1 else -np.inf
        ev_under = (prob_under * (avg_under_2_5 - 1)) - (prob_over * 1) if avg_under_2_5 and avg_under_2_5 > 1 else -np.inf

        recommendation = "No bet"
        bet_on = "None"
        
        if ev_over > EV_THRESHOLD:
            recommendation = "Bet Over 2.5"
            bet_on = "Over 2.5"
        elif ev_under > EV_THRESHOLD:
            recommendation = "Bet Under 2.5"
            bet_on = "Under 2.5"

        return jsonify({
            "home_team": home_team_api,
            "away_team": away_team_api,
            "prob_over_2_5": f"{prob_over:.2%}",
            "prob_under_2_5": f"{prob_under:.2%}",
            "ev_over_2_5": f"{ev_over:.2%}",
            "ev_under_2_5": f"{ev_under:.2%}",
            "recommendation": recommendation,
            "bet_on": bet_on
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Invalid input or prediction error: {e}"}), 400

if __name__ == '__main__':
    # For local development, remove debug=True for production
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000), debug=False)
