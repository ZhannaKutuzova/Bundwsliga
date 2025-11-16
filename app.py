from flask import Flask, request, jsonify
import os
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
preprocessing_pipeline = joblib.load(
    os.path.join(BASE_DIR, "models", "preprocessing_pipeline_final.pkl")
)
model_totals = joblib.load(
    os.path.join(BASE_DIR, "models", "model_totals_final.pkl")
)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "Bundesliga ML API"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        input_data = {
            "HomeTeam": [data["home_team"]],
            "AwayTeam": [data["away_team"]],
            "AvgH": [data["avg_h_odds"]],
            "AvgD": [data["avg_d_odds"]],
            "AvgA": [data["avg_a_odds"]],
            "Avg>2.5": [data["avg_over_odds"]],
            "Avg<2.5": [data["avg_under_odds"]],
        }

        matches_df = pd.DataFrame(input_data)
        X_processed = preprocessing_pipeline.transform(matches_df)

        prob_totals = model_totals.predict_proba(X_processed)[0]
        model_prob_under = float(prob_totals[0])
        model_prob_over = float(prob_totals[1])

        predicted = ">2.5" if model_prob_over > model_prob_under else "<2.5"

        ev_over = (model_prob_over * data["avg_over_odds"]) - 1
        ev_under = (model_prob_under * data["avg_under_odds"]) - 1

        value_bet = False
        bet_on = None
        bet_odds = 0

        if ev_over > 0.05:
            value_bet = True
            bet_on = ">2.5"
            bet_odds = data["avg_over_odds"]
        elif ev_under > 0.05:
            value_bet = True
            bet_on = "<2.5"
            bet_odds = data["avg_under_odds"]

        return jsonify(
            {
                "HomeTeam": data["home_team"],
                "AwayTeam": data["away_team"],
                "Predicted_Totals": predicted,
                "Model_Prob_Over2.5": round(model_prob_over, 4),
                "Model_Prob_Under2.5": round(model_prob_under, 4),
                "EV_Over2.5": round(ev_over, 4),
                "EV_Under2.5": round(ev_under, 4),
                "Value_Bet_Found": value_bet,
                "Bet_On_Outcome": bet_on,
                "Bet_Odds": bet_odds,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
