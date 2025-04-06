from flask import Flask, request, jsonify
from flask_cors import CORS
from multi_agent_advisor_realdata import AdvisorTeam, model

app = Flask(__name__)
CORS(app)

# Initialize the AdvisorTeam
if not model:
    raise RuntimeError("Model not initialized. Ensure GOOGLE_API_KEY is set in the environment.")
advisor_team = AdvisorTeam(model)

@app.route('/consult', methods=['POST'])
def consult():
    """
    Endpoint to handle financial queries.
    Expects a JSON payload with a 'query' field.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request payload"}), 400

    user_query = data['query']
    if not user_query.strip():
        return jsonify({"error": "Query cannot be empty"}), 400

    # Run the consultation process
    response = advisor_team.run_consultation(user_query)
    return jsonify({"response": response})

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(debug=True)