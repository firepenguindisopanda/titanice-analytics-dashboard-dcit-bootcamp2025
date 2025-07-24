from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from lib import TitanicAnalytics

app = Flask(__name__)

# Initialize analytics instance
analytics = TitanicAnalytics()

analytics.load_data(file_path='data/titanic.csv')
analytics.clean_data()
analytics.train_predictive_models()


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/descriptive-stats')
def descriptive_stats():
    """Get descriptive statistics"""
    try:
        stats = analytics.get_descriptive_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/diagnostic-analysis')
def diagnostic_analysis():
    """Get diagnostic analysis results"""
    try:
        diagnostics = analytics.diagnostic_analysis()
        return jsonify(diagnostics)
    except Exception as e:
        print(f"Error in diagnostic analysis: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_survival():
    """Predict survival for given passenger data"""
    try:
        passenger_data = request.json

        # Validate required fields
        required_fields = [
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
        ]
        for field in required_fields:
            if field not in passenger_data:
                return jsonify({'error':
                                f'Missing required field: {field}'}), 400

        # Add default values for derived fields
        passenger_data[
            'Title'] = 'Mr' if passenger_data['Sex'] == 'male' else 'Miss'
        if 'Cabin' not in passenger_data:
            passenger_data['Cabin'] = None

        prediction = analytics.predict_survival(passenger_data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clustering')
def clustering_analysis():
    """Get clustering analysis results"""
    try:
        clusters = analytics.perform_clustering(n_clusters=3)
        return jsonify(clusters)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimization')
def optimization_analysis():
    """Get optimization results for maximum survival"""
    try:
        optimal = analytics.optimize_survival_factors()
        return jsonify(optimal)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-performance')
def model_performance():
    """Get model performance metrics"""
    try:
        # Re-train to get fresh performance metrics
        performance = analytics.train_predictive_models()
        return jsonify(performance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
