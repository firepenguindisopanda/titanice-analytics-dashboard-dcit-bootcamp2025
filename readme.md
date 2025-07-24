# Titanic Analytics Dashboard

`Generated using claude`

`As part of the DCIT AI Bootcamp. As a mentor I assisted the campers with a follow along activity to create an analytics dashboard using llms`

[Link to linkedin post about DCIT bootcamp](https://www.linkedin.com/posts/uwidcit_introducing-this-year-guest-speakers-tracell-activity-7351312903074947072-36E-?utm_source=share&utm_medium=member_desktop&rcm=ACoAACOdFPcBKISwS8FqrESmFMsZpo9GSQh6yk4)

A modern Flask web application for analyzing the Titanic dataset with interactive visualizations and machine learning predictions.

## 🚀 Features

- **Interactive Dashboard**: Modern UI with Bootstrap 5 and Chart.js
- **Data Analytics**: Descriptive statistics, diagnostic analysis, and correlations
- **Machine Learning**: Survival prediction using Random Forest and Logistic Regression
- **Clustering Analysis**: K-means clustering to identify passenger groups
- **Optimization**: Find optimal passenger characteristics for maximum survival probability
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 📁 Project Structure

```
titanic-analytics/
├── app.py                          # Main Flask application
├── titanic_analytics.py            # Analytics library with all ML functions
├── requirements.txt                 # Python dependencies
├── templates/
│   ├── dashboard.html              # Main dashboard page
│   └── predict.html                # Prediction form page
└── README.md                       # This file
```

## 🛠️ Setup Instructions for Replit

1. **Create a new Python Repl** in Replit

2. **Create the file structure**:
   - Create `app.py` in the root directory
   - Create `titanic_analytics.py` in the root directory
   - Create `requirements.txt` in the root directory
   - Create a `templates/` folder
   - Create `dashboard.html` in the `templates/` folder
   - Create `predict.html` in the `templates/` folder

3. **Copy the code**:
   - Copy the Flask server code into `app.py`
   - Copy the analytics library code into `titanic_analytics.py`
   - Copy the requirements into `requirements.txt`
   - Copy the dashboard HTML into `templates/dashboard.html`
   - Copy the prediction HTML into `templates/predict.html`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Access the dashboard**:
   - Open the web preview in Replit
   - The dashboard will be available at your Repl's URL

## 🎯 API Endpoints

- `GET /` - Main dashboard
- `GET /predict` - Prediction form page
- `GET /api/descriptive-stats` - Get dataset statistics
- `GET /api/diagnostic-analysis` - Get survival analysis by various factors
- `POST /api/predict` - Predict survival for passenger data
- `GET /api/clustering` - Get clustering analysis results
- `GET /api/optimization` - Get optimal survival factors
- `GET /api/model-performance` - Get ML model performance metrics

## 📊 Dashboard Sections

### 1. Overview
- Total passengers and survival rate
- Average age and fare statistics
- Key dataset metrics

### 2. Analytics
- **Survival by Class**: Doughnut chart showing survival rates by passenger class
- **Survival by Gender**: Bar chart comparing male vs female survival rates
- **Age Distribution**: Line chart showing passenger age distribution
- **Clustering Analysis**: Scatter plot of passenger clusters
- **Model Performance**: Bar chart comparing ML model accuracies

### 3. Prediction
- Interactive form to predict survival for new passenger data
- Real-time predictions using trained ML models
- Confidence indicators and model comparisons

### 4. Optimization
- Calculate optimal passenger characteristics for maximum survival probability
- Shows the ideal passenger profile based on ML model analysis

## 🔧 Customization

### Adding New Data
Replace the sample data in `app.py` with your own Titanic dataset:

```python
# Load from CSV file
df = pd.read_csv('your_titanic_data.csv')
analytics.load_data(data=df)
```

### Adding New Charts
Add new chart functions in the dashboard.html JavaScript section:

```javascript
function createNewChart(data) {
    const ctx = document.getElementById('newChart').getContext('2d');
    // Chart.js configuration
}
```

### Styling
Modify the CSS variables in the dashboard template to change colors:

```css
:root {
    --primary-color: #your-color;
    --secondary-color: #your-color;
}
```

## 🚀 Deployment

### Replit (Recommended)
- The app is ready to run on Replit
- Just click "Run" after setting up the files

### Local Development
```bash
git clone your-repo
cd titanic-analytics
pip install -r requirements.txt
python app.py
```

### Production
For production deployment, consider:
- Using a production WSGI server like Gunicorn
- Setting up environment variables
- Using a proper database instead of in-memory data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🎨 Technologies Used

- **Backend**: Flask, Pandas, Scikit-learn, NumPy, SciPy
- **Frontend**: Bootstrap 5, Chart.js, JavaScript ES6
- **ML Models**: Random Forest, Logistic Regression, K-means Clustering
- **Optimization**: SciPy optimization algorithms

## 📞 Support

If you encounter any issues or have questions, please create an issue in the repository or contact the development team.

---

**Happy Analyzing! 🚢📊**