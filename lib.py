import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')


class TitanicAnalytics:

    def __init__(self):
        self.data = None
        self.cleaned_data = None
        self.models = {}
        self.scalers = {}

    def load_data(self, file_path=None, data=None):
        """
        Load Titanic dataset from file or DataFrame

        Args:
            file_path (str): Path to CSV file
            data (DataFrame): Pre-loaded DataFrame

        Returns:
            DataFrame: Raw loaded data
        """
        if data is not None:
            self.data = data.copy()
        elif file_path:
            self.data = pd.read_csv(file_path)
        else:
            raise ValueError("Either file_path or data must be provided")

        return self.data

    def clean_data(self):
        """
        Clean and preprocess the Titanic dataset

        Returns:
            DataFrame: Cleaned data
        """
        if self.data is None:
            raise ValueError("Data must be loaded first using load_data()")

        df = self.data.copy()

        # Fill missing ages with median by class and sex
        df['Age'] = df.groupby(
            ['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

        # Fill missing embarked with mode
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

        # Fill missing fare with median by class
        df['Fare'] = df.groupby('Pclass')['Fare'].transform(
            lambda x: x.fillna(x.median()))

        # Create cabin indicator
        df['HasCabin'] = df['Cabin'].notna().astype(int)

        # Extract title from name
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace([
            'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
            'Sir', 'Jonkheer', 'Dona'
        ], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

        # Create family size feature
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Create age groups
        df['AgeGroup'] = pd.cut(df['Age'],
                                bins=[0, 12, 18, 65, 100],
                                labels=['Child', 'Teen', 'Adult', 'Senior'])

        # Create fare groups
        df['FareGroup'] = pd.qcut(df['Fare'],
                                  q=4,
                                  labels=['Low', 'Medium', 'High', 'VeryHigh'])

        self.cleaned_data = df
        return df

    def get_descriptive_stats(self):
        """
        Calculate descriptive statistics

        Returns:
            dict: Dictionary containing various descriptive statistics
        """
        if self.cleaned_data is None:
            raise ValueError("Data must be cleaned first using clean_data()")

        df = self.cleaned_data

        stats = {
            'survival_rate':
            df['Survived'].mean(),
            'total_passengers':
            len(df),
            'age_stats': {
                'mean': df['Age'].mean(),
                'median': df['Age'].median(),
                'std': df['Age'].std()
            },
            'fare_stats': {
                'mean': df['Fare'].mean(),
                'median': df['Fare'].median(),
                'std': df['Fare'].std()
            },
            'class_distribution':
            df['Pclass'].value_counts().to_dict(),
            'sex_distribution':
            df['Sex'].value_counts().to_dict(),
            'embarked_distribution':
            df['Embarked'].value_counts().to_dict(),
            'top_10_fares':
            df.nlargest(10, 'Fare')[['Name', 'Fare',
                                     'Pclass']].to_dict('records'),
            'youngest_passengers':
            df.nsmallest(5, 'Age')[['Name', 'Age',
                                    'Survived']].to_dict('records'),
            'oldest_passengers':
            df.nlargest(5, 'Age')[['Name', 'Age',
                                   'Survived']].to_dict('records')
        }

        return stats

    def diagnostic_analysis(self):
        """
        Perform diagnostic comparisons and correlation analysis

        Returns:
            dict: Dictionary containing diagnostic analysis results
        """
        if self.cleaned_data is None:
            raise ValueError("Data must be cleaned first using clean_data()")

        df = self.cleaned_data

        # Survival by various factors - convert to JSON-serializable format
        def convert_groupby_to_dict(grouped_data):
            result = {'mean': {}, 'count': {}}
            for key, value in grouped_data['mean'].items():
                result['mean'][str(key)] = float(value)
            for key, value in grouped_data['count'].items():
                result['count'][str(key)] = int(value)
            return result

        survival_by_class = convert_groupby_to_dict(
            df.groupby('Pclass')['Survived'].agg(['mean', 'count']).to_dict())
        survival_by_sex = convert_groupby_to_dict(
            df.groupby('Sex')['Survived'].agg(['mean', 'count']).to_dict())

        # Handle AgeGroup with potential NaN values
        age_group_data = df.dropna(
            subset=['AgeGroup']).groupby('AgeGroup')['Survived'].agg(
                ['mean', 'count']).to_dict()
        survival_by_age_group = convert_groupby_to_dict(age_group_data)

        survival_by_embarked = convert_groupby_to_dict(
            df.groupby('Embarked')['Survived'].agg(['mean',
                                                    'count']).to_dict())
        survival_by_family_size = convert_groupby_to_dict(
            df.groupby('FamilySize')['Survived'].agg(['mean',
                                                      'count']).to_dict())

        # Correlations for numerical features
        numerical_cols = [
            'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'FamilySize', 'IsAlone', 'HasCabin'
        ]
        correlations = {}
        corr_data = df[numerical_cols].corr()['Survived'].drop('Survived')
        for key, value in corr_data.items():
            correlations[str(key)] = float(
                value) if not pd.isna(value) else 0.0

        # Cross-tabulations - convert complex keys to strings
        crosstab_data = pd.crosstab([df['Pclass'], df['Sex']],
                                    df['Survived'],
                                    normalize='index')
        class_sex_survival = {}
        for col in crosstab_data.columns:
            col_data = {}
            for idx in crosstab_data.index:
                # Convert tuple index to string
                key_str = f"{idx[0]}_{idx[1]}"  # e.g., "1_female", "2_male"
                col_data[key_str] = float(crosstab_data.loc[idx, col])
            class_sex_survival[str(col)] = col_data

        diagnostics = {
            'survival_by_class': survival_by_class,
            'survival_by_sex': survival_by_sex,
            'survival_by_age_group': survival_by_age_group,
            'survival_by_embarked': survival_by_embarked,
            'survival_by_family_size': survival_by_family_size,
            'correlations': correlations,
            'class_sex_survival': class_sex_survival
        }

        return diagnostics

    def train_predictive_models(self, model_params=None):
        """
        Train predictive models for survival prediction
        Optional model_params dict to override default hyperparameters
        """
        if self.cleaned_data is None:
            raise ValueError("Data must be cleaned first using clean_data()")

        df = self.cleaned_data.copy()

        # Encode categorical
        le_sex = LabelEncoder()
        le_embarked = LabelEncoder()
        le_title = LabelEncoder()
        df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
        df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])
        df['Title_encoded'] = le_title.fit_transform(df['Title'])
        self.encoders = {
            'sex': le_sex,
            'embarked': le_embarked,
            'title': le_title
        }

        # Features and target
        features = [
            'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_encoded', 'FamilySize', 'IsAlone', 'HasCabin',
            'Title_encoded'
        ]
        X = df[features]
        y = df['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        # Scale only for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['features'] = scaler

        # Default hyperparameters
        defaults = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            },
            'logistic_regression': {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 200,
                'random_state': 42
            }
        }

        # Override with user-supplied params
        if model_params:
            for key, params in model_params.items():
                if key in defaults:
                    defaults[key].update(params)

        # Instantiate models
        models = {
            'random_forest':
            RandomForestClassifier(**defaults['random_forest']),
            'logistic_regression':
            LogisticRegression(**defaults['logistic_regression'])
        }

        results = {}
        for name, model in models.items():
            # Choose data representation
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                X_eval, X_pred = X_test_scaled, model
            else:
                model.fit(X_train, y_train)
                X_eval, X_pred = X_test, model

            y_pred = model.predict(X_eval)
            accuracy = accuracy_score(y_test, y_pred)

            # Feature importance or coefficients
            if hasattr(model, 'feature_importances_'):
                fi = model.feature_importances_
            else:
                fi = np.abs(model.coef_[0])

            fi_dict = {f: float(imp) for f, imp in zip(features, fi)}

            self.models[name] = model
            results[name] = {
                'accuracy': float(accuracy),
                'feature_importance': fi_dict
            }

        return results

    def predict_survival(self, passenger_data):
        """
        Predict survival for new passenger data

        Args:
            passenger_data (dict): Dictionary with passenger features

        Returns:
            dict: Prediction results from different models
        """
        if not self.models:
            raise ValueError(
                "Models must be trained first using train_predictive_models()")

        # Create DataFrame from input
        df = pd.DataFrame([passenger_data])

        # Apply same preprocessing
        if 'Sex' in df.columns:
            df['Sex_encoded'] = self.encoders['sex'].transform(df['Sex'])
        if 'Embarked' in df.columns:
            df['Embarked_encoded'] = self.encoders['embarked'].transform(
                df['Embarked'])
        if 'Title' in df.columns:
            df['Title_encoded'] = self.encoders['title'].transform(df['Title'])

        # Create derived features
        df['FamilySize'] = df.get('SibSp', 0) + df.get('Parch', 0) + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['HasCabin'] = df.get('Cabin', pd.NA).notna().astype(int)

        features = [
            'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_encoded', 'FamilySize', 'IsAlone', 'HasCabin',
            'Title_encoded'
        ]

        X = df[features]

        predictions = {}

        for name, model in self.models.items():
            if name == 'logistic_regression':
                X_scaled = self.scalers['features'].transform(X)
                prob = model.predict_proba(X_scaled)[0]
                pred = model.predict(X_scaled)[0]
            else:
                prob = model.predict_proba(X)[0]
                pred = model.predict(X)[0]

            predictions[name] = {
                'survival_prediction': int(pred),
                'survival_probability': float(prob[1]),
                'death_probability': float(prob[0])
            }

        return predictions

    def perform_clustering(self, n_clusters=3):
        """
        Perform clustering analysis on passengers

        Args:
            n_clusters (int): Number of clusters

        Returns:
            dict: Clustering results and analysis
        """
        if self.cleaned_data is None:
            raise ValueError("Data must be cleaned first using clean_data()")

        df = self.cleaned_data.copy()

        # Prepare features for clustering
        le_sex = LabelEncoder()
        le_embarked = LabelEncoder()

        df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
        df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])

        clustering_features = [
            'Pclass', 'Sex_encoded', 'Age', 'Fare', 'FamilySize',
            'Embarked_encoded'
        ]
        X = df[clustering_features].fillna(df[clustering_features].median())

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        df['Cluster'] = clusters

        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_data = df[df['Cluster'] == i]

            # Convert value_counts to proper dictionaries
            class_dist = cluster_data['Pclass'].value_counts().to_dict()
            sex_dist = cluster_data['Sex'].value_counts().to_dict()

            # Ensure all values are JSON serializable
            cluster_analysis[f'cluster_{i}'] = {
                'size': int(len(cluster_data)),
                'survival_rate': float(cluster_data['Survived'].mean()),
                'avg_age': float(cluster_data['Age'].mean()),
                'avg_fare': float(cluster_data['Fare'].mean()),
                'class_distribution': {
                    str(k): int(v)
                    for k, v in class_dist.items()
                },
                'sex_distribution': {
                    str(k): int(v)
                    for k, v in sex_dist.items()
                }
            }

        return {
            'cluster_analysis': cluster_analysis,
            'total_clusters': n_clusters,
            'cluster_assignments': clusters.tolist()
        }

    def optimize_survival_factors(self):
        """
        Prescriptive optimization to find factors that maximize survival probability

        Returns:
            dict: Optimization results
        """
        if not self.models or 'random_forest' not in self.models:
            raise ValueError("Random Forest model must be trained first")

        model = self.models['random_forest']

        # Define optimization function
        def survival_probability(params):
            # params: [Pclass, Sex_encoded, Age, SibSp, Parch, Fare, Embarked_encoded, FamilySize, IsAlone, HasCabin, Title_encoded]
            features = np.array(params).reshape(1, -1)
            prob = model.predict_proba(features)[0][
                1]  # Probability of survival
            return -prob  # Negative because we want to maximize

        # Define constraints and bounds
        bounds = [
            (1, 3),  # Pclass: 1-3
            (0, 1),  # Sex_encoded: 0-1 (female-male)
            (0, 80),  # Age: 0-80
            (0, 8),  # SibSp: 0-8
            (0, 6),  # Parch: 0-6
            (0, 500),  # Fare: 0-500
            (0, 2),  # Embarked_encoded: 0-2
            (1, 11),  # FamilySize: 1-11
            (0, 1),  # IsAlone: 0-1
            (0, 1),  # HasCabin: 0-1
            (0, 4)  # Title_encoded: 0-4 (approximate)
        ]

        # Initial guess (average values)
        x0 = [2, 0, 30, 1, 0, 50, 0, 2, 0, 0, 1]

        # Optimize
        result = minimize(survival_probability,
                          x0,
                          bounds=bounds,
                          method='L-BFGS-B')

        optimal_features = result.x
        max_survival_prob = -result.fun

        # Create readable interpretation
        sex_map = {0: 'Female', 1: 'Male'}
        embarked_map = {0: 'C', 1: 'Q', 2: 'S'}

        optimal_passenger = {
            'pclass': int(round(optimal_features[0])),
            'sex': sex_map[int(round(optimal_features[1]))],
            'age': round(optimal_features[2], 1),
            'sibsp': int(round(optimal_features[3])),
            'parch': int(round(optimal_features[4])),
            'fare': round(optimal_features[5], 2),
            'embarked': embarked_map[int(round(optimal_features[6]))],
            'family_size': int(round(optimal_features[7])),
            'is_alone': int(round(optimal_features[8])),
            'has_cabin': int(round(optimal_features[9])),
            'max_survival_probability': round(max_survival_prob, 3)
        }

        return optimal_passenger


# Convenience functions for easy import
def load_data(file_path=None, data=None):
    """Load Titanic dataset"""
    analytics = TitanicAnalytics()
    return analytics.load_data(file_path, data)


def get_analytics_instance(data=None, file_path=None):
    """Get a configured analytics instance"""
    analytics = TitanicAnalytics()
    if data is not None or file_path is not None:
        analytics.load_data(file_path, data)
        analytics.clean_data()
    return analytics
