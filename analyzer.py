import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ComplaintAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.category_model = LogisticRegression(max_iter=1000)
        self.emotion_model = LogisticRegression(max_iter=1000)
        self.severity_model = LogisticRegression(max_iter=1000)
        self.models_trained = False

    def train(self, df):
        X = df['complaint_text']
        X_vec = self.vectorizer.fit_transform(X)

        self.category_model.fit(X_vec, df['category'])
        self.emotion_model.fit(X_vec, df['emotion'])
        self.severity_model.fit(X_vec, df['severity'])

        self.models_trained = True

    def analyze_complaint(self, complaint_text):
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call train() first.")

        X_vec = self.vectorizer.transform([complaint_text])

        category = self.category_model.predict(X_vec)[0]
        emotion = self.emotion_model.predict(X_vec)[0]
        severity = self.severity_model.predict(X_vec)[0]

        cat_proba = self.category_model.predict_proba(X_vec)[0].max()
        emo_proba = self.emotion_model.predict_proba(X_vec)[0].max()
        sev_proba = self.severity_model.predict_proba(X_vec)[0].max()

        return {
            'category': category,
            'emotion': emotion,
            'severity': severity,
            'confidence': {
                'category': f"{cat_proba:.2%}",
                'emotion': f"{emo_proba:.2%}",
                'severity': f"{sev_proba:.2%}"
            },
            'raw_confidence': {
                'category': cat_proba,
                'emotion': emo_proba,
                'severity': sev_proba
            }
        }

    def evaluate_models(self, df):
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call train() first.")

        X = df['complaint_text']
        X_vec = self.vectorizer.transform(X)

        cat_pred = self.category_model.predict(X_vec)
        cat_accuracy = accuracy_score(df['category'], cat_pred)

        emo_pred = self.emotion_model.predict(X_vec)
        emo_accuracy = accuracy_score(df['emotion'], emo_pred)

        sev_pred = self.severity_model.predict(X_vec)
        sev_accuracy = accuracy_score(df['severity'], sev_pred)

        return {
            'category_accuracy': cat_accuracy,
            'emotion_accuracy': emo_accuracy,
            'severity_accuracy': sev_accuracy,
            'category_report': classification_report(df['category'], cat_pred, output_dict=True),
            'emotion_report': classification_report(df['emotion'], emo_pred, output_dict=True),
            'severity_report': classification_report(df['severity'], sev_pred, output_dict=True)
        }

    def save_models(self, model_dir='models'):
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call train() first.")

        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.vectorizer, f'{model_dir}/complaint_vectorizer.pkl')
        joblib.dump(self.category_model, f'{model_dir}/category_model.pkl')
        joblib.dump(self.emotion_model, f'{model_dir}/emotion_model.pkl')
        joblib.dump(self.severity_model, f'{model_dir}/severity_model.pkl')

    def load_models(self, model_dir='models'):
        try:
            self.vectorizer = joblib.load(f'{model_dir}/complaint_vectorizer.pkl')
            self.category_model = joblib.load(f'{model_dir}/category_model.pkl')
            self.emotion_model = joblib.load(f'{model_dir}/emotion_model.pkl')
            self.severity_model = joblib.load(f'{model_dir}/severity_model.pkl')
            self.models_trained = True
            return True
        except Exception as e:
            return False

    def get_feature_importance(self, top_n=20):
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call train() first.")

        feature_names = self.vectorizer.get_feature_names_out()

        category_coef = self.category_model.coef_
        emotion_coef = self.emotion_model.coef_
        severity_coef = self.severity_model.coef_

        importance_data = {
            'category': {},
            'emotion': {},
            'severity': {}
        }

        for i, class_name in enumerate(self.category_model.classes_):
            top_indices = np.argsort(np.abs(category_coef[i]))[-top_n:]
            importance_data['category'][class_name] = [
                (feature_names[idx], category_coef[i][idx]) for idx in top_indices[::-1]
            ]

        for i, class_name in enumerate(self.emotion_model.classes_):
            top_indices = np.argsort(np.abs(emotion_coef[i]))[-top_n:]
            importance_data['emotion'][class_name] = [
                (feature_names[idx], emotion_coef[i][idx]) for idx in top_indices[::-1]
            ]

        for i, class_name in enumerate(self.severity_model.classes_):
            top_indices = np.argsort(np.abs(severity_coef[i]))[-top_n:]
            importance_data['severity'][class_name] = [
                (feature_names[idx], severity_coef[i][idx]) for idx in top_indices[::-1]
            ]

        return importance_data

    def predict_batch(self, complaint_texts):
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call train() first.")

        results = []
        for text in complaint_texts:
            result = self.analyze_complaint(text)
            results.append(result)
        return results

    def create_confusion_matrices(self, df):
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call train() first.")

        X = df['complaint_text']
        X_vec = self.vectorizer.transform(X)

        cat_pred = self.category_model.predict(X_vec)
        emo_pred = self.emotion_model.predict(X_vec)
        sev_pred = self.severity_model.predict(X_vec)

        cat_cm = confusion_matrix(df['category'], cat_pred)
        emo_cm = confusion_matrix(df['emotion'], emo_pred)
        sev_cm = confusion_matrix(df['severity'], sev_pred)

        return {
            'category': {
                'matrix': cat_cm,
                'labels': self.category_model.classes_
            },
            'emotion': {
                'matrix': emo_cm,
                'labels': self.emotion_model.classes_
            },
            'severity': {
                'matrix': sev_cm,
                'labels': self.severity_model.classes_
            }
        }

    def get_model_info(self):
        if not self.models_trained:
            return {"error": "Models not trained yet"}

        return {
            "vectorizer_features": self.vectorizer.max_features,
            "category_classes": list(self.category_model.classes_),
            "emotion_classes": list(self.emotion_model.classes_),
            "severity_classes": list(self.severity_model.classes_),
            "models_trained": self.models_trained
        }


class AdvancedComplaintAnalyzer(ComplaintAnalyzer):
    def __init__(self):
        super().__init__()
        self.rf_category_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.nb_category_model = MultinomialNB()
        self.ensemble_models = False

    def train_ensemble(self, df):
        X = df['complaint_text']
        X_vec = self.vectorizer.fit_transform(X)

        self.category_model.fit(X_vec, df['category'])
        self.rf_category_model.fit(X_vec, df['category'])
        self.nb_category_model.fit(X_vec, df['category'])

        self.emotion_model.fit(X_vec, df['emotion'])
        self.severity_model.fit(X_vec, df['severity'])

        self.models_trained = True
        self.ensemble_models = True

    def analyze_complaint_ensemble(self, complaint_text):
        if not self.models_trained or not self.ensemble_models:
            return self.analyze_complaint(complaint_text)

        X_vec = self.vectorizer.transform([complaint_text])

        lr_pred = self.category_model.predict_proba(X_vec)[0]
        rf_pred = self.rf_category_model.predict_proba(X_vec)[0]
        nb_pred = self.nb_category_model.predict_proba(X_vec)[0]

        ensemble_pred = (lr_pred + rf_pred + nb_pred) / 3
        category = self.category_model.classes_[np.argmax(ensemble_pred)]

        emotion = self.emotion_model.predict(X_vec)[0]
        severity = self.severity_model.predict(X_vec)[0]

        return {
            'category': category,
            'emotion': emotion,
            'severity': severity,
            'ensemble_confidence': f"{ensemble_pred.max():.2%}",
            'individual_predictions': {
                'logistic_regression': self.category_model.classes_[np.argmax(lr_pred)],
                'random_forest': self.rf_category_model.classes_[np.argmax(rf_pred)],
                'naive_bayes': self.nb_category_model.classes_[np.argmax(nb_pred)]
            }
        }