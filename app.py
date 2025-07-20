import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import io
import requests
import random

from generator import FreeComplaintGenerator


class HuggingFaceComplaintGenerator:
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        self.headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    def generate_complaint(self, category: str, severity: str, emotion: str) -> str:
        prompt = f"""Customer complaint about {category}. Severity: {severity}. Emotion: {emotion}.

Write a realistic banking complaint from an {emotion} customer about {category} issues. Make it {severity} severity.

Complaint:"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 200,
                "temperature": 0.8,
                "do_sample": True,
                "top_p": 0.9
            }
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    complaint = generated_text.split("Complaint:")[-1].strip()
                    return complaint if complaint else self._generate_fallback(category, severity, emotion)

            return self._generate_fallback(category, severity, emotion)

        except Exception as e:
            return self._generate_fallback(category, severity, emotion)

    def _generate_fallback(self, category: str, severity: str, emotion: str) -> str:
        templates = {
            "ATM_FAILURE": [
                f"I am {emotion} because the ATM at your branch completely malfunctioned. This is {severity} and needs immediate attention.",
                f"Your ATM has serious issues and I'm {emotion} about it. The {severity} problem affected my banking.",
                f"The ATM system failed and I'm {emotion}. This {severity} issue must be resolved immediately."
            ],
            "FRAUD_DETECTION": [
                f"I'm {emotion} about unauthorized transactions on my account. This {severity} security breach is unacceptable.",
                f"Someone accessed my account without permission and I'm {emotion}. This is {severity} and frightening.",
                f"Fraudulent activity on my card has me {emotion}. This {severity} situation needs immediate action."
            ],
            "UX_ISSUES": [
                f"Your mobile app is terrible and I'm {emotion}. These {severity} usability issues are frustrating.",
                f"The website interface is confusing and I'm {emotion}. This {severity} design problem needs fixing.",
                f"I'm {emotion} with your digital banking platform. The {severity} user experience issues are unacceptable."
            ]
        }

        return random.choice(templates.get(category, [f"I have a {severity} {category} issue and I'm {emotion}."]))


class FreeEnhancedComplaintGenerator:
    def __init__(self, method: str = "fallback", **kwargs):
        self.method = method
        self.generator = None

        if method == "huggingface":
            self.generator = HuggingFaceComplaintGenerator(kwargs.get('hf_token'))

        self.fallback_generator = HuggingFaceComplaintGenerator()

    def generate_single_complaint(self, category=None):
        if category is None:
            category = random.choice(["ATM_FAILURE", "FRAUD_DETECTION", "UX_ISSUES"])

        severities = {
            "ATM_FAILURE": ["low", "medium", "high", "critical"],
            "FRAUD_DETECTION": ["medium", "high", "critical"],
            "UX_ISSUES": ["low", "medium", "high"]
        }

        emotions = {
            "ATM_FAILURE": ["frustrated", "angry", "worried", "stressed"],
            "FRAUD_DETECTION": ["panicked", "angry", "scared", "worried"],
            "UX_ISSUES": ["frustrated", "confused", "disappointed", "annoyed"]
        }

        severity = random.choice(severities[category])
        emotion = random.choice(emotions[category])

        if self.generator:
            complaint_text = self.generator.generate_complaint(category, severity, emotion)
        else:
            complaint_text = self.fallback_generator._generate_fallback(category, severity, emotion)

        from faker import Faker
        fake = Faker()

        demographics = {
            "age_group": random.choice(["18-30", "31-50", "51-65", "65+"]),
            "tech_savviness": random.choice(["low", "medium", "high"]),
            "banking_experience": random.choice(["novice", "intermediate", "expert"])
        }

        metadata = {
            "complaint_id": fake.uuid4(),
            "customer_id": fake.uuid4(),
            "timestamp": fake.date_time_between(start_date="-30d", end_date="now"),
            "channel": random.choice(["phone", "email", "chat", "branch", "mobile_app", "website"]),
            "customer_name": fake.name(),
            "account_type": random.choice(["checking", "savings", "credit", "business", "investment"]),
            "resolution_time_hours": random.randint(1, 72) if severity != "low" else random.randint(1, 24),
            "location": fake.city() + ", " + fake.state_abbr()
        }

        return {
            "complaint_text": complaint_text,
            "category": category,
            "specific_issue": f"{self.method}_generated_{category.lower()}",
            "severity": severity,
            "emotion": emotion,
            "demographics": demographics,
            "metadata": metadata,
            "word_count": len(complaint_text.split()),
            "character_count": len(complaint_text),
            "generated_at": datetime.now().isoformat(),
            "generation_method": self.method
        }

    def generate_batch_complaints(self, total_count: int, category_distribution=None):
        if category_distribution is None:
            category_distribution = {
                "ATM_FAILURE": 0.4,
                "FRAUD_DETECTION": 0.3,
                "UX_ISSUES": 0.3
            }

        complaints = []
        categories = list(category_distribution.keys())
        weights = list(category_distribution.values())

        for i in range(total_count):
            category = random.choices(categories, weights=weights)[0]
            complaint = self.generate_single_complaint(category)
            complaints.append(complaint)

            if (i + 1) % 25 == 0:
                print(f"Generated {i + 1}/{total_count} complaints using {self.method}...")

        return complaints


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
            'severity_accuracy': sev_accuracy
        }

    def save_models(self):
        if not self.models_trained:
            raise ValueError("Models not trained yet. Call train() first.")

        os.makedirs('models', exist_ok=True)
        joblib.dump(self.vectorizer, 'models/complaint_vectorizer.pkl')
        joblib.dump(self.category_model, 'models/category_model.pkl')
        joblib.dump(self.emotion_model, 'models/emotion_model.pkl')
        joblib.dump(self.severity_model, 'models/severity_model.pkl')

    def load_models(self):
        try:
            self.vectorizer = joblib.load('models/complaint_vectorizer.pkl')
            self.category_model = joblib.load('models/category_model.pkl')
            self.emotion_model = joblib.load('models/emotion_model.pkl')
            self.severity_model = joblib.load('models/severity_model.pkl')
            self.models_trained = True
            return True
        except:
            return False


def initialize_session_state():
    directories = ['data', 'models', 'data/exports', 'data/uploads', 'data/llm_generated']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    if 'generator' not in st.session_state:
        st.session_state.generator = FreeComplaintGenerator()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ComplaintAnalyzer()
    if 'complaints_data' not in st.session_state:
        st.session_state.complaints_data = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = st.session_state.analyzer.load_models()


def create_visualizations(df):
    fig_category = px.pie(df, names='category', title="Complaint Categories Distribution")

    fig_severity = px.histogram(df, x='severity', title="Severity Distribution",
                                color='severity', category_orders={'severity': ['low', 'medium', 'high', 'critical']})

    emotion_counts = df['emotion'].value_counts().reset_index()
    fig_emotion = px.bar(emotion_counts,
                         x='emotion', y='count', title="Emotion Distribution")

    channel_counts = df['channel'].value_counts().reset_index()
    fig_channel = px.bar(channel_counts,
                         x='channel', y='count', title="Channel Distribution")

    return fig_category, fig_severity, fig_emotion, fig_channel


def generate_page():
    st.header("🏭 Generate Synthetic Complaints")

    st.subheader("🤖 Generation Method")

    col1, col2 = st.columns([2, 1])

    with col1:
        generation_method = st.selectbox(
            "Choose Generation Method:",
            ["template", "huggingface"],
            format_func=lambda x: {
                "template": "📝 Template-based (Original)",
                "huggingface": "🤗 Hugging Face (AI-Enhanced)"
            }[x]
        )

    with col2:
        if generation_method == "huggingface":
            st.info("💡 AI-Enhanced Generation")
        else:
            st.info("📝 Traditional Generation")

    llm_config = {}

    if generation_method == "huggingface":
        st.subheader("🤗 Hugging Face Configuration")
        hf_token = st.text_input(
            "Hugging Face Token (Get free at: https://huggingface.co/settings/tokens)",
            type="password",
            help="Free token - just sign up at Hugging Face"
        )
        llm_config['hf_token'] = hf_token

        if not hf_token:
            st.warning("⚠️ Please provide Hugging Face token for AI generation")

    col1, col2 = st.columns(2)

    with col1:
        total_count = st.number_input("Number of Complaints", min_value=10, max_value=5000, value=500, step=10)

        st.subheader("Category Distribution")
        atm_failure = st.slider("ATM Failure %", 0, 100, 40)
        fraud_detection = st.slider("Fraud Detection %", 0, 100, 35)
        ux_issues = 100 - atm_failure - fraud_detection
        st.write(f"UX Issues: {ux_issues}%")

        if atm_failure + fraud_detection > 100:
            st.error("Total percentage cannot exceed 100%")
            return

    with col2:
        st.subheader("Export Options")
        export_format = st.selectbox("Export Format", ["CSV", "JSON"])
        include_metadata = st.checkbox("Include Full Metadata", value=True)

        if generation_method != "template":
            st.subheader("🎯 AI Generation Benefits")
            st.success("✅ More realistic language")
            st.success("✅ Greater variety")
            st.success("✅ Context-aware emotions")

    generate_button_text = {
        "template": "🚀 Generate Complaints (Template)",
        "huggingface": "🤗 Generate with AI (HuggingFace)"
    }

    if st.button(generate_button_text[generation_method], type="primary"):
        if generation_method == "huggingface" and not llm_config.get('hf_token'):
            st.error("Please provide Hugging Face token")
            return

        with st.spinner(f"Generating complaints using {generation_method}..."):
            category_dist = {
                "ATM_FAILURE": atm_failure / 100,
                "FRAUD_DETECTION": fraud_detection / 100,
                "UX_ISSUES": ux_issues / 100
            }

            if generation_method == "template":
                generator = st.session_state.generator
                complaints = generator.generate_batch_complaints(
                    total_count=total_count,
                    category_distribution=category_dist
                )
            else:
                try:
                    enhanced_generator = FreeEnhancedComplaintGenerator(
                        method=generation_method,
                        **llm_config
                    )
                    complaints = enhanced_generator.generate_batch_complaints(
                        total_count=total_count,
                        category_distribution=category_dist
                    )
                except Exception as e:
                    st.error(f"LLM generation failed: {e}")
                    st.info("Falling back to template generation...")
                    generator = st.session_state.generator
                    complaints = generator.generate_batch_complaints(
                        total_count=total_count,
                        category_distribution=category_dist
                    )

            st.session_state.complaints_data = complaints

            os.makedirs('data', exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            method_suffix = f"_{generation_method}" if generation_method != "template" else ""

            if export_format.lower() == "csv":
                filename = f"data/synthetic_complaints_{timestamp}{method_suffix}.csv"
                flattened_data = []
                for complaint in complaints:
                    flat_record = {
                        "complaint_text": complaint["complaint_text"],
                        "category": complaint["category"],
                        "specific_issue": complaint["specific_issue"],
                        "severity": complaint["severity"],
                        "emotion": complaint["emotion"],
                        "age_group": complaint["demographics"]["age_group"],
                        "tech_savviness": complaint["demographics"]["tech_savviness"],
                        "banking_experience": complaint["demographics"]["banking_experience"],
                        "word_count": complaint["word_count"],
                        "character_count": complaint["character_count"],
                        "generation_method": complaint.get("generation_method", "template"),
                        "customer_id": complaint["metadata"]["customer_id"],
                        "timestamp": complaint["metadata"]["timestamp"],
                        "channel": complaint["metadata"]["channel"],
                        "account_type": complaint["metadata"]["account_type"],
                        "resolution_time_hours": complaint["metadata"]["resolution_time_hours"],
                        "location": complaint["metadata"]["location"]
                    }
                    flattened_data.append(flat_record)

                df = pd.DataFrame(flattened_data)
                df.to_csv(filename, index=False)
            else:
                filename = f"data/synthetic_complaints_{timestamp}{method_suffix}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(complaints, f, indent=2, default=str, ensure_ascii=False)

            st.session_state.df = pd.read_csv(filename) if export_format.lower() == "csv" else pd.read_json(filename)

            if generation_method != "template":
                st.success(f"✅ Generated {len(complaints)} AI-enhanced complaints using {generation_method}!")
                st.info(f"🎯 Generation method: {generation_method}")
            else:
                st.success(f"✅ Generated {len(complaints)} template-based complaints!")

            analysis = {}
            if hasattr(st.session_state.generator, 'analyze_generated_data'):
                analysis = st.session_state.generator.analyze_generated_data(complaints)
            else:
                analysis = {
                    "total_complaints": len(complaints),
                    "category_distribution": {},
                    "emotion_distribution": {},
                    "word_count_stats": {
                        "avg": sum(c["word_count"] for c in complaints) / len(complaints),
                        "min": min(c["word_count"] for c in complaints),
                        "max": max(c["word_count"] for c in complaints)
                    }
                }

                for complaint in complaints:
                    cat = complaint["category"]
                    analysis["category_distribution"][cat] = analysis["category_distribution"].get(cat, 0) + 1

                    emo = complaint["emotion"]
                    analysis["emotion_distribution"][emo] = analysis["emotion_distribution"].get(emo, 0) + 1

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Complaints", analysis["total_complaints"])
            with col2:
                st.metric("Avg Word Count", f"{analysis['word_count_stats']['avg']:.1f}")
            with col3:
                st.metric("Categories", len(analysis["category_distribution"]))
            with col4:
                st.metric("Generation Method", generation_method.title())

            if generation_method != "template" and complaints:
                st.subheader("🎯 Sample AI-Generated Complaints")
                sample_complaints = random.sample(complaints, min(3, len(complaints)))

                for i, complaint in enumerate(sample_complaints):
                    with st.expander(f"Sample {i + 1}: {complaint['category']} ({complaint['severity']})"):
                        st.write(f"**Emotion:** {complaint['emotion']}")
                        st.write(f"**Text:** {complaint['complaint_text']}")
                        st.write(
                            f"**Words:** {complaint['word_count']} | **Method:** {complaint.get('generation_method', 'unknown')}")

            if st.session_state.df is not None:
                fig_cat, fig_sev, fig_emo, fig_chan = create_visualizations(st.session_state.df)

                st.plotly_chart(fig_cat, use_container_width=True, key="generate_category_chart")

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_sev, use_container_width=True, key="generate_severity_chart")
                with col2:
                    st.plotly_chart(fig_emo, use_container_width=True, key="generate_emotion_chart")

            with open(filename, 'rb') as f:
                st.download_button(
                    label=f"📥 Download {export_format} File ({generation_method})",
                    data=f.read(),
                    file_name=filename.split('/')[-1],
                    mime='text/csv' if export_format.lower() == 'csv' else 'application/json'
                )


def train_page():
    st.header("🎯 Train ML Models")

    if st.session_state.df is None:
        st.warning("No data available. Please generate complaints first or upload a dataset.")

        uploaded_file = st.file_uploader("Upload Training Data (CSV)", type=['csv'])
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")

    if st.session_state.df is not None:
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(st.session_state.df))
        with col2:
            st.metric("Categories", st.session_state.df['category'].nunique())
        with col3:
            st.metric("Emotions", st.session_state.df['emotion'].nunique())
        with col4:
            st.metric("Severity Levels", st.session_state.df['severity'].nunique())

        st.dataframe(st.session_state.df.head(), use_container_width=True)

        if st.button("🔄 Train Models", type="primary"):
            with st.spinner("Training ML models..."):
                try:
                    st.session_state.analyzer.train(st.session_state.df)

                    evaluation = st.session_state.analyzer.evaluate_models(st.session_state.df)

                    st.success("✅ Models trained successfully!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Category Accuracy", f"{evaluation['category_accuracy']:.3f}")
                    with col2:
                        st.metric("Emotion Accuracy", f"{evaluation['emotion_accuracy']:.3f}")
                    with col3:
                        st.metric("Severity Accuracy", f"{evaluation['severity_accuracy']:.3f}")

                    st.session_state.analyzer.save_models()
                    st.session_state.models_loaded = True
                    st.info("💾 Models saved to 'models/' directory")

                except Exception as e:
                    st.error(f"Training failed: {str(e)}")


def analyze_page():
    st.header("🔍 Analyze Complaints")

    if not st.session_state.models_loaded:
        st.warning("No trained models found. Please train models first.")
        return

    st.subheader("Single Complaint Analysis")

    sample_complaints = [
        "The ATM ate my card and I'm furious about it",
        "Your mobile app keeps crashing when I try to login",
        "Someone stole my credit card information and made purchases",
        "I can't find the transfer button on your confusing website"
    ]

    selected_sample = st.selectbox("Choose a sample complaint:", [""] + sample_complaints)

    complaint_text = st.text_area(
        "Enter complaint text:",
        value=selected_sample,
        height=100,
        placeholder="Type your complaint here..."
    )

    if complaint_text and st.button("🚀 Analyze", type="primary"):
        try:
            result = st.session_state.analyzer.analyze_complaint(complaint_text)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Category", result['category'])
                st.caption(f"Confidence: {result['confidence']['category']}")

            with col2:
                st.metric("Emotion", result['emotion'])
                st.caption(f"Confidence: {result['confidence']['emotion']}")

            with col3:
                st.metric("Severity", result['severity'])
                st.caption(f"Confidence: {result['confidence']['severity']}")

            severity_color = {
                'low': 'green',
                'medium': 'orange',
                'high': 'red',
                'critical': 'darkred'
            }

            st.markdown(f"""
            ### 📊 Analysis Summary
            - **Category**: {result['category']}
            - **Emotion**: {result['emotion']} 
            - **Severity**: <span style="color: {severity_color.get(result['severity'], 'black')}">{result['severity']}</span>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

    st.subheader("Batch Analysis")

    uploaded_file = st.file_uploader("Upload complaints for batch analysis (CSV)", type=['csv'])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)

        if 'complaint_text' not in df_batch.columns:
            st.error("CSV must contain 'complaint_text' column")
        else:
            if st.button("🔄 Analyze Batch", type="secondary"):
                with st.spinner("Analyzing batch..."):
                    results = []
                    for text in df_batch['complaint_text']:
                        result = st.session_state.analyzer.analyze_complaint(text)
                        results.append(result)

                    df_results = pd.DataFrame(results)
                    df_combined = pd.concat([df_batch, df_results], axis=1)

                    st.success(f"✅ Analyzed {len(df_combined)} complaints")
                    st.dataframe(df_combined, use_container_width=True)

                    csv_output = df_combined.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv_output,
                        f"analyzed_complaints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )


def dashboard_page():
    st.header("📊 Analytics Dashboard")

    if st.session_state.df is None:
        st.warning("No data available. Please generate complaints or upload data first.")
        return

    df = st.session_state.df

    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Complaints", len(df))
    with col2:
        avg_words = df['word_count'].mean()
        st.metric("Avg Words", f"{avg_words:.1f}")
    with col3:
        high_severity = len(df[df['severity'].isin(['high', 'critical'])])
        st.metric("High Severity", high_severity)
    with col4:
        fraud_complaints = len(df[df['category'] == 'FRAUD_DETECTION'])
        st.metric("Fraud Cases", fraud_complaints)

    fig_cat, fig_sev, fig_emo, fig_chan = create_visualizations(df)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_cat, use_container_width=True, key="dashboard_category_chart")
    with col2:
        st.plotly_chart(fig_sev, use_container_width=True, key="dashboard_severity_chart")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_emo, use_container_width=True, key="dashboard_emotion_chart")
    with col2:
        st.plotly_chart(fig_chan, use_container_width=True, key="dashboard_channel_chart")

    st.subheader("Detailed Data")

    category_filter = st.multiselect("Filter by Category", df['category'].unique(), default=df['category'].unique())
    severity_filter = st.multiselect("Filter by Severity", df['severity'].unique(), default=df['severity'].unique())

    filtered_df = df[
        (df['category'].isin(category_filter)) &
        (df['severity'].isin(severity_filter))
        ]

    st.dataframe(filtered_df, use_container_width=True)

    csv_output = filtered_df.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Data",
        csv_output,
        f"filtered_complaints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )


def main():
    st.set_page_config(
        page_title="Complaint Analysis System",
        page_icon="🏦",
        layout="wide"
    )

    initialize_session_state()

    st.title("🏦 Customer Complaint Analysis System")
    st.markdown("Generate synthetic complaints and train ML models for automated analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["🏭 Generate", "🎯 Train", "🔍 Analyze", "📊 Dashboard"])

    with tab1:
        generate_page()

    with tab2:
        train_page()

    with tab3:
        analyze_page()

    with tab4:
        dashboard_page()

    st.sidebar.header("System Status")

    if st.session_state.complaints_data:
        st.sidebar.success(f"✅ {len(st.session_state.complaints_data)} complaints generated")
    else:
        st.sidebar.info("ℹ️ No complaints generated yet")

    if st.session_state.models_loaded:
        st.sidebar.success("✅ ML models loaded")
    else:
        st.sidebar.warning("⚠️ ML models not trained")

    if st.session_state.df is not None:
        st.sidebar.success(f"✅ Dataset loaded ({len(st.session_state.df)} records)")
    else:
        st.sidebar.info("ℹ️ No dataset loaded")


if __name__ == "__main__":
    main()