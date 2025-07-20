import requests
import json
import random
import time
from datetime import datetime
from faker import Faker
import pandas as pd
import re


class HuggingFaceComplaintGenerator:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self.api_base_url = "https://api-inference.huggingface.co/models"
        self.primary_model = "microsoft/DialoGPT-medium"
        self.backup_models = [
            "microsoft/DialoGPT-small",
            "facebook/blenderbot-400M-distill",
            "microsoft/DialoGPT-large"
        ]
        self.headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        self.max_retries = 3
        self.retry_delay = 2
        self.timeout = 30

    def generate_complaint(self, category, severity, emotion, demographics=None):
        prompt = self._build_prompt(category, severity, emotion, demographics)

        for attempt in range(self.max_retries):
            try:
                result = self._call_huggingface_api(prompt, self.primary_model)
                if result:
                    return self._process_response(result, category, severity, emotion)

                for backup_model in self.backup_models:
                    result = self._call_huggingface_api(prompt, backup_model)
                    if result:
                        return self._process_response(result, category, severity, emotion)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue

        return self._generate_fallback(category, severity, emotion)

    def _build_prompt(self, category, severity, emotion, demographics=None):
        base_prompts = {
            "ATM_FAILURE": f"Write a realistic banking complaint from an {emotion} customer about ATM problems. Severity: {severity}. Include specific issues like card stuck, cash not dispensed, or machine errors.",
            "FRAUD_DETECTION": f"Write a realistic banking complaint from an {emotion} customer about fraud or unauthorized transactions. Severity: {severity}. Include concerns about security and unauthorized access.",
            "UX_ISSUES": f"Write a realistic banking complaint from an {emotion} customer about digital banking problems. Severity: {severity}. Include app crashes, website issues, or confusing interfaces."
        }

        demographic_context = ""
        if demographics:
            age = demographics.get('age_group', 'middle-aged')
            tech = demographics.get('tech_savviness', 'medium')
            experience = demographics.get('banking_experience', 'intermediate')
            demographic_context = f" Customer is {age}, {tech} tech-savvy, {experience} banking experience."

        emotion_modifiers = {
            "frustrated": "Write in a frustrated, annoyed tone.",
            "angry": "Write in an angry, demanding tone.",
            "worried": "Write in a concerned, anxious tone.",
            "scared": "Write in a frightened, panicked tone.",
            "confused": "Write in a confused, puzzled tone.",
            "disappointed": "Write in a disappointed tone.",
            "panicked": "Write in a panicked, urgent tone.",
            "annoyed": "Write in an annoyed, irritated tone.",
            "impatient": "Write in an impatient, frustrated tone.",
            "stressed": "Write in a stressed, overwhelmed tone."
        }

        severity_modifiers = {
            "low": "This is a minor inconvenience.",
            "medium": "This is a significant problem.",
            "high": "This is a serious issue requiring immediate attention.",
            "critical": "This is an emergency requiring immediate escalation."
        }

        prompt = base_prompts.get(category, f"Write a banking complaint about {category}")
        prompt += f" {emotion_modifiers.get(emotion, '')} {severity_modifiers.get(severity, '')}"
        prompt += demographic_context
        prompt += "\n\nComplaint:"

        return prompt

    def _call_huggingface_api(self, prompt, model):
        api_url = f"{self.api_base_url}/{model}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 200,
                "temperature": 0.8,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1
            }
        }

        response = requests.post(
            api_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')

        return None

    def _process_response(self, generated_text, category, severity, emotion):
        if "Complaint:" in generated_text:
            complaint = generated_text.split("Complaint:")[-1].strip()
        else:
            complaint = generated_text.strip()

        complaint = self._clean_text(complaint)

        if self._validate_complaint(complaint, category, severity, emotion):
            return complaint

        return self._generate_fallback(category, severity, emotion)

    def _clean_text(self, text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        sentences = text.split('.')
        if len(sentences) > 5:
            text = '. '.join(sentences[:5]) + '.'

        return text

    def _validate_complaint(self, complaint, category, severity, emotion):
        if len(complaint) < 20 or len(complaint) > 1000:
            return False

        word_count = len(complaint.split())
        if word_count < 5 or word_count > 200:
            return False

        category_keywords = {
            "ATM_FAILURE": ["atm", "machine", "card", "cash", "withdraw", "deposit", "screen", "error"],
            "FRAUD_DETECTION": ["fraud", "unauthorized", "suspicious", "transaction", "charge", "account", "security"],
            "UX_ISSUES": ["app", "website", "interface", "login", "crash", "slow", "error", "navigation"]
        }

        complaint_lower = complaint.lower()
        keywords = category_keywords.get(category, [])

        if not any(keyword in complaint_lower for keyword in keywords):
            return False

        return True

    def _generate_fallback(self, category, severity, emotion):
        templates = {
            "ATM_FAILURE": [
                f"I am {emotion} because the ATM at your branch completely malfunctioned. This is {severity} and needs immediate attention.",
                f"Your ATM has serious issues and I'm {emotion} about it. The {severity} problem affected my banking experience.",
                f"The ATM system failed when I tried to use it and I'm {emotion}. This {severity} issue must be resolved immediately.",
                f"I'm {emotion} about the ATM malfunction that occurred. This {severity} situation requires your immediate action.",
                f"The ATM at your location had a {severity} failure and I'm {emotion} about the inconvenience caused."
            ],
            "FRAUD_DETECTION": [
                f"I'm {emotion} about unauthorized transactions on my account. This {severity} security breach is unacceptable.",
                f"Someone accessed my account without permission and I'm {emotion}. This is {severity} and needs immediate investigation.",
                f"Fraudulent activity on my card has me {emotion}. This {severity} situation requires immediate action from your security team.",
                f"I discovered suspicious charges and I'm {emotion} about this {severity} security issue that affects my financial safety.",
                f"Unauthorized use of my banking information has me {emotion}. This {severity} fraud incident demands immediate resolution."
            ],
            "UX_ISSUES": [
                f"Your mobile app is terrible and I'm {emotion}. These {severity} usability issues are preventing me from banking properly.",
                f"The website interface is confusing and I'm {emotion}. This {severity} design problem needs immediate fixing.",
                f"I'm {emotion} with your digital banking platform. The {severity} user experience issues are completely unacceptable.",
                f"Your banking app keeps malfunctioning and I'm {emotion}. These {severity} technical problems are very frustrating.",
                f"The online banking system is poorly designed and I'm {emotion}. This {severity} user interface problem affects my daily banking."
            ]
        }

        category_templates = templates.get(category,
                                           [f"I have a {severity} {category} issue and I'm {emotion} about it."])
        return random.choice(category_templates)


class ComplaintQualityValidator:
    def __init__(self):
        self.min_word_count = 10
        self.max_word_count = 300
        self.min_char_count = 50
        self.max_char_count = 1500
        self.profanity_words = set(['damn', 'hell', 'crap', 'stupid', 'idiot'])

    def validate_complaint(self, complaint_text, category, severity, emotion):
        if not self._check_length(complaint_text):
            return False, "Length validation failed"

        if not self._check_content_quality(complaint_text):
            return False, "Content quality validation failed"

        if not self._check_category_relevance(complaint_text, category):
            return False, "Category relevance validation failed"

        if not self._check_profanity(complaint_text):
            return False, "Profanity check failed"

        return True, "Validation passed"

    def _check_length(self, text):
        word_count = len(text.split())
        char_count = len(text)
        return (self.min_word_count <= word_count <= self.max_word_count and
                self.min_char_count <= char_count <= self.max_char_count)

    def _check_content_quality(self, text):
        sentences = text.split('.')
        if len(sentences) < 2:
            return False

        repeated_words = []
        words = text.lower().split()
        for word in set(words):
            if words.count(word) > 5:
                repeated_words.append(word)

        return len(repeated_words) < 3

    def _check_category_relevance(self, text, category):
        category_keywords = {
            "ATM_FAILURE": ["atm", "machine", "card", "cash", "withdraw", "deposit", "screen", "error", "transaction"],
            "FRAUD_DETECTION": ["fraud", "unauthorized", "suspicious", "transaction", "charge", "account", "security",
                                "stolen"],
            "UX_ISSUES": ["app", "website", "interface", "login", "crash", "slow", "error", "navigation", "system"]
        }

        text_lower = text.lower()
        keywords = category_keywords.get(category, [])

        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return matches >= 1

    def _check_profanity(self, text):
        text_lower = text.lower()
        explicit_profanity = ['fuck', 'shit', 'bitch', 'asshole']
        return not any(word in text_lower for word in explicit_profanity)


class EnhancedComplaintGenerator:
    def __init__(self, method="template", **kwargs):
        self.method = method
        self.fake = Faker()
        self.validator = ComplaintQualityValidator()

        if method == "huggingface":
            self.llm_generator = HuggingFaceComplaintGenerator(kwargs.get('hf_token'))
        else:
            self.llm_generator = None

        self.fallback_generator = HuggingFaceComplaintGenerator()

        self.categories = ["ATM_FAILURE", "FRAUD_DETECTION", "UX_ISSUES"]
        self.severities = {
            "ATM_FAILURE": ["low", "medium", "high", "critical"],
            "FRAUD_DETECTION": ["medium", "high", "critical"],
            "UX_ISSUES": ["low", "medium", "high"]
        }
        self.emotions = {
            "ATM_FAILURE": ["frustrated", "angry", "worried", "stressed"],
            "FRAUD_DETECTION": ["panicked", "angry", "scared", "worried"],
            "UX_ISSUES": ["frustrated", "confused", "disappointed", "annoyed"]
        }

    def generate_single_complaint(self, category=None):
        if category is None:
            category = random.choice(self.categories)

        severity = random.choice(self.severities[category])
        emotion = random.choice(self.emotions[category])

        demographics = self._generate_demographics()

        if self.llm_generator and self.method == "huggingface":
            complaint_text = self.llm_generator.generate_complaint(
                category, severity, emotion, demographics
            )
        else:
            complaint_text = self.fallback_generator._generate_fallback(
                category, severity, emotion
            )

        is_valid, validation_message = self.validator.validate_complaint(
            complaint_text, category, severity, emotion
        )

        if not is_valid:
            complaint_text = self.fallback_generator._generate_fallback(
                category, severity, emotion
            )

        metadata = self._generate_metadata(severity)

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
            "generation_method": self.method,
            "validation_status": validation_message
        }

    def generate_batch_complaints(self, total_count, category_distribution=None):
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

    def _generate_demographics(self):
        age_groups = ["18-30", "31-50", "51-65", "65+"]
        tech_levels = ["low", "medium", "high"]
        experience_levels = ["novice", "intermediate", "expert"]

        return {
            "age_group": random.choice(age_groups),
            "tech_savviness": random.choice(tech_levels),
            "banking_experience": random.choice(experience_levels)
        }

    def _generate_metadata(self, severity):
        channels = ["phone", "email", "chat", "branch", "mobile_app", "website"]
        account_types = ["checking", "savings", "credit", "business", "investment"]

        resolution_hours = {
            "low": random.randint(1, 24),
            "medium": random.randint(2, 48),
            "high": random.randint(1, 72),
            "critical": random.randint(1, 8)
        }

        return {
            "complaint_id": self.fake.uuid4(),
            "customer_id": self.fake.uuid4(),
            "timestamp": self.fake.date_time_between(start_date="-30d", end_date="now"),
            "channel": random.choice(channels),
            "customer_name": self.fake.name(),
            "account_type": random.choice(account_types),
            "resolution_time_hours": resolution_hours.get(severity, 24),
            "location": self.fake.city() + ", " + self.fake.state_abbr()
        }

    def export_complaints(self, complaints, output_format="csv", filename=None):
        if not complaints:
            raise ValueError("No complaints to export")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        method_suffix = f"_{self.method}" if self.method != "template" else ""

        if output_format.lower() == "csv":
            if filename is None:
                filename = f"data/llm_generated/synthetic_complaints_{timestamp}{method_suffix}.csv"

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
                    "generation_method": complaint["generation_method"],
                    "validation_status": complaint.get("validation_status", "unknown"),
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
            return filename

        elif output_format.lower() == "json":
            if filename is None:
                filename = f"data/llm_generated/synthetic_complaints_{timestamp}{method_suffix}.json"

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(complaints, f, indent=2, default=str, ensure_ascii=False)
            return filename

        else:
            raise ValueError("Supported formats: 'csv', 'json'")

    def analyze_generated_data(self, complaints):
        if not complaints:
            return {"error": "No complaints to analyze"}

        analysis = {
            "total_complaints": len(complaints),
            "generation_method": self.method,
            "category_distribution": {},
            "severity_distribution": {},
            "emotion_distribution": {},
            "channel_distribution": {},
            "word_count_stats": {
                "avg": sum(c["word_count"] for c in complaints) / len(complaints),
                "min": min(c["word_count"] for c in complaints),
                "max": max(c["word_count"] for c in complaints)
            },
            "character_count_stats": {
                "avg": sum(c["character_count"] for c in complaints) / len(complaints),
                "min": min(c["character_count"] for c in complaints),
                "max": max(c["character_count"] for c in complaints)
            },
            "demographics_breakdown": {
                "age_group": {},
                "tech_savviness": {},
                "banking_experience": {}
            },
            "validation_summary": {}
        }

        for complaint in complaints:
            cat = complaint["category"]
            analysis["category_distribution"][cat] = analysis["category_distribution"].get(cat, 0) + 1

            sev = complaint["severity"]
            analysis["severity_distribution"][sev] = analysis["severity_distribution"].get(sev, 0) + 1

            emo = complaint["emotion"]
            analysis["emotion_distribution"][emo] = analysis["emotion_distribution"].get(emo, 0) + 1

            channel = complaint["metadata"]["channel"]
            analysis["channel_distribution"][channel] = analysis["channel_distribution"].get(channel, 0) + 1

            validation = complaint.get("validation_status", "unknown")
            analysis["validation_summary"][validation] = analysis["validation_summary"].get(validation, 0) + 1

            for demo_type, demo_value in complaint["demographics"].items():
                analysis["demographics_breakdown"][demo_type][demo_value] = \
                    analysis["demographics_breakdown"][demo_type].get(demo_value, 0) + 1

        return analysis

    def get_generation_stats(self):
        return {
            "method": self.method,
            "has_llm": self.llm_generator is not None,
            "supported_categories": self.categories,
            "severity_levels": self.severities,
            "emotion_types": self.emotions,
            "validation_enabled": True
        }