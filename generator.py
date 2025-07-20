import random
import json
from datetime import datetime
from faker import Faker
import pandas as pd
from config import COMPLAINT_TEMPLATES, DEMOGRAPHICS, METADATA_OPTIONS


class FreeComplaintGenerator:
    def __init__(self):
        self.fake = Faker()
        self.complaint_templates = COMPLAINT_TEMPLATES

    def generate_single_complaint(self, category=None):
        if category is None:
            category = random.choice(["ATM_FAILURE", "FRAUD_DETECTION", "UX_ISSUES"])

        if category not in self.complaint_templates:
            raise ValueError(f"Invalid category: {category}")

        templates = self.complaint_templates[category]

        if category == "ATM_FAILURE":
            severity = random.choice(["low", "medium", "high", "critical"])
            emotion = random.choice(list(templates["emotions"].keys()))
            specific_issue = random.choice(
                ["card_stuck", "cash_not_dispensed", "screen_frozen", "receipt_not_printed", "keypad_unresponsive",
                 "out_of_service", "network_error", "wrong_amount_dispensed", "card_not_returned"])
        elif category == "FRAUD_DETECTION":
            severity = random.choice(["medium", "high", "critical"])
            emotion = random.choice(list(templates["emotions"].keys()))
            specific_issue = random.choice(
                ["unauthorized_transaction", "card_skimming", "account_takeover", "phishing_attempt",
                 "suspicious_login", "identity_theft"])
        else:
            severity = random.choice(["low", "medium", "high"])
            emotion = random.choice(list(templates["emotions"].keys()))
            specific_issue = random.choice(
                ["confusing_interface", "slow_loading", "accessibility_issues", "mobile_app_crashes",
                 "unclear_instructions", "poor_navigation"])

        demographics = {
            "age_group": random.choice(DEMOGRAPHICS["age_groups"]),
            "tech_savviness": random.choice(DEMOGRAPHICS["tech_savviness"]),
            "banking_experience": random.choice(DEMOGRAPHICS["banking_experience"])
        }

        template = random.choice(templates["templates"])

        replacements = {
            "emotion": random.choice(templates["emotions"][emotion]),
            "emotion_start": random.choice(
                templates.get("emotion_starts", {}).get(emotion, [templates["emotions"][emotion][0]])),
            "emotion_statement": f"I am {random.choice(templates['emotions'][emotion])} about this situation.",
            "severity_desc": random.choice(templates["severity_descriptions"][severity])
        }

        if category == "ATM_FAILURE":
            replacements.update({
                "location": random.choice(templates["locations"]),
                "time_ref": random.choice(templates["time_refs"]),
                "action": random.choice(templates["actions"]),
                "failure_description": random.choice(templates["failure_descriptions"]),
                "impact_statement": random.choice(templates["impact_statements"]),
                "resolution_request": random.choice(templates["resolution_requests"])
            })
        elif category == "FRAUD_DETECTION":
            replacements.update({
                "fraud_discovery": random.choice(templates["fraud_discoveries"]),
                "time_ref": random.choice(templates["time_refs"]),
                "fraud_details": random.choice(templates["fraud_details"]),
                "impact_statement": random.choice(templates["impact_statements"]),
                "resolution_request": random.choice(templates["resolution_requests"]),
                "fraud_type": random.choice(templates["fraud_types"])
            })
        else:
            replacements.update({
                "platform": random.choice(templates["platforms"]),
                "problem_description": random.choice(templates["problem_descriptions"]),
                "time_ref": random.choice(templates["time_refs"]),
                "task": random.choice(templates["tasks"]),
                "specific_issue": random.choice(templates["specific_issues"]),
                "impact_statement": random.choice(templates["impact_statements"]),
                "comparison": random.choice(templates["comparisons"]),
                "resolution_request": random.choice(templates["resolution_requests"])
            })

        complaint_text = template.format(**replacements)

        metadata = {
            "complaint_id": self.fake.uuid4(),
            "customer_id": self.fake.uuid4(),
            "timestamp": self.fake.date_time_between(start_date="-30d", end_date="now"),
            "channel": random.choice(METADATA_OPTIONS["channels"]),
            "customer_name": self.fake.name(),
            "account_type": random.choice(METADATA_OPTIONS["account_types"]),
            "resolution_time_hours": random.randint(1, 72) if severity != "low" else random.randint(1, 24),
            "location": self.fake.city() + ", " + self.fake.state_abbr()
        }

        return {
            "complaint_text": complaint_text,
            "category": category,
            "specific_issue": specific_issue,
            "severity": severity,
            "emotion": emotion,
            "demographics": demographics,
            "metadata": metadata,
            "word_count": len(complaint_text.split()),
            "character_count": len(complaint_text),
            "generated_at": datetime.now().isoformat()
        }

    def generate_batch_complaints(self, total_count, category_distribution=None):
        if category_distribution is None:
            category_distribution = {
                "ATM_FAILURE": 0.4,
                "FRAUD_DETECTION": 0.3,
                "UX_ISSUES": 0.3
            }

        if abs(sum(category_distribution.values()) - 1.0) > 0.01:
            raise ValueError("Category distribution must sum to 1.0")

        complaints = []
        generated_count = 0

        for category, proportion in category_distribution.items():
            count = int(total_count * proportion)

            for i in range(count):
                complaint = self.generate_single_complaint(category)
                complaints.append(complaint)
                generated_count += 1

        remaining = total_count - len(complaints)
        if remaining > 0:
            for _ in range(remaining):
                complaint = self.generate_single_complaint()
                complaints.append(complaint)

        random.shuffle(complaints)
        return complaints

    def export_training_data(self, complaints, output_format="csv", filename=None):
        if not complaints:
            raise ValueError("No complaints to export")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_format.lower() == "csv":
            if filename is None:
                filename = f"data/synthetic_complaints_{timestamp}.csv"

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
                filename = f"data/synthetic_complaints_{timestamp}.json"

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
            "category_distribution": {},
            "severity_distribution": {},
            "emotion_distribution": {},
            "channel_distribution": {},
            "word_count_stats": {
                "avg": sum(c["word_count"] for c in complaints) / len(complaints),
                "min": min(c["word_count"] for c in complaints),
                "max": max(c["word_count"] for c in complaints)
            },
            "demographics_breakdown": {
                "age_group": {},
                "tech_savviness": {},
                "banking_experience": {}
            }
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

            for demo_type, demo_value in complaint["demographics"].items():
                if demo_type not in analysis["demographics_breakdown"]:
                    analysis["demographics_breakdown"][demo_type] = {}
                analysis["demographics_breakdown"][demo_type][demo_value] = \
                    analysis["demographics_breakdown"][demo_type].get(demo_value, 0) + 1

        return analysis