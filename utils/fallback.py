import random
import re
from datetime import datetime
from faker import Faker


class FallbackManager:
    def __init__(self):
        self.max_retries = 3
        self.fallback_threshold = 2
        self.error_count = 0
        self.template_generator = TemplateGenerator()
        self.error_handler = ErrorHandler()

    def should_use_fallback(self, error_count=None):
        if error_count is not None:
            self.error_count = error_count

        return self.error_count >= self.fallback_threshold

    def generate_fallback_complaint(self, category, severity, emotion, demographics=None):
        try:
            if self.should_use_fallback():
                return self.template_generator.generate_enhanced_template(
                    category, severity, emotion, demographics
                )
            else:
                return self.template_generator.generate_basic_template(
                    category, severity, emotion
                )
        except Exception as e:
            self.error_handler.log_error("Fallback generation failed", e)
            return self.template_generator.generate_emergency_template(
                category, severity, emotion
            )

    def increment_error_count(self):
        self.error_count += 1
        return self.error_count

    def reset_error_count(self):
        self.error_count = 0

    def get_fallback_strategy(self, error_type):
        strategies = {
            'api_timeout': 'enhanced_template',
            'api_error': 'basic_template',
            'validation_failed': 'retry_with_template',
            'rate_limit': 'wait_and_retry',
            'network_error': 'emergency_template'
        }

        return strategies.get(error_type, 'basic_template')


class TemplateGenerator:
    def __init__(self):
        self.fake = Faker()
        self.templates = self._load_templates()
        self.modifiers = self._load_modifiers()

    def generate_basic_template(self, category, severity, emotion):
        templates = self.templates['basic'][category][emotion]
        template = random.choice(templates)

        return template.format(
            severity=severity,
            emotion=emotion,
            category=category
        )

    def generate_enhanced_template(self, category, severity, emotion, demographics=None):
        base_complaint = self.generate_basic_template(category, severity, emotion)

        if demographics:
            base_complaint = self._apply_demographic_modifications(
                base_complaint, demographics
            )

        base_complaint = self._add_contextual_details(base_complaint, category, severity)
        base_complaint = self._apply_emotion_intensifiers(base_complaint, emotion, severity)

        return base_complaint

    def generate_emergency_template(self, category, severity, emotion):
        emergency_templates = {
            'ATM_FAILURE': f"I'm {emotion} about a {severity} ATM problem at your branch.",
            'FRAUD_DETECTION': f"I'm {emotion} about {severity} unauthorized activity on my account.",
            'UX_ISSUES': f"I'm {emotion} with {severity} problems using your digital banking."
        }

        return emergency_templates.get(
            category,
            f"I have a {severity} banking issue and I'm {emotion} about it."
        )

    def _load_templates(self):
        return {
            'basic': {
                'ATM_FAILURE': {
                    'frustrated': [
                        "I'm extremely {emotion} with the ATM malfunction at your branch. This {severity} issue needs immediate attention.",
                        "The ATM failure has left me very {emotion} and unable to access my money. This is {severity}.",
                        "Your malfunctioning ATM is causing me significant {emotion}. This {severity} problem must be fixed.",
                        "I'm {emotion} because the ATM at your location completely failed when I tried to use it.",
                        "The {severity} ATM breakdown has made me {emotion} and unable to complete my banking."
                    ],
                    'angry': [
                        "I'm absolutely {emotion} about the ATM eating my card without warning. This is {severity}.",
                        "The ATM failure has made me extremely {emotion} and I demand immediate action.",
                        "Your broken ATM has left me {emotion} and without access to my funds.",
                        "I'm {emotion} that your {severity} ATM malfunction has disrupted my banking.",
                        "The {severity} ATM problem has made me {emotion} and I want this resolved now."
                    ],
                    'worried': [
                        "I'm very {emotion} about the ATM keeping my card and not returning it. This is {severity}.",
                        "The ATM malfunction has me {emotion} about the security of my account.",
                        "I'm {emotion} about whether my money is safe after this {severity} ATM failure.",
                        "This {severity} ATM issue has me {emotion} about my financial security.",
                        "I'm {emotion} that the ATM malfunction might affect my account balance."
                    ],
                    'stressed': [
                        "I'm very {emotion} because of this {severity} ATM malfunction.",
                        "The ATM failure is causing me {emotion} and I need this resolved.",
                        "This {severity} ATM problem has me {emotion} and unable to access my money.",
                        "I'm {emotion} about the {severity} issues with your ATM service.",
                        "The ATM breakdown is making me {emotion} and affecting my daily banking."
                    ]
                },
                'FRAUD_DETECTION': {
                    'panicked': [
                        "I'm in complete {emotion} after discovering unauthorized charges on my account. This is {severity}.",
                        "The fraudulent transactions have left me absolutely {emotion} about my finances.",
                        "I'm in total {emotion} mode after finding suspicious activity on my card.",
                        "This {severity} fraud situation has me {emotion} and I need immediate help.",
                        "I'm {emotion} about the {severity} unauthorized access to my banking information."
                    ],
                    'angry': [
                        "I'm {emotion} about the unauthorized transactions appearing on my statement. This is {severity}.",
                        "The fraud on my account has made me extremely {emotion} and I want answers.",
                        "I'm {emotion} about someone using my card without my permission.",
                        "This {severity} fraud incident has made me {emotion} and I demand action.",
                        "I'm {emotion} that unauthorized charges have appeared on my {severity} compromised account."
                    ],
                    'scared': [
                        "I'm {emotion} that someone has accessed my banking information illegally. This is {severity}.",
                        "The suspicious charges have left me {emotion} about my financial security.",
                        "I'm {emotion} by the unauthorized access to my personal accounts.",
                        "This {severity} security breach has me {emotion} about my financial safety.",
                        "I'm {emotion} that fraudsters have compromised my {severity} banking security."
                    ],
                    'worried': [
                        "I'm extremely {emotion} about the {severity} unauthorized activity on my account.",
                        "The suspicious transactions have me {emotion} about my financial security.",
                        "I'm {emotion} that my account has been compromised in this {severity} way.",
                        "This {severity} fraud situation has me {emotion} about my banking safety.",
                        "I'm {emotion} about the {severity} security breach affecting my accounts."
                    ]
                },
                'UX_ISSUES': {
                    'frustrated': [
                        "I'm extremely {emotion} with your confusing mobile banking app. This is {severity}.",
                        "The website interface has left me very {emotion} and unable to complete tasks.",
                        "Your digital platform is causing me significant {emotion} every time I use it.",
                        "I'm {emotion} with the {severity} usability problems in your banking system.",
                        "The {severity} interface issues are making me {emotion} and unable to bank properly."
                    ],
                    'confused': [
                        "I'm totally {emotion} by the layout of your online banking system. This is {severity}.",
                        "The app interface has left me {emotion} and unable to find basic functions.",
                        "I'm {emotion} by how difficult it is to navigate your website.",
                        "Your {severity} design problems have me {emotion} about basic banking tasks.",
                        "I'm {emotion} by the {severity} complexity of your digital banking platform."
                    ],
                    'disappointed': [
                        "I'm very {emotion} with the poor quality of your digital services. This is {severity}.",
                        "The app performance has left me extremely {emotion} as a customer.",
                        "Your website's usability issues are very {emotion} for a major bank.",
                        "I'm {emotion} with the {severity} problems in your digital banking experience.",
                        "The {severity} technical issues have left me {emotion} with your service."
                    ],
                    'annoyed': [
                        "I'm really {emotion} with the constant problems in your banking app. This is {severity}.",
                        "The website keeps malfunctioning and I'm {emotion} about it.",
                        "Your digital platform is {emotion} me with {severity} technical issues.",
                        "I'm {emotion} that your {severity} app problems keep preventing me from banking.",
                        "The {severity} interface problems are making me {emotion} every time I try to use it."
                    ]
                }
            }
        }

    def _load_modifiers(self):
        return {
            'demographic': {
                'age_group': {
                    '18-30': {
                        'language_style': 'casual',
                        'tech_references': True,
                        'urgency': 'moderate'
                    },
                    '31-50': {
                        'language_style': 'professional',
                        'tech_references': True,
                        'urgency': 'high'
                    },
                    '51-65': {
                        'language_style': 'formal',
                        'tech_references': False,
                        'urgency': 'high'
                    },
                    '65+': {
                        'language_style': 'very_formal',
                        'tech_references': False,
                        'urgency': 'moderate'
                    }
                },
                'tech_savviness': {
                    'low': {
                        'tech_terms': False,
                        'detail_level': 'basic',
                        'confusion_level': 'high'
                    },
                    'medium': {
                        'tech_terms': True,
                        'detail_level': 'moderate',
                        'confusion_level': 'moderate'
                    },
                    'high': {
                        'tech_terms': True,
                        'detail_level': 'detailed',
                        'confusion_level': 'low'
                    }
                }
            },
            'contextual': {
                'ATM_FAILURE': [
                    "The machine kept my card and never returned it.",
                    "The screen froze completely during my transaction.",
                    "Cash never came out but my account was charged.",
                    "The receipt printer jammed and wouldn't work.",
                    "The keypad stopped responding halfway through."
                ],
                'FRAUD_DETECTION': [
                    "There are charges I never authorized on my statement.",
                    "Someone used my card while it was in my possession.",
                    "Transactions appeared from locations I've never visited.",
                    "Multiple suspicious charges occurred overnight.",
                    "My account shows activity I definitely didn't perform."
                ],
                'UX_ISSUES': [
                    "The app crashes every time I try to log in.",
                    "Pages take forever to load on your website.",
                    "I can't find basic functions in the interface.",
                    "Error messages appear constantly without explanation.",
                    "The navigation menu is confusing and unhelpful."
                ]
            },
            'emotion_intensifiers': {
                'frustrated': {
                    'low': ["mildly frustrated", "somewhat annoyed"],
                    'medium': ["quite frustrated", "really annoyed"],
                    'high': ["extremely frustrated", "incredibly annoyed"],
                    'critical': ["absolutely livid", "beyond frustrated"]
                },
                'angry': {
                    'low': ["upset", "displeased"],
                    'medium': ["angry", "really mad"],
                    'high': ["furious", "extremely angry"],
                    'critical': ["absolutely livid", "enraged"]
                },
                'worried': {
                    'low': ["concerned", "slightly worried"],
                    'medium': ["quite worried", "very concerned"],
                    'high': ["extremely worried", "deeply concerned"],
                    'critical': ["terrified", "in complete panic"]
                }
            }
        }

    def _apply_demographic_modifications(self, complaint, demographics):
        age_group = demographics.get('age_group', '31-50')
        tech_level = demographics.get('tech_savviness', 'medium')

        age_mods = self.modifiers['demographic']['age_group'].get(age_group, {})
        tech_mods = self.modifiers['demographic']['tech_savviness'].get(tech_level, {})

        if age_mods.get('language_style') == 'casual':
            complaint = self._make_casual(complaint)
        elif age_mods.get('language_style') == 'very_formal':
            complaint = self._make_formal(complaint)

        if not tech_mods.get('tech_terms', True):
            complaint = self._simplify_tech_language(complaint)

        return complaint

    def _add_contextual_details(self, complaint, category, severity):
        details = self.modifiers['contextual'].get(category, [])
        if details:
            detail = random.choice(details)
            complaint += f" {detail}"

        if severity in ['high', 'critical']:
            urgency_phrases = [
                "This needs immediate attention.",
                "I need this resolved right away.",
                "This requires urgent action.",
                "Please address this immediately."
            ]
            complaint += f" {random.choice(urgency_phrases)}"

        return complaint

    def _apply_emotion_intensifiers(self, complaint, emotion, severity):
        intensifiers = self.modifiers['emotion_intensifiers'].get(emotion, {})
        level_intensifiers = intensifiers.get(severity, [])

        if level_intensifiers:
            intensifier = random.choice(level_intensifiers)
            complaint = complaint.replace(emotion, intensifier)

        return complaint

    def _make_casual(self, text):
        replacements = {
            'I am': "I'm",
            'cannot': "can't",
            'will not': "won't",
            'do not': "don't",
            'does not': "doesn't",
            'have not': "haven't",
            'This is': "This's"
        }

        for formal, casual in replacements.items():
            text = text.replace(formal, casual)

        return text

    def _make_formal(self, text):
        replacements = {
            "I'm": 'I am',
            "can't": 'cannot',
            "won't": 'will not',
            "don't": 'do not',
            "doesn't": 'does not',
            "haven't": 'have not'
        }

        for casual, formal in replacements.items():
            text = text.replace(casual, formal)

        return text

    def _simplify_tech_language(self, text):
        tech_replacements = {
            'application': 'app',
            'interface': 'screen',
            'malfunction': 'problem',
            'transaction': 'banking',
            'unauthorized': 'without permission',
            'fraudulent': 'fake'
        }

        for tech, simple in tech_replacements.items():
            text = text.replace(tech, simple)

        return text


class ErrorHandler:
    def __init__(self):
        self.error_log = []
        self.error_counts = {}

    def handle_generation_error(self, error_type, error_message, context=None):
        error_entry = {
            'type': error_type,
            'message': str(error_message),
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }

        self.error_log.append(error_entry)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        recovery_strategy = self._get_recovery_strategy(error_type)

        return {
            'error_logged': True,
            'recovery_strategy': recovery_strategy,
            'error_count': self.error_counts[error_type]
        }

    def log_error(self, message, exception=None):
        error_entry = {
            'message': message,
            'exception': str(exception) if exception else None,
            'timestamp': datetime.now().isoformat()
        }

        self.error_log.append(error_entry)

    def _get_recovery_strategy(self, error_type):
        strategies = {
            'api_timeout': 'retry_with_fallback',
            'api_error': 'use_fallback',
            'validation_failed': 'regenerate_with_template',
            'rate_limit': 'wait_and_retry',
            'network_error': 'use_emergency_template',
            'parsing_error': 'use_basic_template'
        }

        return strategies.get(error_type, 'use_fallback')

    def get_error_summary(self):
        return {
            'total_errors': len(self.error_log),
            'error_counts': self.error_counts,
            'recent_errors': self.error_log[-5:] if self.error_log else []
        }

    def clear_error_log(self):
        self.error_log.clear()
        self.error_counts.clear()


def generate_fallback_complaint(category, severity, emotion, demographics=None):
    manager = FallbackManager()
    return manager.generate_fallback_complaint(category, severity, emotion, demographics)


def handle_generation_error(error_type, error_message, context=None):
    handler = ErrorHandler()
    return handler.handle_generation_error(error_type, error_message, context)


def create_emergency_complaint(category, severity, emotion):
    generator = TemplateGenerator()
    return generator.generate_emergency_template(category, severity, emotion)


def apply_demographic_styling(text, demographics):
    generator = TemplateGenerator()
    return generator._apply_demographic_modifications(text, demographics)


def add_contextual_details(text, category, severity):
    generator = TemplateGenerator()
    return generator._add_contextual_details(text, category, severity)


def intensify_emotion(text, emotion, severity):
    generator = TemplateGenerator()
    return generator._apply_emotion_intensifiers(text, emotion, severity)


def should_use_fallback(error_count, threshold=2):
    return error_count >= threshold


def get_fallback_strategy(error_type):
    manager = FallbackManager()
    return manager.get_fallback_strategy(error_type)


def log_generation_attempt(success, method, category, error=None):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'success': success,
        'method': method,
        'category': category,
        'error': str(error) if error else None
    }

    return log_entry


def create_complaint_variations(base_complaint, count=3):
    variations = []

    for i in range(count):
        variation = base_complaint

        if i == 1:
            variation = variation.replace("I'm", "I am")
            variation = variation.replace("can't", "cannot")
        elif i == 2:
            variation = re.sub(r'\bvery\b', 'extremely', variation)
            variation = re.sub(r'\breally\b', 'quite', variation)

        variations.append(variation)

    return variations


def validate_fallback_quality(complaint_text, min_length=30):
    if not complaint_text or len(complaint_text) < min_length:
        return False, "Complaint too short"

    if not complaint_text[0].isupper():
        return False, "Should start with capital letter"

    if not complaint_text.strip().endswith(('.', '!', '?')):
        return False, "Should end with proper punctuation"

    word_count = len(complaint_text.split())
    if word_count < 10:
        return False, "Too few words"

    return True, "Quality validation passed"