COMPLAINT_TEMPLATES = {
    "ATM_FAILURE": {
        "templates": [
            "I am {emotion} about what happened at the {location} ATM {time_ref}. I was trying to {action} but {failure_description}. {impact_statement} {resolution_request}",
            "{emotion_start} I went to the ATM at {location} {time_ref} to {action}. {failure_description} {impact_statement} This is {severity_desc} and needs immediate attention. {resolution_request}",
            "This is {severity_desc}! The ATM at {location} {failure_description} when I tried to {action} {time_ref}. {impact_statement} {emotion_statement} {resolution_request}",
            "I need to report a serious problem with your ATM at {location}. {time_ref} I attempted to {action} but {failure_description}. {impact_statement} {emotion_statement} {resolution_request}"
        ],
        "emotions": {
            "frustrated": ["extremely frustrated", "really frustrated", "very frustrated", "incredibly frustrated"],
            "angry": ["absolutely furious", "really angry", "very angry", "extremely upset", "livid"],
            "disappointed": ["very disappointed", "extremely disappointed", "really disappointed"],
            "worried": ["very concerned", "extremely worried", "really worried", "very anxious"],
            "stressed": ["very stressed", "extremely stressed", "under a lot of stress"]
        },
        "emotion_starts": {
            "frustrated": ["I'm extremely frustrated that", "I'm really frustrated because", "I'm very frustrated that"],
            "angry": ["I'm absolutely furious that", "I'm really angry that", "I'm very upset that"],
            "disappointed": ["I'm very disappointed that", "I'm extremely disappointed because"],
            "worried": ["I'm very concerned that", "I'm really worried because"],
            "stressed": ["I'm very stressed because", "I'm under a lot of stress because"]
        },
        "locations": ["Main Street", "downtown branch", "shopping mall", "grocery store", "gas station", "airport", "train station", "university campus", "hospital", "city center"],
        "time_refs": ["yesterday evening", "this morning", "last night", "this afternoon", "yesterday", "earlier today", "last weekend", "on Friday", "on Monday", "last Tuesday"],
        "actions": ["withdraw $200", "deposit my paycheck", "check my balance", "withdraw some cash", "deposit cash", "transfer money", "get $100", "withdraw $500", "make a deposit"],
        "failure_descriptions": [
            "the machine kept my card and never returned it",
            "the screen froze completely and became unresponsive",
            "the cash never came out but my account was charged",
            "the keypad stopped working halfway through",
            "it displayed an error message and shut down",
            "the transaction timed out and ate my card",
            "it dispensed the wrong amount of money",
            "the receipt printer jammed and wouldn't give me proof",
            "the machine went completely out of service",
            "it charged me twice for the same transaction"
        ],
        "impact_statements": [
            "Now I can't access my money when I need it most.",
            "This left me stranded without cash for the weekend.",
            "I had to explain to my family why I couldn't pay for groceries.",
            "I missed an important payment because of this.",
            "This caused me to overdraft on another account.",
            "I had to borrow money from friends because of this.",
            "This ruined my weekend plans completely.",
            "I couldn't pay for parking and got a ticket.",
            "This made me late for an important appointment.",
            "I'm now worried about the security of my account."
        ],
        "resolution_requests": [
            "Please fix this immediately and return my card.",
            "I need my money back and this machine needs to be repaired.",
            "Please investigate this and credit my account immediately.",
            "I want a full refund and compensation for this trouble.",
            "Please ensure this never happens again to other customers.",
            "I need someone to call me back about this today.",
            "Please have a technician check this machine immediately.",
            "I expect a written explanation and my money back.",
            "This needs to be resolved within 24 hours.",
            "Please contact me immediately to resolve this issue."
        ],
        "severity_descriptions": {
            "low": ["frustrating", "annoying", "inconvenient", "disappointing"],
            "medium": ["unacceptable", "really problematic", "very concerning", "quite serious"],
            "high": ["completely unacceptable", "absolutely ridiculous", "totally outrageous", "extremely serious"],
            "critical": ["absolutely unacceptable", "completely outrageous", "totally unacceptable", "beyond belief"]
        },
        "specific_issues": ["card_stuck", "cash_not_dispensed", "screen_frozen", "receipt_not_printed", "keypad_unresponsive", "out_of_service", "network_error", "wrong_amount_dispensed", "card_not_returned"],
        "severity_levels": ["low", "medium", "high", "critical"]
    },
    "FRAUD_DETECTION": {
        "templates": [
            "I am {emotion} because I discovered {fraud_discovery} {time_ref}. {fraud_details} {impact_statement} {emotion_statement} {resolution_request}",
            "{emotion_start} I noticed {fraud_discovery} {time_ref}. {fraud_details} {impact_statement} This is {severity_desc} and I need immediate help. {resolution_request}",
            "I need to report {fraud_type} immediately. {time_ref} I found {fraud_discovery}. {fraud_details} {impact_statement} {emotion_statement} {resolution_request}",
            "This is {severity_desc}! {fraud_discovery} {time_ref} and {fraud_details}. {impact_statement} {emotion_statement} {resolution_request}"
        ],
        "emotions": {
            "panicked": ["in complete panic", "absolutely panicked", "extremely panicked", "in total panic"],
            "angry": ["absolutely furious", "extremely angry", "really angry", "very upset"],
            "scared": ["really scared", "very frightened", "extremely worried", "terrified"],
            "confused": ["very confused", "extremely confused", "totally bewildered", "really puzzled"],
            "frustrated": ["extremely frustrated", "very frustrated", "really frustrated"],
            "worried": ["extremely worried", "very concerned", "really anxious", "deeply concerned"]
        },
        "emotion_starts": {
            "panicked": ["I'm in complete panic because", "I'm absolutely panicked that", "I'm in total panic because"],
            "angry": ["I'm absolutely furious that", "I'm extremely angry because", "I'm really upset that"],
            "scared": ["I'm really scared because", "I'm terrified that", "I'm very frightened because"],
            "confused": ["I'm very confused because", "I'm totally bewildered that", "I'm really puzzled because"],
            "frustrated": ["I'm extremely frustrated that", "I'm very frustrated because"],
            "worried": ["I'm extremely worried because", "I'm very concerned that", "I'm really anxious because"]
        },
        "fraud_discoveries": [
            "unauthorized transactions on my account",
            "charges I never made",
            "suspicious activity on my credit card",
            "someone used my card without permission",
            "transactions from places I've never been",
            "my card was used while it was in my wallet",
            "charges appearing overnight",
            "purchases I definitely didn't make",
            "my account was accessed without my knowledge",
            "someone has been using my banking information"
        ],
        "time_refs": ["this morning", "last night", "yesterday", "when I checked my statement", "on my phone app", "in my email alert", "during my monthly review", "when I got a notification"],
        "fraud_details": [
            "There are charges totaling over $1,500 that I never authorized.",
            "Someone spent $800 at stores I've never visited.",
            "My card was used in three different states on the same day.",
            "There are multiple small charges that I never made.",
            "Someone withdrew $500 from an ATM across town.",
            "There are online purchases I never made totaling $1,200.",
            "My card was used at restaurants while I was at work.",
            "Someone made purchases while my card was in my possession.",
            "There are charges from last weekend when I was out of town.",
            "Multiple transactions appeared overnight while I was sleeping."
        ],
        "impact_statements": [
            "This has completely drained my account.",
            "I can't pay my bills because of these fraudulent charges.",
            "My family's finances are now in jeopardy.",
            "I'm afraid to use any of my cards now.",
            "This is affecting my ability to buy groceries.",
            "I had to cancel all my cards and change my passwords.",
            "I'm worried they have access to other accounts too.",
            "This is causing me enormous stress and anxiety.",
            "I don't feel safe using your banking services anymore.",
            "I'm losing sleep over this financial security breach."
        ],
        "resolution_requests": [
            "Please reverse all fraudulent charges immediately.",
            "I need new cards and account numbers right away.",
            "Please investigate how this happened and prevent it again.",
            "I want all my money back and better security measures.",
            "Please contact me immediately to resolve this.",
            "I need written confirmation that this will be fixed.",
            "Please expedite the fraud investigation process.",
            "I want compensation for this security breach.",
            "Please ensure my remaining accounts are secure.",
            "I need someone to call me back within the hour."
        ],
        "fraud_types": ["credit card fraud", "identity theft", "account takeover", "unauthorized access", "banking fraud", "card skimming"],
        "severity_descriptions": {
            "medium": ["very serious", "quite concerning", "really problematic", "very worrying"],
            "high": ["extremely serious", "absolutely unacceptable", "completely outrageous", "totally unacceptable"],
            "critical": ["beyond belief", "absolutely catastrophic", "completely devastating", "totally unacceptable"]
        },
        "specific_issues": ["unauthorized_transaction", "card_skimming", "account_takeover", "phishing_attempt", "suspicious_login", "identity_theft"],
        "severity_levels": ["medium", "high", "critical"]
    },
    "UX_ISSUES": {
        "templates": [
            "I am {emotion} with the {platform} {problem_description}. {time_ref} I was trying to {task} but {specific_issue}. {impact_statement} {comparison} {resolution_request}",
            "{emotion_start} the {platform} {problem_description}. {specific_issue} when I tried to {task} {time_ref}. {impact_statement} {comparison} {resolution_request}",
            "The {platform} has {problem_description} and it's {severity_desc}. {time_ref} {specific_issue} while I was trying to {task}. {impact_statement} {resolution_request}",
            "I need to complain about the {platform}. {problem_description} and {specific_issue} every time I try to {task}. {impact_statement} {comparison} {resolution_request}"
        ],
        "emotions": {
            "frustrated": ["extremely frustrated", "very frustrated", "really frustrated", "incredibly frustrated"],
            "confused": ["very confused", "extremely confused", "totally confused", "really puzzled"],
            "disappointed": ["very disappointed", "extremely disappointed", "really disappointed"],
            "annoyed": ["really annoyed", "very annoyed", "extremely annoyed", "quite irritated"],
            "impatient": ["very impatient", "extremely impatient", "really impatient", "getting very frustrated"]
        },
        "emotion_starts": {
            "frustrated": ["I'm extremely frustrated with", "I'm very frustrated that", "I'm really frustrated because"],
            "confused": ["I'm very confused by", "I'm extremely confused about", "I'm totally puzzled by"],
            "disappointed": ["I'm very disappointed with", "I'm extremely disappointed in"],
            "annoyed": ["I'm really annoyed with", "I'm very annoyed that", "I'm quite irritated by"],
            "impatient": ["I'm very impatient with", "I'm getting very frustrated with"]
        },
        "platforms": ["mobile app", "website", "online banking", "ATM interface", "phone system", "customer portal"],
        "problem_descriptions": [
            "keeps crashing constantly",
            "is incredibly slow to load",
            "has a confusing interface",
            "doesn't work properly",
            "has terrible navigation",
            "freezes all the time",
            "has buttons that don't work",
            "gives error messages constantly",
            "logs me out repeatedly",
            "won't accept my login information"
        ],
        "time_refs": ["for the past week", "all month", "yesterday", "this morning", "every time I try to use it", "for several days now", "since the last update"],
        "tasks": [
            "check my account balance",
            "transfer money between accounts",
            "pay my bills online",
            "deposit a check",
            "view my transaction history",
            "update my contact information",
            "set up automatic payments",
            "apply for a loan",
            "contact customer service",
            "download my statements"
        ],
        "specific_issues": [
            "the app crashes and loses all my information",
            "pages take forever to load",
            "buttons don't respond when I tap them",
            "I get error messages that make no sense",
            "the interface is impossible to navigate",
            "text is too small to read",
            "the session times out too quickly",
            "forms don't submit properly",
            "I can't find basic functions",
            "the search feature doesn't work at all"
        ],
        "impact_statements": [
            "I've wasted hours trying to complete simple tasks.",
            "I can't manage my finances effectively.",
            "This is making me consider switching banks.",
            "I've missed payment deadlines because of these issues.",
            "It's impossible to do basic banking tasks.",
            "I have to call customer service for everything now.",
            "This is affecting my daily financial management.",
            "I'm losing confidence in your technology.",
            "These problems are costing me time and money.",
            "I can't recommend your services to others because of this."
        ],
        "comparisons": [
            "Other banks have much better apps.",
            "My previous bank's website was so much easier to use.",
            "Every other financial app I use works better than this.",
            "Even free apps work better than your expensive service.",
            "Your competitors have solved these problems years ago.",
            "This feels like technology from 10 years ago.",
            "Other banks make banking simple, not complicated.",
            "I've never had these problems with other services."
        ],
        "resolution_requests": [
            "Please fix these technical issues immediately.",
            "I need a working app that actually functions properly.",
            "Please improve the user interface design.",
            "I want these problems resolved within a week.",
            "Please test your app before releasing updates.",
            "I need better technical support for these issues.",
            "Please make the app actually usable.",
            "I want compensation for this poor service.",
            "Please hire better app developers.",
            "I need a timeline for when these issues will be fixed."
        ],
        "severity_descriptions": {
            "low": ["frustrating", "annoying", "inconvenient", "disappointing"],
            "medium": ["unacceptable", "really problematic", "very concerning", "quite serious"],
            "high": ["completely unacceptable", "absolutely ridiculous", "totally outrageous", "extremely serious"]
        },
        "specific_issues": ["confusing_interface", "slow_loading", "accessibility_issues", "mobile_app_crashes", "unclear_instructions", "poor_navigation"],
        "severity_levels": ["low", "medium", "high"]
    }
}

DEMOGRAPHICS = {
    "age_groups": ["18-30", "31-50", "51-65", "65+"],
    "tech_savviness": ["low", "medium", "high"],
    "banking_experience": ["novice", "intermediate", "expert"]
}

METADATA_OPTIONS = {
    "channels": ["phone", "email", "chat", "branch", "mobile_app", "website"],
    "account_types": ["checking", "savings", "credit", "business", "investment"]
}

DEFAULT_CATEGORY_DISTRIBUTION = {
    "ATM_FAILURE": 0.4,
    "FRAUD_DETECTION": 0.3,
    "UX_ISSUES": 0.3
}

MODEL_CONFIG = {
    "tfidf_max_features": 5000,
    "tfidf_stop_words": "english",
    "logistic_regression_max_iter": 1000,
    "random_forest_n_estimators": 100,
    "random_state": 42
}

FILE_PATHS = {
    "data_dir": "data",
    "models_dir": "models",
    "exports_dir": "data/exports",
    "uploads_dir": "data/uploads",
    "llm_generated_dir": "data/llm_generated"
}

MODEL_FILES = {
    "vectorizer": "models/complaint_vectorizer.pkl",
    "category_model": "models/category_model.pkl",
    "emotion_model": "models/emotion_model.pkl",
    "severity_model": "models/severity_model.pkl"
}

SAMPLE_COMPLAINTS = [
    "The ATM ate my card and I'm furious about it",
    "Your mobile app keeps crashing when I try to login",
    "Someone stole my credit card information and made purchases",
    "I can't find the transfer button on your confusing website",
    "The machine froze when I was trying to deposit my paycheck",
    "There are unauthorized charges on my account from yesterday",
    "Your website is so slow it takes forever to load my statement",
    "I'm extremely worried about suspicious activity on my card"
]

COLOR_SCHEMES = {
    "severity_colors": {
        "low": "green",
        "medium": "orange",
        "high": "red",
        "critical": "darkred"
    },
    "category_colors": {
        "ATM_FAILURE": "#FF6B6B",
        "FRAUD_DETECTION": "#4ECDC4",
        "UX_ISSUES": "#45B7D1"
    }
}

STOP_WORDS = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
    'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
    'our', 'their', 'a', 'an'
}

VALIDATION_RULES = {
    "required_columns": ["complaint_text", "category", "severity", "emotion"],
    "min_complaint_length": 5,
    "max_complaint_length": 1000,
    "min_word_count": 3,
    "max_word_count": 500
}

APP_CONFIG = {
    "page_title": "Customer Complaint Analysis System",
    "page_icon": "🏦",
    "layout": "wide",
    "max_upload_size": 200,
    "default_complaint_count": 500,
    "max_complaint_count": 5000,
    "min_complaint_count": 10
}

HUGGING_FACE_CONFIG = {
    "api_base_url": "https://api-inference.huggingface.co/models",
    "models": {
        "primary": "microsoft/DialoGPT-medium",
        "backup": [
            "microsoft/DialoGPT-small",
            "facebook/blenderbot-400M-distill",
            "microsoft/DialoGPT-large"
        ]
    },
    "generation_params": {
        "max_length": 200,
        "temperature": 0.8,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1
    },
    "rate_limits": {
        "requests_per_minute": 30,
        "retry_attempts": 3,
        "retry_delay": 2
    },
    "fallback_threshold": 3,
    "timeout_seconds": 30
}

LLM_PROMPTS = {
    "base_prompt": "Customer complaint about {category}. Severity: {severity}. Emotion: {emotion}.\n\nWrite a realistic banking complaint from an {emotion} customer about {category} issues. Make it {severity} severity.\n\nComplaint:",
    "category_prompts": {
        "ATM_FAILURE": "Write a realistic complaint about ATM problems. Customer is {emotion} and severity is {severity}. Include specific ATM issues like card problems, cash dispensing, or machine errors.",
        "FRAUD_DETECTION": "Write a realistic complaint about fraud or unauthorized transactions. Customer is {emotion} and severity is {severity}. Include specific fraud concerns and security issues.",
        "UX_ISSUES": "Write a realistic complaint about digital banking user experience problems. Customer is {emotion} and severity is {severity}. Include specific app or website usability issues."
    },
    "emotion_modifiers": {
        "frustrated": "Write in a frustrated, annoyed tone with specific complaints about inconvenience.",
        "angry": "Write in an angry, upset tone demanding immediate action and resolution.",
        "worried": "Write in a concerned, anxious tone expressing worry about security or money.",
        "scared": "Write in a frightened, panicked tone about serious security breaches.",
        "confused": "Write in a confused, puzzled tone asking for clarification and help.",
        "disappointed": "Write in a disappointed tone expressing unmet expectations."
    },
    "severity_modifiers": {
        "low": "Make this a minor inconvenience that needs attention but isn't urgent.",
        "medium": "Make this a significant problem that needs prompt resolution.",
        "high": "Make this a serious issue requiring immediate attention and action.",
        "critical": "Make this an emergency situation requiring immediate escalation."
    }
}

QUALITY_THRESHOLDS = {
    "min_word_count": 10,
    "max_word_count": 300,
    "min_character_count": 50,
    "max_character_count": 1500,
    "profanity_check": True,
    "duplicate_similarity_threshold": 0.8,
    "coherence_threshold": 0.6
}

GENERATION_METHODS = {
    "template": {
        "name": "Template-based (Original)",
        "description": "Uses predefined templates with variable substitution",
        "speed": "fast",
        "quality": "good",
        "variety": "medium",
        "cost": "free"
    },
    "huggingface": {
        "name": "Hugging Face (AI-Enhanced)",
        "description": "Uses Hugging Face Inference API for AI generation",
        "speed": "medium",
        "quality": "high",
        "variety": "high",
        "cost": "free_with_limits"
    }
}