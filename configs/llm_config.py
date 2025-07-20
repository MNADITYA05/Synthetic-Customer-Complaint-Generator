LLM_CONFIG = {
    "provider": "huggingface",
    "api_base_url": "https://api-inference.huggingface.co/models",
    "timeout_seconds": 30,
    "max_retries": 3,
    "retry_delay": 2,
    "fallback_threshold": 3,
    "rate_limit_requests_per_minute": 30,
    "enable_validation": True,
    "enable_fallback": True,
    "log_requests": True,
    "cache_responses": False
}

HUGGING_FACE_MODELS = {
    "primary": {
        "name": "microsoft/DialoGPT-medium",
        "size": "355MB",
        "quality": "high",
        "speed": "medium"
    },
    "backup": [
        {
            "name": "microsoft/DialoGPT-small",
            "size": "117MB",
            "quality": "medium",
            "speed": "fast"
        },
        {
            "name": "facebook/blenderbot-400M-distill",
            "size": "400MB",
            "quality": "high",
            "speed": "medium"
        },
        {
            "name": "microsoft/DialoGPT-large",
            "size": "774MB",
            "quality": "very_high",
            "speed": "slow"
        }
    ],
    "experimental": [
        {
            "name": "facebook/blenderbot-1B-distill",
            "size": "1GB",
            "quality": "excellent",
            "speed": "slow"
        }
    ]
}

GENERATION_PARAMETERS = {
    "default": {
        "max_length": 200,
        "min_length": 20,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 2,
        "early_stopping": True
    },
    "creative": {
        "max_length": 250,
        "min_length": 30,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 60,
        "do_sample": True,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "early_stopping": False
    },
    "conservative": {
        "max_length": 150,
        "min_length": 15,
        "temperature": 0.6,
        "top_p": 0.8,
        "top_k": 40,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "length_penalty": 1.1,
        "no_repeat_ngram_size": 2,
        "early_stopping": True
    }
}

API_SETTINGS = {
    "headers": {
        "Content-Type": "application/json",
        "User-Agent": "ComplaintGenerator/1.0"
    },
    "request_timeout": 30,
    "connection_timeout": 10,
    "read_timeout": 25,
    "max_connections": 10,
    "retry_status_codes": [429, 500, 502, 503, 504],
    "backoff_factor": 2,
    "jitter": True
}

QUALITY_SETTINGS = {
    "validation": {
        "min_word_count": 10,
        "max_word_count": 300,
        "min_character_count": 50,
        "max_character_count": 1500,
        "min_sentence_count": 2,
        "max_sentence_count": 10,
        "min_category_keywords": 1,
        "max_repetition_ratio": 0.3
    },
    "filtering": {
        "profanity_check": True,
        "spam_detection": True,
        "coherence_check": True,
        "relevance_check": True,
        "duplicate_threshold": 0.8,
        "similarity_threshold": 0.9
    },
    "enhancement": {
        "auto_capitalize": True,
        "fix_spacing": True,
        "remove_duplicates": True,
        "normalize_punctuation": True,
        "spell_check": False,
        "grammar_check": False
    }
}

CATEGORY_KEYWORDS = {
    "ATM_FAILURE": [
        "atm", "machine", "card", "cash", "withdraw", "deposit", "screen",
        "error", "transaction", "dispenser", "receipt", "pin", "balance",
        "keypad", "slot", "malfunction", "frozen", "timeout", "service"
    ],
    "FRAUD_DETECTION": [
        "fraud", "unauthorized", "suspicious", "transaction", "charge",
        "account", "security", "stolen", "hacked", "breach", "identity",
        "scam", "phishing", "compromised", "fake", "illegal", "criminal"
    ],
    "UX_ISSUES": [
        "app", "website", "interface", "login", "crash", "slow", "error",
        "navigation", "system", "browser", "mobile", "desktop", "loading",
        "button", "menu", "search", "form", "page", "design", "usability"
    ]
}

EMOTION_KEYWORDS = {
    "frustrated": ["frustrated", "annoyed", "irritated", "upset", "bothered"],
    "angry": ["angry", "furious", "mad", "livid", "enraged", "outraged"],
    "worried": ["worried", "concerned", "anxious", "nervous", "troubled"],
    "scared": ["scared", "frightened", "terrified", "afraid", "panicked"],
    "confused": ["confused", "puzzled", "bewildered", "lost", "unclear"],
    "disappointed": ["disappointed", "let down", "dissatisfied", "unhappy"],
    "stressed": ["stressed", "overwhelmed", "pressured", "tense"],
    "impatient": ["impatient", "restless", "eager", "hurried"],
    "panicked": ["panicked", "desperate", "frantic", "urgent", "emergency"]
}

SEVERITY_INDICATORS = {
    "low": ["minor", "small", "slight", "inconvenience", "annoying"],
    "medium": ["significant", "concerning", "problematic", "serious"],
    "high": ["major", "severe", "critical", "urgent", "important"],
    "critical": ["emergency", "catastrophic", "devastating", "immediate"]
}

FALLBACK_TEMPLATES = {
    "ATM_FAILURE": {
        "frustrated": [
            "I'm extremely frustrated with the ATM malfunction at your branch.",
            "The ATM failure has left me very frustrated and unable to access my money.",
            "Your malfunctioning ATM is causing me significant frustration."
        ],
        "angry": [
            "I'm absolutely furious about the ATM eating my card without warning.",
            "The ATM failure has made me extremely angry and I demand immediate action.",
            "Your broken ATM has left me livid and without access to my funds."
        ],
        "worried": [
            "I'm very worried about the ATM keeping my card and not returning it.",
            "The ATM malfunction has me concerned about the security of my account.",
            "I'm anxious about whether my money is safe after the ATM failure."
        ]
    },
    "FRAUD_DETECTION": {
        "panicked": [
            "I'm in complete panic after discovering unauthorized charges on my account.",
            "The fraudulent transactions have left me absolutely panicked about my finances.",
            "I'm in total panic mode after finding suspicious activity on my card."
        ],
        "angry": [
            "I'm furious about the unauthorized transactions appearing on my statement.",
            "The fraud on my account has made me extremely angry and I want answers.",
            "I'm livid about someone using my card without my permission."
        ],
        "scared": [
            "I'm terrified that someone has accessed my banking information illegally.",
            "The suspicious charges have left me scared about my financial security.",
            "I'm frightened by the unauthorized access to my personal accounts."
        ]
    },
    "UX_ISSUES": {
        "frustrated": [
            "I'm extremely frustrated with your confusing mobile banking app.",
            "The website interface has left me very frustrated and unable to complete tasks.",
            "Your digital platform is causing me significant frustration every time I use it."
        ],
        "confused": [
            "I'm totally confused by the layout of your online banking system.",
            "The app interface has left me bewildered and unable to find basic functions.",
            "I'm puzzled by how difficult it is to navigate your website."
        ],
        "disappointed": [
            "I'm very disappointed with the poor quality of your digital services.",
            "The app performance has left me extremely disappointed as a customer.",
            "Your website's usability issues are very disappointing for a major bank."
        ]
    }
}

MODEL_PERFORMANCE_METRICS = {
    "response_time_targets": {
        "excellent": 2.0,
        "good": 5.0,
        "acceptable": 10.0,
        "poor": 20.0
    },
    "quality_thresholds": {
        "excellent": 0.9,
        "good": 0.8,
        "acceptable": 0.7,
        "poor": 0.6
    },
    "success_rate_targets": {
        "excellent": 0.95,
        "good": 0.90,
        "acceptable": 0.85,
        "poor": 0.80
    }
}

CACHING_CONFIG = {
    "enable_cache": False,
    "cache_ttl_seconds": 3600,
    "max_cache_size": 1000,
    "cache_compression": True,
    "cache_encryption": False,
    "cache_cleanup_interval": 300
}

LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_requests": True,
    "log_responses": False,
    "log_errors": True,
    "log_performance": True,
    "log_file": "logs/llm_generator.log",
    "max_log_size": "10MB",
    "backup_count": 5,
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}