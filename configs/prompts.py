PROMPT_TEMPLATES = {
    "base_template": "Customer complaint about {category}. Severity: {severity}. Emotion: {emotion}.\n\nWrite a realistic banking complaint from an {emotion} customer about {category} issues. Make it {severity} severity.\n\nComplaint:",

    "detailed_template": "Generate a realistic customer banking complaint with these characteristics:\n\nCategory: {category}\nSeverity: {severity}\nCustomer Emotion: {emotion}\n{demographic_context}\n\nWrite from the customer's perspective (first person). Include specific details and context. Match the emotional tone ({emotion}) and severity level ({severity}). Sound like a real person, not AI-generated.\n\nComplaint:",

    "contextual_template": "You are writing a complaint as a {emotion} banking customer. You experienced a {severity} {category} issue. {context_details}\n\nWrite a realistic complaint that:\n- Expresses {emotion} emotion authentically\n- Reflects {severity} severity level\n- Includes specific banking details\n- Sounds like real customer language\n\nComplaint:",

    "narrative_template": "Scenario: A {demographic_age} customer with {tech_level} technical skills had a {severity} {category} problem. They are feeling {emotion} about the situation. {specific_context}\n\nWrite their complaint as they would express it:\n\nComplaint:"
}

CATEGORY_PROMPTS = {
    "ATM_FAILURE": {
        "base": "Write a realistic complaint about ATM problems. Customer is {emotion} and severity is {severity}. Include specific ATM issues like card problems, cash dispensing, or machine errors.",

        "detailed": "Generate a banking complaint about ATM malfunction. The customer is {emotion} due to a {severity} ATM failure. Include details about:\n- What the customer was trying to do\n- How the ATM failed\n- Impact on the customer\n- What resolution they want\n\nWrite in first person as an {emotion} customer.",

        "specific_issues": {
            "card_stuck": "Write a complaint about an ATM keeping the customer's card. The customer is {emotion} about this {severity} situation.",
            "cash_not_dispensed": "Write a complaint about an ATM not dispensing cash but charging the account. Customer is {emotion}, severity is {severity}.",
            "screen_frozen": "Write a complaint about an ATM screen freezing during transaction. Customer feels {emotion}, issue is {severity}.",
            "receipt_not_printed": "Write a complaint about ATM not providing transaction receipt. Customer is {emotion}, severity level {severity}.",
            "wrong_amount": "Write a complaint about ATM dispensing wrong cash amount. Customer emotion: {emotion}, severity: {severity}."
        }
    },

    "FRAUD_DETECTION": {
        "base": "Write a realistic complaint about fraud or unauthorized transactions. Customer is {emotion} and severity is {severity}. Include specific fraud concerns and security issues.",

        "detailed": "Generate a banking complaint about fraudulent activity. The customer is {emotion} due to {severity} unauthorized transactions. Include details about:\n- How they discovered the fraud\n- What unauthorized activities occurred\n- Financial impact\n- Security concerns\n- Immediate actions needed\n\nWrite as a {emotion} customer in first person.",

        "specific_issues": {
            "unauthorized_transaction": "Write a complaint about unauthorized charges on account. Customer is {emotion}, severity {severity}.",
            "card_skimming": "Write a complaint about suspected card skimming incident. Customer feels {emotion}, severity is {severity}.",
            "account_takeover": "Write a complaint about someone accessing their account illegally. Customer emotion: {emotion}, severity: {severity}.",
            "identity_theft": "Write a complaint about identity theft affecting bank accounts. Customer is {emotion}, severity level {severity}.",
            "phishing_attempt": "Write a complaint about phishing scam affecting their banking. Customer feels {emotion}, severity {severity}."
        }
    },

    "UX_ISSUES": {
        "base": "Write a realistic complaint about digital banking user experience problems. Customer is {emotion} and severity is {severity}. Include specific app or website usability issues.",

        "detailed": "Generate a banking complaint about poor user experience. The customer is {emotion} due to {severity} digital platform problems. Include details about:\n- Which platform (app/website) has issues\n- Specific usability problems\n- Tasks they cannot complete\n- Frequency of issues\n- Comparison to expectations\n\nWrite as an {emotion} customer in first person.",

        "specific_issues": {
            "mobile_app_crashes": "Write a complaint about mobile banking app constantly crashing. Customer is {emotion}, severity {severity}.",
            "slow_loading": "Write a complaint about extremely slow website loading times. Customer feels {emotion}, severity is {severity}.",
            "confusing_interface": "Write a complaint about confusing navigation and interface design. Customer emotion: {emotion}, severity: {severity}.",
            "login_problems": "Write a complaint about repeated login failures and access issues. Customer is {emotion}, severity level {severity}.",
            "feature_missing": "Write a complaint about missing or hard-to-find banking features. Customer feels {emotion}, severity {severity}."
        }
    }
}

EMOTION_MODIFIERS = {
    "frustrated": {
        "descriptors": ["extremely frustrated", "very frustrated", "really frustrated", "incredibly frustrated"],
        "language_style": "Write in a frustrated, annoyed tone with specific complaints about inconvenience and wasted time.",
        "tone_markers": ["This is so frustrating", "I can't believe", "What a waste of time", "This is ridiculous"],
        "intensity_low": "mildly frustrated and inconvenienced",
        "intensity_medium": "quite frustrated and annoyed",
        "intensity_high": "extremely frustrated and fed up"
    },

    "angry": {
        "descriptors": ["absolutely furious", "really angry", "very angry", "extremely upset", "livid"],
        "language_style": "Write in an angry, demanding tone requiring immediate action and resolution.",
        "tone_markers": ["This is unacceptable", "I demand", "This is outrageous", "I'm livid"],
        "intensity_low": "somewhat angry and disappointed",
        "intensity_medium": "quite angry and upset",
        "intensity_high": "absolutely furious and outraged"
    },

    "worried": {
        "descriptors": ["very concerned", "extremely worried", "really worried", "very anxious"],
        "language_style": "Write in a concerned, anxious tone expressing worry about security, money, or consequences.",
        "tone_markers": ["I'm worried that", "This concerns me", "I'm afraid", "What if"],
        "intensity_low": "somewhat concerned and cautious",
        "intensity_medium": "quite worried and anxious",
        "intensity_high": "extremely worried and distressed"
    },

    "scared": {
        "descriptors": ["really scared", "very frightened", "extremely worried", "terrified"],
        "language_style": "Write in a frightened, panicked tone about serious security breaches or financial threats.",
        "tone_markers": ["I'm terrified", "This scares me", "I'm afraid", "What if they"],
        "intensity_low": "somewhat nervous and uneasy",
        "intensity_medium": "quite scared and worried",
        "intensity_high": "absolutely terrified and panicked"
    },

    "confused": {
        "descriptors": ["very confused", "extremely confused", "totally bewildered", "really puzzled"],
        "language_style": "Write in a confused, puzzled tone asking for clarification and help understanding.",
        "tone_markers": ["I don't understand", "This makes no sense", "I'm confused about", "How am I supposed to"],
        "intensity_low": "somewhat puzzled and uncertain",
        "intensity_medium": "quite confused and lost",
        "intensity_high": "completely bewildered and clueless"
    },

    "disappointed": {
        "descriptors": ["very disappointed", "extremely disappointed", "really disappointed"],
        "language_style": "Write in a disappointed tone expressing unmet expectations and reduced trust.",
        "tone_markers": ["I expected better", "This is disappointing", "I thought you were", "I'm let down"],
        "intensity_low": "mildly disappointed and dissatisfied",
        "intensity_medium": "quite disappointed and unhappy",
        "intensity_high": "extremely disappointed and disillusioned"
    },

    "panicked": {
        "descriptors": ["in complete panic", "absolutely panicked", "extremely panicked", "in total panic"],
        "language_style": "Write in a panicked, urgent tone demanding immediate emergency response.",
        "tone_markers": ["This is an emergency", "I need help now", "I'm panicking", "This is urgent"],
        "intensity_low": "somewhat alarmed and rushed",
        "intensity_medium": "quite panicked and urgent",
        "intensity_high": "in complete panic and crisis mode"
    },

    "annoyed": {
        "descriptors": ["really annoyed", "very annoyed", "extremely annoyed", "quite irritated"],
        "language_style": "Write in an annoyed, irritated tone about repeated problems and poor service.",
        "tone_markers": ["This is annoying", "I'm getting tired of", "Enough is enough", "This keeps happening"],
        "intensity_low": "mildly annoyed and bothered",
        "intensity_medium": "quite annoyed and irritated",
        "intensity_high": "extremely annoyed and fed up"
    },

    "stressed": {
        "descriptors": ["very stressed", "extremely stressed", "under a lot of stress"],
        "language_style": "Write in a stressed, overwhelmed tone about pressure and difficult circumstances.",
        "tone_markers": ["I'm under pressure", "This is stressing me out", "I can't handle", "This adds to my stress"],
        "intensity_low": "somewhat stressed and pressured",
        "intensity_medium": "quite stressed and overwhelmed",
        "intensity_high": "extremely stressed and at breaking point"
    }
}

SEVERITY_MODIFIERS = {
    "low": {
        "descriptors": ["minor inconvenience", "small problem", "slight issue", "annoying situation"],
        "impact_language": "This is causing some inconvenience but is manageable.",
        "urgency_level": "Please address this when convenient.",
        "resolution_expectation": "within a few days",
        "tone_adjustment": "Keep tone moderate, not overly dramatic."
    },

    "medium": {
        "descriptors": ["significant problem", "concerning issue", "notable difficulty", "substantial inconvenience"],
        "impact_language": "This is causing real problems and needs attention.",
        "urgency_level": "Please resolve this promptly.",
        "resolution_expectation": "within 24-48 hours",
        "tone_adjustment": "Express clear concern and need for action."
    },

    "high": {
        "descriptors": ["serious problem", "major issue", "significant concern", "urgent matter"],
        "impact_language": "This is causing major disruption and must be resolved quickly.",
        "urgency_level": "This requires immediate attention.",
        "resolution_expectation": "within a few hours",
        "tone_adjustment": "Use urgent, demanding language without being abusive."
    },

    "critical": {
        "descriptors": ["emergency situation", "critical issue", "urgent crisis", "severe problem"],
        "impact_language": "This is an emergency that requires immediate action.",
        "urgency_level": "This is a critical emergency.",
        "resolution_expectation": "immediately",
        "tone_adjustment": "Use emergency language, express extreme urgency."
    }
}

DEMOGRAPHIC_PROMPTS = {
    "age_group": {
        "18-30": "Write as a young adult who is comfortable with technology but may be new to banking.",
        "31-50": "Write as a middle-aged professional with established banking expectations.",
        "51-65": "Write as an experienced adult with specific service expectations.",
        "65+": "Write as a senior customer who values personal service and clear communication."
    },

    "tech_savviness": {
        "low": "Use simpler language about technology, focus on basic banking needs.",
        "medium": "Use moderate technical language, comfortable with digital banking basics.",
        "high": "Can use more technical terms, expects advanced digital features."
    },

    "banking_experience": {
        "novice": "Write as someone new to banking who may not understand all procedures.",
        "intermediate": "Write as someone with basic banking knowledge and experience.",
        "expert": "Write as someone very familiar with banking services and standards."
    }
}

CONTEXT_BUILDERS = {
    "time_context": {
        "morning": "this morning when I was trying to",
        "afternoon": "this afternoon during my lunch break",
        "evening": "yesterday evening after work",
        "weekend": "over the weekend when I needed to",
        "holiday": "during the holiday when banks were closed"
    },

    "location_context": {
        "branch": "at your Main Street branch location",
        "atm": "at the ATM outside your downtown office",
        "online": "while using your online banking platform",
        "mobile": "through your mobile banking app",
        "phone": "when I called your customer service line"
    },

    "urgency_context": {
        "low": "when I have time to deal with this properly",
        "medium": "because I need this resolved soon",
        "high": "as this is affecting my daily banking",
        "critical": "because this is an emergency situation"
    }
}

VALIDATION_PROMPTS = {
    "quality_check": "Ensure the complaint sounds authentic, includes specific details, and matches the requested emotion and severity level.",

    "category_relevance": "Verify the complaint is clearly about {category} and includes relevant keywords and scenarios.",

    "emotion_consistency": "Check that the language and tone consistently reflect {emotion} throughout the complaint.",

    "severity_alignment": "Confirm the urgency and impact described match {severity} severity level.",

    "length_requirement": "Ensure the complaint is between 50-300 words with proper sentence structure.",

    "professionalism": "Maintain complaint authenticity while avoiding excessive profanity or inappropriate language."
}

STYLE_GUIDELINES = {
    "language_style": {
        "formal": "Use proper grammar, complete sentences, and professional language.",
        "informal": "Use conversational tone with contractions and casual expressions.",
        "mixed": "Combine formal concerns with informal emotional expressions."
    },

    "structure_patterns": {
        "chronological": "Start with when the problem occurred, describe what happened, then state the impact.",
        "problem_first": "Lead with the main issue, provide context, then explain consequences.",
        "emotional_opening": "Begin with emotional reaction, explain the cause, then request resolution."
    },

    "detail_levels": {
        "minimal": "Basic problem description with key points only.",
        "moderate": "Include specific details about what went wrong and impact.",
        "comprehensive": "Detailed account with timeline, specific amounts, and complete context."
    }
}

RESPONSE_FORMATTING = {
    "output_format": {
        "plain_text": "Return only the complaint text without formatting.",
        "structured": "Include complaint with metadata tags.",
        "json": "Return structured JSON with complaint and attributes."
    },

    "length_targets": {
        "short": "50-100 words",
        "medium": "100-200 words",
        "long": "200-300 words"
    },

    "cleanup_rules": {
        "remove_quotes": True,
        "fix_spacing": True,
        "capitalize_sentences": True,
        "remove_metadata": True,
        "limit_sentences": 8
    }
}