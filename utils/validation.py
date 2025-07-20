import re
import string
from collections import Counter
import math


class ComplaintValidator:
    def __init__(self):
        self.min_word_count = 10
        self.max_word_count = 300
        self.min_char_count = 50
        self.max_char_count = 1500
        self.min_sentence_count = 2
        self.max_sentence_count = 10
        self.profanity_words = {
            'damn', 'hell', 'crap', 'stupid', 'idiot', 'shit', 'fuck',
            'bitch', 'asshole', 'bastard', 'piss', 'suck'
        }

    def validate_complaint(self, text, category=None, severity=None, emotion=None):
        if not text or not isinstance(text, str):
            return False, "Invalid text input"

        text = text.strip()
        if not text:
            return False, "Empty text"

        length_valid, length_msg = self._validate_length(text)
        if not length_valid:
            return False, length_msg

        quality_valid, quality_msg = self._validate_quality(text)
        if not quality_valid:
            return False, quality_msg

        if category:
            category_valid, category_msg = self._validate_category_relevance(text, category)
            if not category_valid:
                return False, category_msg

        profanity_valid, profanity_msg = self._validate_profanity(text)
        if not profanity_valid:
            return False, profanity_msg

        return True, "Validation passed"

    def _validate_length(self, text):
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len([s for s in text.split('.') if s.strip()])

        if word_count < self.min_word_count:
            return False, f"Too few words: {word_count} < {self.min_word_count}"

        if word_count > self.max_word_count:
            return False, f"Too many words: {word_count} > {self.max_word_count}"

        if char_count < self.min_char_count:
            return False, f"Too few characters: {char_count} < {self.min_char_count}"

        if char_count > self.max_char_count:
            return False, f"Too many characters: {char_count} > {self.max_char_count}"

        if sentence_count < self.min_sentence_count:
            return False, f"Too few sentences: {sentence_count} < {self.min_sentence_count}"

        if sentence_count > self.max_sentence_count:
            return False, f"Too many sentences: {sentence_count} > {self.max_sentence_count}"

        return True, "Length validation passed"

    def _validate_quality(self, text):
        words = text.lower().split()

        if len(set(words)) < len(words) * 0.7:
            return False, "Too much repetition"

        word_counts = Counter(words)
        max_word_frequency = max(word_counts.values())
        if max_word_frequency > len(words) * 0.15:
            return False, "Single word used too frequently"

        if not re.search(r'[.!?]', text):
            return False, "No proper sentence endings"

        if len(re.findall(r'[A-Z]', text)) < 2:
            return False, "Insufficient capitalization"

        consonant_vowel_ratio = self._calculate_consonant_vowel_ratio(text)
        if consonant_vowel_ratio < 0.5 or consonant_vowel_ratio > 3.0:
            return False, "Unusual consonant-vowel ratio"

        return True, "Quality validation passed"

    def _validate_category_relevance(self, text, category):
        category_keywords = {
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

        keywords = category_keywords.get(category, [])
        if not keywords:
            return True, "No keywords defined for category"

        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)

        if matches < 1:
            return False, f"No relevant keywords found for {category}"

        return True, "Category relevance validated"

    def _validate_profanity(self, text):
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        explicit_profanity = {'fuck', 'shit', 'bitch', 'asshole'}
        for word in words:
            if word in explicit_profanity:
                return False, "Contains explicit profanity"

        profanity_count = sum(1 for word in words if word in self.profanity_words)
        if profanity_count > 2:
            return False, "Too much profanity"

        return True, "Profanity check passed"

    def _calculate_consonant_vowel_ratio(self, text):
        vowels = 'aeiouAEIOU'
        letters = [c for c in text if c.isalpha()]

        if not letters:
            return 1.0

        vowel_count = sum(1 for c in letters if c in vowels)
        consonant_count = len(letters) - vowel_count

        if vowel_count == 0:
            return float('inf')

        return consonant_count / vowel_count


class QualityChecker:
    def __init__(self):
        self.coherence_threshold = 0.6
        self.similarity_threshold = 0.8

    def check_content_quality(self, text):
        scores = {}

        scores['readability'] = self._calculate_readability(text)
        scores['coherence'] = self._calculate_coherence(text)
        scores['sentiment_consistency'] = self._check_sentiment_consistency(text)
        scores['structure_quality'] = self._check_structure_quality(text)
        scores['vocabulary_diversity'] = self._calculate_vocabulary_diversity(text)

        overall_score = sum(scores.values()) / len(scores)

        return {
            'overall_score': overall_score,
            'individual_scores': scores,
            'passed': overall_score >= self.coherence_threshold
        }

    def _calculate_readability(self, text):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)

        if avg_sentence_length < 5:
            return 0.3
        elif avg_sentence_length < 15:
            return 0.8
        elif avg_sentence_length < 25:
            return 0.9
        else:
            return 0.5

    def _calculate_coherence(self, text):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5

        coherence_indicators = [
            'and', 'but', 'however', 'therefore', 'because', 'since', 'while',
            'although', 'furthermore', 'moreover', 'consequently', 'meanwhile'
        ]

        text_lower = text.lower()
        indicator_count = sum(1 for indicator in coherence_indicators if indicator in text_lower)

        return min(1.0, indicator_count / len(sentences))

    def _check_sentiment_consistency(self, text):
        positive_words = [
            'good', 'great', 'excellent', 'satisfied', 'happy', 'pleased',
            'wonderful', 'amazing', 'fantastic', 'perfect'
        ]

        negative_words = [
            'bad', 'terrible', 'awful', 'frustrated', 'angry', 'upset',
            'disappointed', 'horrible', 'unacceptable', 'furious'
        ]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count == 0 and negative_count == 0:
            return 0.5

        total_sentiment = positive_count + negative_count
        if total_sentiment == 0:
            return 0.5

        dominant_ratio = max(positive_count, negative_count) / total_sentiment
        return dominant_ratio

    def _check_structure_quality(self, text):
        score = 0.0

        if text[0].isupper():
            score += 0.2

        if text.strip().endswith(('.', '!', '?')):
            score += 0.2

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) >= 2:
            score += 0.3

        if re.search(r'\b(I|my|me)\b', text, re.IGNORECASE):
            score += 0.3

        return score

    def _calculate_vocabulary_diversity(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)

        return min(1.0, diversity_ratio * 2)


class ContentFilter:
    def __init__(self):
        self.spam_patterns = [
            r'click here', r'free money', r'guaranteed', r'act now',
            r'limited time', r'call now', r'www\.', r'http'
        ]

        self.inappropriate_patterns = [
            r'\b(fuck|shit|damn|hell|bitch|asshole)\b',
            r'[A-Z]{3,}',
            r'(.)\1{4,}'
        ]

    def filter_content(self, text):
        filtered_text = text
        issues = []

        for pattern in self.spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Spam pattern detected: {pattern}")

        for pattern in self.inappropriate_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                issues.append(f"Inappropriate content: {pattern}")

        filtered_text = self._clean_text(filtered_text)

        return {
            'filtered_text': filtered_text,
            'issues': issues,
            'is_clean': len(issues) == 0
        }

    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        text = text.strip()

        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]

        if text and not text.endswith(('.', '!', '?')):
            text += '.'

        return text


class DuplicateDetector:
    def __init__(self):
        self.similarity_threshold = 0.8

    def detect_duplicates(self, new_text, existing_texts):
        if not existing_texts:
            return False, 0.0, None

        max_similarity = 0.0
        most_similar = None

        for existing_text in existing_texts:
            similarity = self.calculate_similarity(new_text, existing_text)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = existing_text

        is_duplicate = max_similarity >= self.similarity_threshold

        return is_duplicate, max_similarity, most_similar

    def calculate_similarity(self, text1, text2):
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        if not words1 and not words2:
            return 1.0

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        jaccard_similarity = len(intersection) / len(union)

        length_similarity = 1.0 - abs(len(text1) - len(text2)) / max(len(text1), len(text2))

        combined_similarity = (jaccard_similarity + length_similarity) / 2

        return combined_similarity

    def find_similar_complaints(self, target_text, complaint_list, threshold=0.7):
        similar_complaints = []

        for complaint in complaint_list:
            similarity = self.calculate_similarity(target_text, complaint)
            if similarity >= threshold:
                similar_complaints.append({
                    'text': complaint,
                    'similarity': similarity
                })

        similar_complaints.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_complaints


def validate_complaint_text(text, category=None, severity=None, emotion=None):
    validator = ComplaintValidator()
    return validator.validate_complaint(text, category, severity, emotion)


def check_content_quality(text):
    checker = QualityChecker()
    return checker.check_content_quality(text)


def filter_profanity(text):
    filter_obj = ContentFilter()
    return filter_obj.filter_content(text)


def detect_duplicates(new_text, existing_texts):
    detector = DuplicateDetector()
    return detector.detect_duplicates(new_text, existing_texts)


def calculate_similarity(text1, text2):
    detector = DuplicateDetector()
    return detector.calculate_similarity(text1, text2)


def extract_keywords(text, category=None):
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)

    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
        'our', 'their', 'a', 'an'
    }

    keywords = []
    for word, freq in word_freq.most_common(20):
        if word not in stop_words and len(word) > 2:
            keywords.append({
                'word': word,
                'frequency': freq,
                'relevance': freq / len(words)
            })

    return keywords


def normalize_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    text = re.sub(r'\s+([.!?])', r'\1', text)

    sentences = text.split('.')
    normalized_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            if not sentence[0].isupper():
                sentence = sentence[0].upper() + sentence[1:]
            normalized_sentences.append(sentence)

    return '. '.join(normalized_sentences)


def validate_category_relevance(text, category):
    validator = ComplaintValidator()
    return validator._validate_category_relevance(text, category)


def check_emotion_consistency(text, emotion):
    emotion_indicators = {
        'frustrated': ['frustrated', 'annoying', 'irritating', 'bothersome'],
        'angry': ['angry', 'furious', 'mad', 'livid', 'outraged'],
        'worried': ['worried', 'concerned', 'anxious', 'nervous'],
        'scared': ['scared', 'frightened', 'terrified', 'afraid'],
        'confused': ['confused', 'puzzled', 'bewildered', 'unclear'],
        'disappointed': ['disappointed', 'let down', 'dissatisfied'],
        'panicked': ['panicked', 'frantic', 'desperate', 'urgent'],
        'stressed': ['stressed', 'overwhelmed', 'pressured', 'tense']
    }

    indicators = emotion_indicators.get(emotion, [])
    text_lower = text.lower()

    matches = sum(1 for indicator in indicators if indicator in text_lower)
    consistency_score = min(1.0, matches / max(1, len(indicators) * 0.3))

    return {
        'consistent': consistency_score >= 0.3,
        'score': consistency_score,
        'matches': matches
    }


def validate_severity_alignment(text, severity):
    severity_indicators = {
        'low': ['minor', 'small', 'slight', 'inconvenience'],
        'medium': ['significant', 'concerning', 'problematic'],
        'high': ['major', 'severe', 'serious', 'urgent'],
        'critical': ['emergency', 'critical', 'immediate', 'urgent']
    }

    indicators = severity_indicators.get(severity, [])
    text_lower = text.lower()

    urgency_words = ['immediately', 'urgent', 'asap', 'emergency', 'critical']
    urgency_count = sum(1 for word in urgency_words if word in text_lower)

    severity_matches = sum(1 for indicator in indicators if indicator in text_lower)

    alignment_score = 0.0

    if severity == 'low' and urgency_count == 0:
        alignment_score += 0.5
    elif severity in ['high', 'critical'] and urgency_count > 0:
        alignment_score += 0.5

    if severity_matches > 0:
        alignment_score += 0.5

    return {
        'aligned': alignment_score >= 0.5,
        'score': alignment_score,
        'severity_matches': severity_matches,
        'urgency_count': urgency_count
    }