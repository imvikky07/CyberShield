"""
AI Toxicity Detection Service
Primary: Detoxify (Toxic-BERT) - local inference
Fallback: Keyword-based heuristic (for deployment without GPU/large model)

Deployment note: Detoxify model is ~400MB. For Render free tier,
use the 'unbiased' model (smaller) or enable USE_FALLBACK_AI=true env var.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

# Global model cache
_model = None
_use_fallback = os.environ.get('USE_FALLBACK_AI', 'false').lower() == 'true'

def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from detoxify import Detoxify
        # Use 'unbiased' for smaller footprint (~200MB) vs 'original' (~400MB)
        model_name = os.environ.get('DETOXIFY_MODEL', 'unbiased')
        logger.info(f"Loading Detoxify model: {model_name}")
        _model = Detoxify(model_name)
        logger.info("Detoxify model loaded successfully")
        return _model
    except Exception as e:
        logger.warning(f"Failed to load Detoxify: {e}. Using fallback.")
        return None

# Keyword-based fallback classifier
TOXIC_KEYWORDS = {
    'high': [
        'kill yourself', 'kys', 'die', 'murder', 'rape', 'terrorist',
        'n***er', 'faggot', 'retard', 'worthless', 'disgusting pig',
        'i hate you', 'go to hell', 'piece of shit', 'fuck you',
        'shoot you', 'bomb', 'attack', 'destroy you'
    ],
    'medium': [
        'idiot', 'stupid', 'moron', 'loser', 'ugly', 'fat',
        'hate', 'awful', 'terrible', 'shut up', 'dumb',
        'pathetic', 'useless', 'trash', 'garbage', 'freak'
    ]
}

def _fallback_score(text: str) -> dict:
    """Simple keyword-based toxicity scoring as fallback."""
    text_lower = text.lower()
    score = 0.0

    for word in TOXIC_KEYWORDS['high']:
        if word in text_lower:
            score = max(score, 0.88 + (0.1 * text_lower.count(word)))

    for word in TOXIC_KEYWORDS['medium']:
        if word in text_lower:
            score = max(score, 0.45 + (0.15 * text_lower.count(word)))

    # Cap at 0.99
    score = min(score, 0.99)

    # Detect label
    if score > 0.85:
        label = 'harmful'
    elif score > 0.6:
        label = 'mild'
    else:
        label = 'safe'

    return {
        'toxicity_score': round(score, 4),
        'toxicity_label': label,
        'method': 'fallback_keyword'
    }

def analyze_text(text: str) -> dict:
    """
    Analyze text for toxicity.
    Returns dict with toxicity_score, toxicity_label, status, method.
    """
    if not text or not text.strip():
        return {
            'toxicity_score': 0.0,
            'toxicity_label': 'safe',
            'status': 'visible',
            'method': 'empty'
        }

    result = {}

    if _use_fallback:
        result = _fallback_score(text)
    else:
        model = _load_model()
        if model is None:
            result = _fallback_score(text)
        else:
            try:
                predictions = model.predict(text)
                # Detoxify returns dict with keys: toxicity, severe_toxicity, obscene, threat, insult, identity_attack
                toxicity = float(predictions.get('toxicity', 0.0))
                severe = float(predictions.get('severe_toxicity', 0.0))
                threat = float(predictions.get('threat', 0.0))
                insult = float(predictions.get('insult', 0.0))

                # Weighted composite score
                score = max(toxicity, severe * 1.2, threat * 1.1, insult * 0.9)
                score = min(score, 0.99)

                if score > 0.85:
                    label = 'harmful'
                elif score > 0.6:
                    label = 'mild'
                else:
                    label = 'safe'

                result = {
                    'toxicity_score': round(score, 4),
                    'toxicity_label': label,
                    'method': 'detoxify',
                    'breakdown': {
                        'toxicity': round(toxicity, 4),
                        'severe_toxicity': round(severe, 4),
                        'threat': round(threat, 4),
                        'insult': round(insult, 4)
                    }
                }
            except Exception as e:
                logger.error(f"Detoxify inference error: {e}")
                result = _fallback_score(text)

    # Determine post status based on score
    score = result['toxicity_score']
    if score > 0.85:
        result['status'] = 'blocked'
        result['is_flagged'] = True
    elif score > 0.6:
        result['status'] = 'withheld'
        result['is_flagged'] = True
    else:
        result['status'] = 'visible'
        result['is_flagged'] = False

    return result

def get_model_info() -> dict:
    """Return info about which AI backend is being used."""
    if _use_fallback:
        return {'backend': 'keyword_fallback', 'model': 'none'}
    model = _load_model()
    if model is None:
        return {'backend': 'keyword_fallback', 'model': 'none'}
    return {'backend': 'detoxify', 'model': os.environ.get('DETOXIFY_MODEL', 'unbiased')}
