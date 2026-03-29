from flask import Blueprint, request, jsonify
from ..services.ai_service import analyze_text, get_model_info

ai_bp = Blueprint('ai', __name__)

@ai_bp.route('/analyze', methods=['POST'])
def analyze():
    """
    Real-time text analysis endpoint.
    Called from frontend while user types (debounced).
    No auth required — just returns toxicity score.
    """
    data = request.get_json()
    if not data or not data.get('text'):
        return jsonify({'toxicity_score': 0.0, 'toxicity_label': 'safe', 'is_flagged': False}), 200

    text = data['text'].strip()
    if len(text) < 5:
        return jsonify({'toxicity_score': 0.0, 'toxicity_label': 'safe', 'is_flagged': False}), 200

    # Limit text length for real-time analysis
    text = text[:500]

    result = analyze_text(text)
    return jsonify({
        'toxicity_score': result['toxicity_score'],
        'toxicity_label': result['toxicity_label'],
        'is_flagged': result['is_flagged'],
        'status': result['status']
    }), 200

@ai_bp.route('/info', methods=['GET'])
def model_info():
    return jsonify(get_model_info()), 200
