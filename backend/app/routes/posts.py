from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request
from ..models import Post, User
from .. import db
from ..services.ai_service import analyze_text

posts_bp = Blueprint('posts', __name__)

@posts_bp.route('/', methods=['GET'])
def get_feed():
    """Public feed — only visible posts."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    posts = Post.query.filter_by(status='visible') \
        .order_by(Post.created_at.desc()) \
        .paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'posts': [p.to_dict() for p in posts.items],
        'total': posts.total,
        'pages': posts.pages,
        'current_page': page
    }), 200


@posts_bp.route('/my', methods=['GET'])
@jwt_required()
def get_my_posts():
    """Current user's own posts (all statuses)."""
    user_id = int(get_jwt_identity())
    posts = Post.query.filter_by(user_id=user_id) \
        .order_by(Post.created_at.desc()).all()
    return jsonify({'posts': [p.to_dict() for p in posts]}), 200


@posts_bp.route('/', methods=['POST'])
@jwt_required()
def create_post():
    user_id = int(get_jwt_identity())
    data = request.get_json()

    if not data or not data.get('content'):
        return jsonify({'error': 'Content is required'}), 400

    content = data['content'].strip()
    if len(content) < 1:
        return jsonify({'error': 'Post cannot be empty'}), 400
    if len(content) > 2000:
        return jsonify({'error': 'Post too long (max 2000 chars)'}), 400

    # AI toxicity analysis
    analysis = analyze_text(content)

    post = Post(
        content=content,
        user_id=user_id,
        status=analysis['status'],
        is_flagged=analysis['is_flagged'],
        toxicity_score=analysis['toxicity_score'],
        toxicity_label=analysis['toxicity_label']
    )

    db.session.add(post)
    db.session.commit()

    response_data = {
        'message': 'Post created',
        'post': post.to_dict(),
        'moderation': {
            'status': analysis['status'],
            'toxicity_score': analysis['toxicity_score'],
            'toxicity_label': analysis['toxicity_label'],
            'is_flagged': analysis['is_flagged']
        }
    }

    status_code = 201
    if analysis['status'] == 'blocked':
        
    # Auto-delete immediately — don't even save to DB
        db.session.delete(post)
        db.session.commit()
        return jsonify({
            'message': 'Post rejected',
            'post': None,
            'moderation': {
                'status': 'deleted',
                'toxicity_score': analysis['toxicity_score'],
                'toxicity_label': analysis['toxicity_label'],
                'is_flagged': True
            },
            'warning': '🚫 Your message was automatically deleted — harmful content detected.'
        }), 200

    elif analysis['status'] == 'withheld':
        response_data['warning'] = '⚠ Your post is under review by moderators.'
        status_code = 202

    return jsonify(response_data), status_code


@posts_bp.route('/<int:post_id>/report', methods=['POST'])
@jwt_required()
def report_post(post_id):
    user_id = int(get_jwt_identity())
    post = Post.query.get_or_404(post_id)

    if post.user_id == user_id:
        return jsonify({'error': 'You cannot report your own post'}), 400

    # Flag it and push to withheld for admin review if not already moderated
    post.is_flagged = True
    if post.status == 'visible':
        post.status = 'withheld'

    db.session.commit()
    return jsonify({'message': 'Post reported. Our moderators will review it.'}), 200


@posts_bp.route('/<int:post_id>', methods=['DELETE'])
@jwt_required()
def delete_post(post_id):
    user_id = int(get_jwt_identity())
    post = Post.query.get_or_404(post_id)

    user = User.query.get(user_id)
    if post.user_id != user_id and user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    post.status = 'deleted'
    db.session.commit()

    return jsonify({'message': 'Post deleted'}), 200
