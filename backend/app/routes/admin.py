from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from ..models import Post, User
from .. import db
from functools import wraps

admin_bp = Blueprint('admin', __name__)

def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        if not user or user.role != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return fn(*args, **kwargs)
    return wrapper

@admin_bp.route('/stats', methods=['GET'])
@admin_required
def get_stats():
    total = Post.query.count()
    visible = Post.query.filter_by(status='visible').count()
    withheld = Post.query.filter_by(status='withheld').count()
    blocked = Post.query.filter_by(status='blocked').count()
    deleted = Post.query.filter_by(status='deleted').count()
    flagged = Post.query.filter_by(is_flagged=True).count()
    users_count = User.query.count()

    return jsonify({
        'stats': {
            'total_posts': total,
            'visible': visible,
            'withheld': withheld,
            'blocked': blocked,
            'deleted': deleted,
            'flagged': flagged,
            'total_users': users_count
        }
    }), 200

@admin_bp.route('/posts', methods=['GET'])
@admin_required
def get_moderation_queue():
    status_filter = request.args.get('status', 'withheld')
    page = request.args.get('page', 1, type=int)

    query = Post.query
    if status_filter == 'all':
        query = query.filter(Post.status.in_(['withheld', 'blocked', 'flagged']))
    elif status_filter != 'all_posts':
        query = query.filter_by(status=status_filter)

    posts = query.order_by(Post.created_at.desc()) \
        .paginate(page=page, per_page=20, error_out=False)

    return jsonify({
        'posts': [p.to_dict() for p in posts.items],
        'total': posts.total,
        'pages': posts.pages
    }), 200

@admin_bp.route('/posts/<int:post_id>/approve', methods=['POST'])
@admin_required
def approve_post(post_id):
    post = Post.query.get_or_404(post_id)
    post.status = 'visible'
    post.is_flagged = False
    db.session.commit()
    return jsonify({'message': 'Post approved', 'post': post.to_dict()}), 200

@admin_bp.route('/posts/<int:post_id>/delete', methods=['POST'])
@admin_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    post.status = 'deleted'
    db.session.commit()
    return jsonify({'message': 'Post deleted', 'post': post.to_dict()}), 200

@admin_bp.route('/posts/<int:post_id>/block', methods=['POST'])
@admin_required
def block_post(post_id):
    post = Post.query.get_or_404(post_id)
    post.status = 'blocked'
    post.is_flagged = True
    db.session.commit()
    return jsonify({'message': 'Post blocked', 'post': post.to_dict()}), 200

@admin_bp.route('/users', methods=['GET'])
@admin_required
def get_users():
    users = User.query.all()
    return jsonify({'users': [u.to_dict() for u in users]}), 200

@admin_bp.route('/users/<int:user_id>/toggle', methods=['POST'])
@admin_required
def toggle_user(user_id):
    user = User.query.get_or_404(user_id)
    user.is_active = not user.is_active
    db.session.commit()
    status = 'enabled' if user.is_active else 'disabled'
    return jsonify({'message': f'User {status}', 'user': user.to_dict()}), 200
