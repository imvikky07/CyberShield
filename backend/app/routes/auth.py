from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from ..models import User
from .. import db
import re

auth_bp = Blueprint('auth', __name__)

def validate_email(email):
    return re.match(r'^[^@]+@[^@]+\.[^@]+$', email)

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    role = data.get('role', 'user')

    # Validation
    if not username or len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if not email or not validate_email(email):
        return jsonify({'error': 'Invalid email address'}), 400
    if not password or len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    # Check uniqueness
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken'}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    # Only allow admin role with secret key
    if role == 'admin':
        admin_key = data.get('admin_key', '')
        if admin_key != 'CYBERSHIELD_ADMIN_2024':
            role = 'user'

    user = User(username=username, email=email, role=role)
    user.set_password(password)

    db.session.add(user)
    db.session.commit()

    token = create_access_token(identity=str(user.id))

    return jsonify({
        'message': 'Account created successfully',
        'token': token,
        'user': user.to_dict()
    }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    identifier = data.get('email', data.get('username', '')).strip()
    password = data.get('password', '')

    if not identifier or not password:
        return jsonify({'error': 'Email/username and password required'}), 400

    # Find by email or username
    user = User.query.filter(
        (User.email == identifier.lower()) | (User.username == identifier)
    ).first()

    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid credentials'}), 401

    if not user.is_active:
        return jsonify({'error': 'Account is disabled'}), 403

    token = create_access_token(identity=str(user.id))

    return jsonify({
        'message': 'Login successful',
        'token': token,
        'user': user.to_dict()
    }), 200


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_me():
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'user': user.to_dict()}), 200


@auth_bp.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    """Get all users (for friend search)."""
    current_user_id = int(get_jwt_identity())
    query = request.args.get('q', '')

    users_query = User.query.filter(User.id != current_user_id)
    if query:
        users_query = users_query.filter(User.username.ilike(f'%{query}%'))

    users = users_query.limit(20).all()
    return jsonify({'users': [u.to_dict() for u in users]}), 200
