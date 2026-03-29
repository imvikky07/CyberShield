from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from ..models import FriendRequest, User
from .. import db
from sqlalchemy import or_, and_

friends_bp = Blueprint('friends', __name__)

@friends_bp.route('/send/<int:receiver_id>', methods=['POST'])
@jwt_required()
def send_request(receiver_id):
    sender_id = int(get_jwt_identity())

    if sender_id == receiver_id:
        return jsonify({'error': 'Cannot send request to yourself'}), 400

    receiver = User.query.get(receiver_id)
    if not receiver:
        return jsonify({'error': 'User not found'}), 404

    # Check existing
    existing = FriendRequest.query.filter(
        or_(
            and_(FriendRequest.sender_id == sender_id, FriendRequest.receiver_id == receiver_id),
            and_(FriendRequest.sender_id == receiver_id, FriendRequest.receiver_id == sender_id)
        )
    ).first()

    if existing:
        if existing.status == 'accepted':
            return jsonify({'error': 'Already friends'}), 409
        if existing.status == 'pending':
            return jsonify({'error': 'Request already pending'}), 409
        # If rejected, allow re-send
        existing.status = 'pending'
        existing.sender_id = sender_id
        existing.receiver_id = receiver_id
        db.session.commit()
        return jsonify({'message': 'Friend request sent', 'request': existing.to_dict()}), 200

    fr = FriendRequest(sender_id=sender_id, receiver_id=receiver_id)
    db.session.add(fr)
    db.session.commit()

    return jsonify({'message': 'Friend request sent', 'request': fr.to_dict()}), 201


@friends_bp.route('/requests', methods=['GET'])
@jwt_required()
def get_requests():
    user_id = int(get_jwt_identity())
    direction = request.args.get('direction', 'received')

    if direction == 'received':
        requests = FriendRequest.query.filter_by(
            receiver_id=user_id, status='pending'
        ).all()
    else:
        requests = FriendRequest.query.filter_by(
            sender_id=user_id, status='pending'
        ).all()

    return jsonify({'requests': [r.to_dict() for r in requests]}), 200


@friends_bp.route('/respond/<int:request_id>', methods=['POST'])
@jwt_required()
def respond_request(request_id):
    user_id = int(get_jwt_identity())
    data = request.get_json()
    action = data.get('action')  # 'accept' or 'reject'

    if action not in ['accept', 'reject']:
        return jsonify({'error': 'Action must be accept or reject'}), 400

    fr = FriendRequest.query.get_or_404(request_id)

    if fr.receiver_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403

    fr.status = 'accepted' if action == 'accept' else 'rejected'
    db.session.commit()

    return jsonify({'message': f'Request {fr.status}', 'request': fr.to_dict()}), 200


@friends_bp.route('/list', methods=['GET'])
@jwt_required()
def get_friends():
    user_id = int(get_jwt_identity())

    accepted = FriendRequest.query.filter(
        and_(
            FriendRequest.status == 'accepted',
            or_(
                FriendRequest.sender_id == user_id,
                FriendRequest.receiver_id == user_id
            )
        )
    ).all()

    friends = []
    for fr in accepted:
        friend_id = fr.receiver_id if fr.sender_id == user_id else fr.sender_id
        friend = User.query.get(friend_id)
        if friend:
            friends.append(friend.to_dict())

    return jsonify({'friends': friends}), 200


@friends_bp.route('/status/<int:other_user_id>', methods=['GET'])
@jwt_required()
def get_friendship_status(other_user_id):
    user_id = int(get_jwt_identity())

    fr = FriendRequest.query.filter(
        or_(
            and_(FriendRequest.sender_id == user_id, FriendRequest.receiver_id == other_user_id),
            and_(FriendRequest.sender_id == other_user_id, FriendRequest.receiver_id == user_id)
        )
    ).first()

    if not fr:
        return jsonify({'status': 'none'}), 200

    return jsonify({
        'status': fr.status,
        'request_id': fr.id,
        'is_sender': fr.sender_id == user_id
    }), 200
