from .. import db
from datetime import datetime

class FriendRequest(db.Model):
    __tablename__ = 'friend_requests'

    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending / accepted / rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('sender_id', 'receiver_id', name='unique_friend_request'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'sender': {
                'id': self.sender.id,
                'username': self.sender.username,
                'avatar_url': self.sender.avatar_url
            } if self.sender else None,
            'receiver': {
                'id': self.receiver.id,
                'username': self.receiver.username,
                'avatar_url': self.receiver.avatar_url
            } if self.receiver else None
        }
