from .. import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='user')  # 'user' or 'admin'
    bio = db.Column(db.String(300), default='')
    avatar_url = db.Column(db.String(500), default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    posts = db.relationship('Post', backref='author', lazy='dynamic')
    sent_requests = db.relationship('FriendRequest',
                                     foreign_keys='FriendRequest.sender_id',
                                     backref='sender', lazy='dynamic')
    received_requests = db.relationship('FriendRequest',
                                         foreign_keys='FriendRequest.receiver_id',
                                         backref='receiver', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'bio': self.bio,
            'avatar_url': self.avatar_url,
            'created_at': self.created_at.isoformat()
        }
