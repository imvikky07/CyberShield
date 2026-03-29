from .. import db
from datetime import datetime

class Post(db.Model):
    __tablename__ = 'posts'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    status = db.Column(db.String(20), default='visible')  # visible / withheld / blocked / deleted
    is_flagged = db.Column(db.Boolean, default=False)
    toxicity_score = db.Column(db.Float, default=0.0)
    toxicity_label = db.Column(db.String(50), default='safe')  # safe / mild / harmful
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self, include_author=True):
        data = {
            'id': self.id,
            'content': self.content,
            'user_id': self.user_id,
            'status': self.status,
            'is_flagged': self.is_flagged,
            'toxicity_score': round(self.toxicity_score, 4),
            'toxicity_label': self.toxicity_label,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_author and self.author:
            data['author'] = {
                'id': self.author.id,
                'username': self.author.username,
                'avatar_url': self.author.avatar_url
            }
        return data
