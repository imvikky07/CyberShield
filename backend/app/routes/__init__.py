from .auth import auth_bp
from .posts import posts_bp
from .admin import admin_bp
from .friends import friends_bp
from .ai import ai_bp

__all__ = ['auth_bp', 'posts_bp', 'admin_bp', 'friends_bp', 'ai_bp']
