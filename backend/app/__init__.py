from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from flask_migrate import Migrate
from .config import config
import os

db = SQLAlchemy()
jwt = JWTManager()
migrate = Migrate()

def create_app(config_name=None):
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')

    app = Flask(__name__)
    app.config.from_object(config[config_name])

    db.init_app(app)
    jwt.init_app(app)
    migrate.init_app(app, db)

    CORS(app,
         resources={r"/api/*": {"origins": app.config['CORS_ORIGINS']}},
         supports_credentials=True)

    # Register blueprints
    from .routes.auth import auth_bp
    from .routes.posts import posts_bp
    from .routes.admin import admin_bp
    from .routes.friends import friends_bp
    from .routes.ai import ai_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(posts_bp, url_prefix='/api/posts')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(friends_bp, url_prefix='/api/friends')
    app.register_blueprint(ai_bp, url_prefix='/api/ai')

    @app.route('/api/health')
    def health():
        return {'status': 'ok', 'message': 'CyberShield API running'}

    return app
