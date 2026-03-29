from app import create_app, db
from app.models import User, Post, FriendRequest
import os

app = create_app(os.environ.get('FLASK_ENV', 'default'))

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Post': Post, 'FriendRequest': FriendRequest}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
