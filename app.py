from datetime import datetime
from flask import Flask, render_template, request, redirect, session, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sqlalchemy.exc import IntegrityError
import os
from huggingface_hub import login



login(token="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")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("JungleLee/bert-toxic-comment-classification")
model = AutoModelForSequenceClassification.from_pretrained("JungleLee/bert-toxic-comment-classification")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def is_cyberbullying(message: str) -> bool:
    result = classifier(message)[0]
    return result['label'].lower() in ['toxic', 'label_1'] and result['score'] > 0.8

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "db.sqlite3")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    content = db.Column(db.Text, nullable=False)
    flagged = db.Column(db.Boolean, default=False)
    user = db.relationship('User', backref='messages')


class FlaggedMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Owner of the original message
    flagged_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Who flagged it
    content = db.Column(db.Text, nullable=False)
    message_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=False)



class DeletedMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    message = db.relationship('Message', backref=db.backref('deleted_messages', lazy=True))
    user = db.relationship('User', backref=db.backref('deleted_messages', lazy=True))


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = User(
            username=request.form['username'],
            email=request.form.get('email'),
            password=generate_password_hash(request.form['password'])
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('user_login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        username = request.form['username']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, request.form['password']):
            session.update({'user_id': user.id, 'username': user.username, 'is_admin': False})
            print(f"Logged in as: {session['username']}")  # Debugging
            return redirect(url_for('chat'))
        return "Invalid credentials"
    return render_template('user_login.html')


@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin123':
            session.update({'user_id': 0, 'username': 'admin', 'is_admin': True})
            return redirect(url_for('admin_dashboard'))
        return "Invalid admin credentials"
    return render_template('admin_login.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'user_id' not in session or session.get('is_admin'):
        return redirect(url_for('user_login'))

    if request.method == 'POST':
        content = request.form.get('message', '').strip()

        if content:
            flagged = is_cyberbullying(content)

            # Add to Message table
            msg = Message(user_id=session['user_id'], content=content, flagged=flagged)
            db.session.add(msg)
            db.session.commit()

            if flagged:
                # AI detected cyberbullying: delete immediately
                deleted = DeletedMessage(
                    user_id=session['user_id'],
                    message_id=msg.id,
                    content=content
                )
                db.session.add(deleted)
                db.session.delete(msg)
                db.session.commit()

                flash('Your message was detected as cyberbullying and has been deleted by the system.', 'danger')

    # ✅ Load latest 5 messages with user info
    from sqlalchemy.orm import joinedload
    messages = Message.query.options(joinedload(Message.user))\
                .order_by(Message.id.desc()).limit(5).all()

    return render_template('chat.html', username=session['username'], messages=messages)


@app.route('/flag/<int:msg_id>', methods=['POST'])
def flag_message(msg_id):
    # Ensure the user is logged in and not an admin
    if 'user_id' not in session or session.get('is_admin'):
        return redirect(url_for('user_login'))

    # Retrieve the message by its ID
    message = Message.query.get_or_404(msg_id)

    # Prevent users from flagging their own messages
    if message.user_id == session['user_id']:
        flash('You cannot flag your own message.', 'warning')
        return redirect(url_for('chat'))

    # Check if the message has already been flagged by this user
    already_flagged = FlaggedMessage.query.filter_by(message_id=msg_id, flagged_by=session['user_id']).first()
    if already_flagged:
        flash('You have already flagged this message.', 'info')
        return redirect(url_for('chat'))

    # Add the flagged message to the FlaggedMessage table
    flagged = FlaggedMessage(
        user_id=message.user_id,
        content=message.content,
        message_id=message.id,
        flagged_by=session['user_id']  # You'll need to have this column in the model
    )
    db.session.add(flagged)
    db.session.commit()

    flash("Message flagged for review.", "success")
    return redirect(url_for('chat'))



@app.route('/admin/delete_flagged/<int:id>', methods=['POST'])
def delete_flagged_message(id):
    flagged_message = FlaggedMessage.query.get(id)
    
    if not flagged_message:
        flash('Message not found', 'danger')
        return redirect(url_for('admin_dashboard'))

    # Create a new DeletedMessage with the same message_id
    deleted_message = DeletedMessage(
        user_id=flagged_message.user_id,  # or whatever logic you use for user_id
        message_id=flagged_message.message_id,  # Ensure this is not None
        content=flagged_message.content
    )

    # Add to the deleted message table
    db.session.add(deleted_message)
    
    # Optionally, remove the flagged message after adding it to DeletedMessage
    db.session.delete(flagged_message)

    try:
        db.session.commit()
        flash('Message deleted and archived successfully', 'success')
    except IntegrityError as e:
        db.session.rollback()  # Rollback in case of an error
        flash('Failed to delete the message', 'danger')
    print(f"Flagged message ID: {flagged_message.message_id}")
    return redirect(url_for('admin_dashboard'))





@app.route('/admin/dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))

    # Basic stats
    total_messages = Message.query.count()
    flagged_messages_count = FlaggedMessage.query.count()
    total_users = User.query.count()

    # Deleted messages (AI-removed)
    deleted_messages = db.session.query(
        User.username,
        DeletedMessage.content,
        DeletedMessage.timestamp
    ).join(User, DeletedMessage.user_id == User.id).all()

    # Flagged messages for admin to review
    flagged_messages = db.session.query(
        Message.id,
        Message.content,
        User.username
    ).join(User, User.id == Message.user_id
    ).join(FlaggedMessage, FlaggedMessage.message_id == Message.id).all()

    test_result = None
    if request.method == 'POST':
        test_text = request.form.get('test_text')
        if test_text:
            result = classifier(test_text)[0]
            test_result = f"{result['label']} ({round(result['score'], 2)})"

    return render_template('admin_dashboard.html',
        total_messages=total_messages,
        flagged_messages_count=flagged_messages_count,
        total_users=total_users,
        deleted_messages=deleted_messages,
        flagged_messages=flagged_messages,
        test_result=test_result
    )


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run()
