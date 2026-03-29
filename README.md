# 🛡 CyberShield — AI-Powered Cyberbullying Detection Platform

> A full-stack, production-ready web app that uses AI (Toxic-BERT via Detoxify) to detect and moderate toxic content in real time.

---

## 🧱 Tech Stack

| Layer       | Technology                              |
|-------------|------------------------------------------|
| Backend     | Flask, SQLAlchemy ORM, JWT Auth          |
| Frontend    | React 18 + Vite                          |
| Database    | PostgreSQL (Supabase)                    |
| AI Model    | Detoxify (Toxic-BERT) + keyword fallback |
| Deploy BE   | Render                                   |
| Deploy FE   | Vercel                                   |

---

## 📁 Project Structure

```
cybershield/
├── backend/
│   ├── app/
│   │   ├── models/         # SQLAlchemy models (User, Post, FriendRequest)
│   │   ├── routes/         # Flask blueprints (auth, posts, admin, friends, ai)
│   │   ├── services/       # AI toxicity service (Detoxify + fallback)
│   │   ├── __init__.py     # App factory
│   │   └── config.py       # Config classes (dev/prod)
│   ├── migrations/         # Flask-Migrate migrations
│   ├── run.py              # Entry point
│   ├── requirements.txt
│   ├── render.yaml         # Render deploy config
│   └── .env.example
│
└── frontend/
    ├── src/
    │   ├── api/            # Axios API client (all endpoints)
    │   ├── components/     # Navbar, PostCard, LoginPrompt, LoadingScreen
    │   ├── context/        # AuthContext (loading/unauth/auth states)
    │   ├── hooks/          # useDebounce
    │   ├── pages/          # Feed, Login, Signup, Profile, AdminPanel, Friends
    │   └── App.jsx         # Routes + protected route logic
    ├── index.html
    ├── vite.config.js
    ├── vercel.json
    └── .env.example
```

---

## ⚡ Core Features

- ✅ **JWT Authentication** — signup/login, role-based (user/admin)
- ✅ **Auth State Machine** — loading → unauthenticated → authenticated (no flash bugs)
- ✅ **Public Feed** — visible without login; login/signup in top-right
- ✅ **AI Post Moderation** — Toxic-BERT scores every post on submit
  - `> 0.85` → `blocked`
  - `0.60–0.85` → `withheld` (pending admin review)
  - `< 0.60` → `visible`
- ✅ **Real-Time AI Warning** — debounced typing analysis with `⚠ This message may be harmful`
- ✅ **Login Prompt Popup** — shown when unauthenticated user tries to post
- ✅ **Admin Panel** — stats dashboard, moderation queue, approve/block/delete
- ✅ **Profile Page** — user's posts by status, toxicity scores, delete
- ✅ **Friend System** — send/accept/reject requests, friends list, user search
- ✅ **AI Fallback** — keyword-based classifier if Detoxify unavailable (free tier)

---

## 🚀 Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL (local or Supabase)

---

### 1. Clone & Setup Backend

```bash
cd cybershield/backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from example:
```bash
cp .env.example .env
# Edit .env with your DATABASE_URL, SECRET_KEY, etc.
```

Initialize database:
```bash
flask db init
flask db migrate -m "initial"
flask db upgrade
```

Run backend:
```bash
python run.py
# → http://localhost:5000
```

---

### 2. Setup Frontend

```bash
cd cybershield/frontend
npm install
```

Create `.env`:
```bash
cp .env.example .env
# VITE_API_URL=http://localhost:5000/api
```

Run frontend:
```bash
npm run dev
# → http://localhost:5173
```

---

## ☁️ Deployment

### Step 1 — Database (Supabase)

1. Go to [supabase.com](https://supabase.com) → New project
2. Go to **Settings → Database**
3. Copy the **Connection string (URI)** — it looks like:
   ```
   postgresql://postgres:[password]@db.xxx.supabase.co:5432/postgres
   ```
4. Save this as `DATABASE_URL`

---

### Step 2 — Backend (Render)

1. Push `backend/` to a GitHub repo
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn run:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
   - **Python version:** 3.11
5. Add Environment Variables:
   ```
   FLASK_ENV=production
   SECRET_KEY=<generate a long random string>
   JWT_SECRET_KEY=<generate another long random string>
   DATABASE_URL=<your Supabase URI>
   CORS_ORIGINS=https://your-vercel-app.vercel.app
   USE_FALLBACK_AI=true    # set false + upgrade plan for real Detoxify
   ```
6. After deploy, run migrations via Render Shell:
   ```bash
   flask db upgrade
   ```

> **Note on AI Model:** Detoxify + PyTorch = ~800MB disk + ~1GB RAM.
> Render free tier has 512MB RAM — use `USE_FALLBACK_AI=true` on free tier.
> Upgrade to Render Starter ($7/mo) to run real Detoxify. Set `USE_FALLBACK_AI=false`.

---

### Step 3 — Frontend (Vercel)

1. Push `frontend/` to GitHub (can be same repo, different folder)
2. Go to [vercel.com](https://vercel.com) → New Project → Import repo
3. Settings:
   - **Framework:** Vite
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Add Environment Variable:
   ```
   VITE_API_URL=https://your-render-backend.onrender.com/api
   ```
5. Deploy ✅
6. Go back to Render → update `CORS_ORIGINS` with your Vercel URL

---

## 👤 Creating an Admin Account

On signup, provide the secret admin key:
- **Admin Key:** `CYBERSHIELD_ADMIN_2024`

This grants `role: admin` and access to `/admin` panel.

To change the key, edit `backend/app/routes/auth.py` line:
```python
if admin_key != 'CYBERSHIELD_ADMIN_2024':
```

---

## 🤖 AI Architecture

### Primary: Detoxify (Toxic-BERT)
```python
from detoxify import Detoxify
model = Detoxify('unbiased')   # ~200MB | 'original' = ~400MB
result = model.predict("some text")
# Returns: toxicity, severe_toxicity, obscene, threat, insult, identity_attack
```

Composite score formula:
```
score = max(toxicity, severe_toxicity × 1.2, threat × 1.1, insult × 0.9)
```

### Fallback: Keyword Classifier
When `USE_FALLBACK_AI=true` or Detoxify fails to load:
- High-severity keyword list → score `0.88–0.99`
- Medium-severity keyword list → score `0.45–0.65`
- No keywords matched → score `0.0`

### Real-Time Analysis (Frontend)
- User types in composer → debounced 700ms → `POST /api/ai/analyze`
- If score `> 0.5` → show `⚠ This message may be harmful`
- If score `> 0.85` → pulsing red warning border on composer

---

## 🔌 API Reference

### Auth
| Method | Endpoint          | Auth | Description         |
|--------|-------------------|------|---------------------|
| POST   | /api/auth/signup  | No   | Register user       |
| POST   | /api/auth/login   | No   | Login, get JWT      |
| GET    | /api/auth/me      | JWT  | Get current user    |
| GET    | /api/auth/users   | JWT  | Search users        |

### Posts
| Method | Endpoint          | Auth  | Description         |
|--------|-------------------|-------|---------------------|
| GET    | /api/posts/       | No    | Public feed         |
| GET    | /api/posts/my     | JWT   | My posts            |
| POST   | /api/posts/       | JWT   | Create + AI scan    |
| DELETE | /api/posts/:id    | JWT   | Delete own post     |

### Admin
| Method | Endpoint                      | Auth  | Description         |
|--------|-------------------------------|-------|---------------------|
| GET    | /api/admin/stats              | Admin | Platform stats      |
| GET    | /api/admin/posts?status=...   | Admin | Moderation queue    |
| POST   | /api/admin/posts/:id/approve  | Admin | Approve post        |
| POST   | /api/admin/posts/:id/delete   | Admin | Delete post         |
| POST   | /api/admin/posts/:id/block    | Admin | Block post          |
| GET    | /api/admin/users              | Admin | All users           |
| POST   | /api/admin/users/:id/toggle   | Admin | Enable/disable user |

### Friends
| Method | Endpoint                  | Auth | Description         |
|--------|---------------------------|------|---------------------|
| POST   | /api/friends/send/:id     | JWT  | Send request        |
| GET    | /api/friends/requests     | JWT  | Incoming requests   |
| POST   | /api/friends/respond/:id  | JWT  | Accept / reject     |
| GET    | /api/friends/list         | JWT  | My friends          |
| GET    | /api/friends/status/:id   | JWT  | Friendship status   |

### AI
| Method | Endpoint         | Auth | Description              |
|--------|------------------|------|--------------------------|
| POST   | /api/ai/analyze  | No   | Real-time text analysis  |
| GET    | /api/ai/info     | No   | Model info               |

---

## 🎨 Design System

- **Font:** Space Grotesk (headings/body) + JetBrains Mono (code/scores)
- **Theme:** Dark cyber — deep navy backgrounds, cyan primary, red danger, yellow warning
- **CSS:** CSS Modules per component, global variables in `index.css`
- **Animations:** fadeIn, slideDown, shimmer skeleton, warningPulse, spin

---

## 🔒 Security Notes

- Passwords hashed with Werkzeug `pbkdf2:sha256`
- JWT tokens expire in 24 hours
- Admin routes protected by server-side role check
- CORS restricted to configured origins only
- Content length limited to 2000 chars on both FE and BE
- SQL injection prevented by SQLAlchemy ORM

---

## 📈 Extending the Project

Ideas for v2:
- WebSocket real-time notifications (friend requests, moderation results)
- Image/media post support
- Post comments with toxicity checks
- Appeal system for blocked posts
- Analytics dashboard with Chart.js
- Email verification via SendGrid
- Rate limiting (Flask-Limiter)
- Redis caching for feed

---

Built with ❤️ — resume-ready, deployable, and production-quality.
