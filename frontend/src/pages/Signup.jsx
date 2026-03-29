import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { authAPI } from '../api'
import { useAuth } from '../context/AuthContext'
import styles from './Auth.module.css'

export default function Signup() {
  const navigate = useNavigate()
  const { login } = useAuth()

  const [form, setForm] = useState({
    username: '', email: '', password: '', confirmPassword: '', adminKey: ''
  })
  const [showAdminKey, setShowAdminKey] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    setForm(f => ({ ...f, [e.target.name]: e.target.value }))
    setError('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.username || !form.email || !form.password) {
      setError('All fields are required.')
      return
    }
    if (form.password !== form.confirmPassword) {
      setError('Passwords do not match.')
      return
    }
    if (form.password.length < 6) {
      setError('Password must be at least 6 characters.')
      return
    }

    setLoading(true)
    try {
      const payload = {
        username: form.username,
        email: form.email,
        password: form.password,
      }
      if (form.adminKey) {
        payload.role = 'admin'
        payload.admin_key = form.adminKey
      }
      const res = await authAPI.signup(payload)
      login(res.data.token, res.data.user)
      navigate('/')
    } catch (err) {
      setError(err.response?.data?.error || 'Signup failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={`${styles.card} fade-in`}>
        <div className={styles.brand}>
          <svg width="36" height="36" viewBox="0 0 48 48" fill="none">
            <path d="M24 4L6 12V24C6 34.5 14 43.5 24 46C34 43.5 42 34.5 42 24V12L24 4Z"
              fill="none" stroke="#00d4ff" strokeWidth="2.5" strokeLinejoin="round"/>
            <path d="M16 24L21 29L32 18" stroke="#00d4ff" strokeWidth="2.5"
              strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>CyberShield</span>
        </div>

        <h1 className={styles.title}>Create account</h1>
        <p className={styles.subtitle}>Join the AI-protected community</p>

        {error && <div className={styles.errorBox}>{error}</div>}

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.field}>
            <label className={styles.label}>Username</label>
            <input
              name="username"
              type="text"
              value={form.username}
              onChange={handleChange}
              placeholder="cooluser42"
              className={styles.input}
              autoFocus
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Email</label>
            <input
              name="email"
              type="email"
              value={form.email}
              onChange={handleChange}
              placeholder="you@example.com"
              className={styles.input}
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Password</label>
            <input
              name="password"
              type="password"
              value={form.password}
              onChange={handleChange}
              placeholder="Min. 6 characters"
              className={styles.input}
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Confirm Password</label>
            <input
              name="confirmPassword"
              type="password"
              value={form.confirmPassword}
              onChange={handleChange}
              placeholder="Repeat password"
              className={styles.input}
            />
          </div>

          <button
            type="button"
            onClick={() => setShowAdminKey(v => !v)}
            className={styles.toggleAdmin}
          >
            {showAdminKey ? '▲ Hide' : '▼ Have an admin key?'}
          </button>

          {showAdminKey && (
            <div className={`${styles.field} slide-down`}>
              <label className={styles.label}>Admin Key (optional)</label>
              <input
                name="adminKey"
                type="password"
                value={form.adminKey}
                onChange={handleChange}
                placeholder="Enter admin key"
                className={styles.input}
              />
            </div>
          )}

          <button type="submit" disabled={loading} className={styles.submitBtn}>
            {loading ? <span className={styles.spinner} /> : 'Create Account'}
          </button>
        </form>

        <p className={styles.switchText}>
          Already have an account?{' '}
          <Link to="/login" className={styles.switchLink}>Sign in</Link>
        </p>
      </div>
    </div>
  )
}
