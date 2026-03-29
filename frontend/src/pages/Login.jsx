import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { authAPI } from '../api'
import { useAuth } from '../context/AuthContext'
import styles from './Auth.module.css'

export default function Login() {
  const navigate = useNavigate()
  const { login } = useAuth()

  const [form, setForm] = useState({ identifier: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    setForm(f => ({ ...f, [e.target.name]: e.target.value }))
    setError('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.identifier || !form.password) {
      setError('All fields are required.')
      return
    }
    setLoading(true)
    try {
      const res = await authAPI.login({
        email: form.identifier,
        username: form.identifier,
        password: form.password,
      })
      login(res.data.token, res.data.user)
      navigate('/')
    } catch (err) {
      setError(err.response?.data?.error || 'Login failed. Check your credentials.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={`${styles.card} fade-in`}>
        {/* Brand */}
        <div className={styles.brand}>
          <svg width="36" height="36" viewBox="0 0 48 48" fill="none">
            <path d="M24 4L6 12V24C6 34.5 14 43.5 24 46C34 43.5 42 34.5 42 24V12L24 4Z"
              fill="none" stroke="#00d4ff" strokeWidth="2.5" strokeLinejoin="round"/>
            <path d="M16 24L21 29L32 18" stroke="#00d4ff" strokeWidth="2.5"
              strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>CyberShield</span>
        </div>

        <h1 className={styles.title}>Welcome back</h1>
        <p className={styles.subtitle}>Sign in to your account</p>

        {error && <div className={styles.errorBox}>{error}</div>}

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.field}>
            <label className={styles.label}>Email or Username</label>
            <input
              name="identifier"
              type="text"
              value={form.identifier}
              onChange={handleChange}
              placeholder="you@example.com"
              className={styles.input}
              autoComplete="username"
              autoFocus
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Password</label>
            <input
              name="password"
              type="password"
              value={form.password}
              onChange={handleChange}
              placeholder="••••••••"
              className={styles.input}
              autoComplete="current-password"
            />
          </div>

          <button type="submit" disabled={loading} className={styles.submitBtn}>
            {loading ? (
              <span className={styles.spinner} />
            ) : 'Sign In'}
          </button>
        </form>

        <p className={styles.switchText}>
          Don't have an account?{' '}
          <Link to="/signup" className={styles.switchLink}>Create one</Link>
        </p>

        <div className={styles.adminHint}>
          <span>🛡</span>
          <span>Admin? Use your admin credentials to access the panel.</span>
        </div>
      </div>
    </div>
  )
}
