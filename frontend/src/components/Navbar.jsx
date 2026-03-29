import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import styles from './Navbar.module.css'

export default function Navbar() {
  const { status, user, logout } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  const isActive = (path) => location.pathname === path

  return (
    <nav className={styles.nav}>
      <div className={styles.inner}>
        {/* Brand */}
        <Link to="/" className={styles.brand}>
          <svg width="28" height="28" viewBox="0 0 48 48" fill="none">
            <path d="M24 4L6 12V24C6 34.5 14 43.5 24 46C34 43.5 42 34.5 42 24V12L24 4Z"
              fill="none" stroke="#00d4ff" strokeWidth="2.5" strokeLinejoin="round"/>
            <path d="M16 24L21 29L32 18" stroke="#00d4ff" strokeWidth="2.5"
              strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>CyberShield</span>
        </Link>

        {/* Nav links */}
        <div className={styles.links}>
          <Link to="/" className={`${styles.link} ${isActive('/') ? styles.active : ''}`}>
            Feed
          </Link>
          {status === 'authenticated' && (
            <>
              <Link to="/friends" className={`${styles.link} ${isActive('/friends') ? styles.active : ''}`}>
                Friends
              </Link>
              {user?.role === 'admin' && (
                <Link to="/admin" className={`${styles.link} ${styles.adminLink} ${isActive('/admin') ? styles.active : ''}`}>
                  Admin
                </Link>
              )}
            </>
          )}
        </div>

        {/* Auth area */}
        <div className={styles.auth}>
          {status === 'unauthenticated' && (
            <>
              <Link to="/login" className={styles.btnGhost}>Login</Link>
              <Link to="/signup" className={styles.btnPrimary}>Sign Up</Link>
            </>
          )}
          {status === 'authenticated' && (
            <>
              <Link to="/profile" className={styles.userChip}>
                <span className={styles.avatar}>
                  {user?.username?.[0]?.toUpperCase() || '?'}
                </span>
                <span className={styles.username}>{user?.username}</span>
                {user?.role === 'admin' && <span className={styles.roleBadge}>admin</span>}
              </Link>
              <button onClick={handleLogout} className={styles.btnLogout}>
                Logout
              </button>
            </>
          )}
        </div>
      </div>
    </nav>
  )
}
