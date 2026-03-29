import { useNavigate } from 'react-router-dom'
import styles from './LoginPrompt.module.css'

export default function LoginPrompt({ onClose }) {
  const navigate = useNavigate()

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.icon}>🔒</div>
        <h2 className={styles.title}>Login Required</h2>
        <p className={styles.text}>
          Please login or create an account to post on CyberShield.
        </p>
        <div className={styles.actions}>
          <button onClick={() => navigate('/login')} className={styles.btnPrimary}>
            Login
          </button>
          <button onClick={() => navigate('/signup')} className={styles.btnSecondary}>
            Create Account
          </button>
        </div>
        <button className={styles.close} onClick={onClose} aria-label="Close">×</button>
      </div>
    </div>
  )
}
