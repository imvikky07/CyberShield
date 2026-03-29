import styles from './LoadingScreen.module.css'

export default function LoadingScreen() {
  return (
    <div className={styles.screen}>
      <div className={styles.logo}>
        <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
          <path d="M24 4L6 12V24C6 34.5 14 43.5 24 46C34 43.5 42 34.5 42 24V12L24 4Z"
            fill="none" stroke="#00d4ff" strokeWidth="2" strokeLinejoin="round"/>
          <path d="M16 24L21 29L32 18" stroke="#00d4ff" strokeWidth="2.5"
            strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <div className={styles.spinner} />
      <p className={styles.text}>Initializing CyberShield</p>
    </div>
  )
}
