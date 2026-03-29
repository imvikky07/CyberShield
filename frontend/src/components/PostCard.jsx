import styles from './PostCard.module.css'

const LABEL_CONFIG = {
  safe: { color: 'green', icon: '✓' },
  mild: { color: 'yellow', icon: '⚠' },
  harmful: { color: 'red', icon: '✕' },
}

function ToxicityBadge({ score, label }) {
  const cfg = LABEL_CONFIG[label] || LABEL_CONFIG.safe
  return (
    <span className={`${styles.badge} ${styles[`badge_${cfg.color}`]}`}>
      {cfg.icon} {Math.round(score * 100)}%
    </span>
  )
}

export default function PostCard({ post, showMeta = false, onDelete }) {
  const timeAgo = (iso) => {
    const diff = Date.now() - new Date(iso)
    const m = Math.floor(diff / 60000)
    if (m < 1) return 'just now'
    if (m < 60) return `${m}m ago`
    const h = Math.floor(m / 60)
    if (h < 24) return `${h}h ago`
    return `${Math.floor(h / 24)}d ago`
  }

  return (
    <article className={`${styles.card} fade-in`}>
      <div className={styles.header}>
        <div className={styles.avatar}>
          {post.author?.username?.[0]?.toUpperCase() || '?'}
        </div>
        <div className={styles.meta}>
          <span className={styles.username}>{post.author?.username || 'Unknown'}</span>
          <span className={styles.time}>{timeAgo(post.created_at)}</span>
        </div>
        <div className={styles.badges}>
          {showMeta && (
            <ToxicityBadge score={post.toxicity_score} label={post.toxicity_label} />
          )}
          {post.is_flagged && (
            <span className={`${styles.badge} ${styles.badge_flag}`}>🚩 Flagged</span>
          )}
        </div>
      </div>

      <p className={styles.content}>{post.content}</p>

      {showMeta && (
        <div className={styles.metaRow}>
          <span className={styles.metaItem}>
            <span className={styles.metaLabel}>Status</span>
            <span className={`${styles.statusDot} ${styles[`status_${post.status}`]}`} />
            {post.status}
          </span>
          <span className={styles.metaItem}>
            <span className={styles.metaLabel}>Score</span>
            <code className={styles.score}>{post.toxicity_score?.toFixed(4)}</code>
          </span>
          {onDelete && (
            <button onClick={() => onDelete(post.id)} className={styles.deleteBtn}>
              Delete
            </button>
          )}
        </div>
      )}
    </article>
  )
}
