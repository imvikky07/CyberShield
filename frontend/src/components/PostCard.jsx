import { useState } from 'react'
import styles from './PostCard.module.css'

const LABEL_CONFIG = {
  safe:    { color: 'green',  icon: '✓' },
  mild:    { color: 'yellow', icon: '⚠' },
  harmful: { color: 'red',    icon: '✕' },
}

function ToxicityBadge({ score, label }) {
  const cfg = LABEL_CONFIG[label] || LABEL_CONFIG.safe
  return (
    <span className={`${styles.badge} ${styles[`badge_${cfg.color}`]}`}>
      {cfg.icon} {Math.round(score * 100)}%
    </span>
  )
}

export default function PostCard({
  post,
  showMeta = false,
  onDelete,       // passed for own posts (Profile page)
  onReport,       // passed for feed posts (Feed page)
  currentUserId,  // logged-in user id to decide which action to show
}) {
  const [reported, setReported] = useState(false)
  const [reporting, setReporting] = useState(false)
  const [reportMsg, setReportMsg] = useState('')

  const timeAgo = (iso) => {
    const diff = Date.now() - new Date(iso)
    const m = Math.floor(diff / 60000)
    if (m < 1) return 'just now'
    if (m < 60) return `${m}m ago`
    const h = Math.floor(m / 60)
    if (h < 24) return `${h}h ago`
    return `${Math.floor(h / 24)}d ago`
  }

  const isOwn = currentUserId && post.user_id === currentUserId

  const handleReport = async () => {
    if (reported || !onReport) return
    setReporting(true)
    try {
      await onReport(post.id)
      setReported(true)
      setReportMsg('Reported — moderators will review this post.')
    } catch (err) {
      setReportMsg(err.response?.data?.error || 'Could not report. Try again.')
    } finally {
      setReporting(false)
    }
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

      {/* Feed action row: report for others, delete for own */}
      {(onReport || (onDelete && !showMeta)) && (
        <div className={styles.actionRow}>
          {onReport && !isOwn && (
            reportMsg ? (
              <span className={`${styles.reportMsg} ${reported ? styles.reportMsgOk : styles.reportMsgErr}`}>
                {reportMsg}
              </span>
            ) : (
              <button
                onClick={handleReport}
                disabled={reporting}
                className={styles.reportBtn}
              >
                {reporting ? 'Reporting…' : '🚩 Report'}
              </button>
            )
          )}
          {onDelete && isOwn && !showMeta && (
            <button onClick={() => onDelete(post.id)} className={styles.deleteBtn}>
              🗑 Delete
            </button>
          )}
        </div>
      )}

      {/* Meta row for Profile page */}
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
              🗑 Delete
            </button>
          )}
        </div>
      )}
    </article>
  )
}
