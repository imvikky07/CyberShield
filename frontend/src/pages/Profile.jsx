import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { postsAPI } from '../api'
import PostCard from '../components/PostCard'
import styles from './Profile.module.css'

const STATUS_COLORS = {
  visible: '#00ff88',
  withheld: '#ffd600',
  blocked: '#ff3d6b',
  deleted: '#3a5070',
}

export default function Profile() {
  const { user } = useAuth()
  const [posts, setPosts] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    const fetch = async () => {
      setLoading(true)
      try {
        const res = await postsAPI.getMyPosts()
        setPosts(res.data.posts)
      } catch { /* ignore */ }
      finally { setLoading(false) }
    }
    fetch()
  }, [])

  const handleDelete = async (id) => {
    if (!confirm('Delete this post?')) return
    try {
      await postsAPI.deletePost(id)
      setPosts(prev => prev.map(p => p.id === id ? { ...p, status: 'deleted' } : p))
    } catch { alert('Failed to delete.') }
  }

  const filtered = filter === 'all' ? posts : posts.filter(p => p.status === filter)

  const stats = {
    total: posts.length,
    visible: posts.filter(p => p.status === 'visible').length,
    withheld: posts.filter(p => p.status === 'withheld').length,
    blocked: posts.filter(p => p.status === 'blocked').length,
  }

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        {/* Profile header */}
        <div className={`${styles.header} fade-in`}>
          <div className={styles.avatarLg}>
            {user?.username?.[0]?.toUpperCase()}
          </div>
          <div className={styles.info}>
            <h1 className={styles.username}>{user?.username}</h1>
            <p className={styles.email}>{user?.email}</p>
            <span className={`${styles.roleBadge} ${user?.role === 'admin' ? styles.admin : ''}`}>
              {user?.role}
            </span>
          </div>
        </div>

        {/* Stats row */}
        <div className={styles.statsRow}>
          {Object.entries(stats).map(([key, val]) => (
            <div key={key} className={styles.statCard}>
              <span className={styles.statVal}
                style={{ color: STATUS_COLORS[key] || 'var(--cyan)' }}>
                {val}
              </span>
              <span className={styles.statLabel}>{key}</span>
            </div>
          ))}
        </div>

        {/* Filter tabs */}
        <div className={styles.tabs}>
          {['all', 'visible', 'withheld', 'blocked', 'deleted'].map(s => (
            <button
              key={s}
              onClick={() => setFilter(s)}
              className={`${styles.tab} ${filter === s ? styles.tabActive : ''}`}
            >
              {s}
            </button>
          ))}
        </div>

        {/* Posts */}
        {loading ? (
          <div className={styles.loading}>
            <div className={styles.spinner} />
          </div>
        ) : filtered.length === 0 ? (
          <div className={styles.empty}>
            <span>📭</span>
            <p>No {filter === 'all' ? '' : filter} posts yet.</p>
          </div>
        ) : (
          <div className={styles.postList}>
            {filtered.map(post => (
              <PostCard
                key={post.id}
                post={post}
                showMeta
                onDelete={post.status !== 'deleted' ? handleDelete : null}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
