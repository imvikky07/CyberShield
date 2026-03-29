import { useState, useEffect } from 'react'
import { adminAPI } from '../api'
import styles from './AdminPanel.module.css'

function StatCard({ label, value, color }) {
  return (
    <div className={styles.statCard}>
      <span className={styles.statVal} style={{ color }}>{value ?? '—'}</span>
      <span className={styles.statLabel}>{label}</span>
    </div>
  )
}

function PostRow({ post, onApprove, onDelete, onBlock }) {
  const score = post.toxicity_score ?? 0
  const pct = Math.round(score * 100)
  const barColor = score > 0.85 ? 'var(--red)' : score > 0.6 ? 'var(--yellow)' : 'var(--green)'

  return (
    <div className={`${styles.postRow} fade-in`}>
      <div className={styles.postMeta}>
        <div className={styles.rowAvatar}>{post.author?.username?.[0]?.toUpperCase() || '?'}</div>
        <div>
          <span className={styles.rowUser}>{post.author?.username || 'unknown'}</span>
          <span className={styles.rowTime}>{new Date(post.created_at).toLocaleString()}</span>
        </div>
        <span className={`${styles.statusPill} ${styles[`pill_${post.status}`]}`}>
          {post.status}
        </span>
      </div>

      <p className={styles.postContent}>{post.content}</p>

      {/* Toxicity bar */}
      <div className={styles.toxRow}>
        <span className={styles.toxLabel}>Toxicity</span>
        <div className={styles.toxBar}>
          <div
            className={styles.toxFill}
            style={{ width: `${pct}%`, background: barColor }}
          />
        </div>
        <code className={styles.toxScore} style={{ color: barColor }}>{pct}%</code>
      </div>

      <div className={styles.rowActions}>
        {post.status !== 'visible' && (
          <button onClick={() => onApprove(post.id)} className={styles.btnApprove}>
            ✓ Approve
          </button>
        )}
        {post.status !== 'blocked' && (
          <button onClick={() => onBlock(post.id)} className={styles.btnBlock}>
            ⊘ Block
          </button>
        )}
        {post.status !== 'deleted' && (
          <button onClick={() => onDelete(post.id)} className={styles.btnDelete}>
            ✕ Delete
          </button>
        )}
      </div>
    </div>
  )
}

export default function AdminPanel() {
  const [stats, setStats] = useState(null)
  const [posts, setPosts] = useState([])
  const [filter, setFilter] = useState('withheld')
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('posts') // 'posts' | 'users'
  const [users, setUsers] = useState([])

  const fetchStats = async () => {
    try {
      const res = await adminAPI.getStats()
      setStats(res.data.stats)
    } catch { /* ignore */ }
  }

  const fetchPosts = async (status = filter) => {
    setLoading(true)
    try {
      const res = await adminAPI.getPosts(status)
      setPosts(res.data.posts)
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }

  const fetchUsers = async () => {
    setLoading(true)
    try {
      const res = await adminAPI.getUsers()
      setUsers(res.data.users)
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchStats() }, [])

  useEffect(() => {
    if (tab === 'posts') fetchPosts(filter)
    else fetchUsers()
  }, [tab, filter])

  const handleApprove = async (id) => {
    await adminAPI.approvePost(id)
    setPosts(prev => prev.filter(p => p.id !== id))
    fetchStats()
  }

  const handleDelete = async (id) => {
    await adminAPI.deletePost(id)
    setPosts(prev => prev.filter(p => p.id !== id))
    fetchStats()
  }

  const handleBlock = async (id) => {
    await adminAPI.blockPost(id)
    setPosts(prev => prev.map(p => p.id === id ? { ...p, status: 'blocked' } : p))
    fetchStats()
  }

  const handleToggleUser = async (id) => {
    await adminAPI.toggleUser(id)
    fetchUsers()
  }

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        {/* Header */}
        <div className={styles.pageHeader}>
          <div>
            <h1 className={styles.title}>
              <span className={styles.shield}>🛡</span> Admin Panel
            </h1>
            <p className={styles.subtitle}>AI Moderation Dashboard</p>
          </div>
        </div>

        {/* Stats */}
        {stats && (
          <div className={styles.statsGrid}>
            <StatCard label="Total Posts" value={stats.total_posts} color="var(--cyan)" />
            <StatCard label="Visible" value={stats.visible} color="var(--green)" />
            <StatCard label="Withheld" value={stats.withheld} color="var(--yellow)" />
            <StatCard label="Blocked" value={stats.blocked} color="var(--red)" />
            <StatCard label="Flagged" value={stats.flagged} color="#ff8c00" />
            <StatCard label="Users" value={stats.total_users} color="var(--purple)" />
          </div>
        )}

        {/* Main tabs */}
        <div className={styles.mainTabs}>
          <button
            onClick={() => setTab('posts')}
            className={`${styles.mainTab} ${tab === 'posts' ? styles.mainTabActive : ''}`}
          >
            Moderation Queue
          </button>
          <button
            onClick={() => setTab('users')}
            className={`${styles.mainTab} ${tab === 'users' ? styles.mainTabActive : ''}`}
          >
            Users
          </button>
        </div>

        {/* Posts tab */}
        {tab === 'posts' && (
          <>
            <div className={styles.filterRow}>
              {['withheld', 'blocked', 'all'].map(s => (
                <button
                  key={s}
                  onClick={() => setFilter(s)}
                  className={`${styles.filterBtn} ${filter === s ? styles.filterActive : ''}`}
                >
                  {s}
                </button>
              ))}
              <button
                onClick={() => fetchPosts(filter)}
                className={styles.refreshBtn}
              >↻ Refresh</button>
            </div>

            {loading ? (
              <div className={styles.loadingBox}>
                <div className={styles.spinner} />
              </div>
            ) : posts.length === 0 ? (
              <div className={styles.empty}>
                <span>✅</span>
                <p>No {filter} posts — queue is clean!</p>
              </div>
            ) : (
              <div className={styles.postList}>
                {posts.map(post => (
                  <PostRow
                    key={post.id}
                    post={post}
                    onApprove={handleApprove}
                    onDelete={handleDelete}
                    onBlock={handleBlock}
                  />
                ))}
              </div>
            )}
          </>
        )}

        {/* Users tab */}
        {tab === 'users' && (
          <div className={styles.usersTable}>
            {loading ? (
              <div className={styles.loadingBox}><div className={styles.spinner} /></div>
            ) : (
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Email</th>
                    <th>Role</th>
                    <th>Joined</th>
                    <th>Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map(u => (
                    <tr key={u.id} className={styles.tableRow}>
                      <td>
                        <div className={styles.userCell}>
                          <span className={styles.miniAvatar}>{u.username[0].toUpperCase()}</span>
                          {u.username}
                        </div>
                      </td>
                      <td className={styles.dimText}>{u.email}</td>
                      <td>
                        <span className={`${styles.rolePill} ${u.role === 'admin' ? styles.roleAdmin : ''}`}>
                          {u.role}
                        </span>
                      </td>
                      <td className={styles.dimText}>
                        {new Date(u.created_at).toLocaleDateString()}
                      </td>
                      <td>
                        <span className={`${styles.activePill}`}>active</span>
                      </td>
                      <td>
                        {u.role !== 'admin' && (
                          <button
                            onClick={() => handleToggleUser(u.id)}
                            className={styles.toggleBtn}
                          >
                            Toggle
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
