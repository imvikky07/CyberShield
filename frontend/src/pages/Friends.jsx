import { useState, useEffect } from 'react'
import { friendsAPI, authAPI } from '../api'
import { useAuth } from '../context/AuthContext'
import styles from './Friends.module.css'

function UserSearchCard({ u, onSend }) {
  const [status, setStatus] = useState('none') // none | pending | sent | friends
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    friendsAPI.getStatus(u.id)
      .then(r => {
        const s = r.data.status
        if (s === 'accepted') setStatus('friends')
        else if (s === 'pending') setStatus(r.data.is_sender ? 'sent' : 'pending')
      })
      .catch(() => {})
  }, [u.id])

  const handleSend = async () => {
    setLoading(true)
    try {
      await friendsAPI.sendRequest(u.id)
      setStatus('sent')
      onSend?.()
    } catch (err) {
      alert(err.response?.data?.error || 'Failed to send request')
    } finally {
      setLoading(false)
    }
  }

  const actionLabel = {
    none: 'Add Friend',
    sent: 'Request Sent',
    pending: 'Respond',
    friends: '✓ Friends',
  }[status]

  return (
    <div className={styles.userCard}>
      <div className={styles.uAvatar}>{u.username[0].toUpperCase()}</div>
      <div className={styles.uInfo}>
        <span className={styles.uName}>{u.username}</span>
        <span className={styles.uEmail}>{u.email}</span>
      </div>
      <button
        onClick={status === 'none' ? handleSend : undefined}
        disabled={loading || status !== 'none'}
        className={`${styles.addBtn} ${status !== 'none' ? styles.addBtnDone : ''}`}
      >
        {loading ? '…' : actionLabel}
      </button>
    </div>
  )
}

function RequestCard({ req, onRespond }) {
  const [loading, setLoading] = useState(null)

  const handle = async (action) => {
    setLoading(action)
    try {
      await friendsAPI.respondRequest(req.id, action)
      onRespond()
    } catch { /* ignore */ }
    finally { setLoading(null) }
  }

  return (
    <div className={styles.userCard}>
      <div className={styles.uAvatar}>{req.sender?.username?.[0]?.toUpperCase()}</div>
      <div className={styles.uInfo}>
        <span className={styles.uName}>{req.sender?.username}</span>
        <span className={styles.uEmail}>Wants to be your friend</span>
      </div>
      <div className={styles.reqActions}>
        <button
          onClick={() => handle('accept')}
          disabled={!!loading}
          className={styles.acceptBtn}
        >
          {loading === 'accept' ? '…' : '✓ Accept'}
        </button>
        <button
          onClick={() => handle('reject')}
          disabled={!!loading}
          className={styles.rejectBtn}
        >
          {loading === 'reject' ? '…' : '✕'}
        </button>
      </div>
    </div>
  )
}

export default function Friends() {
  const { user } = useAuth()
  const [tab, setTab] = useState('friends') // friends | requests | search
  const [friends, setFriends] = useState([])
  const [requests, setRequests] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searching, setSearching] = useState(false)
  const [loading, setLoading] = useState(false)

  const fetchFriends = async () => {
    setLoading(true)
    try {
      const res = await friendsAPI.getFriends()
      setFriends(res.data.friends)
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }

  const fetchRequests = async () => {
    setLoading(true)
    try {
      const res = await friendsAPI.getRequests('received')
      setRequests(res.data.requests)
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }

  useEffect(() => {
    if (tab === 'friends') fetchFriends()
    else if (tab === 'requests') fetchRequests()
  }, [tab])

  const handleSearch = async (e) => {
    const q = e.target.value
    setSearchQuery(q)
    if (q.length < 2) { setSearchResults([]); return }
    setSearching(true)
    try {
      const res = await authAPI.searchUsers(q)
      setSearchResults(res.data.users)
    } catch { /* ignore */ }
    finally { setSearching(false) }
  }

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <div className={styles.pageHeader}>
          <h1 className={styles.title}>Friends</h1>
          <span className={styles.count}>{friends.length} connected</span>
        </div>

        {/* Tabs */}
        <div className={styles.tabs}>
          {['friends', 'requests', 'search'].map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`${styles.tab} ${tab === t ? styles.tabActive : ''}`}
            >
              {t === 'requests' && requests.length > 0 && tab !== 'requests'
                ? `Requests (${requests.length})`
                : t.charAt(0).toUpperCase() + t.slice(1)
              }
            </button>
          ))}
        </div>

        {/* Friends list */}
        {tab === 'friends' && (
          loading ? (
            <div className={styles.loadingBox}><div className={styles.spinner} /></div>
          ) : friends.length === 0 ? (
            <div className={styles.empty}>
              <span>👥</span>
              <p>No friends yet. Search for people to add!</p>
              <button onClick={() => setTab('search')} className={styles.actionBtn}>
                Find Friends
              </button>
            </div>
          ) : (
            <div className={styles.list}>
              {friends.map(f => (
                <div key={f.id} className={styles.userCard}>
                  <div className={styles.uAvatar}>{f.username[0].toUpperCase()}</div>
                  <div className={styles.uInfo}>
                    <span className={styles.uName}>{f.username}</span>
                    <span className={styles.uEmail}>{f.email}</span>
                  </div>
                  <span className={styles.friendBadge}>✓ Friend</span>
                </div>
              ))}
            </div>
          )
        )}

        {/* Requests */}
        {tab === 'requests' && (
          loading ? (
            <div className={styles.loadingBox}><div className={styles.spinner} /></div>
          ) : requests.length === 0 ? (
            <div className={styles.empty}>
              <span>📬</span>
              <p>No pending friend requests.</p>
            </div>
          ) : (
            <div className={styles.list}>
              {requests.map(req => (
                <RequestCard
                  key={req.id}
                  req={req}
                  onRespond={() => {
                    fetchRequests()
                    fetchFriends()
                  }}
                />
              ))}
            </div>
          )
        )}

        {/* Search */}
        {tab === 'search' && (
          <div className={styles.searchSection}>
            <div className={styles.searchBox}>
              <span className={styles.searchIcon}>🔍</span>
              <input
                type="text"
                value={searchQuery}
                onChange={handleSearch}
                placeholder="Search by username…"
                className={styles.searchInput}
                autoFocus
              />
              {searching && <span className={styles.searchingDot}>…</span>}
            </div>

            {searchQuery.length < 2 ? (
              <p className={styles.searchHint}>Type at least 2 characters to search</p>
            ) : searchResults.length === 0 && !searching ? (
              <div className={styles.empty}>
                <span>🔎</span>
                <p>No users found for "{searchQuery}"</p>
              </div>
            ) : (
              <div className={styles.list}>
                {searchResults.map(u => (
                  <UserSearchCard
                    key={u.id}
                    u={u}
                    onSend={fetchRequests}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
