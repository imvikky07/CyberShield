import { useState, useEffect, useRef } from 'react'
import { postsAPI, aiAPI } from '../api'
import { useAuth } from '../context/AuthContext'
import { useDebounce } from '../hooks/useDebounce'
import PostCard from '../components/PostCard'
import LoginPrompt from '../components/LoginPrompt'
import styles from './Feed.module.css'

export default function Feed() {
  const { status, user } = useAuth()
  const [posts, setPosts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)

  // Composer
  const [content, setContent] = useState('')
  const [posting, setPosting] = useState(false)
  const [postMsg, setPostMsg] = useState(null)
  const [showLoginPrompt, setShowLoginPrompt] = useState(false)

  // AI real-time warning
  const [aiWarning, setAiWarning] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const debouncedContent = useDebounce(content, 700)

  const fetchFeed = async (p = 1, append = false) => {
    try {
      setLoading(true)
      const res = await postsAPI.getFeed(p)
      const { posts: newPosts, pages } = res.data
      setPosts(prev => append ? [...prev, ...newPosts] : newPosts)
      setHasMore(p < pages)
      setPage(p)
    } catch {
      setError('Failed to load feed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchFeed(1) }, [])

  // Real-time AI analysis while typing
  useEffect(() => {
    if (!debouncedContent || debouncedContent.length < 10) {
      setAiWarning(null)
      return
    }
    const run = async () => {
      setAnalyzing(true)
      try {
        const res = await aiAPI.analyze(debouncedContent)
        const { toxicity_score, toxicity_label } = res.data
        if (toxicity_score > 0.5) {
          setAiWarning({ score: toxicity_score, label: toxicity_label })
        } else {
          setAiWarning(null)
        }
      } catch { /* silently ignore */ }
      finally { setAnalyzing(false) }
    }
    run()
  }, [debouncedContent])

  const handleComposerClick = () => {
    if (status !== 'authenticated') setShowLoginPrompt(true)
  }

  const handlePost = async () => {
    if (!content.trim()) return
    setPosting(true)
    setPostMsg(null)
    try {
      const res = await postsAPI.createPost(content)
      const { moderation } = res.data
      if (moderation.status === 'visible') {
        setPostMsg({ type: 'success', text: '✓ Post published successfully!' })
        setContent('')
        setAiWarning(null)
        await fetchFeed(1)
      } else if (moderation.status === 'withheld') {
        setPostMsg({ type: 'warn', text: `⚠ Your post is under review (toxicity: ${(moderation.toxicity_score * 100).toFixed(0)}%).` })
        setContent('')
        setAiWarning(null)
      } else {
        setPostMsg({ type: 'error', text: `✕ Post blocked — harmful content detected (${(moderation.toxicity_score * 100).toFixed(0)}%). Please revise.` })
      }
    } catch (err) {
      setPostMsg({ type: 'error', text: err.response?.data?.error || 'Failed to post. Try again.' })
    } finally {
      setPosting(false)
    }
  }

  // Report handler — any logged-in user can report other people's posts
  const handleReport = async (postId) => {
    await postsAPI.reportPost(postId)
  }

  // Delete handler — only for own posts shown in feed
  const handleDelete = async (postId) => {
    if (!confirm('Delete this post?')) return
    await postsAPI.deletePost(postId)
    setPosts(prev => prev.filter(p => p.id !== postId))
  }

  const getWarningClass = () => {
    if (!aiWarning) return ''
    if (aiWarning.score > 0.85) return styles.warnHigh
    return styles.warnMid
  }

  return (
    <div className={styles.page}>
      {showLoginPrompt && <LoginPrompt onClose={() => setShowLoginPrompt(false)} />}

      <div className={styles.container}>
        {/* Composer */}
        <div className={`${styles.composer} ${getWarningClass()}`}>
          <div className={styles.composerHeader}>
            {status === 'authenticated' ? (
              <div className={styles.composerAvatar}>
                {user?.username?.[0]?.toUpperCase()}
              </div>
            ) : (
              <div className={`${styles.composerAvatar} ${styles.composerAvatarGuest}`}>?</div>
            )}
            <span className={styles.composerTitle}>
              {status === 'authenticated' ? `Post as ${user.username}` : 'Share your thoughts'}
            </span>
            {analyzing && <span className={styles.analyzing}>analyzing…</span>}
          </div>

          <textarea
            className={styles.textarea}
            placeholder={status === 'authenticated' ? "What's happening? AI moderation is active…" : "Click to write a post…"}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            onClick={handleComposerClick}
            readOnly={status !== 'authenticated'}
            maxLength={2000}
            rows={4}
          />

          {aiWarning && (
            <div className={styles.aiWarning}>
              <span>⚠</span>
              <div>
                <strong>This message may be harmful</strong>
                <p>Toxicity score: {(aiWarning.score * 100).toFixed(0)}% — {aiWarning.label}. Your post may be withheld or blocked.</p>
              </div>
            </div>
          )}

          {postMsg && (
            <div className={`${styles.postMsg} ${styles[`postMsg_${postMsg.type}`]}`}>
              {postMsg.text}
            </div>
          )}

          <div className={styles.composerFooter}>
            <span className={styles.charCount}>{content.length}/2000</span>
            <div className={styles.composerActions}>
              <div className={styles.shieldTag}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L3 6V12C3 17.25 7 22.25 12 23C17 22.25 21 17.25 21 12V6L12 2Z"
                    fill="none" stroke="currentColor" strokeWidth="2"/>
                </svg>
                AI Protected
              </div>
              <button
                onClick={status === 'authenticated' ? handlePost : handleComposerClick}
                disabled={posting || (status === 'authenticated' && !content.trim())}
                className={styles.postBtn}
              >
                {posting ? 'Posting…' : 'Post'}
              </button>
            </div>
          </div>
        </div>

        {/* Feed header */}
        <div className={styles.feedHeader}>
          <h2 className={styles.feedTitle}>Public Feed</h2>
          <button onClick={() => fetchFeed(1)} className={styles.refreshBtn} disabled={loading}>
            {loading ? '…' : '↻ Refresh'}
          </button>
        </div>

        {error && <div className={styles.errorMsg}>{error}</div>}

        {loading && posts.length === 0 ? (
          <div className={styles.skeletonList}>
            {[1,2,3].map(i => <div key={i} className={styles.skeleton} />)}
          </div>
        ) : posts.length === 0 ? (
          <div className={styles.empty}>
            <span>📭</span>
            <p>No posts yet. Be the first to post!</p>
          </div>
        ) : (
          <div className={styles.postList}>
            {posts.map(post => (
              <PostCard
                key={post.id}
                post={post}
                currentUserId={user?.id}
                onReport={status === 'authenticated' ? handleReport : null}
                onDelete={status === 'authenticated' ? handleDelete : null}
              />
            ))}
          </div>
        )}

        {hasMore && !loading && (
          <button
            className={styles.loadMore}
            onClick={() => fetchFeed(page + 1, true)}
          >
            Load more
          </button>
        )}
      </div>
    </div>
  )
}

