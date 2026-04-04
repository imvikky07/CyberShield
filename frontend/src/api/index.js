import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || '/api'

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 15000,
})

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('cs_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Handle 401 globally
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('cs_token')
      localStorage.removeItem('cs_user')
      window.dispatchEvent(new Event('auth:logout'))
    }
    return Promise.reject(err)
  }
)

export default api

// ── Auth ──────────────────────────────────────────────
export const authAPI = {
  signup: (data) => api.post('/auth/signup', data),
  login: (data) => api.post('/auth/login', data),
  me: () => api.get('/auth/me'),
  searchUsers: (q) => api.get('/auth/users', { params: { q } }),
}

// ── Posts ─────────────────────────────────────────────
export const postsAPI = {
  getFeed: (page = 1) => api.get('/posts/', { params: { page } }),
  getMyPosts: () => api.get('/posts/my'),
  createPost: (content) => api.post('/posts/', { content }),
  deletePost: (id) => api.delete(`/posts/${id}`),
  reportPost: (id) => api.post(`/posts/${id}/report`),
}

// ── Admin ─────────────────────────────────────────────
export const adminAPI = {
  getStats: () => api.get('/admin/stats'),
  getPosts: (status = 'withheld', page = 1) =>
    api.get('/admin/posts', { params: { status, page } }),
  approvePost: (id) => api.post(`/admin/posts/${id}/approve`),
  deletePost: (id) => api.post(`/admin/posts/${id}/delete`),
  blockPost: (id) => api.post(`/admin/posts/${id}/block`),
  getUsers: () => api.get('/admin/users'),
  toggleUser: (id) => api.post(`/admin/users/${id}/toggle`),
}

// ── Friends ───────────────────────────────────────────
export const friendsAPI = {
  sendRequest: (userId) => api.post(`/friends/send/${userId}`),
  getRequests: (direction = 'received') =>
    api.get('/friends/requests', { params: { direction } }),
  respondRequest: (id, action) =>
    api.post(`/friends/respond/${id}`, { action }),
  getFriends: () => api.get('/friends/list'),
  getStatus: (userId) => api.get(`/friends/status/${userId}`),
}

// ── AI ────────────────────────────────────────────────
export const aiAPI = {
  analyze: (text) => api.post('/ai/analyze', { text }),
}
