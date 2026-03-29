import { createContext, useContext, useEffect, useReducer, useCallback } from 'react'
import { authAPI } from '../api'

const AuthContext = createContext(null)

const initialState = {
  status: 'loading', // 'loading' | 'unauthenticated' | 'authenticated'
  user: null,
  token: null,
}

function authReducer(state, action) {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, status: 'loading' }
    case 'LOGIN':
      return { status: 'authenticated', user: action.user, token: action.token }
    case 'LOGOUT':
      return { status: 'unauthenticated', user: null, token: null }
    default:
      return state
  }
}

export function AuthProvider({ children }) {
  const [state, dispatch] = useReducer(authReducer, initialState)

  // Bootstrap: verify token from localStorage on mount
  useEffect(() => {
    const bootstrap = async () => {
      const token = localStorage.getItem('cs_token')
      if (!token) {
        dispatch({ type: 'LOGOUT' })
        return
      }
      try {
        const res = await authAPI.me()
        dispatch({ type: 'LOGIN', user: res.data.user, token })
      } catch {
        localStorage.removeItem('cs_token')
        localStorage.removeItem('cs_user')
        dispatch({ type: 'LOGOUT' })
      }
    }
    bootstrap()
  }, [])

  // Listen for global logout events (401 from interceptor)
  useEffect(() => {
    const handler = () => dispatch({ type: 'LOGOUT' })
    window.addEventListener('auth:logout', handler)
    return () => window.removeEventListener('auth:logout', handler)
  }, [])

  const login = useCallback((token, user) => {
    localStorage.setItem('cs_token', token)
    localStorage.setItem('cs_user', JSON.stringify(user))
    dispatch({ type: 'LOGIN', user, token })
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('cs_token')
    localStorage.removeItem('cs_user')
    dispatch({ type: 'LOGOUT' })
  }, [])

  return (
    <AuthContext.Provider value={{ ...state, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
