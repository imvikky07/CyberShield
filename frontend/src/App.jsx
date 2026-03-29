import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import Navbar from './components/Navbar'
import Feed from './pages/Feed'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Profile from './pages/Profile'
import AdminPanel from './pages/AdminPanel'
import Friends from './pages/Friends'
import LoadingScreen from './components/LoadingScreen'

function ProtectedRoute({ children, adminOnly = false }) {
  const { status, user } = useAuth()

  if (status === 'loading') return <LoadingScreen />
  if (status === 'unauthenticated') return <Navigate to="/login" replace />
  if (adminOnly && user?.role !== 'admin') return <Navigate to="/" replace />

  return children
}

function GuestRoute({ children }) {
  const { status } = useAuth()
  if (status === 'loading') return <LoadingScreen />
  if (status === 'authenticated') return <Navigate to="/" replace />
  return children
}

function AppRoutes() {
  const { status } = useAuth()
  if (status === 'loading') return <LoadingScreen />

  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<Feed />} />
        <Route path="/login" element={<GuestRoute><Login /></GuestRoute>} />
        <Route path="/signup" element={<GuestRoute><Signup /></GuestRoute>} />
        <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
        <Route path="/friends" element={<ProtectedRoute><Friends /></ProtectedRoute>} />
        <Route path="/admin" element={<ProtectedRoute adminOnly><AdminPanel /></ProtectedRoute>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  )
}
