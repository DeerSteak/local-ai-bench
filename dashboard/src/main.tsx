import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import Dashboard from './benchmark_dashboard'

const rootElement = document.getElementById('root')
if (!rootElement) throw new Error('#root element not found')

createRoot(rootElement).render(
  <StrictMode>
    <Dashboard />
  </StrictMode>,
)
