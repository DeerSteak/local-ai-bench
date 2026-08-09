import { readFileSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { parseSuiteVersion } from './src/utils/version.ts'

function suiteVersion() {
  const path = fileURLToPath(new URL('../scripts/runtime/config.py', import.meta.url))
  try {
    return parseSuiteVersion(readFileSync(path, 'utf8'))
  } catch {
    return null
  }
}

export default defineConfig({
  plugins: [react()],
  define: {
    'import.meta.env.VITE_SUITE_VERSION': JSON.stringify(suiteVersion()),
  },
  build: {
    chunkSizeWarningLimit: 2000,
  },
})
