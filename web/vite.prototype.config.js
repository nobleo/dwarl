import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { mergeConfig } from 'vite'
import base from './vite.config.js'

const script = fileURLToPath(new URL('./src/prototype/train_preview.py', import.meta.url))
export default mergeConfig(base, { define: {'import.meta.env.TRAIN_DEMO': 'true'}, plugins: [{
  name: 'prototype-train-preview',
  configureServer(server) {
    server.middlewares.use('/prototype-train', async (req, res) => {
      if (req.method !== 'POST' || req.headers['content-type'] !== 'application/json') {
        res.writeHead(405).end()
        return
      }
      let body = ''
      for await (const chunk of req) body += chunk
      const child = spawn('python', [script])
      let output = ''
      child.stdin.end(body)
      child.stdout.on('data', chunk => { output += chunk })
      child.stderr.pipe(process.stderr)
      child.on('close', code => {
        res.writeHead(code ? 500 : 200, { 'Content-Type': 'application/json' })
        res.end(code ? JSON.stringify({error:'Python preview failed.'}) : output)
      })
    })
  },
}] })
