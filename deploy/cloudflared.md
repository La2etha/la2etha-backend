# Exposing the local GPU backend with Cloudflare Tunnel (free)

The backend runs on your RTX 2060 box; Cloudflare Tunnel gives it a public HTTPS
URL with **no paid plan, no port-forwarding, and no card on file**. Two options:

## Option A — Quick Tunnel (zero setup, ephemeral URL)

Best for a demo. No Cloudflare account needed. The URL changes each run.

```bash
# 1. Install cloudflared (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
#    Windows:  winget install --id Cloudflare.cloudflared
#    macOS:    brew install cloudflared
#    Linux:    download the binary from the link above

# 2. Start the backend (in its own terminal)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Start a quick tunnel to it (in another terminal)
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` prints a `https://<random>.trycloudflare.com` URL. Point the frontend
at it: set `VITE_API_BASE_URL` to that URL and rebuild/redeploy the frontend.

## Option B — Named Tunnel (stable URL, free)

Best if you want the same URL across restarts. Requires a free Cloudflare account
and a domain on Cloudflare (a free subdomain works).

```bash
cloudflared tunnel login                     # opens browser; free account
cloudflared tunnel create lahza              # creates a tunnel + credentials file
cloudflared tunnel route dns lahza api.yourdomain.com
cloudflared tunnel run lahza                 # uses ~/.cloudflared/config.yml below
```

`~/.cloudflared/config.yml`:

```yaml
tunnel: lahza
credentials-file: /home/<you>/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

## Notes

- **CORS**: add the frontend's deployed origin to `CORS_ORIGINS` in the backend
  `.env` (comma-separated) so the browser can call the tunnel.
- **The worker stays local** — only the API is exposed. Redis/Postgres never leave
  the box. Run the RQ worker alongside uvicorn as usual.
- Keep the tunnel process running for as long as you need the backend reachable.
