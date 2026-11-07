# TRACE support bot (Cloudflare Worker)

The AI support chat behind [gt7trace.netlify.app](https://gt7trace.netlify.app).
`src/worker.js` holds the chat logic and the knowledge base (the `DOCS` string).

## Updating the docs
Edit `DOCS` in `src/worker.js`, commit and push to `main`. GitHub Actions checks the
syntax and runs `wrangler deploy --keep-vars`, so the bot is live in about a minute.
You can also run the workflow by hand from the Actions tab.

## One-time setup
Repo secrets (Settings > Secrets and variables > Actions):
- `CLOUDFLARE_API_TOKEN` : API token with **Edit Cloudflare Workers** permission
- `CLOUDFLARE_ACCOUNT_ID` : your Cloudflare account ID

`GEMINI_KEY` and `GROQ_KEY` are Worker secrets set in Cloudflare. Never commit them.

All rights reserved. See `LICENSE`.
