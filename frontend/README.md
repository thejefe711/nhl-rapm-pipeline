# NHL Analytics Frontend

A modern Next.js frontend for NHL RAPM analytics.

## Setup

1. Install Node.js (v18+) from [nodejs.org](https://nodejs.org/)

2. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

3. Configure environment:
   ```bash
   # Copy example env file
   cp .env.example .env.local
   
   # Edit .env.local to point to your backend
   # NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

4. Run development server:
   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000)

## Pages

- **/** - Dashboard with overview and quick stats
- **/players** - Player search with autocomplete
- **/players/[id]** - Player detail with RAPM, AI explanation
- **/leaderboards** - Top players by metric and season
- **/compare** - Side-by-side player comparison

## Production Build

```bash
npm run build
npm run start
```

## Deployment

### Vercel (Recommended)
1. Push to GitHub
2. Connect repo to [vercel.com](https://vercel.com)
3. Set environment variable: `NEXT_PUBLIC_API_URL=https://your-backend.up.railway.app`
4. Deploy

### Other Platforms
The `npm run build` creates a standalone output in `.next/standalone/`.
