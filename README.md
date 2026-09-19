# Ankle ROM PoC

ROM = |angle(dorsiflexion) − angle(rest)|, angle measured at `bottom_heel` between `5th_metatarsal` and `shin`.
Stateless: nothing is stored server-side.

```
ROM_calc/
├── backend/            FastAPI + MMPose  → Railway (Dockerfile)
│   ├── main.py         app, startup model load, /api/calculate_rom
│   ├── pose.py         gdown download + MMPose inference wrapper
│   ├── geometry.py     angle math, skeleton drawing, base64 encoding
│   ├── configs/        put hrnet_w32_ankle.py (MMPose config) here
│   ├── requirements.txt
│   └── Dockerfile
└── frontend/           Next.js (App Router) + Tailwind → Vercel
    ├── app/page.tsx
    ├── components/     Dropzone, ResultCard
    └── lib/api.ts
```

## Backend env vars
| Var | Purpose |
|---|---|
| `MODEL_URL` | Google Drive link to the `.pth` (currently a placeholder) |
| `MODEL_CONFIG` | Path to the MMPose config (default `configs/hrnet_w32_ankle.py`) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (your Vercel URL) |
| `DEVICE` | `cpu` (default) or `cuda:0` |

Run locally: `cd backend && docker build -t rom . && docker run -p 8000:8000 rom`

## Frontend
```
cd frontend && cp .env.example .env.local && npm install && npm run dev
```
On Vercel set `NEXT_PUBLIC_API_URL` to the Railway URL.
