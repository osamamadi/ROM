# Ankle ROM PoC

ROM = |angle(dorsiflexion) − angle(rest)|; angle = included angle between heel*→5th_metatarsal
and the vertical through the malleolus, where heel* = (malleolus.x, bottom_heel.y).
Stateless: nothing is stored server-side.

```
ROM_calc/
├── backend/            FastAPI + MMPose  → Railway (Dockerfile)
│   ├── main.py         app, startup model load, /api/calculate_rom
│   ├── pose.py         gdown download + MMPose inference wrapper (config path fix-ups)
│   ├── imaging.py      EXIF-transpose / RGB / ≤2048px downscale (matches training)
│   ├── geometry.py     heel*/angle/ROM math, overlay drawing, base64 encoding
│   ├── tests/          unit tests (python -m unittest discover -s tests)
│   ├── configs/        hrnet_w32_ankle_v2.py + ankle_v2_metainfo.py
│   ├── requirements.txt
│   └── Dockerfile
└── frontend/           Next.js (App Router) + Tailwind → Vercel
    ├── app/page.tsx
    ├── components/     Dropzone, ResultCard
    └── lib/api.ts
```

Keypoints (in model order): `bottom_heel`, `5th_metatarsal`, `malleolus`.
Both photos must be taken from the same fixed camera position (the angle uses the image vertical).

## Backend env vars
| Var | Purpose |
|---|---|
| `MODEL_URL` | Google Drive link to the v2 checkpoint `epoch_160.pth` (default is set; downloaded with gdown on startup) |
| `MODEL_CONFIG` | Path to the MMPose config (default `configs/hrnet_w32_ankle_v2.py`) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (your Vercel URL) |
| `DEVICE` | `cpu` (default) or `cuda:0` |

Run locally: `cd backend && docker build -t rom . && docker run -p 8000:8000 rom`

## Frontend
```
cd frontend && cp .env.example .env.local && npm install && npm run dev
```
On Vercel set `NEXT_PUBLIC_API_URL` to the Railway URL.
