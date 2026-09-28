# rom-calc

The deployed ROM measurement product: upload two ankle photos (rest + dorsiflexion),
get the range-of-motion angle back. Stateless — nothing is stored server-side.

ROM = |angle(dorsiflexion) − angle(rest)|; angle = included angle between
heel\*→5th_metatarsal and the vertical through the malleolus, where
heel\* = (malleolus.x, bottom_heel.y).

This is one of two repos in the ROM project — see the [project-level README](../README.md)
for how this fits together with `rom-annotation` and the shared `assets/`/`research/` folders.

```
rom-calc/
├── dashboard/              Next.js (App Router) + Tailwind → deploys to Vercel
│   ├── app/page.tsx        upload UI
│   ├── components/         Dropzone, ResultCard
│   ├── lib/api.ts          calls the infrastructure API
│   └── .env.example
└── infrastructure/         FastAPI + MMPose → deploys to Railway (Dockerfile)
    ├── main.py             app entrypoint, startup model load, /api/calculate_rom
    ├── pose.py             gdown checkpoint download + MMPose inference wrapper
    ├── imaging.py          EXIF-transpose / RGB / ≤2048px downscale (matches training)
    ├── geometry.py         heel*/angle/ROM math, overlay drawing, base64 encoding
    ├── memory.py           RSS logging (diagnosed a first-boot OOM on Railway)
    ├── slim_checkpoint.py  strips the training-only weights from the checkpoint on first boot
    ├── configs/            hrnet_w32_ankle_v2.py + ankle_v2_metainfo.py (MMPose model config)
    ├── tests/               unit tests
    ├── tools/verify_slim.py manual check that slim_checkpoint didn't change inference output
    ├── requirements.txt / constraints.txt
    ├── Dockerfile / .dockerignore
    └── .env.example
```

Keypoints (in model order): `bottom_heel`, `5th_metatarsal`, `malleolus`.
Both photos must be taken from the same fixed camera position (the angle uses the image vertical).

## Run it locally

**Backend**
```bash
cd infrastructure
cp .env.example .env
docker build -t rom-infra .
docker run -p 8000:8000 --env-file .env rom-infra
```
Or without Docker: `pip install -r requirements.txt -c constraints.txt && python main.py`

**Frontend**
```bash
cd dashboard
cp .env.example .env.local
npm install
npm run dev
```

## Tests
```bash
cd infrastructure
python -m unittest discover -s tests -v
```

## Deployment

- **Backend → Railway**: builds `infrastructure/Dockerfile`. Set the variables listed in
  `infrastructure/.env.example` under the Railway project's Variables tab.
- **Frontend → Vercel**: builds `dashboard/`. Set `NEXT_PUBLIC_API_URL` to the Railway
  backend's public URL.
- **Access handover**: see [`../HANDOVER.md`](../HANDOVER.md) for how deploy access was
  transferred (collaborator invites, not shared passwords).

## Where the training data lives

This repo does not contain the photos or annotation CSVs used to train the model —
see `assets/` at the project root (synced via Drive, not git). `infrastructure/configs/`
only holds the MMPose model architecture/metadata, not the dataset itself.
