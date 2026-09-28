Place the two files from the v2 training run folder here:

- `hrnet_w32_ankle_v2.py` — the training config (`config.py` in the run folder)
- `ankle_v2_metainfo.py`  — the keypoint metainfo (`metainfo.py` in the run folder)

Keypoint order is load-bearing: `0 bottom_heel`, `1 5th_metatarsal`, `2 malleolus`.

The backend rewrites the Colab-only paths at load time (`metainfo.from_file` -> `ankle_v2_metainfo.py`,
`load_from = None`; `DATA_ROOT` is never read at inference), so the files can be copied over unmodified.
