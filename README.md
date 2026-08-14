# SceneDemo

Construction-site image analysis system: object detection, scene-graph construction, and safety reasoning. The repository contains application code and algorithm source only. Model weights, generated data, logs, and local databases are intentionally excluded.

## Prerequisites

- Python 3.10 or later (the original environment uses Python 3.13)
- Node.js 18 or later with npm
- MySQL 8 (create an empty database named `scene_behavior`)
- NVIDIA GPU deployment: a CUDA-compatible PyTorch build matching the host driver

## Restore a new machine

1. Clone the repository and copy your weight archive into the same project-relative paths recorded in `backend/instance/algorithm_runtime.example.json`.
2. Copy `backend/instance/algorithm_runtime.example.json` to `backend/instance/algorithm_runtime.json`. Set each `device` field to `cuda` for GPU inference or `cpu` for CPU inference. The `__PROJECT_ROOT__` and `__BACKEND_ROOT__` placeholders are expanded automatically.
3. Create the MySQL database:

   ```sql
   CREATE DATABASE scene_behavior CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```

4. Set database and secret environment variables. Use `backend/.env.example` as the value reference; do not commit an actual `.env` file.
5. Install dependencies:

   ```powershell
   cd backend
   python -m pip install -r requirements.txt
   cd ..\frontend
   npm ci
   ```

6. Start the backend and frontend in separate terminals:

   ```powershell
   cd backend
   python run.py
   ```

   ```powershell
   cd frontend
   npm run dev -- --host 0.0.0.0
   ```

The frontend is served on port 80 and proxies API requests to Flask on port 5000. On first launch, the backend initializes database tables and seed accounts automatically.

## Weight locations

The runtime configuration is authoritative. Its default locations include:

- `weight/stw/best.pt` and `weight/yolo/best.pt` for detection
- `backend/resources/scene_graph/model_0006000.pth`, `glove.6B.300d.pt`, and `VG-SGG-with-attri.h5` for scene graphs
- `backend/resources/hazard_reasoning/best_model.pt` and the local BERT model files under `backend/resources/hf_models/bert-base-uncased/` for safety reasoning

All of the above are ignored by Git. Copy them from your secure weight archive after cloning.
