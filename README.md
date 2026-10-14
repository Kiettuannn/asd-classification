# ASD Classification

This repository contains two independently runnable applications:

- `frontend/`: React and Vite user interface.
- `backend/`: FastAPI service and the `asd_classification` Python package.

Shared research assets remain at the repository root: `data/`, `artifacts/`, and
`notebooks/`. Backend defaults use the root `data/` and `artifacts/` directories.
Set `ASD_DATA_DIR` and `ASD_ARTIFACTS_DIR` in `backend/.env` to use different
storage locations; begin with `backend/.env.example`. Frontend API configuration
remains in `frontend/.env.example`.

## Frontend

```sh
cd frontend
npm install
npm run dev
npm run lint
npm test
npm run build
```

## Backend

```sh
cd backend
uv sync --extra api --extra dev
uv run uvicorn asd_classification.api.app:create_app --factory --reload
uv run pytest
```

The backend CLI is available through `uv run asd --help`.
