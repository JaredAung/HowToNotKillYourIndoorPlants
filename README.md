# How To Not Kill Your Indoor Plants

Indoor plant care tracking app with a FastAPI backend and Next.js frontend.

## Setup

### Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API runs at **http://localhost:8000**

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **http://localhost:3000**

## Project Structure

```
├── backend/          # FastAPI API
│   ├── main.py
│   └── requirements.txt
├── frontend/         # Next.js app
└── README.md
```
