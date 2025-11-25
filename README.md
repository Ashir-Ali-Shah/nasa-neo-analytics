# NASA NEO Analytics Dashboard 🚀

A full-stack application that tracks Near-Earth Objects (asteroids) using NASA data, Machine Learning for hazard prediction, and an AI chatbot.

## 🛠️ Tech Stack
- **Frontend:** React, Recharts
- **Backend:** FastAPI, Python, XGBoost
- **AI:** RAG System

## ⚙️ How to Run

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
# Create a .env file here with your NASA_API_KEY and OPENROUTER_API_KEY
uvicorn main:app --reload
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```
