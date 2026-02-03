# 🌌 NASA NEO Analytics & Hazard Prediction 🚀

A full-stack planetary defense analytics platform that tracks Near-Earth Objects (asteroids) using NASA's NeoWs API. It features an interactive dashboard for real-time tracking, an AI chatbot for data analysis, and a high-performance Machine Learning model to predict hazardous asteroids.

## 📊 Model Performance Metrics

The core of this project is an **XGBoost Classifier** trained on **127,347** historic asteroid records to predict if a new object is potentially hazardous.

### Key Results
* **Overall Accuracy:** **94.18%** 🎯
* **Total Records Analyzed:** 127,347
* **Test Set Size:** 25,470 samples (20% split)

### Detailed Classification Report
The dataset is **imbalanced** (far more non-hazardous asteroids than hazardous ones), so `Accuracy` alone is not enough. We prioritize **Recall** for the `True` class (hazardous) to minimize missed threats.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **False** (Safe) | 0.95 | 0.99 | 0.97 | 23,693 |
| **True** (Hazardous) | **0.74** | **0.25** | **0.38** | **1,777** |
| | | | | |
| **Macro Avg** | 0.84 | 0.62 | 0.67 | 25,470 |
| **Weighted Avg** | 0.93 | 0.94 | 0.93 | 25,470 |

* **Precision (0.74):** When the model says an asteroid is hazardous, it is correct 74% of the time.
* **Recall (0.25):** The model correctly identified 25% of all actual hazardous asteroids in the test set. *Future work aims to improve this using techniques like SMOTE or class weighting.*

### Confusion Matrix
| | Predicted Safe | Predicted Hazardous |
| :--- | :---: | :---: |
| **Actual Safe** | **23,535** (TN) | 158 (FP) |
| **Actual Hazardous** | 1,324 (FN) | **453** (TP) |

### 🧠 Feature Importance
Which physical properties make an asteroid dangerous? The model found **Absolute Magnitude** (brightness/size) to be the single most critical factor.

| Rank | Feature | Importance Score |
| :--- | :--- | :--- |
| 1 | **Absolute Magnitude** | **0.7001** |
| 2 | Estimated Diameter (Min) | 0.1216 |
| 3 | Miss Distance | 0.0871 |
| 4 | Estimated Diameter (Max) | 0.0469 |
| 5 | Relative Velocity | 0.0444 |

---

## 🤖 Robot Scientist - LangGraph Agentic RAG

> **Premium Skill: Agentic Engineering** — This project showcases LangGraph, one of the most in-demand AI frameworks in 2026 alongside LlamaIndex and CrewAI.

This project features the **"Robot Scientist,"** an autonomous planetary defense agent built with **LangGraph** (from the LangChain ecosystem) using the **ReAct (Reasoning + Acting)** protocol.

### Architecture (LangGraph StateGraph)
```
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │  agent  │◄──────┐
    └────┬────┘       │
         │            │
   ┌─────┴─────┐      │
   │           │      │
   ▼           ▼      │
┌─────┐    ┌───────┐  │
│ END │    │ tools │──┘
└─────┘    └───────┘
```

### How It Works (ReAct Protocol)
1. **Thought**: What information is missing? Which tool is best?
2. **Action**: Call the appropriate tool (NASA feed, ML model, or knowledge base)
3. **Observation**: Analyze the tool output
4. **Repeat** if needed, then deliver a **Mission Control Briefing**

### Available Tools
| Tool | Description | Priority |
| :--- | :--- | :--- |
| `fetch_live_nasa_feed` | Real-time asteroid data from NASA NeoWs API | Use first for current/future queries |
| `predict_hazard_xgboost` | ML model trained on 127,347 asteroids (94.18% accuracy) | Use for hazard classification |
| `search_knowledge_base` | Historical NEO data from Weaviate | Use for context and analogies |

### Key Principle
> **NO MANUAL CALCULATIONS**: The Robot Scientist is forbidden from performing complex physics math. It must use the `predict_hazard_xgboost` ML tool for all hazard assessments.

### Example Queries
- *"What is the most dangerous asteroid approaching this week?"*
- *"Is asteroid 2024 XY hazardous? Use the ML model to predict."*
- *"Compare today's closest approach to Chelyabinsk"*

### API Endpoints
```bash
# Main agent endpoint (LangGraph ReAct loop)
POST /api/agent/query
{
  "question": "Is there anything dangerous coming this weekend?",
  "include_reasoning": true
}

# Agent status (shows LangGraph structure)
GET /api/agent/status

# Quick NEO analysis by name or parameters
POST /api/agent/analyze-neo?neo_name=2024%20XY
```

---

## 🛠️ Tech Stack
* **Frontend:** React, Recharts (Data Visualization), Tailwind CSS
* **Backend:** FastAPI, Python, Weaviate (Vector DB), Redis (Caching)
* **Machine Learning:** XGBoost (94.18% accuracy on 127,347 samples)
* **MLOps:** MLflow (Experiment Tracking & Model Registry)
* **Agentic AI:** LangGraph >= 0.2.0, LangChain >= 0.3.0 (ReAct Protocol)
* **LLM Provider:** OpenRouter (Qwen, Claude, GPT-4 compatible)
* **DevOps:** Docker, Docker Compose
* **Monitoring:** Prometheus, Grafana

## ⚙️ How to Run

### Prerequisites
* Docker & Docker Compose installed
* API Keys for NASA and OpenRouter

### 1. Clone & Configure
```bash
git clone [https://github.com/your-username/nasa-neo-analytics.git](https://github.com/your-username/nasa-neo-analytics.git)
cd nasa-neo-analytics

# Create a .env file in the backend folder
echo "NASA_API_KEY=your_key_here" > backend/.env
echo "OPENROUTER_API_KEY=your_key_here" >> backend/.env
````

### 2\. Launch with Docker

Run the entire stack (Frontend + Backend + Database) with one command:

```bash
docker compose up --build
```

  * **Dashboard:** [http://localhost:5173](https://www.google.com/search?q=http://localhost:5173)
  * **API Docs:** [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)
  * **Prometheus:** [http://localhost:9090](http://localhost:9090)
  * **Grafana:** [http://localhost:3000](http://localhost:3000) (Login: admin/admin)
  * **MLflow:** [http://localhost:5001](http://localhost:5001) (Experiment Tracking UI)

## 📂 Project Structure

```text
nasa-neo-analytics/
├── backend/             # FastAPI & ML Logic
│   ├── main.py          # API Endpoints
│   ├── xgb_neo_classifier.pkl # Trained Model
│   └── Dockerfile
├── frontend/            # React Dashboard
│   ├── src/             # UI Components
│   └── Dockerfile
└── docker-compose.yml   # Orchestration
├── monitoring/          # Prometheus & Grafana Configs
```

-----

## 📈 Dataset Info

The model was trained on data fetched dynamically from NASA.

  * **Hazardous Distribution:** \~90% Safe / \~10% Hazardous
  * **Missing Values:** Imputed using mean strategy.
  * **Scaling:** Standard Scaler applied to all numerical features.

How to run the above?

docker compose up --build -d

