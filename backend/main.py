from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import requests
import uvicorn
import numpy as np
import math
import pickle
import joblib
import os
import traceback
import json
import asyncio
from openai import OpenAI
from collections import defaultdict
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import caching layer and NASA service
from core.cache import (
    cache_response, 
    get_cache_stats, 
    clear_cache, 
    get_cached_keys_info,
    get_redis_client
)
from services.nasa_service import NasaService
from services.agent_service import LangGraphAgentService, create_langgraph_agent
from core.calculations import (
    calculate_kinetic_energy as calc_ke,
    calculate_impact_probability as calc_ip,
    calculate_risk_score as calc_rs,
    get_risk_category,
    full_risk_assessment
)

# Try to import MLflow service
try:
    from services.mlflow_service import MLflowService
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: MLflow service not available.")

# Try to import Weaviate, but don't fail if not available
try:
    import weaviate
    from weaviate.classes.init import Auth
    from weaviate.classes.query import MetadataQuery
    from weaviate.classes.config import Property, DataType
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    print("WARNING: Weaviate not installed. RAG features will be disabled.")

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence-transformers not installed. Using fallback embeddings.")

app = FastAPI(
    title="NASA NEO Advanced Analytics API with RAG", 
    description="Backend API with risk scoring, ML predictions, and RAG",
    version="4.0.1"
)

# Instrument with Prometheus
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
NASA_API_KEY = os.getenv("NASA_API_KEY")
NASA_API_URL = 'https://api.nasa.gov/neo/rest/v1/feed' # Keep the URL hardcoded unless it changes
EARTH_RADIUS_KM = 6371
LUNAR_DISTANCE_KM = 384400

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen-2.5-7b-instruct")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT")

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Global variables
xgb_model = None
scaler = None
model_load_error = None

weaviate_client = None
embedding_model = None
weaviate_error = None
neo_documents = []  # Fallback in-memory storage

def init_weaviate():
    """Initialize Weaviate client"""
    global weaviate_client, embedding_model, weaviate_error
    
    if not WEAVIATE_AVAILABLE:
        weaviate_error = "Weaviate library not installed. Install with: pip install weaviate-client"
        print(f"✗ {weaviate_error}")
        return
    
    try:
        print("\n" + "="*70)
        print("INITIALIZING WEAVIATE")
        print("="*70)

        # Prefer explicit Weaviate URL if provided
        if WEAVIATE_URL:
            parsed = urlparse(WEAVIATE_URL)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            grpc_port = int(WEAVIATE_GRPC_PORT) if WEAVIATE_GRPC_PORT else 50051
            weaviate_client = weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
            weaviate_client.connect()
            print(f"✓ Connected to Weaviate at {host}:{port}")
        else:
            # Try embedded mode first
            try:
                weaviate_client = weaviate.WeaviateClient(
                    embedded_options=weaviate.embedded.EmbeddedOptions(
                        persistence_data_path="./weaviate_data",
                        binary_path="./weaviate_binary"
                    )
                )
                weaviate_client.connect()
                print("✓ Weaviate embedded mode connected")
            except Exception as embed_error:
                print(f"Embedded mode failed: {str(embed_error)}")
                # Try connecting to local instance
                try:
                    weaviate_client = weaviate.connect_to_local()
                    print("✓ Connected to local Weaviate instance")
                except Exception as local_error:
                    print(f"Local connection failed: {str(local_error)}")
                    raise Exception("Could not connect to Weaviate. Please install Weaviate or use fallback mode.")
        
        # Create schema if needed
        try:
            collections = weaviate_client.collections.list_all()
            if "NEODocument" not in [c.name for c in collections]:
                create_neo_schema()
                print("✓ Created NEODocument schema")
            else:
                print("✓ NEODocument schema exists")
        except Exception as e:
            print(f"Schema check: {str(e)}")
            create_neo_schema()
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print("\nInitializing embedding model...")
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Embedding model loaded (all-MiniLM-L6-v2)")
            except Exception as e:
                print(f"✗ Embedding model failed: {str(e)}")
                embedding_model = None
        else:
            print("✗ sentence-transformers not available, using fallback")
            embedding_model = None
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"✗ Weaviate initialization failed: {str(e)}")
        print(traceback.format_exc())
        weaviate_error = str(e)
        weaviate_client = None
        embedding_model = None
        print("\n⚠ Using fallback in-memory storage for RAG")

def create_neo_schema():
    """Create Weaviate schema for NEO documents"""
    global weaviate_client
    
    try:
        # Delete if exists
        try:
            weaviate_client.collections.delete("NEODocument")
        except:
            pass
        
        # Create collection
        weaviate_client.collections.create(
            name="NEODocument",
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="neo_id", data_type=DataType.INT),
                Property(name="name", data_type=DataType.TEXT),
                Property(name="date", data_type=DataType.TEXT),
                Property(name="risk_score", data_type=DataType.NUMBER),
                Property(name="risk_category", data_type=DataType.TEXT),
                Property(name="diameter_km", data_type=DataType.NUMBER),
                Property(name="velocity_kms", data_type=DataType.NUMBER),
                Property(name="miss_distance_km", data_type=DataType.NUMBER),
                Property(name="kinetic_energy_mt", data_type=DataType.NUMBER),
                Property(name="is_hazardous", data_type=DataType.BOOL)
            ],
            vectorizer_config=None
        )
        print("✓ Schema created")
    except Exception as e:
        print(f"✗ Schema creation failed: {str(e)}")
        raise

def get_semantic_embedding(text: str) -> List[float]:
    """Generate semantic embeddings"""
    global embedding_model
    
    try:
        if embedding_model is not None:
            embedding = embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        else:
            return create_simple_embedding(text)
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return create_simple_embedding(text)

def create_simple_embedding(text: str, dim: int = 384) -> List[float]:
    """Fallback simple embedding"""
    text = text.lower()
    embedding = [0.0] * dim
    
    words = text.split()
    for i, word in enumerate(words[:dim]):
        idx = hash(word) % dim
        embedding[idx] += 1.0 / (i + 1)
    
    norm = math.sqrt(sum(x*x for x in embedding))
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding

def load_models():
    """Load ML models"""
    global xgb_model, scaler, model_load_error
    
    print("\n" + "="*70)
    print("LOADING ML MODELS")
    print("="*70)
    
    try:
        import xgboost as xgb
        print(f"✓ XGBoost installed: {xgb.__version__}")
    except ImportError:
        print("✗ XGBoost not installed")
        model_load_error = "XGBoost not installed"
        return
    
    # Try to load model
    model_paths = ['xgb_neo_classifier.pkl', './xgb_neo_classifier.pkl', 
                   '../xgb_neo_classifier.pkl', './models/xgb_neo_classifier.pkl']
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Found model: {path}")
            try:
                with open(path, 'rb') as f:
                    xgb_model = pickle.load(f)
                print(f"✓ Model loaded")
                break
            except Exception as e:
                print(f"✗ Load failed: {e}")
    
    # Try to load scaler
    scaler_paths = ['scaler.joblib', './scaler.joblib', 
                    '../scaler.joblib', './models/scaler.joblib']
    
    for path in scaler_paths:
        if os.path.exists(path):
            print(f"Found scaler: {path}")
            try:
                scaler = joblib.load(path)
                print(f"✓ Scaler loaded")
                break
            except Exception as e:
                print(f"✗ Load failed: {e}")
    
    print("="*70 + "\n")

# Initialize on startup
load_models()
init_weaviate()

# Pydantic models
class NEOData(BaseModel):
    neo_id: int
    name: str
    date: str
    absolute_magnitude: Optional[float]
    estimated_diameter_min: float
    estimated_diameter_max: float
    orbiting_body: str
    relative_velocity: float
    miss_distance: float
    is_hazardous: bool

class RiskScoredNEO(BaseModel):
    neo_id: int
    name: str
    date: str
    risk_score: float
    impact_probability: float
    kinetic_energy_mt: float
    lunar_distances: float
    diameter_km: float
    velocity_kms: float
    miss_distance_km: float
    is_hazardous: bool
    risk_category: str
    follow_up_priority: str
    absolute_magnitude: Optional[float] = None

class ImpactAnalysis(BaseModel):
    neo_id: int
    name: str
    impact_probability: float
    impact_corridor_width_km: float
    geographic_footprint: List[Dict[str, float]]
    potential_impact_locations: List[str]

class AdvancedAnalyticsResponse(BaseModel):
    top_50_risks: List[RiskScoredNEO]
    top_10_impact_analysis: List[ImpactAnalysis]
    temporal_clusters: Optional[List[Dict]] = []
    overall_insights: List[str]

class PredictionInput(BaseModel):
    absolute_magnitude: float
    estimated_diameter_min: float
    estimated_diameter_max: float
    relative_velocity: float
    miss_distance: float
    
    @validator('absolute_magnitude')
    def validate_magnitude(cls, v):
        if v < 0 or v > 35:
            raise ValueError('Absolute magnitude must be between 0 and 35')
        return v

class PredictionResponse(BaseModel):
    is_hazardous: bool
    probability: float
    confidence: float
    risk_level: str
    input_features: Dict[str, float]
    scaled_features: List[float]
    interpretation: str

class NEOIndexRequest(BaseModel):
    neos: List[Dict]

class RAGQueryRequest(BaseModel):
    question: str
    search_type: Optional[str] = "hybrid"

class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    search_type: str
    semantic_similarity_scores: Optional[List[float]] = None
    kb_document_count: int

# Helper functions
def fetch_nasa_data(start_date: str, end_date: str) -> dict:
    """Fetch data from NASA API (synchronous fallback)"""
    params = {
        'start_date': start_date,
        'end_date': end_date,
        'api_key': NASA_API_KEY
    }
    
    try:
        response = requests.get(NASA_API_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"NASA API error: {str(e)}")


async def fetch_nasa_data_cached(start_date: str, end_date: str) -> dict:
    """
    Fetch data from NASA API with Redis caching.
    
    This is the primary method that should be used for all NASA API calls.
    It uses the NasaService which has the @cache_response decorator applied.
    """
    try:
        return await NasaService.fetch_asteroid_feed(start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NASA API error: {str(e)}")

def parse_neo_data(api_response: dict) -> List[NEOData]:
    """Parse NASA API response"""
    records = []
    near_earth_objects = api_response.get('near_earth_objects', {})
    
    for date, neos in near_earth_objects.items():
        for neo in neos:
            for approach in neo.get('close_approach_data', []):
                record = NEOData(
                    neo_id=int(neo['id']),
                    name=neo['name'],
                    date=date,
                    absolute_magnitude=float(neo.get('absolute_magnitude_h')) if neo.get('absolute_magnitude_h') else None,
                    estimated_diameter_min=float(neo['estimated_diameter']['kilometers']['estimated_diameter_min']),
                    estimated_diameter_max=float(neo['estimated_diameter']['kilometers']['estimated_diameter_max']),
                    orbiting_body=approach['orbiting_body'],
                    relative_velocity=float(approach['relative_velocity']['kilometers_per_hour']),
                    miss_distance=float(approach['miss_distance']['kilometers']),
                    is_hazardous=neo['is_potentially_hazardous_asteroid']
                )
                records.append(record)
    
    return records

def calculate_kinetic_energy(diameter_km: float, velocity_kms: float) -> float:
    """Calculate kinetic energy in megatons"""
    density_kg_m3 = 2600
    radius_m = (diameter_km * 1000) / 2
    volume_m3 = (4/3) * math.pi * (radius_m ** 3)
    mass_kg = volume_m3 * density_kg_m3
    velocity_ms = velocity_kms * 1000
    energy_joules = 0.5 * mass_kg * (velocity_ms ** 2)
    energy_mt = energy_joules / 4.184e15
    return energy_mt

def calculate_impact_probability(miss_distance_km: float, diameter_km: float) -> float:
    """Calculate impact probability"""
    normalized_distance = miss_distance_km / LUNAR_DISTANCE_KM
    if normalized_distance < 0.1:
        prob = 1.0 / (1.0 + normalized_distance * 100)
    else:
        prob = math.exp(-normalized_distance * 10) * 0.01
    prob *= (diameter_km / 1.0)
    return min(prob, 1.0)

def calculate_risk_score(miss_distance_km: float, kinetic_energy_mt: float, 
                        impact_probability: float, w1: float = 0.4, 
                        w2: float = 0.35, w3: float = 0.25) -> float:
    """Calculate composite risk score"""
    lunar_distances = miss_distance_km / LUNAR_DISTANCE_KM
    component1 = w1 * (1.0 / max(lunar_distances, 0.001))
    component2 = w2 * math.log10(max(kinetic_energy_mt, 0.001))
    component3 = w3 * impact_probability
    risk_score = component1 + component2 + component3
    return max(risk_score, 0)

def impact_corridor_analysis(neo: RiskScoredNEO) -> ImpactAnalysis:
    """Analyze impact corridor"""
    uncertainty_km = neo.miss_distance_km * 0.05
    corridor_width = 2 * uncertainty_km
    
    num_locations = 10
    geographic_footprint = []
    potential_locations = []
    
    for i in range(num_locations):
        lat = np.random.uniform(-60, 60)
        lon = np.random.uniform(-180, 180)
        impact_energy = neo.kinetic_energy_mt * np.random.uniform(0.8, 1.0)
        crater_diameter = 0.02 * (impact_energy ** 0.33) * (neo.diameter_km ** 0.33)
        
        geographic_footprint.append({
            'latitude': float(lat),
            'longitude': float(lon),
            'impact_energy_mt': float(impact_energy),
            'crater_diameter_km': float(crater_diameter),
            'destruction_radius_km': float(crater_diameter * 10)
        })
        
        if -30 <= lat <= 30:
            if -100 <= lon <= -60:
                potential_locations.append("North America")
            elif -20 <= lon <= 50:
                potential_locations.append("Europe/Africa")
            elif 60 <= lon <= 150:
                potential_locations.append("Asia/Pacific")
    
    potential_locations = list(set(potential_locations)) if potential_locations else ["Ocean"]
    
    return ImpactAnalysis(
        neo_id=neo.neo_id,
        name=neo.name,
        impact_probability=neo.impact_probability,
        impact_corridor_width_km=corridor_width,
        geographic_footprint=geographic_footprint,
        potential_impact_locations=potential_locations
    )

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with status"""
    kb_count = 0
    if weaviate_client:
        try:
            collection = weaviate_client.collections.get("NEODocument")
            response = collection.aggregate.over_all(total_count=True)
            kb_count = response.total_count
        except:
            kb_count = len(neo_documents)
    else:
        kb_count = len(neo_documents)
    
    # Check Redis cache status
    redis_connected = False
    try:
        client = await get_redis_client()
        redis_connected = client is not None
    except:
        pass
    
    # Check MLflow status
    mlflow_connected = False
    if MLFLOW_AVAILABLE:
        try:
            mlflow_connected = MLflowService.is_available()
        except:
            pass
    
    return {
        "status": "online",
        "service": "NASA NEO Advanced Analytics API",
        "version": "5.1.0",  # Version bump for MLflow integration
        "ml_models_loaded": {
            "xgboost": xgb_model is not None,
            "scaler": scaler is not None
        },
        "weaviate_connected": weaviate_client is not None,
        "redis_cache_connected": redis_connected,
        "mlflow_connected": mlflow_connected,
        "semantic_search_enabled": embedding_model is not None,
        "knowledge_base_documents": kb_count,
        "ready_for_predictions": xgb_model is not None and scaler is not None,
        "ready_for_rag": True,  # Always ready (fallback available)
        "ready_for_agent": True,  # Agentic RAG system
        "caching_enabled": redis_connected,
        "agent_system": {
            "status": "operational",
            "type": "Agentic RAG with OpenAI Tool Calling",
            "tools": ["search_knowledge_base", "fetch_live_nasa_feed", "calculate_risk"]
        },
        "endpoints": {
            "analytics": [
                "/api/neo/advanced-analytics",
                "/api/neo/predict",
                "/api/neo/model-status",
                "/api/neo/model-metrics",
                "/api/neo/feature-importance",
                "/api/neo/evaluate-model"
            ],
            "agent": [
                "/api/agent/query",
                "/api/agent/status",
                "/api/agent/analyze-neo"
            ],
            "rag": [
                "/api/rag/kb-status",
                "/api/rag/index-neos",
                "/api/rag/query",
                "/api/rag/auto-index",
                "/api/rag/clear-kb",
                "/api/rag/reinit"
            ],
            "cache": [
                "/api/cache/stats",
                "/api/cache/health",
                "/api/cache/keys",
                "/api/cache/clear",
                "/api/cache/warm"
            ],
            "mlflow": [
                "/api/mlflow/status",
                "/api/mlflow/log-evaluation",
                "/api/mlflow/full-evaluation",
                "/api/mlflow/best-run",
                "/api/mlflow/recent-runs",
                "/api/mlflow/register-model",
                "/api/mlflow/promote-model",
                "/api/mlflow/compare-runs",
                "/api/mlflow/registered-models"
            ],
            "monitoring": [
                "/api/nasa/status"
            ]
        }
    }

@app.get("/api/neo/advanced-analytics", response_model=AdvancedAnalyticsResponse)
async def get_advanced_analytics(days: int = 30):
    """Get advanced NEO analytics with Redis caching"""
    all_records = []
    end_date = datetime.now()
    current_start = end_date - timedelta(days=days)
    
    num_chunks = (days // 7) + (1 if days % 7 != 0 else 0)
    
    # Collect all chunk tasks for parallel execution
    chunk_tasks = []
    chunk_params = []
    
    temp_start = current_start
    for chunk in range(num_chunks):
        chunk_end = min(temp_start + timedelta(days=6), end_date)
        start_str = temp_start.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        chunk_params.append((start_str, end_str))
        temp_start = chunk_end + timedelta(days=1)
        if temp_start > end_date:
            break
    
    # Fetch all chunks (cached via Redis)
    for start_str, end_str in chunk_params:
        try:
            # Use cached async fetch - this will hit Redis cache if available
            raw_data = await fetch_nasa_data_cached(start_str, end_str)
            chunk_data = parse_neo_data(raw_data)
            all_records.extend(chunk_data)
        except Exception as e:
            print(f"Chunk error ({start_str} to {end_str}): {str(e)}")
            continue
    
    if not all_records:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Remove duplicates
    seen = set()
    unique_records = []
    for record in all_records:
        key = (record.neo_id, record.miss_distance)
        if key not in seen:
            seen.add(key)
            unique_records.append(record)
    
    # Calculate risk scores
    scored_neos = []
    for neo in unique_records:
        diameter = (neo.estimated_diameter_min + neo.estimated_diameter_max) / 2
        velocity_kms = neo.relative_velocity / 3600
        
        ke = calculate_kinetic_energy(diameter, velocity_kms)
        ip = calculate_impact_probability(neo.miss_distance, diameter)
        rs = calculate_risk_score(neo.miss_distance, ke, ip)
        
        if rs > 10:
            risk_cat = "CRITICAL"
            priority = "IMMEDIATE"
        elif rs > 5:
            risk_cat = "HIGH"
            priority = "URGENT"
        elif rs > 2:
            risk_cat = "MODERATE"
            priority = "SCHEDULED"
        else:
            risk_cat = "LOW"
            priority = "ROUTINE"
        
        scored_neos.append(RiskScoredNEO(
            neo_id=neo.neo_id,
            name=neo.name,
            date=neo.date,
            risk_score=rs,
            impact_probability=ip,
            kinetic_energy_mt=ke,
            lunar_distances=neo.miss_distance / LUNAR_DISTANCE_KM,
            diameter_km=diameter,
            velocity_kms=velocity_kms,
            miss_distance_km=neo.miss_distance,
            is_hazardous=neo.is_hazardous,
            risk_category=risk_cat,
            follow_up_priority=priority,
            absolute_magnitude=neo.absolute_magnitude
        ))
    
    scored_neos.sort(key=lambda x: x.risk_score, reverse=True)
    top_50 = scored_neos[:50]
    
    closest_10 = sorted(scored_neos, key=lambda x: x.miss_distance_km)[:10]
    impact_analyses = [impact_corridor_analysis(neo) for neo in closest_10]
    
    # Calculate temporal clusters dynamically
    date_to_high_risk = defaultdict(list)
    for neo in scored_neos:
        if neo.risk_category in ["HIGH", "CRITICAL"]:
            date_to_high_risk[neo.date].append(neo)
    
    # Convert to sorted datetime objects
    parsed_dates = {d: datetime.strptime(d, '%Y-%m-%d') for d in date_to_high_risk}
    sorted_dates = sorted(parsed_dates.values())
    
    temporal_clusters = []
    if sorted_dates:
        current_start = sorted_dates[0]
        current_end = sorted_dates[0]
        current_neos = date_to_high_risk[current_start.strftime('%Y-%m-%d')]
        
        for i in range(1, len(sorted_dates)):
            current_date = sorted_dates[i]
            if current_date == current_end + timedelta(days=1):
                # Consecutive, extend cluster
                current_end = current_date
                current_neos.extend(date_to_high_risk[current_date.strftime('%Y-%m-%d')])
            else:
                # End current cluster if it qualifies (neo_count > 1)
                if len(current_neos) > 1:
                    cluster = {
                        "start_date": current_start.strftime('%Y-%m-%d'),
                        "end_date": current_end.strftime('%Y-%m-%d'),
                        "duration_days": (current_end - current_start).days + 1,
                        "neo_count": len(current_neos),
                        "total_risk_score": sum(n.risk_score for n in current_neos),
                        "neos": [n.name for n in current_neos]
                    }
                    temporal_clusters.append(cluster)
                
                # Start new cluster
                current_start = current_date
                current_end = current_date
                current_neos = date_to_high_risk[current_date.strftime('%Y-%m-%d')]
        
        # Add the last cluster if it qualifies
        if len(current_neos) > 1:
            cluster = {
                "start_date": current_start.strftime('%Y-%m-%d'),
                "end_date": current_end.strftime('%Y-%m-%d'),
                "duration_days": (current_end - current_start).days + 1,
                "neo_count": len(current_neos),
                "total_risk_score": sum(n.risk_score for n in current_neos),
                "neos": [n.name for n in current_neos]
            }
            temporal_clusters.append(cluster)
    
    # Generate insights
    insights = []
    immediate = [n for n in top_50 if n.follow_up_priority == "IMMEDIATE"]
    if immediate:
        insights.append(f"{len(immediate)} asteroids require IMMEDIATE follow-up")
    
    high_risk = [n for n in top_50 if n.risk_category in ["CRITICAL", "HIGH"]]
    if high_risk:
        insights.append(f"{len(high_risk)} high/critical risk asteroids detected")
    
    hazardous = sum(1 for n in scored_neos if n.is_hazardous)
    insights.append(f"{hazardous} potentially hazardous asteroids")
    
    if temporal_clusters:
        insights.append(f"{len(temporal_clusters)} high-risk periods detected")
    
    return AdvancedAnalyticsResponse(
        top_50_risks=top_50,
        top_10_impact_analysis=impact_analyses,
        temporal_clusters=temporal_clusters,
        overall_insights=insights
    )

@app.post("/api/neo/predict", response_model=PredictionResponse)
async def predict_hazardous(input_data: PredictionInput):
    """Predict if NEO is hazardous"""
    if xgb_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    try:
        features = np.array([[
            input_data.absolute_magnitude,
            input_data.estimated_diameter_min,
            input_data.estimated_diameter_max,
            input_data.relative_velocity,
            input_data.miss_distance
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = xgb_model.predict(features_scaled)[0]
        probabilities = xgb_model.predict_proba(features_scaled)[0]
        
        hazardous_probability = float(probabilities[1])
        confidence = float(max(probabilities))
        
        if hazardous_probability >= 0.8:
            risk_level = "CRITICAL"
            interpretation = f"Extremely high probability ({hazardous_probability*100:.2f}%) of being hazardous."
        elif hazardous_probability >= 0.6:
            risk_level = "HIGH"
            interpretation = f"High probability ({hazardous_probability*100:.2f}%) of being hazardous."
        elif hazardous_probability >= 0.4:
            risk_level = "MODERATE"
            interpretation = f"Moderate probability ({hazardous_probability*100:.2f}%) of being hazardous."
        else:
            risk_level = "LOW"
            interpretation = f"Low probability ({hazardous_probability*100:.2f}%) of being hazardous."
        
        # Log prediction to MLflow for monitoring
        if MLFLOW_AVAILABLE:
            try:
                MLflowService.log_prediction(
                    input_features={
                        'absolute_magnitude': input_data.absolute_magnitude,
                        'estimated_diameter_min': input_data.estimated_diameter_min,
                        'estimated_diameter_max': input_data.estimated_diameter_max,
                        'relative_velocity': input_data.relative_velocity,
                        'miss_distance': input_data.miss_distance
                    },
                    prediction=bool(prediction),
                    probability=hazardous_probability,
                    confidence=confidence,
                    risk_level=risk_level
                )
            except Exception as mlflow_error:
                print(f"MLflow logging error (non-critical): {mlflow_error}")
        
        return PredictionResponse(
            is_hazardous=bool(prediction),
            probability=hazardous_probability,
            confidence=confidence,
            risk_level=risk_level,
            input_features={
                'absolute_magnitude': input_data.absolute_magnitude,
                'estimated_diameter_min': input_data.estimated_diameter_min,
                'estimated_diameter_max': input_data.estimated_diameter_max,
                'relative_velocity': input_data.relative_velocity,
                'miss_distance': input_data.miss_distance
            },
            scaled_features=features_scaled[0].tolist(),
            interpretation=interpretation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/neo/model-status")
async def get_model_status():
    """Get ML model status"""
    return {
        "xgboost_model_loaded": xgb_model is not None,
        "scaler_loaded": scaler is not None,
        "ready_for_predictions": xgb_model is not None and scaler is not None,
        "last_load_error": model_load_error
    }

async def evaluate_model_on_live_data(days: int = 30) -> dict:
    """
    Evaluate the ML model against real NASA API data.
    
    Fetches NEO data, extracts features and ground truth labels,
    runs predictions, and computes classification metrics.
    
    Returns:
        Dictionary containing accuracy, precision, recall, F1, confusion matrix
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    
    if xgb_model is None or scaler is None:
        raise ValueError("ML models not loaded")
    
    # Collect NEO data from NASA API
    all_records = []
    end_date = datetime.now()
    current_start = end_date - timedelta(days=days)
    
    num_chunks = (days // 7) + (1 if days % 7 != 0 else 0)
    
    for chunk in range(num_chunks):
        chunk_end = min(current_start + timedelta(days=6), end_date)
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        
        try:
            raw_data = await fetch_nasa_data_cached(start_str, end_str)
            chunk_data = parse_neo_data(raw_data)
            all_records.extend(chunk_data)
        except Exception as e:
            print(f"Evaluation chunk error: {str(e)}")
            continue
        
        current_start = chunk_end + timedelta(days=1)
        if current_start > end_date:
            break
    
    if len(all_records) < 10:
        raise ValueError(f"Insufficient data for evaluation: only {len(all_records)} records")
    
    # Remove duplicates
    seen = set()
    unique_records = []
    for record in all_records:
        key = (record.neo_id, record.miss_distance)
        if key not in seen:
            seen.add(key)
            unique_records.append(record)
    
    # Extract features and labels
    features_list = []
    labels = []
    
    for neo in unique_records:
        # Skip records with missing data
        if neo.absolute_magnitude is None:
            continue
            
        features_list.append([
            neo.absolute_magnitude,
            neo.estimated_diameter_min,
            neo.estimated_diameter_max,
            neo.relative_velocity,
            neo.miss_distance
        ])
        labels.append(1 if neo.is_hazardous else 0)
    
    if len(features_list) < 10:
        raise ValueError(f"Insufficient valid records: only {len(features_list)} with complete data")
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y_true = np.array(labels)
    
    # Scale features and predict
    X_scaled = scaler.transform(X)
    y_pred = xgb_model.predict(X_scaled)
    y_proba = xgb_model.predict_proba(X_scaled)[:, 1]
    
    # Compute metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    
    # Handle edge case where there might be only one class in the data
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        precision = 1.0 if y_true[0] == y_pred[0] else 0.0
        recall = 1.0 if y_true[0] == y_pred[0] else 0.0
        f1 = 1.0 if y_true[0] == y_pred[0] else 0.0
        roc_auc = None
    else:
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
        except:
            roc_auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = 0, 0, 0, 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        if y_true[0] == 0:
            tn = int(cm[0, 0])
        else:
            tp = int(cm[0, 0])
    
    # Count class distribution
    hazardous_count = int(np.sum(y_true == 1))
    non_hazardous_count = int(np.sum(y_true == 0))
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc else None,
        "evaluation_samples": len(y_true),
        "hazardous_count": hazardous_count,
        "non_hazardous_count": non_hazardous_count,
        "confusion_matrix": {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "evaluation_date": datetime.now().isoformat(),
        "data_range_days": days,
        "data_source": "NASA NeoWs API (Live)",
        "metrics_type": "dynamic_evaluation"
    }


@app.get("/api/neo/model-metrics")
async def get_model_metrics(force_evaluation: bool = False):
    """
    Get ML model performance metrics - computed dynamically from live NASA data.
    
    Metrics are cached in Redis for 1 hour to avoid repeated expensive evaluations.
    Use force_evaluation=true to trigger a fresh evaluation.
    """
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        # Try to get cached metrics from Redis first
        redis_client = await get_redis_client()
        cached_metrics = None
        
        if redis_client and not force_evaluation:
            try:
                cached_data = await redis_client.get("model_metrics:evaluation")
                if cached_data:
                    cached_metrics = json.loads(cached_data)
                    print("✓ Using cached model metrics from Redis")
            except Exception as e:
                print(f"Redis cache read error: {e}")
        
        # Get model parameters
        model_params = {}
        n_estimators = 100
        
        try:
            n_estimators = xgb_model.n_estimators if hasattr(xgb_model, 'n_estimators') else 100
            if hasattr(xgb_model, 'get_params'):
                model_params = xgb_model.get_params()
            if hasattr(xgb_model, 'get_booster'):
                booster = xgb_model.get_booster()
                n_estimators = booster.num_boosted_rounds() if hasattr(booster, 'num_boosted_rounds') else n_estimators
        except Exception as e:
            print(f"Error getting model params: {e}")
        
        # Get feature importances
        importances = xgb_model.feature_importances_
        max_importance = float(max(importances))
        importance_variance = float(np.var(importances))
        objective = model_params.get('objective', 'binary:logistic')
        
        # Base metrics structure
        metrics = {
            "model_type": "XGBoost Classifier",
            "n_estimators": n_estimators,
            "max_depth": model_params.get('max_depth', 6),
            "learning_rate": model_params.get('learning_rate', 0.1),
            "objective": objective,
            "feature_count": len(importances),
            "max_feature_importance": max_importance,
            "importance_variance": importance_variance,
            "model_loaded": True,
            "scaler_loaded": scaler is not None
        }
        
        # Use cached metrics if available
        if cached_metrics:
            metrics["performance"] = cached_metrics
            metrics["metrics_source"] = "redis_cache"
            metrics["cache_hit"] = True
            return metrics
        
        # Compute metrics dynamically from live NASA data
        try:
            print("🔄 Computing model metrics from live NASA data...")
            evaluation_results = await evaluate_model_on_live_data(days=30)
            
            # Cache the results in Redis for 1 hour
            if redis_client:
                try:
                    await redis_client.setex(
                        "model_metrics:evaluation",
                        3600,  # 1 hour TTL
                        json.dumps(evaluation_results)
                    )
                    print("💾 Model metrics cached in Redis (1h TTL)")
                except Exception as e:
                    print(f"Redis cache write error: {e}")
            
            metrics["performance"] = evaluation_results
            metrics["metrics_source"] = "live_evaluation"
            metrics["cache_hit"] = False
            
        except Exception as eval_error:
            print(f"Live evaluation failed: {eval_error}")
            # Fallback: return model properties without performance metrics
            metrics["performance"] = {
                "note": f"Live evaluation temporarily unavailable: {str(eval_error)}",
                "model_ready": True,
                "features_trained": 5,
                "feature_names": [
                    "Absolute Magnitude",
                    "Estimated Diameter (Min)", 
                    "Estimated Diameter (Max)",
                    "Relative Velocity",
                    "Miss Distance"
                ]
            }
            metrics["metrics_source"] = "fallback"
            metrics["evaluation_error"] = str(eval_error)
        
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")


@app.post("/api/neo/evaluate-model")
async def trigger_model_evaluation(days: int = 30):
    """
    Trigger a fresh model evaluation against live NASA data.
    
    This endpoint forces a new evaluation regardless of cached results.
    Results are cached in Redis for subsequent requests.
    
    Args:
        days: Number of days of NASA data to use for evaluation (default: 30)
    """
    if xgb_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    try:
        print(f"🚀 Starting model evaluation with {days} days of data...")
        
        evaluation_results = await evaluate_model_on_live_data(days=min(days, 60))
        
        # Cache the results in Redis
        redis_client = await get_redis_client()
        if redis_client:
            try:
                await redis_client.setex(
                    "model_metrics:evaluation",
                    3600,  # 1 hour TTL
                    json.dumps(evaluation_results)
                )
                evaluation_results["cached"] = True
                evaluation_results["cache_ttl_seconds"] = 3600
            except Exception as e:
                print(f"Redis cache error: {e}")
                evaluation_results["cached"] = False
        else:
            evaluation_results["cached"] = False
        
        return {
            "status": "success",
            "message": f"Model evaluated on {evaluation_results['evaluation_samples']} samples",
            "evaluation": evaluation_results
        }
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/neo/feature-importance")
async def get_feature_importance():
    """Get feature importance from the XGBoost model"""
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        # Feature names in the order they were trained
        feature_names = [
            "Absolute Magnitude",
            "Estimated Diameter (Min)",
            "Estimated Diameter (Max)",
            "Relative Velocity",
            "Miss Distance"
        ]
        
        # Get feature importances from the model
        importances = xgb_model.feature_importances_
        
        # Create sorted list of features by importance
        feature_importance_list = []
        for name, importance in zip(feature_names, importances):
            feature_importance_list.append({
                "feature": name,
                "importance": float(importance),
                "percentage": float(importance * 100)
            })
        
        # Sort by importance descending
        feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)
        
        # Add rank
        for i, item in enumerate(feature_importance_list):
            item["rank"] = i + 1
        
        return {
            "feature_importances": feature_importance_list,
            "model_type": "XGBoost Classifier",
            "total_features": len(feature_names),
            "top_feature": feature_importance_list[0]["feature"] if feature_importance_list else None,
            "description": "Feature importance scores indicate how much each feature contributes to the model's hazard predictions. Higher scores mean greater influence on determining if an asteroid is potentially hazardous."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")


# ==================== MLflow Endpoints ====================

@app.get("/api/mlflow/status")
async def get_mlflow_status():
    """
    Get MLflow service status and experiment summary.
    
    Returns connection status, experiment info, and recent run statistics.
    """
    if not MLFLOW_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "MLflow service not installed",
            "tracking_uri": None
        }
    
    try:
        return MLflowService.get_status()
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/api/mlflow/log-evaluation")
async def log_evaluation_to_mlflow(days: int = 30, run_name: Optional[str] = None):
    """
    Evaluate the model and log results to MLflow.
    
    This endpoint:
    1. Evaluates the model against live NASA data
    2. Logs all metrics (accuracy, precision, recall, F1, ROC-AUC) to MLflow
    3. Stores the confusion matrix and evaluation details as artifacts
    
    Args:
        days: Number of days of NASA data to use (default: 30)
        run_name: Optional custom name for this MLflow run
    """
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow service not available")
    
    if xgb_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    try:
        # Run model evaluation
        print(f"🚀 Evaluating model for MLflow logging ({days} days of data)...")
        evaluation_results = await evaluate_model_on_live_data(days=min(days, 60))
        
        # Log to MLflow
        run_id = MLflowService.log_model_evaluation(
            metrics=evaluation_results,
            model_name="xgb_neo_classifier",
            run_name=run_name
        )
        
        if run_id:
            return {
                "status": "success",
                "message": "Evaluation logged to MLflow",
                "run_id": run_id,
                "metrics": {
                    "accuracy": evaluation_results.get("accuracy"),
                    "precision": evaluation_results.get("precision"),
                    "recall": evaluation_results.get("recall"),
                    "f1_score": evaluation_results.get("f1_score"),
                    "roc_auc": evaluation_results.get("roc_auc"),
                    "evaluation_samples": evaluation_results.get("evaluation_samples")
                },
                "mlflow_ui": "http://localhost:5000"
            }
        else:
            return {
                "status": "partial",
                "message": "Evaluation completed but MLflow logging failed",
                "evaluation": evaluation_results
            }
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/api/mlflow/best-run")
async def get_best_mlflow_run(metric: str = "f1_score"):
    """
    Get the best performing model run based on a specified metric.
    
    Args:
        metric: The metric to optimize for (default: f1_score)
                Options: accuracy, precision, recall, f1_score, roc_auc
    """
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow service not available")
    
    try:
        best_run = MLflowService.get_best_run(metric_name=metric)
        
        if best_run:
            return {
                "status": "success",
                "optimized_for": metric,
                "best_run": best_run
            }
        else:
            return {
                "status": "no_runs",
                "message": "No evaluation runs found. Run /api/mlflow/log-evaluation first."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mlflow/recent-runs")
async def get_recent_mlflow_runs(limit: int = 10):
    """
    Get the most recent model evaluation runs from MLflow.
    
    Args:
        limit: Maximum number of runs to return (default: 10)
    """
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow service not available")
    
    try:
        runs = MLflowService.get_recent_runs(limit=min(limit, 50))
        
        return {
            "status": "success",
            "count": len(runs),
            "runs": runs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mlflow/register-model")
async def register_model_to_mlflow():
    """
    Register the current XGBoost model to MLflow Model Registry.
    
    This creates a versioned model entry that can be used for:
    - Model versioning and lineage tracking
    - A/B testing different model versions
    - Production deployment workflows
    """
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow service not available")
    
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    try:
        # Get current model metrics for registration
        model_path = "xgb_neo_classifier.pkl"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found on disk")
        
        # Get feature importances as metrics
        importances = xgb_model.feature_importances_
        metrics = {
            "max_feature_importance": float(max(importances)),
            "min_feature_importance": float(min(importances)),
            "feature_count": len(importances)
        }
        
        version = MLflowService.log_model_registration(
            model_path=model_path,
            model_name="xgb_neo_classifier",
            metrics=metrics
        )
        
        if version:
            return {
                "status": "success",
                "message": f"Model registered as version {version}",
                "model_name": "xgb_neo_classifier",
                "version": version,
                "mlflow_ui": "http://localhost:5001/#/models"
            }
        else:
            raise HTTPException(status_code=500, detail="Model registration failed")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mlflow/full-evaluation")
async def full_mlflow_evaluation(
    days: int = 30,
    run_name: Optional[str] = None,
    register_model: bool = False,
    stage: str = "None"
):
    """
    🌟 PRODUCTION-READY: Full model evaluation with artifacts and optional registration.
    
    This endpoint:
    1. Evaluates the model against live NASA data
    2. Logs all metrics (accuracy, precision, recall, F1, ROC-AUC)
    3. Creates and logs a Confusion Matrix visualization
    4. Creates and logs a Feature Importance plot
    5. Logs the model with its signature (input/output schema)
    6. Logs an input example for documentation
    7. Optionally registers the model to the Model Registry with a stage
    
    Args:
        days: Number of days of NASA data to use (default: 30)
        run_name: Optional custom name for this MLflow run
        register_model: If True, registers model to Model Registry
        stage: Stage to assign if registering ("None", "Staging", "Production")
    
    Returns:
        Full logging results including run_id, model_version, and artifact list
    """
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow service not available")
    
    if xgb_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    try:
        # Run model evaluation
        print(f"🚀 Running FULL evaluation for MLflow ({days} days of data)...")
        evaluation_results = await evaluate_model_on_live_data(days=min(days, 60))
        
        # Create a sample input for signature
        sample_input = np.array([[22.0, 0.1, 0.2, 50000.0, 500000.0]])
        
        # Log full evaluation with all artifacts
        result = MLflowService.log_full_evaluation(
            metrics=evaluation_results,
            model=xgb_model,
            scaler=scaler,
            sample_input=sample_input,
            run_name=run_name,
            register_model=register_model,
            model_stage=stage
        )
        
        result["evaluation_metrics"] = {
            "accuracy": evaluation_results.get("accuracy"),
            "precision": evaluation_results.get("precision"),
            "recall": evaluation_results.get("recall"),
            "f1_score": evaluation_results.get("f1_score"),
            "roc_auc": evaluation_results.get("roc_auc"),
            "evaluation_samples": evaluation_results.get("evaluation_samples"),
            "confusion_matrix": evaluation_results.get("confusion_matrix")
        }
        result["mlflow_ui"] = "http://localhost:5001"
        
        return result
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Full evaluation failed: {str(e)}")


@app.post("/api/mlflow/promote-model")
async def promote_model_stage(version: str, stage: str = "Staging"):
    """
    Promote a model version to a new stage (Staging or Production).
    
    This allows you to:
    - Move a tested model from "None" to "Staging" for QA
    - Move a validated model from "Staging" to "Production" for live use
    - Archive old models by moving them to "Archived"
    
    Args:
        version: Model version number to promote (e.g., "1", "2")
        stage: Target stage ("Staging", "Production", "Archived", "None")
    """
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow service not available")
    
    valid_stages = ["None", "Staging", "Production", "Archived"]
    if stage not in valid_stages:
        raise HTTPException(status_code=400, detail=f"Invalid stage. Must be one of: {valid_stages}")
    
    try:
        success = MLflowService.transition_model_stage(
            model_name="NASA_NEO_Classifier",
            version=version,
            stage=stage
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Model version {version} promoted to {stage}",
                "model_name": "NASA_NEO_Classifier",
                "version": version,
                "new_stage": stage,
                "mlflow_ui": "http://localhost:5001/#/models/NASA_NEO_Classifier"
            }
        else:
            raise HTTPException(status_code=500, detail="Stage transition failed")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mlflow/compare-runs")
async def compare_mlflow_runs(run_ids: str):
    """
    Compare multiple runs side by side.
    
    This is useful for:
    - Comparing XGBoost vs Random Forest performance
    - Comparing different hyperparameter configurations
    - Identifying which model has fewer False Negatives (critical for hazard detection)
    
    Args:
        run_ids: Comma-separated run IDs to compare (e.g., "abc123,def456")
    
    Returns:
        Comparison data with metrics for each run and best performer per metric
    """
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow service not available")
    
    try:
        run_id_list = [rid.strip() for rid in run_ids.split(",") if rid.strip()]
        
        if len(run_id_list) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 run IDs to compare")
        
        if len(run_id_list) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 runs can be compared at once")
        
        comparison = MLflowService.compare_runs(run_id_list)
        
        return {
            "status": "success",
            "comparison": comparison,
            "mlflow_ui": "http://localhost:5001"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mlflow/registered-models")
async def get_registered_models():
    """
    Get all registered models and their versions from Model Registry.
    
    Shows:
    - All registered model names
    - Each version with its stage (None/Staging/Production/Archived)
    - Run ID that created each version
    """
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow service not available")
    
    try:
        models = MLflowService.get_registered_models()
        
        return {
            "status": "success",
            "model_count": len(models),
            "models": models,
            "production_uri": MLflowService.get_production_model_uri()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/kb-status")
async def get_kb_status():
    """Get knowledge base status"""
    if weaviate_client:
        try:
            collection = weaviate_client.collections.get("NEODocument")
            response = collection.aggregate.over_all(total_count=True)
            count = response.total_count
            
            return {
                "weaviate_connected": True,
                "collection_exists": True,
                "document_count": count,
                "semantic_search_enabled": embedding_model is not None,
                "embedding_model": "all-MiniLM-L6-v2" if embedding_model else "fallback",
                "storage_mode": "weaviate",
                "error": None
            }
        except Exception as e:
            return {
                "weaviate_connected": True,
                "collection_exists": False,
                "document_count": 0,
                "semantic_search_enabled": False,
                "storage_mode": "fallback",
                "error": str(e)
            }
    else:
        return {
            "weaviate_connected": False,
            "collection_exists": False,
            "document_count": len(neo_documents),
            "semantic_search_enabled": False,
            "storage_mode": "in-memory",
            "error": weaviate_error
        }

@app.post("/api/rag/auto-index")
async def auto_index_from_nasa():
    """Auto-index NEO data from NASA with Redis caching"""
    global neo_documents
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        all_records = []
        current_start = start_date
        
        for chunk in range(5):
            chunk_end = min(current_start + timedelta(days=6), end_date)
            start_str = current_start.strftime('%Y-%m-%d')
            end_str = chunk_end.strftime('%Y-%m-%d')
            
            try:
                # Use cached async fetch - leverages Redis caching
                raw_data = await fetch_nasa_data_cached(start_str, end_str)
                chunk_data = parse_neo_data(raw_data)
                all_records.extend(chunk_data)
            except Exception as e:
                print(f"Chunk error: {str(e)}")
                continue
            
            current_start = chunk_end + timedelta(days=1)
            if current_start > end_date:
                break
        
        if not all_records:
            raise HTTPException(status_code=404, detail="No NASA data available")
        
        # Remove duplicates
        seen = set()
        unique_records = []
        for record in all_records:
            key = (record.neo_id, record.miss_distance)
            if key not in seen:
                seen.add(key)
                unique_records.append(record)
        
        indexed_count = 0
        
        for neo in unique_records:
            diameter = (neo.estimated_diameter_min + neo.estimated_diameter_max) / 2
            velocity_kms = neo.relative_velocity / 3600
            
            ke = calculate_kinetic_energy(diameter, velocity_kms)
            ip = calculate_impact_probability(neo.miss_distance, diameter)
            rs = calculate_risk_score(neo.miss_distance, ke, ip)
            
            if rs > 10:
                risk_cat = "CRITICAL"
            elif rs > 5:
                risk_cat = "HIGH"
            elif rs > 2:
                risk_cat = "MODERATE"
            else:
                risk_cat = "LOW"
            
            content = f"""NEO Name: {neo.name}
NEO ID: {neo.neo_id}
Date: {neo.date}
Risk Score: {rs:.2f}
Risk Category: {risk_cat}
Diameter: {diameter:.4f} km
Velocity: {velocity_kms:.2f} km/s
Miss Distance: {neo.miss_distance:.2f} km ({neo.miss_distance/384400:.3f} lunar distances)
Kinetic Energy: {ke:.2f} Mt
Potentially Hazardous: {'Yes' if neo.is_hazardous else 'No'}

This is a {risk_cat.lower()} risk near-Earth object approaching on {neo.date}."""
            
            doc = {
                "content": content,
                "neo_id": neo.neo_id,
                "name": neo.name,
                "date": neo.date,
                "risk_score": float(rs),
                "risk_category": risk_cat,
                "diameter_km": float(diameter),
                "velocity_kms": float(velocity_kms),
                "miss_distance_km": float(neo.miss_distance),
                "kinetic_energy_mt": float(ke),
                "is_hazardous": bool(neo.is_hazardous)
            }
            
            if weaviate_client:
                try:
                    embedding = get_semantic_embedding(content)
                    collection = weaviate_client.collections.get("NEODocument")
                    collection.data.insert(
                        properties=doc,
                        vector=embedding
                    )
                    indexed_count += 1
                except Exception as e:
                    print(f"Weaviate insert error: {str(e)}")
                    neo_documents.append(doc)
                    indexed_count += 1
            else:
                neo_documents.append(doc)
                indexed_count += 1
        
        total_count = 0
        if weaviate_client:
            try:
                collection = weaviate_client.collections.get("NEODocument")
                response = collection.aggregate.over_all(total_count=True)
                total_count = response.total_count
            except:
                total_count = len(neo_documents)
        else:
            total_count = len(neo_documents)
        
        return {
            "status": "success",
            "indexed_count": indexed_count,
            "total_documents": total_count,
            "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "storage_mode": "weaviate" if weaviate_client else "in-memory"
        }
    
    except Exception as e:
        print(f"Auto-indexing error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Auto-indexing failed: {str(e)}")

@app.post("/api/rag/index-neos")
async def index_neos(request: NEOIndexRequest):
    """Index NEO documents"""
    global neo_documents
    
    try:
        indexed_count = 0
        
        for neo_data in request.neos:
            content = f"""NEO Name: {neo_data['name']}
NEO ID: {neo_data['neo_id']}
Date: {neo_data['date']}
Risk Score: {neo_data['risk_score']:.2f}
Risk Category: {neo_data['risk_category']}
Diameter: {neo_data['diameter_km']:.4f} km
Velocity: {neo_data['velocity_kms']:.2f} km/s
Miss Distance: {neo_data['miss_distance_km']:.2f} km ({neo_data['miss_distance_km']/384400:.3f} lunar distances)
Kinetic Energy: {neo_data['kinetic_energy_mt']:.2f} megatons
Potentially Hazardous: {'Yes' if neo_data['is_hazardous'] else 'No'}

This is a {neo_data['risk_category'].lower()} risk near-Earth object approaching on {neo_data['date']}."""
            
            doc = {
                "content": content,
                "neo_id": neo_data['neo_id'],
                "name": neo_data['name'],
                "date": neo_data['date'],
                "risk_score": neo_data['risk_score'],
                "risk_category": neo_data['risk_category'],
                "diameter_km": neo_data['diameter_km'],
                "velocity_kms": neo_data['velocity_kms'],
                "miss_distance_km": neo_data['miss_distance_km'],
                "kinetic_energy_mt": neo_data['kinetic_energy_mt'],
                "is_hazardous": neo_data['is_hazardous']
            }
            
            if weaviate_client:
                try:
                    embedding = get_semantic_embedding(content)
                    collection = weaviate_client.collections.get("NEODocument")
                    collection.data.insert(
                        properties=doc,
                        vector=embedding
                    )
                    indexed_count += 1
                except Exception as e:
                    print(f"Weaviate insert error: {str(e)}")
                    neo_documents.append(doc)
                    indexed_count += 1
            else:
                neo_documents.append(doc)
                indexed_count += 1
        
        total_count = 0
        if weaviate_client:
            try:
                collection = weaviate_client.collections.get("NEODocument")
                response = collection.aggregate.over_all(total_count=True)
                total_count = response.total_count
            except:
                total_count = len(neo_documents)
        else:
            total_count = len(neo_documents)
        
        return {
            "status": "success",
            "indexed_count": indexed_count,
            "total_documents": total_count,
            "storage_mode": "weaviate" if weaviate_client else "in-memory"
        }
    
    except Exception as e:
        print(f"Indexing error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """Query the NEO knowledge base"""
    global neo_documents
    
    try:
        # Get current document count
        current_count = 0
        if weaviate_client:
            try:
                collection = weaviate_client.collections.get("NEODocument")
                response = collection.aggregate.over_all(total_count=True)
                current_count = response.total_count
            except:
                current_count = len(neo_documents)
        else:
            current_count = len(neo_documents)
        
        if current_count == 0:
            return RAGQueryResponse(
                answer="The NEO knowledge base is empty. Please click 'Auto-Index from NASA' to populate it with recent asteroid data.",
                sources=[],
                search_type=request.search_type,
                semantic_similarity_scores=[],
                kb_document_count=0
            )
        
        sources = []
        context_parts = []
        similarity_scores = []
        
        # Try Weaviate search first
        if weaviate_client:
            try:
                collection = weaviate_client.collections.get("NEODocument")
                
                if request.search_type == "semantic":
                    query_embedding = get_semantic_embedding(request.question)
                    results = collection.query.near_vector(
                        near_vector=query_embedding,
                        limit=5,
                        return_metadata=MetadataQuery(distance=True)
                    )
                    
                    for obj in results.objects:
                        sources.append({
                            "neo_id": obj.properties.get("neo_id"),
                            "name": obj.properties.get("name"),
                            "risk_score": obj.properties.get("risk_score"),
                            "risk_category": obj.properties.get("risk_category"),
                            "date": obj.properties.get("date"),
                            "similarity": 1 - obj.metadata.distance if obj.metadata.distance else None
                        })
                        context_parts.append(obj.properties.get("content", ""))
                        if obj.metadata.distance:
                            similarity_scores.append(1 - obj.metadata.distance)
                
                elif request.search_type == "keyword":
                    results = collection.query.bm25(
                        query=request.question,
                        limit=5,
                        return_metadata=MetadataQuery(score=True)
                    )
                    
                    for obj in results.objects:
                        sources.append({
                            "neo_id": obj.properties.get("neo_id"),
                            "name": obj.properties.get("name"),
                            "risk_score": obj.properties.get("risk_score"),
                            "risk_category": obj.properties.get("risk_category"),
                            "date": obj.properties.get("date"),
                            "bm25_score": obj.metadata.score if obj.metadata.score else None
                        })
                        context_parts.append(obj.properties.get("content", ""))
                
                else:  # hybrid
                    query_embedding = get_semantic_embedding(request.question)
                    results = collection.query.hybrid(
                        query=request.question,
                        vector=query_embedding,
                        limit=5,
                        alpha=0.7,
                        return_metadata=MetadataQuery(score=True)
                    )
                    
                    for obj in results.objects:
                        sources.append({
                            "neo_id": obj.properties.get("neo_id"),
                            "name": obj.properties.get("name"),
                            "risk_score": obj.properties.get("risk_score"),
                            "risk_category": obj.properties.get("risk_category"),
                            "date": obj.properties.get("date"),
                            "hybrid_score": obj.metadata.score if obj.metadata.score else None
                        })
                        context_parts.append(obj.properties.get("content", ""))
            
            except Exception as weaviate_error:
                print(f"Weaviate search error: {str(weaviate_error)}")
                # Fall back to in-memory search
                pass
        
        # Fallback to in-memory search if Weaviate failed or not available
        if not sources and neo_documents:
            query_lower = request.question.lower()
            scored_docs = []
            
            for doc in neo_documents:
                content_lower = doc["content"].lower()
                score = sum(1 for word in query_lower.split() if word in content_lower)
                
                if score > 0:
                    scored_docs.append((doc, score))
            
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            for doc, score in scored_docs[:5]:
                sources.append({
                    "neo_id": doc["neo_id"],
                    "name": doc["name"],
                    "risk_score": doc["risk_score"],
                    "risk_category": doc["risk_category"],
                    "date": doc["date"],
                    "relevance_score": score
                })
                context_parts.append(doc["content"])
        
        context = "\n\n".join(context_parts[:3])
        
        if not context:
            return RAGQueryResponse(
                answer="No relevant NEO data found for this query. Try asking about risk scores, specific asteroids, or general NEO information.",
                sources=[],
                search_type=request.search_type,
                semantic_similarity_scores=[],
                kb_document_count=current_count
            )
        
        # Generate answer using LLM
        try:
            system_prompt = """You are an expert NASA scientist specializing in Near-Earth Objects (NEOs) and planetary defense. 
Answer questions accurately based on the provided NEO data. Be concise but informative.
Focus on risk assessment, orbital characteristics, and potential impact scenarios.
If you don't have specific information in the context, say so clearly."""
            
            user_prompt = f"""Based on the following NEO data, answer this question: {request.question}

Context:
{context}

Provide a clear, accurate answer focusing on the most relevant information."""

            completion = openrouter_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = completion.choices[0].message.content
            
        except Exception as e:
            print(f"LLM error: {str(e)}")
            # Fallback to simple response
            answer = f"Based on the NEO database, here are the most relevant asteroids:\n\n{context[:500]}..."
        
        return RAGQueryResponse(
            answer=answer,
            sources=sources,
            search_type=request.search_type,
            semantic_similarity_scores=similarity_scores if similarity_scores else None,
            kb_document_count=current_count
        )
    
    except Exception as e:
        print(f"Query error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.delete("/api/rag/clear-kb")
async def clear_knowledge_base():
    """Clear the knowledge base"""
    global neo_documents
    
    try:
        count_before = 0
        
        if weaviate_client:
            try:
                collection = weaviate_client.collections.get("NEODocument")
                response = collection.aggregate.over_all(total_count=True)
                count_before = response.total_count
                
                collection.data.delete_many(where=None)
            except Exception as e:
                print(f"Weaviate clear error: {str(e)}")
        
        # Also clear in-memory storage
        neo_documents = []
        
        return {
            "status": "success",
            "message": "Knowledge base cleared",
            "documents_deleted": count_before + len(neo_documents)
        }
    except Exception as e:
        print(f"Clear error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@app.post("/api/neo/reload-models")
async def reload_models():
    """Reload ML models"""
    try:
        load_models()
        return {
            "status": "success" if (xgb_model and scaler) else "failed",
            "xgboost_loaded": xgb_model is not None,
            "scaler_loaded": scaler is not None,
            "ready_for_predictions": xgb_model is not None and scaler is not None,
            "error": model_load_error
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/api/rag/reinit")
async def reinitialize_weaviate():
    """Reinitialize Weaviate"""
    try:
        init_weaviate()
        
        count = 0
        if weaviate_client:
            try:
                collection = weaviate_client.collections.get("NEODocument")
                response = collection.aggregate.over_all(total_count=True)
                count = response.total_count
            except:
                count = 0
        
        return {
            "status": "success" if weaviate_client else "using_fallback",
            "weaviate_connected": weaviate_client is not None,
            "document_count": count,
            "semantic_search_enabled": embedding_model is not None,
            "error": weaviate_error
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# =============================================================================
# AGENTIC RAG SYSTEM - Autonomous AI with Tool Calling
# =============================================================================

# Pydantic models for Agent endpoints
class AgentQueryRequest(BaseModel):
    """Request model for agent queries."""
    question: str
    include_reasoning: Optional[bool] = False

class AgentQueryResponse(BaseModel):
    """Response model for agent queries."""
    answer: str
    tools_used: List[str]
    sources: List[Dict]
    iterations: int
    reasoning_steps: Optional[List[Dict]] = None


async def search_knowledge_base_for_agent(query: str) -> List[Dict]:
    """
    Knowledge base search function to inject into the Agent.
    
    This function wraps the existing Weaviate/fallback search logic
    and returns documents in a format the agent can use.
    """
    global weaviate_client, embedding_model, neo_documents
    
    sources = []
    
    # Try Weaviate search first
    if weaviate_client:
        try:
            collection = weaviate_client.collections.get("NEODocument")
            query_embedding = get_semantic_embedding(query)
            
            results = collection.query.hybrid(
                query=query,
                vector=query_embedding,
                limit=5,
                alpha=0.7,
                return_metadata=MetadataQuery(score=True)
            )
            
            for obj in results.objects:
                sources.append({
                    "neo_id": obj.properties.get("neo_id"),
                    "name": obj.properties.get("name"),
                    "date": obj.properties.get("date"),
                    "risk_score": obj.properties.get("risk_score"),
                    "risk_category": obj.properties.get("risk_category"),
                    "diameter_km": obj.properties.get("diameter_km"),
                    "velocity_kms": obj.properties.get("velocity_kms"),
                    "miss_distance_km": obj.properties.get("miss_distance_km"),
                    "content": obj.properties.get("content", "")
                })
            
            return sources
        
        except Exception as e:
            print(f"Weaviate search error in agent: {str(e)}")
    
    # Fallback to in-memory search
    if neo_documents:
        query_lower = query.lower()
        scored_docs = []
        
        for doc in neo_documents:
            content_lower = doc.get("content", "").lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            
            if score > 0:
                scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        for doc, score in scored_docs[:5]:
            sources.append(doc)
    
    return sources


async def predict_hazard_for_agent(params: Dict) -> Dict:
    """
    XGBoost prediction function to inject into the Agent.
    
    This wraps the existing ML model to make predictions based on
    asteroid parameters provided by the agent.
    """
    global xgb_model, scaler
    
    if xgb_model is None or scaler is None:
        raise ValueError("ML models not loaded")
    
    try:
        # Extract features in the correct order for the model
        absolute_magnitude = params.get("absolute_magnitude", 25.0)
        diameter_min = params.get("estimated_diameter_min", 0.01)
        diameter_max = params.get("estimated_diameter_max", 0.02)
        velocity = params.get("relative_velocity", 50000)  # kph
        miss_distance = params.get("miss_distance", 1000000)  # km
        
        # Create feature array: [magnitude, diameter_min, diameter_max, velocity, miss_distance]
        features = np.array([[
            absolute_magnitude,
            diameter_min,
            diameter_max,
            velocity,
            miss_distance
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get prediction and probability
        prediction = xgb_model.predict(features_scaled)[0]
        probabilities = xgb_model.predict_proba(features_scaled)[0]
        
        hazard_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        
        # Determine risk level based on probability
        if hazard_probability >= 0.8:
            risk_level = "CRITICAL"
        elif hazard_probability >= 0.5:
            risk_level = "HIGH"
        elif hazard_probability >= 0.25:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return {
            "is_hazardous": bool(prediction),
            "hazard_probability": round(hazard_probability, 4),
            "confidence": round(max(probabilities) * 100, 2),
            "risk_level": risk_level,
            "input_features": {
                "absolute_magnitude": absolute_magnitude,
                "diameter_min_km": diameter_min,
                "diameter_max_km": diameter_max,
                "velocity_kph": velocity,
                "miss_distance_km": miss_distance
            },
            "model_info": {
                "type": "XGBoost Classifier",
                "training_samples": 127347,
                "accuracy": "94.18%"
            }
        }
    
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")


# Create global LangGraph agent instance
neo_agent: Optional[LangGraphAgentService] = None

def get_agent() -> LangGraphAgentService:
    """Get or create the LangGraph Robot Scientist agent instance."""
    global neo_agent
    if neo_agent is None:
        neo_agent = create_langgraph_agent(
            knowledge_base_search_fn=search_knowledge_base_for_agent,
            ml_predict_fn=predict_hazard_for_agent
        )
    return neo_agent


@app.post("/api/agent/query", response_model=AgentQueryResponse)
async def query_agent(request: AgentQueryRequest):
    """
    Query the Robot Scientist - Autonomous Planetary Defense Agent.
    
    This is the main endpoint for the agentic RAG system using ReAct protocol.
    The Robot Scientist can:
    
    1. **Think** about what information is needed (Thought phase)
    2. **Act** by calling appropriate tools (Action phase)
    3. **Observe** tool results and iterate if needed (Observation phase)
    4. **Brief** mission control with synthesized intelligence (Final Answer)
    
    Available Tools:
    - **fetch_live_nasa_feed**: Real-time asteroid data from NASA
    - **predict_hazard_xgboost**: ML model prediction (94.18% accuracy)
    - **search_knowledge_base**: Historical context from Weaviate
    
    Args:
        question: Natural language question about NEOs
        include_reasoning: If true, includes ReAct steps in response
    
    Returns:
        Mission briefing with analysis, tools used, and sources
    
    Example questions:
    - "What is the most dangerous asteroid approaching this week?"
    - "Is asteroid 2024 XY hazardous? Use the ML model to predict."
    - "Compare today's closest approach to Chelyabinsk"
    """
    try:
        agent = get_agent()
        result = await agent.process_query(request.question)
        
        response = AgentQueryResponse(
            answer=result.get("answer", "I couldn't process your query."),
            tools_used=result.get("tools_used", []),
            sources=result.get("sources", []),
            iterations=result.get("iterations", 0)
        )
        
        if request.include_reasoning:
            response.reasoning_steps = result.get("reasoning_steps", [])
        
        return response
    
    except Exception as e:
        print(f"Agent query error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/api/agent/status")
async def get_agent_status():
    """
    Get the status of the LangGraph Robot Scientist agent.
    
    Returns information about:
    - Agent framework (LangGraph)
    - Available tools and their status
    - ML model availability
    - Knowledge base status
    """
    kb_count = 0
    if weaviate_client:
        try:
            collection = weaviate_client.collections.get("NEODocument")
            response = collection.aggregate.over_all(total_count=True)
            kb_count = response.total_count
        except:
            kb_count = len(neo_documents)
    else:
        kb_count = len(neo_documents)
    
    ml_available = xgb_model is not None and scaler is not None
    
    return {
        "status": "operational",
        "agent_name": "Robot Scientist",
        "framework": {
            "name": "LangGraph",
            "version": ">= 0.2.0",
            "ecosystem": "LangChain",
            "architecture": "StateGraph with conditional edges"
        },
        "reasoning_protocol": "ReAct (Reasoning + Acting)",
        "llm_model": CHAT_MODEL,
        "llm_provider": "OpenRouter",
        "available_tools": [
            {
                "name": "fetch_live_nasa_feed",
                "description": "Fetch real-time asteroid data from NASA NeoWs API",
                "status": "available",
                "priority": "Use first for current/future asteroid queries"
            },
            {
                "name": "predict_hazard_xgboost",
                "description": "XGBoost ML model trained on 127,347 asteroids (94.18% accuracy)",
                "status": "available" if ml_available else "unavailable",
                "priority": "Use for hazard classification instead of manual calculations"
            },
            {
                "name": "search_knowledge_base", 
                "description": "Search Weaviate for historical NEO data and past events",
                "status": "available" if (weaviate_client or neo_documents) else "empty",
                "priority": "Use for historical context and analogies"
            }
        ],
        "ml_model": {
            "loaded": ml_available,
            "type": "XGBoost Classifier",
            "training_samples": 127347,
            "accuracy": "94.18%"
        },
        "knowledge_base": {
            "backend": "weaviate" if weaviate_client else "in-memory",
            "document_count": kb_count,
            "semantic_search": embedding_model is not None
        },
        "graph_structure": {
            "nodes": ["agent", "tools"],
            "entry_point": "agent",
            "conditional_edges": "agent -> tools | END",
            "loop_edge": "tools -> agent"
        },
        "max_iterations": 5,
        "capabilities": [
            "LangGraph StateGraph",
            "ReAct reasoning loop",
            "Memory/Checkpointing",
            "Real-time NASA data access",
            "ML-powered hazard prediction",
            "Historical context retrieval",
            "Mission control briefings"
        ],
        "premium_skill": "Agentic Engineering (LangGraph, LlamaIndex, CrewAI)"
    }


@app.post("/api/agent/analyze-neo")
async def agent_analyze_neo(
    neo_name: Optional[str] = None,
    diameter_km: Optional[float] = None,
    velocity_kms: Optional[float] = None,
    miss_distance_km: Optional[float] = None
):
    """
    Quick analysis of a specific NEO using the agent.
    
    Provide either a NEO name (agent will search for it) or 
    physical parameters (agent will calculate risk).
    
    Args:
        neo_name: Name of the asteroid to analyze (searches KB and live data)
        diameter_km: Diameter in kilometers (for direct calculation)
        velocity_kms: Velocity in km/s (for direct calculation)
        miss_distance_km: Miss distance in km (for direct calculation)
    
    Returns:
        Agent analysis of the NEO
    """
    if neo_name:
        query = f"Analyze the asteroid {neo_name}. Find its data and calculate its risk level."
    elif diameter_km and velocity_kms and miss_distance_km:
        query = f"""Calculate the risk assessment for an asteroid with:
- Diameter: {diameter_km} km
- Velocity: {velocity_kms} km/s
- Miss distance: {miss_distance_km} km

Provide a complete threat analysis."""
    else:
        raise HTTPException(
            status_code=400, 
            detail="Provide either neo_name OR all three parameters (diameter_km, velocity_kms, miss_distance_km)"
        )
    
    agent = get_agent()
    result = await agent.process_query(query)
    
    return {
        "analysis": result.get("answer"),
        "tools_used": result.get("tools_used", []),
        "sources": result.get("sources", [])
    }


# =============================================================================
# CACHE MANAGEMENT ENDPOINTS (Redis Look-Aside Caching)
# =============================================================================

@app.get("/api/cache/stats")
async def get_cache_statistics():
    """
    Get Redis cache statistics for observability.
    
    Returns hit/miss ratios, connection status, and memory usage.
    Useful for monitoring cache effectiveness and debugging.
    """
    try:
        stats = await get_cache_stats()
        return {
            "status": "success",
            "cache": stats,
            "cache_enabled": stats.get("redis_connected", False),
            "recommendation": (
                "Cache is performing well" 
                if stats.get("hit_ratio_percent", 0) > 50 
                else "Consider warming the cache with common queries"
            )
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "cache_enabled": False
        }


@app.delete("/api/cache/clear")
async def clear_redis_cache(pattern: str = "nasa:*"):
    """
    Clear cached entries matching a pattern.
    
    Args:
        pattern: Redis key pattern (default: all NASA cache keys)
    
    Use cases:
    - Clear all: pattern="nasa:*"
    - Clear feed cache only: pattern="neo_feed:*"
    - Clear lookup cache only: pattern="neo_lookup:*"
    """
    try:
        result = await clear_cache(pattern)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


@app.get("/api/cache/keys")
async def get_cached_keys(pattern: str = "nasa:*"):
    """
    Get information about cached keys for debugging.
    
    Returns key names and their remaining TTL.
    Limited to first 50 keys for performance.
    """
    try:
        keys = await get_cached_keys_info(pattern)
        return {
            "status": "success",
            "pattern": pattern,
            "key_count": len(keys),
            "keys": keys
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get keys: {str(e)}")


@app.get("/api/cache/health")
async def check_cache_health():
    """
    Health check for Redis cache.
    
    Returns connection status and latency.
    """
    import time
    
    start = time.time()
    client = await get_redis_client()
    
    if client is None:
        return {
            "status": "degraded",
            "redis_connected": False,
            "message": "Redis unavailable - operating in pass-through mode",
            "impact": "All requests go directly to NASA API (rate limit risk)"
        }
    
    try:
        await client.ping()
        latency_ms = (time.time() - start) * 1000
        
        return {
            "status": "healthy",
            "redis_connected": True,
            "latency_ms": round(latency_ms, 2),
            "message": "Redis cache is operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "redis_connected": False,
            "error": str(e),
            "message": "Redis connection failed"
        }


@app.post("/api/cache/warm")
async def warm_cache(days: int = 7):
    """
    Pre-warm the cache with recent NASA data.
    
    Fetches the last N days of data to populate the cache,
    reducing latency for subsequent requests.
    
    Args:
        days: Number of days to cache (default: 7, max: 30)
    """
    days = min(days, 30)  # Cap at 30 days
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        warmed_chunks = 0
        failed_chunks = 0
        
        current_start = start_date
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=6), end_date)
            start_str = current_start.strftime('%Y-%m-%d')
            end_str = chunk_end.strftime('%Y-%m-%d')
            
            try:
                # This will cache the data via the decorator
                await fetch_nasa_data_cached(start_str, end_str)
                warmed_chunks += 1
            except Exception as e:
                print(f"Warm cache error: {str(e)}")
                failed_chunks += 1
            
            current_start = chunk_end + timedelta(days=1)
        
        return {
            "status": "success",
            "message": f"Cache warmed for {days} days of data",
            "chunks_warmed": warmed_chunks,
            "chunks_failed": failed_chunks,
            "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache warming failed: {str(e)}")


@app.get("/api/nasa/status")
async def check_nasa_api_status():
    """
    Check NASA API availability and rate limit status.
    
    Makes a minimal request to verify the API is reachable.
    Useful for monitoring and debugging.
    """
    try:
        status = await NasaService.get_api_status()
        return status
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("NASA NEO ADVANCED ANALYTICS API v5.0.0")
    print("With Agentic RAG - Autonomous AI with Tool Calling")
    print("="*70)
    
    print("\nAvailable endpoints:")
    print("  Analytics:")
    print("    - GET  /api/neo/advanced-analytics?days=30")
    print("    - POST /api/neo/predict")
    print("    - GET  /api/neo/model-status")
    print("    - GET  /api/neo/model-metrics (dynamic - evaluates on live data)")
    print("    - GET  /api/neo/feature-importance")
    print("    - POST /api/neo/evaluate-model?days=30 (force fresh evaluation)")
    print("  Agentic RAG (NEW!):")
    print("    - POST /api/agent/query (main agent endpoint)")
    print("    - GET  /api/agent/status")
    print("    - POST /api/agent/analyze-neo")
    print("  RAG (Basic):")
    print("    - GET  /api/rag/kb-status")
    print("    - POST /api/rag/auto-index")
    print("    - POST /api/rag/index-neos")
    print("    - POST /api/rag/query")
    print("    - DELETE /api/rag/clear-kb")
    print("    - POST /api/rag/reinit")
    print("  Cache (Redis Look-Aside):")
    print("    - GET  /api/cache/stats")
    print("    - GET  /api/cache/health")
    print("    - GET  /api/cache/keys")
    print("    - DELETE /api/cache/clear")
    print("    - POST /api/cache/warm")
    print("  Monitoring:")
    print("    - GET  /api/nasa/status")
    
    print(f"\nML Models:")
    print(f"  XGBoost: {'✓ Loaded' if xgb_model else '✗ Not Found'}")
    print(f"  Scaler:  {'✓ Loaded' if scaler else '✗ Not Found'}")
    
    print(f"\nAgentic RAG System:")
    print(f"  Status: ✓ Operational")
    print(f"  Tools:  search_knowledge_base, fetch_live_nasa_feed, calculate_risk")
    print(f"  LLM:    {CHAT_MODEL} (via OpenRouter)")
    
    print(f"\nRAG System:")
    print(f"  Weaviate: {'✓ Connected' if weaviate_client else '✗ Using Fallback'}")
    print(f"  Storage:  {'Weaviate' if weaviate_client else 'In-Memory'}")
    print(f"  Semantic: {'✓ Enabled' if embedding_model else '✗ Fallback'}")
    
    print(f"\nRedis Cache (Look-Aside):")
    print(f"  URL: {os.getenv('REDIS_URL', 'redis://localhost:6379/0')}")
    print(f"  Feed TTL: 24 hours (86400s)")
    print(f"  Lookup TTL: 7 days (604800s)")
    print(f"  Note: Cache status available at /api/cache/health")
    
    if weaviate_client:
        try:
            collection = weaviate_client.collections.get("NEODocument")
            response = collection.aggregate.over_all(total_count=True)
            count = response.total_count
            print(f"\n{'✓'*35}")
            print(f"✓ Agentic RAG System Ready! Documents: {count}")
            print(f"{'✓'*35}")
        except:
            print(f"\n{'✓'*35}")
            print(f"✓ Agentic RAG System Ready (empty KB)")
            print(f"{'✓'*35}")
    else:
        print(f"\n{'⚠'*35}")
        print(f"⚠ Using In-Memory Fallback Storage")
        print(f"{'⚠'*35}")
    
    print(f"\n{'='*70}")
    print("Server starting on http://0.0.0.0:8000")
    print(f"{'='*70}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)