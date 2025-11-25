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
from openai import OpenAI
from collections import defaultdict

# Try to import Weaviate, but don't fail if not available
try:
    import weaviate
    from weaviate.classes.init import Auth
    from weaviate.classes.query import MetadataQuery
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
NASA_API_KEY = 'API_Key'
NASA_API_URL = 'api url'

EARTH_RADIUS_KM = 6371
LUNAR_DISTANCE_KM = 384400

OPENROUTER_API_KEY = "API_Key"
CHAT_MODEL = "qwen/qwen-2.5-7b-instruct"

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
                {"name": "content", "dataType": ["text"]},
                {"name": "neo_id", "dataType": ["int"]},
                {"name": "name", "dataType": ["text"]},
                {"name": "date", "dataType": ["text"]},
                {"name": "risk_score", "dataType": ["number"]},
                {"name": "risk_category", "dataType": ["text"]},
                {"name": "diameter_km", "dataType": ["number"]},
                {"name": "velocity_kms", "dataType": ["number"]},
                {"name": "miss_distance_km", "dataType": ["number"]},
                {"name": "kinetic_energy_mt", "dataType": ["number"]},
                {"name": "is_hazardous", "dataType": ["boolean"]}
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
    """Fetch data from NASA API"""
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
    
    return {
        "status": "online",
        "service": "NASA NEO Advanced Analytics API",
        "version": "4.0.1",
        "ml_models_loaded": {
            "xgboost": xgb_model is not None,
            "scaler": scaler is not None
        },
        "weaviate_connected": weaviate_client is not None,
        "semantic_search_enabled": embedding_model is not None,
        "knowledge_base_documents": kb_count,
        "ready_for_predictions": xgb_model is not None and scaler is not None,
        "ready_for_rag": True,  # Always ready (fallback available)
        "endpoints": [
            "/api/neo/advanced-analytics",
            "/api/neo/predict",
            "/api/neo/model-status",
            "/api/rag/kb-status",
            "/api/rag/index-neos",
            "/api/rag/query",
            "/api/rag/auto-index"
        ]
    }

@app.get("/api/neo/advanced-analytics", response_model=AdvancedAnalyticsResponse)
async def get_advanced_analytics(days: int = 30):
    """Get advanced NEO analytics"""
    all_records = []
    end_date = datetime.now()
    current_start = end_date - timedelta(days=days)
    
    num_chunks = (days // 7) + (1 if days % 7 != 0 else 0)
    
    for chunk in range(num_chunks):
        chunk_end = min(current_start + timedelta(days=6), end_date)
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        
        try:
            raw_data = fetch_nasa_data(start_str, end_str)
            chunk_data = parse_neo_data(raw_data)
            all_records.extend(chunk_data)
        except Exception as e:
            print(f"Chunk error: {str(e)}")
            continue
        
        current_start = chunk_end + timedelta(days=1)
        if current_start > end_date:
            break
    
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
            follow_up_priority=priority
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
    """Auto-index NEO data from NASA"""
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
                raw_data = fetch_nasa_data(start_str, end_str)
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

if __name__ == "__main__":
    print("\n" + "="*70)
    print("NASA NEO ADVANCED ANALYTICS API v4.0.1")
    print("="*70)
    
    print("\nAvailable endpoints:")
    print("  - GET  /")
    print("  - GET  /api/neo/advanced-analytics?days=30")
    print("  - POST /api/neo/predict")
    print("  - GET  /api/neo/model-status")
    print("  - POST /api/neo/reload-models")
    print("  - GET  /api/rag/kb-status")
    print("  - POST /api/rag/auto-index")
    print("  - POST /api/rag/index-neos")
    print("  - POST /api/rag/query")
    print("  - DELETE /api/rag/clear-kb")
    print("  - POST /api/rag/reinit")
    
    print(f"\nML Models:")
    print(f"  XGBoost: {'✓ Loaded' if xgb_model else '✗ Not Found'}")
    print(f"  Scaler:  {'✓ Loaded' if scaler else '✗ Not Found'}")
    
    print(f"\nRAG System:")
    print(f"  Weaviate: {'✓ Connected' if weaviate_client else '✗ Using Fallback'}")
    print(f"  Storage:  {'Weaviate' if weaviate_client else 'In-Memory'}")
    print(f"  Semantic: {'✓ Enabled' if embedding_model else '✗ Fallback'}")
    print(f"  OpenRouter: ✓ Configured")
    
    if weaviate_client:
        try:
            collection = weaviate_client.collections.get("NEODocument")
            response = collection.aggregate.over_all(total_count=True)
            count = response.total_count
            print(f"\n{'✓'*35}")
            print(f"✓ RAG System Ready! Documents: {count}")
            print(f"{'✓'*35}")
        except:
            print(f"\n{'✓'*35}")
            print(f"✓ RAG System Ready (empty)")
            print(f"{'✓'*35}")
    else:
        print(f"\n{'⚠'*35}")
        print(f"⚠ Using In-Memory Fallback Storage")
        print(f"{'⚠'*35}")
    
    print(f"\n{'='*70}")
    print("Server starting on http://0.0.0.0:8000")
    print(f"{'='*70}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)