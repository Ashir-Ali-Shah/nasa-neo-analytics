"""
Core Calculations Module for NASA NEO Analytics

This module contains all physics and risk calculation functions
extracted for use by both the API endpoints and the Agent service.

These functions are designed to be tool-compatible for the Agentic RAG system.
"""

import math
from typing import Dict, Any

# Constants
LUNAR_DISTANCE_KM = 384400
EARTH_RADIUS_KM = 6371


def calculate_kinetic_energy(diameter_km: float, velocity_kms: float) -> float:
    """
    Calculate kinetic energy of an asteroid impact in megatons of TNT equivalent.
    
    This uses the standard spherical body assumption with average asteroid density.
    
    Args:
        diameter_km: Estimated diameter in kilometers
        velocity_kms: Relative velocity in kilometers per second
    
    Returns:
        Kinetic energy in megatons of TNT equivalent
    
    Example:
        >>> calculate_kinetic_energy(0.1, 20)  # 100m asteroid at 20 km/s
        ~2.5 MT (comparable to a large nuclear weapon)
    """
    density_kg_m3 = 2600  # Average S-type asteroid density
    radius_m = (diameter_km * 1000) / 2
    volume_m3 = (4/3) * math.pi * (radius_m ** 3)
    mass_kg = volume_m3 * density_kg_m3
    velocity_ms = velocity_kms * 1000
    energy_joules = 0.5 * mass_kg * (velocity_ms ** 2)
    energy_mt = energy_joules / 4.184e15  # Convert to megatons TNT
    return energy_mt


def calculate_impact_probability(miss_distance_km: float, diameter_km: float) -> float:
    """
    Calculate simplified impact probability based on proximity and size.
    
    This is a heuristic model, not an orbital mechanics calculation.
    Real impact probabilities require Monte Carlo simulations.
    
    Args:
        miss_distance_km: Closest approach distance in kilometers
        diameter_km: Estimated diameter in kilometers
    
    Returns:
        Probability value between 0 and 1
    """
    normalized_distance = miss_distance_km / LUNAR_DISTANCE_KM
    if normalized_distance < 0.1:
        prob = 1.0 / (1.0 + normalized_distance * 100)
    else:
        prob = math.exp(-normalized_distance * 10) * 0.01
    prob *= (diameter_km / 1.0)
    return min(prob, 1.0)


def calculate_risk_score(
    miss_distance_km: float, 
    kinetic_energy_mt: float, 
    impact_probability: float, 
    w1: float = 0.4, 
    w2: float = 0.35, 
    w3: float = 0.25
) -> float:
    """
    Calculate composite risk score using weighted factors.
    
    The score combines:
    - Proximity (inverse of lunar distances)
    - Destructive potential (log of kinetic energy)
    - Impact probability
    
    Args:
        miss_distance_km: Closest approach distance in kilometers
        kinetic_energy_mt: Kinetic energy in megatons
        impact_probability: Probability of impact (0-1)
        w1: Weight for proximity factor (default 0.4)
        w2: Weight for energy factor (default 0.35)
        w3: Weight for probability factor (default 0.25)
    
    Returns:
        Composite risk score (higher = more dangerous)
        - > 10: CRITICAL
        - > 5: HIGH
        - > 2: MODERATE
        - <= 2: LOW
    """
    lunar_distances = miss_distance_km / LUNAR_DISTANCE_KM
    component1 = w1 * (1.0 / max(lunar_distances, 0.001))
    component2 = w2 * math.log10(max(kinetic_energy_mt, 0.001))
    component3 = w3 * impact_probability
    risk_score = component1 + component2 + component3
    return max(risk_score, 0)


def get_risk_category(risk_score: float) -> str:
    """
    Convert numeric risk score to categorical label.
    
    Args:
        risk_score: Numeric risk score
    
    Returns:
        Category string: CRITICAL, HIGH, MODERATE, or LOW
    """
    if risk_score > 10:
        return "CRITICAL"
    elif risk_score > 5:
        return "HIGH"
    elif risk_score > 2:
        return "MODERATE"
    else:
        return "LOW"


def km_to_lunar_distances(km: float) -> float:
    """Convert kilometers to lunar distances."""
    return km / LUNAR_DISTANCE_KM


def full_risk_assessment(
    diameter_km: float, 
    velocity_kms: float, 
    miss_distance_km: float
) -> Dict[str, Any]:
    """
    Perform a complete risk assessment for an asteroid.
    
    This is the primary tool function used by the Agent.
    
    Args:
        diameter_km: Estimated diameter in kilometers
        velocity_kms: Relative velocity in km/s
        miss_distance_km: Closest approach distance in kilometers
    
    Returns:
        Dictionary containing all risk metrics and category
    """
    kinetic_energy = calculate_kinetic_energy(diameter_km, velocity_kms)
    impact_prob = calculate_impact_probability(miss_distance_km, diameter_km)
    risk_score = calculate_risk_score(miss_distance_km, kinetic_energy, impact_prob)
    risk_category = get_risk_category(risk_score)
    lunar_dist = km_to_lunar_distances(miss_distance_km)
    
    return {
        "diameter_km": round(diameter_km, 4),
        "velocity_kms": round(velocity_kms, 2),
        "miss_distance_km": round(miss_distance_km, 2),
        "lunar_distances": round(lunar_dist, 4),
        "kinetic_energy_mt": round(kinetic_energy, 4),
        "impact_probability": round(impact_prob, 6),
        "risk_score": round(risk_score, 4),
        "risk_category": risk_category,
        "assessment_notes": _generate_assessment_notes(risk_category, lunar_dist, kinetic_energy)
    }


def _generate_assessment_notes(category: str, lunar_dist: float, energy_mt: float) -> str:
    """Generate human-readable assessment notes."""
    notes = []
    
    if lunar_dist < 0.1:
        notes.append("EXTREMELY CLOSE APPROACH - Within 1/10th lunar distance")
    elif lunar_dist < 1:
        notes.append("Close approach - Closer than the Moon")
    elif lunar_dist < 5:
        notes.append("Moderate proximity approach")
    
    if energy_mt > 100:
        notes.append("Extinction-level impact energy")
    elif energy_mt > 10:
        notes.append("Regional destruction potential")
    elif energy_mt > 1:
        notes.append("City-destruction potential")
    elif energy_mt > 0.01:
        notes.append("Localized damage potential")
    
    if category == "CRITICAL":
        notes.append("IMMEDIATE MONITORING REQUIRED")
    elif category == "HIGH":
        notes.append("Priority tracking recommended")
    
    return "; ".join(notes) if notes else "Routine monitoring"
