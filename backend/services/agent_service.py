"""
Agentic RAG Service for NASA NEO Analytics

This module implements an autonomous AI agent that can:
1. Search the vector knowledge base (historical data)
2. Fetch live data from NASA's NeoWs API
3. Calculate risk assessments using physics models
4. Synthesize information and respond to user queries

Architecture:
- Uses OpenAI-compatible function calling (via OpenRouter)
- Implements a reasoning loop that handles tool execution
- Combines RAG (retrieval) with real-time data access (agentic)

This is designed to showcase intermediate AI engineering skills:
- Clean tool abstractions with JSON schemas
- Robust error handling in the agent loop
- Separation of concerns (tools vs reasoning)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from openai import OpenAI

from core.calculations import full_risk_assessment, km_to_lunar_distances
from services.nasa_service import NasaService

# Configure logging
logger = logging.getLogger("agent_service")
logger.setLevel(logging.INFO)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen-2.5-7b-instruct")

# Initialize OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# =============================================================================
# SYSTEM PROMPT - Engineered for NASA NEO Analyst Agent
# =============================================================================

SYSTEM_PROMPT = """# Role & Identity
You are the **NASA NEO Advanced Analyst**, an elite autonomous AI integrated into the Planetary Defense Coordination Office.
Your mandate is to monitor, analyze, and communicate risks regarding Near-Earth Objects (NEOs).
You have access to live NASA datasets, physics engines, and historical archives.

# Operational Directives
1.  **Data-First Authority:** Never guess or hallucinate orbital parameters. Always query your tools for data. If data is missing for a specific date/ID, state that explicitly.
2.  **Multidimensional Analysis:** When analyzing an object:
    *   **Proximity:** Check miss distance (km) and lunar distances (LD).
    *   **Impact:** Use your calculator to determine kinetic energy and impact probability.
    *   **Context:** Compare it to historical objects in your knowledge base.
3.  **Risk Calibration:**
    *   **CRITICAL:** >80% Hazard Probability or Risk Score > 10.
    *   **HIGH:** Risk Score > 5.
    *   **MODERATE:** Risk Score > 2.
    *   **LOW:** All others.
    *   *Tone:* Be clinical and precise for low risks. Be urgent and directive for high/critical risks.

# Tool Usage Protocols
You have access to the following instruments. Use them proactively:

- **`search_knowledge_base`**:
  *   USE WHEN: The user asks about historical events, specific past data, or general NEO concepts.
  *   DO NOT USE: For real-time updates on future dates (use live feed instead).

- **`fetch_live_nasa_feed`**:
  *   USE WHEN: The user asks "What is approaching this week?", "Any hazards today?", or specific date ranges.
  *   CONSTRAINT: Interval must be <= 7 days. For longer ranges, make multiple calls.

- **`calculate_risk`**:
  *   USE WHEN: You have raw data for an object but need to assess its threat level.
  *   MANDATORY: Always run this for any object passing < 1 Lunar Distance.

# Response Guidelines
- **Structure:** Use clear Markdown headers.
- **Visuals:** Use bullet points for stats.
- **Unit Conversion:** Always provide distance in both km and Lunar Distances (LD).
- **Transparency:** If you used a tool, mention it: "Based on my real-time calculations..."
- **Conciseness:** Be informative but not verbose. Users want actionable intelligence.

# Current Date Context
Today's date is: {current_date}
Use this to interpret relative dates like "this week", "tomorrow", "next few days".
"""

# =============================================================================
# TOOL DEFINITIONS - JSON Schema for OpenAI Function Calling
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the vector knowledge base for historical NEO data, past events, definitions, and archived asteroid information. Use for questions about past data or general NEO concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant NEO documents. Be specific about what you're looking for."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_live_nasa_feed",
            "description": "Fetch real-time asteroid close approach data from NASA's NeoWs API for a specific date range. Maximum 7 days per request. Use for current/future asteroid tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (max 7 days from start)"
                    }
                },
                "required": ["start_date", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_risk",
            "description": "Calculate comprehensive risk assessment for an asteroid using physics models. Returns kinetic energy, impact probability, risk score, and risk category. Use when you have asteroid parameters and need threat assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diameter_km": {
                        "type": "number",
                        "description": "Estimated diameter of the asteroid in kilometers"
                    },
                    "velocity_kms": {
                        "type": "number",
                        "description": "Relative velocity in kilometers per second"
                    },
                    "miss_distance_km": {
                        "type": "number",
                        "description": "Closest approach distance in kilometers"
                    }
                },
                "required": ["diameter_km", "velocity_kms", "miss_distance_km"]
            }
        }
    }
]


# =============================================================================
# AGENT SERVICE CLASS
# =============================================================================

class AgentService:
    """
    Autonomous AI Agent for NASA NEO Analytics.
    
    This agent implements a reasoning loop that:
    1. Receives user query
    2. Decides which tools to use
    3. Executes tools and observes results
    4. Synthesizes a final response
    
    The agent uses OpenAI-compatible function calling for tool selection.
    """
    
    def __init__(self, knowledge_base_search_fn: Optional[Callable] = None):
        """
        Initialize the Agent Service.
        
        Args:
            knowledge_base_search_fn: Async function to search the vector KB.
                                      Should accept (query: str) and return list of docs.
        """
        self.kb_search_fn = knowledge_base_search_fn
        self.max_iterations = 5  # Prevent infinite loops
        logger.info("🤖 AgentService initialized")
    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query through the agent reasoning loop.
        
        This is the main entry point for the agentic RAG system.
        
        Args:
            user_query: The user's natural language question
        
        Returns:
            Dictionary containing:
            - answer: The agent's final response
            - tools_used: List of tools that were called
            - reasoning_steps: Debug info about the agent's process
            - sources: Any sources/documents retrieved
        """
        logger.info(f"📥 Processing query: {user_query[:100]}...")
        
        # Build the system prompt with current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        system_prompt = SYSTEM_PROMPT.format(current_date=current_date)
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        tools_used = []
        reasoning_steps = []
        all_sources = []
        
        # Agent reasoning loop
        for iteration in range(self.max_iterations):
            logger.info(f"🔄 Agent iteration {iteration + 1}/{self.max_iterations}")
            
            try:
                # Call the LLM with tools
                response = openrouter_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=1500,
                    temperature=0.7
                )
                
                assistant_message = response.choices[0].message
                
                # Check if the model wants to call tools
                if assistant_message.tool_calls:
                    # Process each tool call
                    messages.append(assistant_message)
                    
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        logger.info(f"🔧 Tool call: {tool_name}({tool_args})")
                        reasoning_steps.append({
                            "action": "tool_call",
                            "tool": tool_name,
                            "arguments": tool_args
                        })
                        
                        # Execute the tool
                        tool_result, sources = await self._execute_tool(tool_name, tool_args)
                        tools_used.append(tool_name)
                        all_sources.extend(sources)
                        
                        reasoning_steps.append({
                            "action": "tool_result",
                            "tool": tool_name,
                            "result_preview": str(tool_result)[:500]
                        })
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result, default=str)
                        })
                
                else:
                    # No tool calls - model is ready to respond
                    final_answer = assistant_message.content
                    logger.info("✅ Agent completed reasoning")
                    
                    return {
                        "answer": final_answer,
                        "tools_used": list(set(tools_used)),
                        "reasoning_steps": reasoning_steps,
                        "sources": all_sources,
                        "iterations": iteration + 1
                    }
            
            except Exception as e:
                logger.error(f"❌ Agent error in iteration {iteration + 1}: {str(e)}")
                reasoning_steps.append({
                    "action": "error",
                    "error": str(e)
                })
                
                # Try to recover with a fallback response
                if iteration == self.max_iterations - 1:
                    return {
                        "answer": f"I encountered an issue while processing your query. Error: {str(e)}. Please try rephrasing your question.",
                        "tools_used": tools_used,
                        "reasoning_steps": reasoning_steps,
                        "sources": all_sources,
                        "iterations": iteration + 1,
                        "error": str(e)
                    }
        
        # Max iterations reached
        logger.warning("⚠️ Max iterations reached without final answer")
        return {
            "answer": "I apologize, but I couldn't complete the analysis in time. Please try a more specific question.",
            "tools_used": tools_used,
            "reasoning_steps": reasoning_steps,
            "sources": all_sources,
            "iterations": self.max_iterations
        }
    
    async def _execute_tool(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any]
    ) -> tuple[Any, List[Dict]]:
        """
        Execute a tool and return its result.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
        
        Returns:
            Tuple of (tool_result, sources_list)
        """
        sources = []
        
        try:
            if tool_name == "search_knowledge_base":
                return await self._tool_search_kb(tool_args["query"])
            
            elif tool_name == "fetch_live_nasa_feed":
                return await self._tool_fetch_nasa(
                    tool_args["start_date"],
                    tool_args["end_date"]
                )
            
            elif tool_name == "calculate_risk":
                result = self._tool_calculate_risk(
                    tool_args["diameter_km"],
                    tool_args["velocity_kms"],
                    tool_args["miss_distance_km"]
                )
                return result, []
            
            else:
                return {"error": f"Unknown tool: {tool_name}"}, []
        
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {str(e)}")
            return {"error": str(e)}, []
    
    async def _tool_search_kb(self, query: str) -> tuple[Dict, List[Dict]]:
        """Search the knowledge base using the injected search function."""
        if self.kb_search_fn is None:
            return {
                "status": "unavailable",
                "message": "Knowledge base search is not configured"
            }, []
        
        try:
            results = await self.kb_search_fn(query)
            
            sources = []
            documents = []
            
            for doc in results[:5]:  # Limit to top 5
                sources.append({
                    "neo_id": doc.get("neo_id"),
                    "name": doc.get("name"),
                    "risk_category": doc.get("risk_category")
                })
                documents.append({
                    "name": doc.get("name", "Unknown"),
                    "date": doc.get("date"),
                    "risk_score": doc.get("risk_score"),
                    "risk_category": doc.get("risk_category"),
                    "content_preview": doc.get("content", "")[:300]
                })
            
            return {
                "status": "success",
                "document_count": len(documents),
                "documents": documents
            }, sources
        
        except Exception as e:
            return {"status": "error", "error": str(e)}, []
    
    async def _tool_fetch_nasa(
        self, 
        start_date: str, 
        end_date: str
    ) -> tuple[Dict, List[Dict]]:
        """Fetch live data from NASA API."""
        try:
            raw_data = await NasaService.fetch_asteroid_feed(start_date, end_date)
            
            # Parse the response into a clean format
            neo_objects = raw_data.get("near_earth_objects", {})
            
            asteroids = []
            for date, neos in neo_objects.items():
                for neo in neos:
                    approach_data = neo.get("close_approach_data", [{}])[0]
                    diameter = neo.get("estimated_diameter", {}).get("kilometers", {})
                    
                    asteroids.append({
                        "id": neo.get("id"),
                        "name": neo.get("name"),
                        "date": date,
                        "is_potentially_hazardous": neo.get("is_potentially_hazardous_asteroid", False),
                        "absolute_magnitude": neo.get("absolute_magnitude_h"),
                        "diameter_km_min": diameter.get("estimated_diameter_min"),
                        "diameter_km_max": diameter.get("estimated_diameter_max"),
                        "velocity_kph": float(approach_data.get("relative_velocity", {}).get("kilometers_per_hour", 0)),
                        "velocity_kms": float(approach_data.get("relative_velocity", {}).get("kilometers_per_hour", 0)) / 3600,
                        "miss_distance_km": float(approach_data.get("miss_distance", {}).get("kilometers", 0)),
                        "miss_distance_lunar": float(approach_data.get("miss_distance", {}).get("lunar", 0))
                    })
            
            # Sort by miss distance (closest first)
            asteroids.sort(key=lambda x: x["miss_distance_km"])
            
            return {
                "status": "success",
                "date_range": f"{start_date} to {end_date}",
                "total_count": len(asteroids),
                "asteroids": asteroids[:20]  # Return top 20 closest
            }, []
        
        except Exception as e:
            return {"status": "error", "error": str(e)}, []
    
    def _tool_calculate_risk(
        self, 
        diameter_km: float, 
        velocity_kms: float, 
        miss_distance_km: float
    ) -> Dict[str, Any]:
        """Calculate risk assessment using physics models."""
        try:
            assessment = full_risk_assessment(
                diameter_km=diameter_km,
                velocity_kms=velocity_kms,
                miss_distance_km=miss_distance_km
            )
            return {
                "status": "success",
                "assessment": assessment
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_agent(knowledge_base_search_fn: Optional[Callable] = None) -> AgentService:
    """
    Factory function to create an AgentService instance.
    
    Args:
        knowledge_base_search_fn: Optional async function for KB search
    
    Returns:
        Configured AgentService instance
    """
    return AgentService(knowledge_base_search_fn=knowledge_base_search_fn)
