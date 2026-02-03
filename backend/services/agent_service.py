"""
LangGraph-based Agentic RAG Service for NASA NEO Analytics

This module implements the "Robot Scientist" agent using LangGraph,
the premium agentic engineering framework from the LangChain ecosystem.

Architecture:
- Uses LangGraph's StateGraph for explicit state management
- Implements ReAct pattern with conditional edges
- Integrates custom tools for NASA data, ML predictions, and knowledge base
- Supports checkpointing for conversation persistence

This implementation showcases intermediate-to-advanced AI engineering skills:
- LangGraph state machines
- Custom tool definitions with Pydantic
- Conditional routing logic
- Async tool execution
- Error handling and fallbacks

Framework: LangGraph >= 0.2.0
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Annotated, TypedDict
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from services.nasa_service import NasaService

# Configure logging
logger = logging.getLogger("langgraph_agent")
logger.setLevel(logging.INFO)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen-2.5-7b-instruct")

# =============================================================================
# SYSTEM PROMPT - Robot Scientist with ReAct Protocol
# =============================================================================

SYSTEM_PROMPT = """### ROLE
You are the "Robot Scientist," an autonomous planetary defense agent powered by LangGraph. Your mission is to analyze Near-Earth Objects (NEOs) using real-time NASA data and predictive machine learning models.

### TOOLS
1. **fetch_live_nasa_feed**: Retrieves current trajectory, velocity, and diameter data for asteroids in a date range.
2. **predict_hazard_xgboost**: Uses a specialized ML model trained on 127,347 historical asteroids to classify if an object is "Hazardous". Returns probability and confidence scores.
3. **search_knowledge_base**: Queries the Weaviate vector database for historical impact events (e.g., Chelyabinsk, Tunguska) for context.

### OPERATIONAL GUIDELINES
1. **DATA PRIORITIZATION**: Always check the live NASA feed first for specific object telemetry.
2. **PREDICTION LOGIC**: Do not calculate kinetic energy manually. Use the `predict_hazard_xgboost` tool. Trust the model's probability score.
3. **ANALOGY & CONTEXT**: Use the knowledge base to compare findings with past events.
4. **NO MANUAL CALCULATIONS**: You are forbidden from performing complex physics math. Use the ML tool.

### REASONING PROTOCOL (ReAct)
- **Thought**: What information is missing? Which tool is best?
- **Action**: Call the appropriate tool.
- **Observation**: Analyze the tool output.
- **Final Answer**: Synthesize findings into a mission-control briefing.

### RESPONSE FORMAT
- Use **bold** for asteroid names and critical metrics
- Use bullet points for key statistics
- Include risk level prominently (CRITICAL/HIGH/MODERATE/LOW)
- Reference the ML model's prediction when available

### TONE
Analytical, urgent but calm, and scientifically rigorous.

### CURRENT DATE
Today is: {current_date}
"""


# =============================================================================
# STATE DEFINITION - LangGraph TypedDict
# =============================================================================

class AgentState(TypedDict):
    """
    State schema for the Robot Scientist agent.
    
    LangGraph uses TypedDict to define the structure of state
    that flows through the graph nodes.
    """
    messages: Annotated[List, add]  # Conversation history (uses reducer)
    tools_used: List[str]           # Track which tools were called
    sources: List[Dict]             # Sources from knowledge base
    iteration: int                  # Current reasoning iteration
    final_answer: Optional[str]     # Final response when done


# =============================================================================
# TOOL DEFINITIONS - LangChain @tool Decorator
# =============================================================================

# These will be populated by the factory function
_kb_search_fn: Optional[Callable] = None
_ml_predict_fn: Optional[Callable] = None


@tool
async def fetch_live_nasa_feed(start_date: str, end_date: str) -> str:
    """
    Fetch real-time asteroid close approach data from NASA's NeoWs API.
    
    Use this tool first when users ask about current or upcoming asteroids.
    Maximum 7 days per request.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (max 7 days from start)
    
    Returns:
        JSON string with asteroid data including name, size, velocity, and miss distance
    """
    try:
        raw_data = await NasaService.fetch_asteroid_feed(start_date, end_date)
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
                    "miss_distance_km": float(approach_data.get("miss_distance", {}).get("kilometers", 0)),
                    "miss_distance_lunar": float(approach_data.get("miss_distance", {}).get("lunar", 0))
                })
        
        asteroids.sort(key=lambda x: x["miss_distance_km"])
        
        return json.dumps({
            "status": "success",
            "data_source": "NASA NeoWs API (Live)",
            "date_range": f"{start_date} to {end_date}",
            "total_count": len(asteroids),
            "closest_approaches": asteroids[:15],
            "hazardous_count": sum(1 for a in asteroids if a["is_potentially_hazardous"])
        }, default=str)
    
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
async def predict_hazard_xgboost(
    absolute_magnitude: float,
    estimated_diameter_min: float,
    estimated_diameter_max: float,
    relative_velocity: float,
    miss_distance: float
) -> str:
    """
    Use the trained XGBoost ML model to predict if an asteroid is potentially hazardous.
    
    The model was trained on 127,347 historical asteroid records and achieved 94.18% accuracy.
    ALWAYS use this tool instead of manual calculations for hazard assessment.
    
    Args:
        absolute_magnitude: Absolute magnitude (H) of the asteroid. Lower = larger. Range: 15-30.
        estimated_diameter_min: Minimum estimated diameter in kilometers
        estimated_diameter_max: Maximum estimated diameter in kilometers
        relative_velocity: Relative velocity in kilometers per hour
        miss_distance: Miss distance in kilometers
    
    Returns:
        JSON string with hazard prediction, probability, confidence, and risk level
    """
    global _ml_predict_fn
    
    if _ml_predict_fn is None:
        return json.dumps({
            "status": "unavailable",
            "message": "XGBoost model not loaded. Report raw NASA metrics instead.",
            "recommendation": "Use NASA's is_potentially_hazardous flag as fallback."
        })
    
    try:
        params = {
            "absolute_magnitude": absolute_magnitude,
            "estimated_diameter_min": estimated_diameter_min,
            "estimated_diameter_max": estimated_diameter_max,
            "relative_velocity": relative_velocity,
            "miss_distance": miss_distance
        }
        result = await _ml_predict_fn(params)
        return json.dumps({
            "status": "success",
            "model": "XGBoost Classifier (trained on 127,347 asteroids)",
            "model_accuracy": "94.18%",
            "prediction": result
        }, default=str)
    
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
async def search_knowledge_base(query: str) -> str:
    """
    Search the Weaviate vector knowledge base for historical NEO data and past events.
    
    Use for historical context, comparisons with events like Chelyabinsk or Tunguska,
    and general NEO concepts.
    
    Args:
        query: Search query to find relevant historical documents
    
    Returns:
        JSON string with matching documents from the knowledge base
    """
    global _kb_search_fn
    
    if _kb_search_fn is None:
        return json.dumps({
            "status": "unavailable",
            "message": "Knowledge base not configured."
        })
    
    try:
        results = await _kb_search_fn(query)
        
        documents = []
        for doc in results[:5]:
            documents.append({
                "name": doc.get("name", "Unknown"),
                "date": doc.get("date"),
                "risk_score": doc.get("risk_score"),
                "risk_category": doc.get("risk_category"),
                "diameter_km": doc.get("diameter_km"),
                "content_preview": doc.get("content", "")[:300]
            })
        
        return json.dumps({
            "status": "success",
            "data_source": "Weaviate Vector Database",
            "document_count": len(documents),
            "documents": documents
        }, default=str)
    
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# =============================================================================
# LANGGRAPH NODES - Processing Functions
# =============================================================================

def create_agent_node(llm_with_tools):
    """
    Create the agent node that decides what to do next.
    
    This node processes the current state and either:
    - Calls a tool (continues the loop)
    - Generates a final response (ends the loop)
    """
    async def agent_node(state: AgentState) -> Dict:
        logger.info(f"🤖 Agent node - Iteration {state.get('iteration', 0) + 1}")
        
        # Build messages with system prompt
        current_date = datetime.now().strftime("%Y-%m-%d")
        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(current_date=current_date))
        
        messages = [system_msg] + state["messages"]
        
        # Invoke the LLM
        response = await llm_with_tools.ainvoke(messages)
        
        # Update iteration count
        new_iteration = state.get("iteration", 0) + 1
        
        return {
            "messages": [response],
            "iteration": new_iteration
        }
    
    return agent_node


async def tool_executor_node(state: AgentState) -> Dict:
    """
    Execute tools requested by the agent.
    
    This node processes tool calls from the last AI message
    and returns the results as ToolMessages.
    """
    logger.info("🔧 Tool executor node")
    
    last_message = state["messages"][-1]
    tools_used = state.get("tools_used", [])
    sources = state.get("sources", [])
    
    tool_messages = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"  → Calling tool: {tool_name}")
            tools_used.append(tool_name)
            
            # Route to appropriate tool
            if tool_name == "fetch_live_nasa_feed":
                result = await fetch_live_nasa_feed.ainvoke(tool_args)
            elif tool_name == "predict_hazard_xgboost":
                result = await predict_hazard_xgboost.ainvoke(tool_args)
            elif tool_name == "search_knowledge_base":
                result = await search_knowledge_base.ainvoke(tool_args)
                # Extract sources from KB results
                try:
                    parsed = json.loads(result)
                    if parsed.get("status") == "success":
                        sources.extend(parsed.get("documents", []))
                except:
                    pass
            else:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})
            
            tool_messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )
    
    return {
        "messages": tool_messages,
        "tools_used": tools_used,
        "sources": sources
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function that determines the next node.
    
    Returns:
        - "tools" if the agent wants to call tools
        - "end" if the agent is ready to respond
    """
    last_message = state["messages"][-1]
    iteration = state.get("iteration", 0)
    
    # Safety limit on iterations
    if iteration >= 5:
        logger.warning("⚠️ Max iterations reached")
        return "end"
    
    # Check if there are tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("  → Routing to tools")
        return "tools"
    
    logger.info("  → Routing to end")
    return "end"


# =============================================================================
# LANGGRAPH AGENT CLASS
# =============================================================================

class LangGraphAgentService:
    """
    LangGraph-based Robot Scientist Agent.
    
    This class encapsulates the entire LangGraph workflow:
    - State management
    - Tool integration
    - ReAct reasoning loop
    - Memory/checkpointing
    """
    
    def __init__(
        self,
        knowledge_base_search_fn: Optional[Callable] = None,
        ml_predict_fn: Optional[Callable] = None
    ):
        """
        Initialize the LangGraph agent.
        
        Args:
            knowledge_base_search_fn: Async function for KB search
            ml_predict_fn: Async function for XGBoost prediction
        """
        global _kb_search_fn, _ml_predict_fn
        _kb_search_fn = knowledge_base_search_fn
        _ml_predict_fn = ml_predict_fn
        
        # Initialize LLM with OpenRouter
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=2000
        )
        
        # Bind tools to LLM
        self.tools = [fetch_live_nasa_feed, predict_hazard_xgboost, search_knowledge_base]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Memory for conversation persistence
        self.memory = MemorySaver()
        
        # Compile the graph with checkpointing
        self.app = self.graph.compile(checkpointer=self.memory)
        
        logger.info("🚀 LangGraph Robot Scientist initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        
        Graph Structure:
        
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
        """
        # Create state graph
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("agent", create_agent_node(self.llm_with_tools))
        graph.add_node("tools", tool_executor_node)
        
        # Set entry point
        graph.set_entry_point("agent")
        
        # Add conditional edge from agent
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # Add edge from tools back to agent
        graph.add_edge("tools", "agent")
        
        return graph
    
    async def process_query(
        self, 
        user_query: str,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process a user query through the LangGraph agent.
        
        Args:
            user_query: Natural language question
            thread_id: Conversation thread ID for memory
        
        Returns:
            Dictionary with answer, tools used, sources, and reasoning steps
        """
        logger.info(f"📥 Processing query: {user_query[:100]}...")
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "tools_used": [],
            "sources": [],
            "iteration": 0,
            "final_answer": None
        }
        
        # Configuration with thread ID for memory
        config = {"configurable": {"thread_id": thread_id}}
        
        # Track reasoning steps
        reasoning_steps = []
        
        try:
            # Stream through the graph
            async for event in self.app.astream(initial_state, config):
                for node_name, node_output in event.items():
                    if node_name == "agent":
                        reasoning_steps.append({
                            "phase": "thought",
                            "node": node_name,
                            "iteration": node_output.get("iteration", 0)
                        })
                    elif node_name == "tools":
                        reasoning_steps.append({
                            "phase": "action_observation",
                            "node": node_name,
                            "tools_called": node_output.get("tools_used", [])[-1:] if node_output.get("tools_used") else []
                        })
            
            # Get final state
            final_state = self.app.get_state(config)
            messages = final_state.values.get("messages", [])
            
            # Extract final answer from last AI message
            final_answer = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_answer = msg.content
                    break
            
            if final_answer is None:
                final_answer = "I apologize, but I couldn't complete the analysis. Please try again."
            
            return {
                "answer": final_answer,
                "tools_used": list(set(final_state.values.get("tools_used", []))),
                "sources": final_state.values.get("sources", []),
                "reasoning_steps": reasoning_steps,
                "iterations": final_state.values.get("iteration", 0),
                "agent_type": "LangGraph Robot Scientist (ReAct)",
                "framework": "LangGraph >= 0.2.0"
            }
        
        except Exception as e:
            logger.error(f"❌ LangGraph agent error: {str(e)}")
            return {
                "answer": f"⚠️ **Mission Control Alert**: Agent encountered an error: {str(e)}",
                "tools_used": [],
                "sources": [],
                "reasoning_steps": reasoning_steps,
                "iterations": 0,
                "error": str(e)
            }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_langgraph_agent(
    knowledge_base_search_fn: Optional[Callable] = None,
    ml_predict_fn: Optional[Callable] = None
) -> LangGraphAgentService:
    """
    Factory function to create a LangGraph-based Robot Scientist agent.
    
    Args:
        knowledge_base_search_fn: Optional async function for KB search
        ml_predict_fn: Optional async function for XGBoost prediction
    
    Returns:
        Configured LangGraphAgentService instance
    """
    return LangGraphAgentService(
        knowledge_base_search_fn=knowledge_base_search_fn,
        ml_predict_fn=ml_predict_fn
    )


# =============================================================================
# LEGACY COMPATIBILITY - Keep old interface working
# =============================================================================

# Alias for backward compatibility
AgentService = LangGraphAgentService
create_agent = create_langgraph_agent
