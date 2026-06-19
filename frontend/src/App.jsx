import React, { useState, useEffect } from 'react';
import RiskTab from './components/tabs/RiskTab';
import StatisticalAnalysisTab from './components/tabs/StatisticalAnalysisTab';
import PredictTab from './components/tabs/PredictTab';
import ChatTab from './components/tabs/ChatTab';
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Activity, Database, AlertTriangle, Rocket, Target, TrendingUp, MapPin, Zap, CheckCircle, XCircle, Loader, Eye, Map, Building, Radio, Telescope, RefreshCw, Brain, BarChart3 } from 'lucide-react';
import Swal from 'sweetalert2';
import withReactContent from 'sweetalert2-react-content';

// Optional: if you want React content inside alerts
const MySwal = withReactContent(Swal);
const API_BASE = 'http://localhost:8000';

const AdvancedNEODashboard = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [days, setDays] = useState(30);
  const [activeTab, setActiveTab] = useState('risk');
  const [modelStatus, setModelStatus] = useState(null);
  
  const [predictionInput, setPredictionInput] = useState({
    absolute_magnitude: 22.0,
    estimated_diameter_min: 0.15,
    estimated_diameter_max: 0.25,
    relative_velocity: 50000,
    miss_distance: 10000000
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState(null);
  const [apiError, setApiError] = useState(null);
  
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [useAgent, setUseAgent] = useState(true);
  const [kbStatus, setKbStatus] = useState(null);
  const [autoIndexing, setAutoIndexing] = useState(false);

  const [selectedNEO, setSelectedNEO] = useState(null);
  
  const [featureImportance, setFeatureImportance] = useState(null);
  const [featureImportanceLoading, setFeatureImportanceLoading] = useState(false);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [modelMetricsLoading, setModelMetricsLoading] = useState(false);

  useEffect(() => {
    fetchAdvancedData();
    checkModelStatus();
    checkKnowledgeBaseStatus();
    fetchFeatureImportance();
    fetchModelMetrics();
  }, [days]);

  const fetchModelMetrics = async () => {
    setModelMetricsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/neo/model-metrics`);
      if (response.ok) {
        const data = await response.json();
        setModelMetrics(data);
        console.log('Model Metrics:', data);
      }
    } catch (err) {
      console.error('Error fetching model metrics:', err);
    } finally {
      setModelMetricsLoading(false);
    }
  };

  const fetchFeatureImportance = async () => {
    setFeatureImportanceLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/neo/feature-importance`);
      if (response.ok) {
        const data = await response.json();
        setFeatureImportance(data);
        console.log('Feature Importance:', data);
      }
    } catch (err) {
      console.error('Error fetching feature importance:', err);
    } finally {
      setFeatureImportanceLoading(false);
    }
  };

  const checkModelStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/neo/model-status`);
      if (response.ok) {
        const status = await response.json();
        setModelStatus(status);
        console.log('Model Status:', status);
      } else {
        console.error('Failed to fetch model status');
        setModelStatus({ ready_for_predictions: false });
      }
    } catch (err) {
      console.error('Error checking model status:', err);
      setModelStatus({ ready_for_predictions: false });
    }
  };

  const checkKnowledgeBaseStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/rag/kb-status`);
      if (response.ok) {
        const status = await response.json();
        setKbStatus(status);
        console.log('Status:', status);
      }
    } catch (err) {
      console.error('Error checking KB status:', err);
    }
  };

  const handleChatSubmit = async () => {
    if (!chatInput.trim() || chatLoading) return;
    
    const userMessage = { role: 'user', content: chatInput };
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);
    
    try {
      const endpoint = useAgent ? `${API_BASE}/api/agent/query` : `${API_BASE}/api/rag/query`;
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: chatInput })
      });
      
      if (!response.ok) {
        throw new Error('Failed to get response');
      }
      
      const result = await response.json();
      const assistantMessage = { 
        role: 'assistant', 
        content: result.answer,
        sources: result.sources 
      };
      setChatMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Chat error:', err);
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error processing your question. Please try again.' 
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setChatLoading(false);
    }
  };

  const autoIndexFromNASA = async () => {
    if (!data) return;
    
    setAutoIndexing(true);
    try {
      const response = await fetch(`${API_BASE}/api/rag/auto-index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const result = await response.json();
        Swal.fire({
          icon: 'success',
          title: 'Success!',
          text: `Successfully indexed ${result.indexed_count} NEOs into knowledge base!`,
        });
        checkKnowledgeBaseStatus();
      } else {
        throw new Error('Failed to index data');
      }
      } catch (err) {
        console.error('Indexing error:', err);
        Swal.fire({
          icon: 'error',
          title: 'Error',
          text: 'Failed to index data. Please ensure backend is running.',
        });
      } finally {
        setAutoIndexing(false);
      }
  };
  const indexCurrentData = async () => {
    if (!data) return;
    
    try {
      const response = await fetch(`${API_BASE}/api/rag/index-neos`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          neos: data.top_50_risks.map(neo => ({
            neo_id: neo.neo_id,
            name: neo.name,
            date: neo.date,
            risk_score: neo.risk_score,
            risk_category: neo.risk_category,
            diameter_km: neo.diameter_km,
            velocity_kms: neo.velocity_kms,
            miss_distance_km: neo.miss_distance_km,
            kinetic_energy_mt: neo.kinetic_energy_mt,
            is_hazardous: neo.is_hazardous
          }))
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        alert(`Successfully indexed ${result.indexed_count} NEOs into knowledge base!`);
        checkKnowledgeBaseStatus();
      }
    } catch (err) {
      console.error('Indexing error:', err);
      alert('Failed to index data');
    }
  };

  const handlePrediction = async () => {
    setPredicting(true);
    setPredictionResult(null);
    setPredictionError(null);
    
    const errors = [];
    if (predictionInput.absolute_magnitude < 0 || predictionInput.absolute_magnitude > 35) {
      errors.push('Absolute magnitude must be between 0 and 35');
    }
    if (predictionInput.estimated_diameter_min <= 0 || predictionInput.estimated_diameter_max <= 0) {
      errors.push('Diameters must be positive values');
    }
    if (predictionInput.estimated_diameter_min > predictionInput.estimated_diameter_max) {
      errors.push('Minimum diameter cannot exceed maximum diameter');
    }
    if (predictionInput.relative_velocity <= 0) {
      errors.push('Relative velocity must be positive');
    }
    if (predictionInput.miss_distance <= 0) {
      errors.push('Miss distance must be positive');
    }

    if (errors.length > 0) {
      setPredictionError(errors.join('. '));
      setPredicting(false);
      return;
    }
    
    try {
      console.log('Sending prediction request to:', `${API_BASE}/api/neo/predict`);
      console.log('Payload:', predictionInput);
      
      const response = await fetch(`${API_BASE}/api/neo/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(predictionInput)
      });

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        let errorMessage = 'Prediction request failed';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch (e) {
          errorMessage = `Server returned ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log('Prediction result:', result);
      setPredictionResult(result);
    } catch (err) {
      console.error('Prediction error:', err);
      setPredictionError(err.message || 'Failed to get prediction. Please ensure the backend server is running at http://localhost:8000 and ML models are loaded.');
    } finally {
      setPredicting(false);
    }
  };

  const handleInputChange = (field, value) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      setPredictionInput(prev => ({
        ...prev,
        [field]: numValue
      }));
    }
  };

  const fetchAdvancedData = async () => {
    setLoading(true);
    setApiError(null);
    try {
      const response = await fetch(`${API_BASE}/api/neo/advanced-analytics?days=${days}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch analytics: ${response.status} ${response.statusText}`);
      }
      const result = await response.json();
      setData(result);
    } catch (err) {
      console.error('Error:', err);
      setApiError(err.message || 'Failed to connect to backend server. Please ensure it is running at http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  const loadSampleNEO = (neo) => {
    if (neo) {
      setPredictionInput({
        absolute_magnitude: neo.absolute_magnitude || 22.0,
        estimated_diameter_min: neo.diameter_km * 0.9,
        estimated_diameter_max: neo.diameter_km * 1.1,
        relative_velocity: neo.velocity_kms * 3600,
        miss_distance: neo.miss_distance_km
      });
    }
  };
  const calculateObservingPriority = (neo) => {
    const riskWeight = 0.6;
    const visibilityWeight = 0.4;
    
    const maxRisk = Math.max(...data.top_50_risks.map(n => n.risk_score));
    const normalizedRisk = neo.risk_score / maxRisk;
    
    const visibilityScore = Math.max(0, 1 - (neo.lunar_distances / 10));
    
    const observingPriority = (normalizedRisk * riskWeight) + (visibilityScore * visibilityWeight);
    
    return {
      priority: observingPriority * 100,
      visibility: visibilityScore * 100,
      difficulty: neo.lunar_distances < 1 ? 'Easy' : neo.lunar_distances < 3 ? 'Moderate' : neo.lunar_distances < 6 ? 'Challenging' : 'Very Difficult'
    };
  };

  const calculateDamageZones = (neo) => {
    const energyMT = neo.kinetic_energy_mt;
    
    const blastRadius = Math.pow(energyMT, 0.33) * 2.2;
    const thermalRadius = Math.pow(energyMT, 0.41) * 3.5;
    const radiationRadius = Math.pow(energyMT, 0.19) * 1.5;
    const craterDiameter = neo.diameter_km * 20;
    
    return {
      totalDestruction: blastRadius,
      severeBlast: blastRadius * 1.8,
      moderateBlast: blastRadius * 3.5,
      thermalBurns: thermalRadius,
      lightDamage: blastRadius * 5,
      craterDiameter: craterDiameter,
      affectedArea: Math.PI * Math.pow(blastRadius * 5, 2)
    };
  };
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex items-center justify-center">
        <div className="text-center">
          <Rocket className="w-20 h-20 text-indigo-400 animate-bounce mx-auto mb-4" />
          <p className="text-2xl text-slate-700 font-semibold">Loading Advanced Analytics...</p>
          <p className="text-indigo-500 mt-2">Risk scoring, Monte Carlo, and impact analysis</p>
        </div>
      </div>
    );
  }

  if (apiError) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex items-center justify-center p-6">
        <div className="max-w-md bg-white rounded-2xl shadow-sm p-8 border border-red-100">
          <XCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-slate-800 mb-3 text-center">Connection Error</h2>
          <p className="text-slate-600 mb-6 text-center">{apiError}</p>
          <button
            onClick={fetchAdvancedData}
            className="w-full px-6 py-3 bg-indigo-500 text-white rounded-xl hover:bg-indigo-600 transition-colors font-semibold shadow-sm"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const COLORS = ['#A5B4FC', '#C4B5FD', '#F9A8D4', '#FCD34D', '#86EFAC', '#7DD3FC'];
  const RISK_COLORS = {
    'CRITICAL': '#FCA5A5',
    'HIGH': '#FDBA74',
    'MODERATE': '#FCD34D',
    'LOW': '#86EFAC'
  };
  const observingTargets = data.top_50_risks.map(neo => ({
    ...neo,
    ...calculateObservingPriority(neo)
  })).sort((a, b) => b.priority - a.priority);
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-5xl font-bold text-slate-800 mb-3 flex items-center gap-4">
            <Target className="text-indigo-500" />
            NASA NEO Advanced Risk Analytics
          </h1>
          <p className="text-indigo-600 text-lg">Analysis • Risk Scoring • Predictions • AI Assistant</p>
          
          <div className="mt-6 flex gap-3 items-center flex-wrap">
            <select 
              value={days} 
              onChange={(e) => setDays(parseInt(e.target.value))}
              className="px-5 py-3 bg-white rounded-xl border border-slate-200 text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-300 shadow-sm"
            >
              <option value={7}>7 Days</option>
              <option value={14}>14 Days</option>
              <option value={30}>30 Days</option>
              <option value={60}>60 Days</option>
            </select>
            <button 
              onClick={fetchAdvancedData}
              className="px-5 py-3 bg-indigo-500 text-white rounded-xl hover:bg-indigo-600 transition-colors font-semibold shadow-sm"
            >
              Refresh Analytics
            </button>
          </div>
        </div>

        {data.overall_insights && data.overall_insights.length > 0 && (
          <div className="mb-8 bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
            <h3 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
              <TrendingUp className="text-indigo-500" />
              Key Insights
            </h3>
            <div className="space-y-2">
              {data.overall_insights.map((insight, idx) => (
                <div key={idx} className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-indigo-400 rounded-full mt-2" />
                  <p className="text-slate-600">{insight}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white rounded-2xl p-6 border border-red-100 shadow-sm">
            <AlertTriangle className="text-red-400 mb-2" size={32} />
            <p className="text-red-600 text-sm mb-1 font-medium">Critical Risk</p>
            <p className="text-4xl font-bold text-red-500">
              {data.top_50_risks ? data.top_50_risks.filter(n => n.risk_category === 'CRITICAL').length : 0}
            </p>
          </div>
          
          <div className="bg-white rounded-2xl p-6 border border-orange-100 shadow-sm">
            <Zap className="text-orange-400 mb-2" size={32} />
            <p className="text-orange-600 text-sm mb-1 font-medium">High Risk</p>
            <p className="text-4xl font-bold text-orange-500">
              {data.top_50_risks ? data.top_50_risks.filter(n => n.risk_category === 'HIGH').length : 0}
            </p>
          </div>
          
          <div className="bg-white rounded-2xl p-6 border border-indigo-100 shadow-sm">
            <Target className="text-indigo-400 mb-2" size={32} />
            <p className="text-indigo-600 text-sm mb-1 font-medium">Immediate Follow-up</p>
            <p className="text-4xl font-bold text-indigo-500">
              {data.top_50_risks ? data.top_50_risks.filter(n => n.follow_up_priority === 'IMMEDIATE').length : 0}
            </p>
          </div>
          
          <div className="bg-white rounded-2xl p-6 border border-blue-100 shadow-sm">
            <Activity className="text-blue-400 mb-2" size={32} />
            <p className="text-blue-600 text-sm mb-1 font-medium">High-Risk Periods</p>
            <p className="text-4xl font-bold text-blue-500">
              {data.temporal_clusters ? data.temporal_clusters.length : 0}
            </p>
          </div>
        </div>

        <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
          {['risk', 'statistical-analysis', 'predict', 'chat'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 rounded-xl font-semibold transition-all whitespace-nowrap shadow-sm ${
                activeTab === tab
                  ? 'bg-indigo-500 text-white'
                  : 'bg-white text-indigo-600 border border-indigo-100 hover:bg-indigo-50'
              }`}
            >
              {tab === 'predict' ? 'ML Prediction' : tab === 'chat' ? 'Ask AI' : tab.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
            </button>
          ))}
        </div>

        {activeTab === 'risk' && <RiskTab data={data} />}
        {activeTab === 'statistical-analysis' && <StatisticalAnalysisTab data={data} />}
        {activeTab === 'predict' && (
          <PredictTab 
            predictionInput={predictionInput}
            handleInputChange={handleInputChange}
            predictionError={predictionError}
            handlePrediction={handlePrediction}
            predicting={predicting}
            modelStatus={modelStatus}
            predictionResult={predictionResult}
            loadSampleNEO={loadSampleNEO}
            data={data}
          />
        )}
        {activeTab === 'chat' && (
          <ChatTab 
            autoIndexFromNASA={autoIndexFromNASA}
            autoIndexing={autoIndexing}
            chatMessages={chatMessages}
            kbStatus={kbStatus}
            setChatInput={setChatInput}
            chatInput={chatInput}
            handleChatSubmit={handleChatSubmit}
            chatLoading={chatLoading}
            useAgent={useAgent}
            setUseAgent={setUseAgent}
          />
        )}
      </div>
    </div>
  );
};

export default AdvancedNEODashboard;