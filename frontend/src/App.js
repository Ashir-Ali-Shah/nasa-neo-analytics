import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Activity, Database, AlertTriangle, Rocket, Target, TrendingUp, MapPin, Zap, CheckCircle, XCircle, Loader, Eye, Map, Building, Radio, Telescope, RefreshCw } from 'lucide-react';
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
  const [kbStatus, setKbStatus] = useState(null);
  const [autoIndexing, setAutoIndexing] = useState(false);

  const [selectedNEO, setSelectedNEO] = useState(null);

  useEffect(() => {
    fetchAdvancedData();
    checkModelStatus();
    checkKnowledgeBaseStatus();
  }, [days]);

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
      const response = await fetch(`${API_BASE}/api/rag/query`, {
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
        absolute_magnitude: 22.0,
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
              {data.temporal_clusters ? data.temporal_clusters.filter(c => c.is_high_risk_period).length : 0}
            </p>
          </div>
        </div>

        <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
          {['risk', 'statistical-analysis', 'impact', 'predict', 'chat'].map((tab) => (
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

        {activeTab === 'risk' && data.top_50_risks && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
              <h3 className="text-2xl font-bold text-slate-800 mb-6">Top 50 Highest Risk Objects</h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <div>
                  <h4 className="text-lg font-semibold text-slate-700 mb-4">Risk Category Composition</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart 
                      data={[
                        {
                          name: 'Risk Distribution',
                          CRITICAL: data.top_50_risks.filter(n => n.risk_category === 'CRITICAL').length,
                          HIGH: data.top_50_risks.filter(n => n.risk_category === 'HIGH').length,
                          MODERATE: data.top_50_risks.filter(n => n.risk_category === 'MODERATE').length,
                          LOW: data.top_50_risks.filter(n => n.risk_category === 'LOW').length
                        }
                      ]}
                      layout="vertical"
                    >
                      <XAxis 
                        type="number" 
                        tick={{ fill: '#94A3B8', fontSize: 12 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis 
                        type="category" 
                        dataKey="name"
                        tick={{ fill: '#94A3B8', fontSize: 12 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'white', 
                          border: '1px solid #E2E8F0',
                          borderRadius: '12px',
                          color: '#1E293B'
                        }}
                        cursor={{ fill: 'rgba(165, 180, 252, 0.1)' }}
                      />
                      <Legend 
                        wrapperStyle={{ paddingTop: '10px' }}
                        iconType="square"
                      />
                      <Bar dataKey="CRITICAL" stackId="a" fill={RISK_COLORS.CRITICAL} radius={[0, 0, 0, 0]} name="Critical" />
                      <Bar dataKey="HIGH" stackId="a" fill={RISK_COLORS.HIGH} radius={[0, 0, 0, 0]} name="High" />
                      <Bar dataKey="MODERATE" stackId="a" fill={RISK_COLORS.MODERATE} radius={[0, 0, 0, 0]} name="Moderate" />
                      <Bar dataKey="LOW" stackId="a" fill={RISK_COLORS.LOW} radius={[0, 8, 8, 0]} name="Low" />
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="mt-4 grid grid-cols-4 gap-2">
                    {['CRITICAL', 'HIGH', 'MODERATE', 'LOW'].map((category) => {
                      const count = data.top_50_risks.filter(n => n.risk_category === category).length;
                      const percentage = ((count / data.top_50_risks.length) * 100).toFixed(1);
                      return (
                        <div 
                          key={category} 
                          className="p-3 rounded-lg border-2"
                          style={{ 
                            backgroundColor: RISK_COLORS[category] + '20',
                            borderColor: RISK_COLORS[category]
                          }}
                        >
                          <p className="text-xs font-semibold text-slate-600 mb-1">{category}</p>
                          <p className="text-2xl font-bold" style={{ color: RISK_COLORS[category] }}>
                            {count}
                          </p>
                          <p className="text-xs text-slate-500 mt-1">{percentage}%</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div>
                  <h4 className="text-lg font-semibold text-slate-700 mb-4">Risk Score Distribution</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data.top_50_risks.slice(0, 20)}>
                      <XAxis 
                        dataKey="name" 
                        tick={{ fill: '#94A3B8', fontSize: 10 }}
                        angle={-45}
                        textAnchor="end"
                        height={100}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis 
                        tick={{ fill: '#94A3B8' }} 
                        axisLine={false}
                        tickLine={false}
                        grid={false}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'white', 
                          border: '1px solid #E2E8F0',
                          borderRadius: '12px',
                          color: '#1E293B'
                        }}
                      />
                      <Bar dataKey="risk_score" fill="#A5B4FC" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div>
                  <h4 className="text-lg font-semibold text-slate-700 mb-4">Threat Matrix: Risk vs Impact Energy</h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
                      <XAxis 
                        type="number" 
                        dataKey="risk_score" 
                        name="Risk Score"
                        tick={{ fill: '#64748B', fontSize: 12 }}
                        label={{ 
                          value: 'Risk Score', 
                          position: 'insideBottom', 
                          offset: -10, 
                          fill: '#475569',
                          style: { fontSize: 14, fontWeight: 600 }
                        }}
                        domain={['auto', 'auto']}
                        axisLine={{ stroke: '#E2E8F0' }}
                        tickLine={{ stroke: '#E2E8F0' }}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="kinetic_energy_mt" 
                        name="Kinetic Energy (MT)"
                        tick={{ fill: '#64748B', fontSize: 12 }}
                        label={{ 
                          value: 'Kinetic Energy (MT)', 
                          angle: -90, 
                          position: 'insideLeft', 
                          fill: '#475569',
                          style: { fontSize: 14, fontWeight: 600 }
                        }}
                        domain={[0, 'auto']}
                        axisLine={{ stroke: '#E2E8F0' }}
                        tickLine={{ stroke: '#E2E8F0' }}
                      />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3', stroke: '#A5B4FC' }}
                        contentStyle={{ 
                          backgroundColor: 'white', 
                          border: '2px solid #E2E8F0',
                          borderRadius: '12px',
                          padding: '12px',
                          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                        }}
                        labelStyle={{ color: '#1F2937', fontWeight: 600, marginBottom: 8 }}
                        itemStyle={{ color: '#6B7280', fontSize: 13 }}
                        formatter={(value, name) => [
                          typeof value === 'number' ? value.toFixed(4) : value,
                          name === 'risk_score' ? 'Risk Score' : 'Energy (MT)'
                        ]}
                        labelFormatter={(value, payload) => {
                          if (payload && payload.length > 0) {
                            return `${payload[0].payload.name || 'Unknown'}`;
                          }
                          return '';
                        }}
                      />
                      <Legend 
                        wrapperStyle={{ paddingTop: '20px' }}
                        iconType="circle"
                      />
                      <Scatter 
                        name="NEO Threat Analysis" 
                        data={data.top_50_risks} 
                        fill="#A5B4FC"
                      >
                        {data.top_50_risks.map((entry, index) => {
                          const color = RISK_COLORS[entry.risk_category] || '#A5B4FC';
                          return (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={color}
                              stroke="#fff"
                              strokeWidth={2}
                            />
                          );
                        })}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                  <div className="mt-4 p-4 bg-red-50 rounded-lg border border-red-200">
                    <p className="text-sm text-red-900">
                      <span className="font-semibold">Threat Matrix:</span> Objects in the upper-right represent both high risk scores and high impact energy (highest threat). Upper-left shows high energy but lower risk score. Lower-right shows high risk but lower energy.
                    </p>
                  </div>
                </div>

                <div>
                  <h4 className="text-lg font-semibold text-slate-700 mb-4">Close Approach: Distance vs Velocity</h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
                      <XAxis 
                        type="number" 
                        dataKey="lunar_distances" 
                        name="Distance (LD)"
                        tick={{ fill: '#64748B', fontSize: 12 }}
                        label={{ 
                          value: 'Miss Distance (Lunar Distances)', 
                          position: 'insideBottom', 
                          offset: -10, 
                          fill: '#475569',
                          style: { fontSize: 14, fontWeight: 600 }
                        }}
                        domain={[0, 'auto']}
                        axisLine={{ stroke: '#E2E8F0' }}
                        tickLine={{ stroke: '#E2E8F0' }}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="velocity_kms" 
                        name="Velocity (km/s)"
                        tick={{ fill: '#64748B', fontSize: 12 }}
                        label={{ 
                          value: 'Velocity (km/s)', 
                          angle: -90, 
                          position: 'insideLeft', 
                          fill: '#475569',
                          style: { fontSize: 14, fontWeight: 600 }
                        }}
                        domain={['auto', 'auto']}
                        axisLine={{ stroke: '#E2E8F0' }}
                        tickLine={{ stroke: '#E2E8F0' }}
                      />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3', stroke: '#7DD3FC' }}
                        contentStyle={{ 
                          backgroundColor: 'white', 
                          border: '2px solid #E2E8F0',
                          borderRadius: '12px',
                          padding: '12px',
                          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                        }}
                        labelStyle={{ color: '#1F2937', fontWeight: 600, marginBottom: 8 }}
                        itemStyle={{ color: '#6B7280', fontSize: 13 }}
                        formatter={(value, name) => [
                          typeof value === 'number' ? value.toFixed(4) : value,
                          name === 'lunar_distances' ? 'Distance (LD)' : 'Velocity (km/s)'
                        ]}
                        labelFormatter={(value, payload) => {
                          if (payload && payload.length > 0) {
                            const neo = payload[0].payload;
                            return `${neo.name || 'Unknown'} (Ø ${neo.diameter_km?.toFixed(4)} km)`;
                          }
                          return '';
                        }}
                      />
                      <Legend 
                        wrapperStyle={{ paddingTop: '20px' }}
                        iconType="circle"
                      />
                      <Scatter 
                        name="Close Approach Analysis" 
                        data={data.top_50_risks} 
                        fill="#7DD3FC"
                      >
                        {data.top_50_risks.map((entry, index) => {
                          const sizeScale = Math.sqrt(entry.diameter_km) * 80;
                          const color = RISK_COLORS[entry.risk_category] || '#7DD3FC';
                          return (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={color}
                              stroke="#fff"
                              strokeWidth={2}
                              r={Math.max(4, Math.min(sizeScale, 12))}
                            />
                          );
                        })}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                  <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <p className="text-sm text-blue-900">
                      <span className="font-semibold">Proximity Analysis:</span> Objects in the lower-left corner are closest and slowest (potentially easier to track). Upper-left shows close but fast-moving objects (higher monitoring priority). Point size represents asteroid diameter.
                    </p>
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50">
                      <th className="text-left py-4 px-4 text-sm font-bold text-slate-700">Rank</th>
                      <th className="text-left py-4 px-4 text-sm font-bold text-slate-700">NEO Name</th>
                      <th className="text-left py-4 px-4 text-sm font-bold text-slate-700">Risk Category</th>
                      <th className="text-left py-4 px-4 text-sm font-bold text-slate-700 min-w-[180px]">Risk Score</th>
                      <th className="text-left py-4 px-4 text-sm font-bold text-slate-700 min-w-[180px]">Energy (MT)</th>
                      <th className="text-left py-4 px-4 text-sm font-bold text-slate-700 min-w-[180px]">Distance (LD)</th>
                      <th className="text-left py-4 px-4 text-sm font-bold text-slate-700 min-w-[180px]">Velocity (km/s)</th>
                      <th className="text-left py-4 px-4 text-sm font-bold text-slate-700 min-w-[180px]">Diameter (km)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.top_50_risks.map((neo, idx) => {
                      const maxRiskScore = Math.max(...data.top_50_risks.map(n => n.risk_score));
                      const maxEnergy = Math.max(...data.top_50_risks.map(n => n.kinetic_energy_mt));
                      const maxDistance = Math.max(...data.top_50_risks.map(n => n.lunar_distances));
                      const maxVelocity = Math.max(...data.top_50_risks.map(n => n.velocity_kms));
                      const maxDiameter = Math.max(...data.top_50_risks.map(n => n.diameter_km));
                      
                      const riskPercent = (neo.risk_score / maxRiskScore) * 100;
                      const energyPercent = (neo.kinetic_energy_mt / maxEnergy) * 100;
                      const distancePercent = 100 - ((neo.lunar_distances / maxDistance) * 100);
                      const velocityPercent = (neo.velocity_kms / maxVelocity) * 100;
                      const diameterPercent = (neo.diameter_km / maxDiameter) * 100;
                      
                      return (
                        <tr 
                          key={neo.neo_id} 
                          className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
                        >
                          <td className="py-3 px-4">
                            <div className="bg-indigo-400 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold text-sm">
                              {idx + 1}
                            </div>
                          </td>
                          <td className="py-3 px-4 text-slate-800 font-semibold">{neo.name}</td>
                          <td className="py-3 px-4">
                            <span 
                              className="px-3 py-1 rounded-full text-xs font-bold inline-block"
                              style={{ 
                                backgroundColor: RISK_COLORS[neo.risk_category] + '40', 
                                color: neo.risk_category === 'CRITICAL' ? '#DC2626' : 
                                       neo.risk_category === 'HIGH' ? '#EA580C' : 
                                       neo.risk_category === 'MODERATE' ? '#CA8A04' : '#16A34A'
                              }}
                            >
                              {neo.risk_category}
                            </span>
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-2">
                              <div className="flex-1">
                                <div className="w-full bg-slate-100 rounded-full h-2">
                                  <div 
                                    className="bg-gradient-to-r from-red-300 to-red-400 h-2 rounded-full transition-all"
                                    style={{ width: `${riskPercent}%` }}
                                  />
                                </div>
                              </div>
                              <span className="text-xs font-bold text-slate-700 min-w-[45px] text-right">
                                {neo.risk_score.toFixed(2)}
                              </span>
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-2">
                              <div className="flex-1">
                                <div className="w-full bg-slate-100 rounded-full h-2">
                                  <div 
                                    className="bg-gradient-to-r from-orange-300 to-orange-400 h-2 rounded-full transition-all"
                                    style={{ width: `${energyPercent}%` }}
                                  />
                                </div>
                              </div>
                              <span className="text-xs font-bold text-slate-700 min-w-[45px] text-right">
                                {neo.kinetic_energy_mt.toFixed(2)}
                              </span>
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-2">
                              <div className="flex-1">
                                <div className="w-full bg-slate-100 rounded-full h-2">
                                  <div 
                                    className="bg-gradient-to-r from-amber-300 to-amber-400 h-2 rounded-full transition-all"
                                    style={{ width: `${distancePercent}%` }}
                                  />
                                </div>
                              </div>
                              <span className="text-xs font-bold text-slate-700 min-w-[45px] text-right">
                                {neo.lunar_distances.toFixed(3)}
                              </span>
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-2">
                              <div className="flex-1">
                                <div className="w-full bg-slate-100 rounded-full h-2">
                                  <div 
                                    className="bg-gradient-to-r from-blue-300 to-blue-400 h-2 rounded-full transition-all"
                                    style={{ width: `${velocityPercent}%` }}
                                  />
                                </div>
                              </div>
                              <span className="text-xs font-bold text-slate-700 min-w-[45px] text-right">
                                {neo.velocity_kms.toFixed(2)}
                              </span>
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-2">
                              <div className="flex-1">
                                <div className="w-full bg-slate-100 rounded-full h-2">
                                  <div 
                                    className="bg-gradient-to-r from-indigo-300 to-indigo-400 h-2 rounded-full transition-all"
                                    style={{ width: `${diameterPercent}%` }}
                                  />
                                </div>
                              </div>
                              <span className="text-xs font-bold text-slate-700 min-w-[45px] text-right">
                                {neo.diameter_km.toFixed(4)}
                              </span>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

{activeTab === 'statistical-analysis' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 border-2 border-indigo-100 shadow-lg">
              <div className="flex items-center gap-3 mb-6">
                <Telescope className="text-indigo-500" size={32} />
                <div>
                  <h3 className="text-2xl font-bold text-slate-800">Amateur Astronomer's Observing Planner</h3>
                  <p className="text-slate-600">Prioritized targets combining scientific urgency with visual accessibility</p>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div>
                  <h4 className="text-lg font-semibold text-slate-700 mb-4">Observing Priority Matrix</h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
                      <XAxis 
                        type="number" 
                        dataKey="visibility" 
                        name="Visibility Score"
                        tick={{ fill: '#64748B', fontSize: 12 }}
                        label={{ 
                          value: 'Visibility Score (0-100)', 
                          position: 'insideBottom', 
                          offset: -10, 
                          fill: '#475569',
                          style: { fontSize: 14, fontWeight: 600 }
                        }}
                        domain={[0, 100]}
                        axisLine={{ stroke: '#E2E8F0' }}
                        tickLine={{ stroke: '#E2E8F0' }}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="priority" 
                        name="Observing Priority"
                        tick={{ fill: '#64748B', fontSize: 12 }}
                        label={{ 
                          value: 'Observing Priority (0-100)', 
                          angle: -90, 
                          position: 'insideLeft', 
                          fill: '#475569',
                          style: { fontSize: 14, fontWeight: 600 }
                        }}
                        domain={[0, 100]}
                        axisLine={{ stroke: '#E2E8F0' }}
                        tickLine={{ stroke: '#E2E8F0' }}
                      />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3', stroke: '#6366F1' }}
                        contentStyle={{ 
                          backgroundColor: 'white', 
                          border: '2px solid #E2E8F0',
                          borderRadius: '12px',
                          padding: '12px',
                          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                        }}
                        formatter={(value, name) => [
                          typeof value === 'number' ? value.toFixed(1) : value,
                          name === 'priority' ? 'Priority' : 'Visibility'
                        ]}
                        labelFormatter={(value, payload) => {
                          if (payload && payload.length > 0) {
                            const neo = payload[0].payload;
                            return `${neo.name} - ${neo.difficulty}`;
                          }
                          return '';
                        }}
                      />
                      <Scatter 
                        name="Observing Targets" 
                        data={observingTargets.slice(0, 30)} 
                        fill="#6366F1"
                      >
                        {observingTargets.slice(0, 30).map((entry, index) => {
                          const color = RISK_COLORS[entry.risk_category] || '#6366F1';
                          return (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={color}
                              stroke="#fff"
                              strokeWidth={2}
                            />
                          );
                        })}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                  <div className="mt-4 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                    <p className="text-sm text-indigo-900">
                      <span className="font-semibold">Upper-right quadrant:</span> High priority & high visibility - ideal observing targets. <span className="font-semibold">Lower-right:</span> Easy to see but lower priority. <span className="font-semibold">Upper-left:</span> Important but challenging observations.
                    </p>
                  </div>
                </div>

                <div>
                  <h4 className="text-lg font-semibold text-slate-700 mb-4">Top 15 Observing Targets</h4>
                  <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
                    {observingTargets.slice(0, 15).map((neo, idx) => (
                      <div 
                        key={neo.neo_id}
                        className="p-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl border border-indigo-200 hover:shadow-md transition-shadow"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <div className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold text-sm flex-shrink-0">
                              {idx + 1}
                            </div>
                            <div>
                              <h5 className="font-bold text-slate-800 text-sm">{neo.name}</h5>
                              <p className="text-xs text-slate-600">{neo.date}</p>
                            </div>
                          </div>
                          <span 
                            className="px-2 py-1 rounded-full text-xs font-bold"
                            style={{ 
                              backgroundColor: RISK_COLORS[neo.risk_category] + '40', 
                              color: neo.risk_category === 'CRITICAL' ? '#DC2626' : 
                                     neo.risk_category === 'HIGH' ? '#EA580C' : 
                                     neo.risk_category === 'MODERATE' ? '#CA8A04' : '#16A34A'
                            }}
                          >
                            {neo.risk_category}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-xs">
                          <div>
                            <p className="text-slate-500 mb-1">Priority Score</p>
                            <div className="flex items-center gap-2">
                              <div className="flex-1 bg-white rounded-full h-2">
                                <div 
                                  className="bg-gradient-to-r from-indigo-400 to-purple-500 h-2 rounded-full"
                                  style={{ width: `${neo.priority}%` }}
                                />
                              </div>
                              <span className="font-bold text-slate-700">{neo.priority.toFixed(0)}</span>
                            </div>
                          </div>
                          <div>
                            <p className="text-slate-500 mb-1">Visibility</p>
                            <div className="flex items-center gap-2">
                              <div className="flex-1 bg-white rounded-full h-2">
                                <div 
                                  className="bg-gradient-to-r from-green-400 to-emerald-500 h-2 rounded-full"
                                  style={{ width: `${neo.visibility}%` }}
                                />
                              </div>
                              <span className="font-bold text-slate-700">{neo.visibility.toFixed(0)}</span>
                            </div>
                          </div>
                          <div className="col-span-2 flex items-center justify-between pt-2 border-t border-indigo-100">
                            <span className="text-slate-600">
                              <Eye size={12} className="inline mr-1" />
                              {neo.difficulty}
                            </span>
                            <span className="text-slate-600">
                              {neo.lunar_distances.toFixed(2)} LD
                            </span>
                            <span className="text-slate-600">
                              Ø {neo.diameter_km.toFixed(3)} km
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-indigo-50 rounded-xl p-5 border border-indigo-200">
                <h4 className="font-semibold text-indigo-900 mb-3 flex items-center gap-2">
                  <Radio size={20} />
                  Observing Guidelines
                </h4>
                <ul className="space-y-2 text-sm text-indigo-800">
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500 font-bold">•</span>
                    <span><strong>Priority Score:</strong> Combines scientific urgency (60%) with visual accessibility (40%)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500 font-bold">•</span>
                    <span><strong>Visibility Score:</strong> Based on distance - closer objects score higher</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500 font-bold">•</span>
                    <span><strong>Difficulty Levels:</strong> Easy (&lt;1 LD), Moderate (1-3 LD), Challenging (3-6 LD), Very Difficult (&gt;6 LD)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500 font-bold">•</span>
                    <span><strong>Contribution:</strong> Your observations help refine orbital calculations and improve planetary defense</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-6 border-2 border-orange-100 shadow-lg">
              <div className="flex items-center gap-3 mb-6">
                <Building className="text-orange-500" size={32} />
                <div>
                  <h3 className="text-2xl font-bold text-slate-800">Civil Planning & Infrastructure Auditor</h3>
                  <p className="text-slate-600">Impact damage zones for emergency preparedness and infrastructure resilience</p>
                </div>
              </div>

              <div className="mb-6">
                <h4 className="text-lg font-semibold text-slate-700 mb-4">Select NEO for Impact Analysis</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                  {data.top_50_risks.slice(0, 15).map((neo, idx) => (
                    <button
                      key={neo.neo_id}
                      onClick={() => setSelectedNEO(neo)}
                      className={`p-3 rounded-xl border-2 transition-all text-left ${
                        selectedNEO?.neo_id === neo.neo_id
                          ? 'border-orange-500 bg-orange-50 shadow-md'
                          : 'border-slate-200 bg-white hover:border-orange-300'
                      }`}
                    >
                      <div className="font-bold text-xs text-slate-800 mb-1 truncate">{neo.name}</div>
                      <div className="text-xs text-slate-600">{neo.kinetic_energy_mt.toFixed(2)} MT</div>
                      <div 
                        className="mt-2 px-2 py-1 rounded text-xs font-bold text-center"
                        style={{ 
                          backgroundColor: RISK_COLORS[neo.risk_category] + '40', 
                          color: neo.risk_category === 'CRITICAL' ? '#DC2626' : 
                                 neo.risk_category === 'HIGH' ? '#EA580C' : 
                                 neo.risk_category === 'MODERATE' ? '#CA8A04' : '#16A34A'
                        }}
                      >
                        {neo.risk_category}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {selectedNEO ? (
                <>
                  <div className="bg-orange-50 rounded-xl p-5 mb-6 border border-orange-200">
                    <h4 className="font-bold text-orange-900 mb-3 flex items-center gap-2">
                      <AlertTriangle size={20} />
                      Selected: {selectedNEO.name}
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-orange-600 mb-1">Impact Energy</p>
                        <p className="text-2xl font-bold text-orange-900">{selectedNEO.kinetic_energy_mt.toFixed(2)} MT</p>
                      </div>
                      <div>
                        <p className="text-orange-600 mb-1">Diameter</p>
                        <p className="text-2xl font-bold text-orange-900">{selectedNEO.diameter_km.toFixed(3)} km</p>
                      </div>
                      <div>
                        <p className="text-orange-600 mb-1">Velocity</p>
                        <p className="text-2xl font-bold text-orange-900">{selectedNEO.velocity_kms.toFixed(2)} km/s</p>
                      </div>
                      <div>
                        <p className="text-orange-600 mb-1">Risk Score</p>
                        <p className="text-2xl font-bold text-orange-900">{selectedNEO.risk_score.toFixed(2)}</p>
                      </div>
                    </div>
                  </div>

                  {(() => {
                    const zones = calculateDamageZones(selectedNEO);
                    return (
                      <>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                          <div>
                            <h4 className="text-lg font-semibold text-slate-700 mb-4">Damage Zone Visualization</h4>
                            <ResponsiveContainer width="100%" height={400}>
                              <RadarChart data={[
                                { zone: 'Total Destruction', radius: zones.totalDestruction, fullMark: zones.lightDamage },
                                { zone: 'Severe Blast', radius: zones.severeBlast, fullMark: zones.lightDamage },
                                { zone: 'Moderate Blast', radius: zones.moderateBlast, fullMark: zones.lightDamage },
                                { zone: 'Thermal Burns', radius: zones.thermalBurns, fullMark: zones.lightDamage },
                                { zone: 'Light Damage', radius: zones.lightDamage, fullMark: zones.lightDamage },
                                { zone: 'Crater', radius: zones.craterDiameter, fullMark: zones.lightDamage }
                              ]}>
                                <PolarGrid stroke="#E2E8F0" />
                                <PolarAngleAxis 
                                  dataKey="zone" 
                                  tick={{ fill: '#64748B', fontSize: 12 }}
                                />
                                <PolarRadiusAxis 
                                  angle={90} 
                                  domain={[0, 'auto']} 
                                  tick={{ fill: '#64748B', fontSize: 10 }}
                                />
                                <Radar 
                                  name="Radius (km)" 
                                  dataKey="radius" 
                                  stroke="#F97316" 
                                  fill="#F97316" 
                                  fillOpacity={0.6} 
                                />
                                <Tooltip 
                                  contentStyle={{ 
                                    backgroundColor: 'white', 
                                    border: '2px solid #E2E8F0',
                                    borderRadius: '12px'
                                  }}
                                />
                              </RadarChart>
                            </ResponsiveContainer>
                          </div>

                          <div>
                            <h4 className="text-lg font-semibold text-slate-700 mb-4">Impact Zone Radii</h4>
                            <div className="space-y-4">
                              {[
                                { name: 'Total Destruction', radius: zones.totalDestruction, color: '#DC2626', desc: 'Complete structural collapse' },
                                { name: 'Severe Blast Damage', radius: zones.severeBlast, color: '#EA580C', desc: 'Major structural damage' },
                                { name: 'Moderate Blast', radius: zones.moderateBlast, color: '#F59E0B', desc: 'Significant property damage' },
                                { name: 'Thermal Radiation', radius: zones.thermalBurns, color: '#F97316', desc: 'Third-degree burns' },
                                { name: 'Light Damage', radius: zones.lightDamage, color: '#FBBF24', desc: 'Broken windows, minor injuries' },
                                { name: 'Crater Diameter', radius: zones.craterDiameter, color: '#78350F', desc: 'Ground zero impact crater' }
                              ].map((zone, idx) => (
                                <div key={idx} className="p-4 rounded-xl border-2" style={{ borderColor: zone.color + '40', backgroundColor: zone.color + '10' }}>
                                  <div className="flex items-center justify-between mb-2">
                                    <h5 className="font-bold text-slate-800 text-sm">{zone.name}</h5>
                                    <span className="text-2xl font-bold" style={{ color: zone.color }}>
                                      {zone.radius.toFixed(1)} km
                                    </span>
                                  </div>
                                  <p className="text-xs text-slate-600">{zone.desc}</p>
                                  <div className="mt-2 w-full bg-slate-100 rounded-full h-2">
                                    <div 
                                      className="h-2 rounded-full transition-all"
                                      style={{ 
                                        width: `${(zone.radius / zones.lightDamage) * 100}%`,
                                        backgroundColor: zone.color
                                      }}
                                    />
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="bg-red-50 rounded-xl p-5 border border-red-200">
                          <h4 className="font-semibold text-red-900 mb-3 flex items-center gap-2">
                            <Map size={20} />
                            Infrastructure Audit Recommendations
                          </h4>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="bg-white rounded-lg p-4">
                              <h5 className="font-bold text-red-800 mb-2 text-sm">Critical Infrastructure Within {zones.totalDestruction.toFixed(1)} km</h5>
                              <ul className="space-y-2 text-xs text-slate-700">
                                <li className="flex items-start gap-2">
                                  <AlertTriangle size={12} className="text-red-500 mt-0.5 flex-shrink-0" />
                                  <span>Evacuate all hospitals, schools, and government buildings</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <AlertTriangle size={12} className="text-red-500 mt-0.5 flex-shrink-0" />
                                  <span>Shut down nuclear facilities and chemical plants</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <AlertTriangle size={12} className="text-red-500 mt-0.5 flex-shrink-0" />
                                  <span>Relocate emergency services and first responders</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <AlertTriangle size={12} className="text-red-500 mt-0.5 flex-shrink-0" />
                                  <span>Complete structural failure expected - no survivability</span>
                                </li>
                              </ul>
                            </div>
                            <div className="bg-white rounded-lg p-4">
                              <h5 className="font-bold text-orange-800 mb-2 text-sm">Moderate Zone ({zones.severeBlast.toFixed(1)} - {zones.moderateBlast.toFixed(1)} km)</h5>
                              <ul className="space-y-2 text-xs text-slate-700">
                                <li className="flex items-start gap-2">
                                  <Zap size={12} className="text-orange-500 mt-0.5 flex-shrink-0" />
                                  <span>Reinforce emergency shelters and basements</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <Zap size={12} className="text-orange-500 mt-0.5 flex-shrink-0" />
                                  <span>Stockpile emergency supplies and medical equipment</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <Zap size={12} className="text-orange-500 mt-0.5 flex-shrink-0" />
                                  <span>Establish backup communication systems</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <Zap size={12} className="text-orange-500 mt-0.5 flex-shrink-0" />
                                  <span>Major structural damage - partial building collapse likely</span>
                                </li>
                              </ul>
                            </div>
                            <div className="bg-white rounded-lg p-4">
                              <h5 className="font-bold text-yellow-800 mb-2 text-sm">Light Damage Zone (up to {zones.lightDamage.toFixed(1)} km)</h5>
                              <ul className="space-y-2 text-xs text-slate-700">
                                <li className="flex items-start gap-2">
                                  <CheckCircle size={12} className="text-yellow-600 mt-0.5 flex-shrink-0" />
                                  <span>Prepare emergency glass replacement services</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <CheckCircle size={12} className="text-yellow-600 mt-0.5 flex-shrink-0" />
                                  <span>Set up triage centers for minor injuries</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <CheckCircle size={12} className="text-yellow-600 mt-0.5 flex-shrink-0" />
                                  <span>Secure outdoor equipment and vehicles</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <CheckCircle size={12} className="text-yellow-600 mt-0.5 flex-shrink-0" />
                                  <span>Most structures intact - broken windows and light debris</span>
                                </li>
                              </ul>
                            </div>
                            <div className="bg-white rounded-lg p-4">
                              <h5 className="font-bold text-slate-800 mb-2 text-sm">Total Affected Area</h5>
                              <div className="space-y-3">
                                <div>
                                  <p className="text-xs text-slate-600 mb-1">Impact Footprint</p>
                                  <p className="text-3xl font-bold text-slate-900">{zones.affectedArea.toFixed(0)} km²</p>
                                  <p className="text-xs text-slate-500 mt-1">
                                    {zones.affectedArea > 10000 ? 'Metropolitan' : zones.affectedArea > 1000 ? 'Major city' : zones.affectedArea > 100 ? 'Urban' : 'Localized'} scale disaster
                                  </p>
                                </div>
                                <div className="pt-3 border-t border-slate-200">
                                  <p className="text-xs text-slate-600 mb-1">Crater Diameter</p>
                                  <p className="text-2xl font-bold text-slate-900">{zones.craterDiameter.toFixed(1)} km</p>
                                  <p className="text-xs text-slate-500 mt-1">Ground zero permanent deformation</p>
                                </div>
                              </div>
                            </div>
                          </div>
                          <div className="mt-4 p-4 bg-white rounded-lg">
                            <h5 className="font-bold text-slate-800 mb-2 text-sm flex items-center gap-2">
                              <Database size={16} />
                              Civil Planning Action Items
                            </h5>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                              <div className="p-3 bg-red-50 rounded-lg">
                                <p className="font-semibold text-red-800 mb-1">Immediate (0-24h)</p>
                                <ul className="space-y-1 text-slate-700">
                                  <li>• Activate emergency operations center</li>
                                  <li>• Begin evacuations of critical zones</li>
                                  <li>• Deploy early warning systems</li>
                                </ul>
                              </div>
                              <div className="p-3 bg-orange-50 rounded-lg">
                                <p className="font-semibold text-orange-800 mb-1">Short-term (1-7 days)</p>
                                <ul className="space-y-1 text-slate-700">
                                  <li>• Establish temporary shelters</li>
                                  <li>• Pre-position emergency supplies</li>
                                  <li>• Coordinate with regional authorities</li>
                                </ul>
                              </div>
                              <div className="p-3 bg-blue-50 rounded-lg">
                                <p className="font-semibold text-blue-800 mb-1">Long-term (7+ days)</p>
                                <ul className="space-y-1 text-slate-700">
                                  <li>• Infrastructure hardening projects</li>
                                  <li>• Public awareness campaigns</li>
                                  <li>• Emergency drill exercises</li>
                                </ul>
                              </div>
                            </div>
                          </div>
                        </div>
                      </>
                    );
                  })()}
                </>
              ) : (
                <div className="h-64 flex items-center justify-center bg-orange-50 rounded-xl border-2 border-dashed border-orange-200">
                  <div className="text-center">
                    <Map className="w-16 h-16 text-orange-300 mx-auto mb-3" />
                    <p className="text-slate-600 font-semibold">Select a NEO above to analyze impact zones</p>
                    <p className="text-slate-500 text-sm mt-1">View detailed damage radii and infrastructure recommendations</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'impact' && data.top_10_impact_analysis && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 border-2 border-purple-100 shadow-lg">
              <h3 className="text-2xl font-bold text-gray-900 mb-6">Impact Probability & Geographic Footprint</h3>
              <p className="text-gray-700 mb-6">Top 10 closest approaches</p>
              
              {data.top_10_impact_analysis.slice(0, 5).map((impact, idx) => (
                <div key={impact.neo_id} className="mb-6 p-5 bg-purple-50 rounded-xl border border-purple-200">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h4 className="text-xl font-bold text-gray-900">{idx + 1}. {impact.name}</h4>
                      <p className="text-purple-700 text-sm mt-1">
                        Impact Probability: {(impact.impact_probability * 100).toFixed(6)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-gray-600 text-sm">Corridor Width</p>
                      <p className="text-2xl font-bold text-gray-900">{(impact.impact_corridor_width_km / 1000).toFixed(0)}k km</p>
                    </div>
                  </div>

                  <div className="mb-4">
                    <p className="text-gray-600 text-sm mb-2">Potential Impact Regions:</p>
                    <div className="flex gap-2 flex-wrap">
                      {impact.potential_impact_locations.map((loc, i) => (
                        <span key={i} className="px-3 py-1 bg-red-50 text-red-700 rounded-full text-sm font-semibold flex items-center gap-2">
                          <MapPin size={14} />
                          {loc}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'predict' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
              <h3 className="text-2xl font-bold text-slate-800 mb-2">Hazard Prediction</h3> 
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="text-lg font-semibold text-slate-700 mb-4">Input Parameters</h4>
                  
                  <div>
                    <label className="block text-slate-600 text-sm mb-2 font-medium">Absolute Magnitude (H)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={predictionInput.absolute_magnitude}
                      onChange={(e) => handleInputChange('absolute_magnitude', e.target.value)}
                      className="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-300"
                      placeholder="e.g., 22.0"
                    />
                    <p className="text-xs text-slate-500 mt-1">Brightness measurement (typically 15-30, lower = larger/brighter)</p>
                  </div>

                  <div>
                    <label className="block text-slate-600 text-sm mb-2 font-medium">Estimated Diameter Min (km)</label>
                    <input
                      type="number"
                      step="0.001"
                      value={predictionInput.estimated_diameter_min}
                      onChange={(e) => handleInputChange('estimated_diameter_min', e.target.value)}
                      className="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-300"
                      placeholder="e.g., 0.15"
                    />
                    <p className="text-xs text-slate-500 mt-1">Minimum estimated diameter in kilometers</p>
                  </div>

                  <div>
                    <label className="block text-slate-600 text-sm mb-2 font-medium">Estimated Diameter Max (km)</label>
                    <input
                      type="number"
                      step="0.001"
                      value={predictionInput.estimated_diameter_max}
                      onChange={(e) => handleInputChange('estimated_diameter_max', e.target.value)}
                      className="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-300"
                      placeholder="e.g., 0.25"
                    />
                    <p className="text-xs text-slate-500 mt-1">Maximum estimated diameter in kilometers</p>
                  </div>

                  <div>
                    <label className="block text-slate-600 text-sm mb-2 font-medium">Relative Velocity (km/h)</label>
                    <input
                      type="number"
                      step="100"
                      value={predictionInput.relative_velocity}
                      onChange={(e) => handleInputChange('relative_velocity', e.target.value)}
                      className="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-300"
                      placeholder="e.g., 50000"
                    />
                    <p className="text-xs text-slate-500 mt-1">Velocity relative to Earth (typical range: 10,000-100,000 km/h)</p>
                  </div>

                  <div>
                    <label className="block text-slate-600 text-sm mb-2 font-medium">Miss Distance (km)</label>
                    <input
                      type="number"
                      step="10000"
                      value={predictionInput.miss_distance}
                      onChange={(e) => handleInputChange('miss_distance', e.target.value)}
                      className="w-full px-4 py-3 bg-white border border-slate-200 rounded-xl text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-300"
                      placeholder="e.g., 10000000"
                    />
                    <p className="text-xs text-slate-500 mt-1">Closest approach distance (Lunar distance ≈ 384,400 km)</p>
                  </div>

                  {predictionError && (
                    <div className="p-4 bg-red-50 border-2 border-red-300 rounded-xl">
                      <p className="text-red-700 text-sm font-medium">{predictionError}</p>
                    </div>
                  )}

                  <button
                    onClick={handlePrediction}
                    disabled={predicting || !modelStatus?.ready_for_predictions}
                    className={`w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-semibold transition-all flex items-center justify-center gap-2 shadow-lg ${
                      predicting || !modelStatus?.ready_for_predictions
                        ? 'opacity-50 cursor-not-allowed'
                        : 'hover:from-purple-700 hover:to-pink-700 hover:shadow-xl'
                    }`}
                  >
                    {predicting ? (
                      <>
                        <Loader className="animate-spin" size={20} />
                        Predicting...
                      </>
                    ) : (
                      <>
                        <Zap size={20} />
                        Predict Hazard Status
                      </>
                    )}
                  </button>
                </div>

                <div>
                  <h4 className="text-lg font-semibold text-slate-700 mb-4">Prediction Result</h4>
                  
                  {predictionResult ? (
                    <div className="space-y-4">
                      <div className={`p-6 rounded-xl border-2 ${
                        predictionResult.is_hazardous 
                          ? 'bg-red-50 border-red-400' 
                          : 'bg-green-50 border-green-400'
                      }`}>
                        <div className="flex items-center gap-3 mb-3">
                          <AlertTriangle className={predictionResult.is_hazardous ? 'text-red-600' : 'text-green-600'} size={32} />
                          <div>
                            <p className={`text-2xl font-bold ${
                              predictionResult.is_hazardous ? 'text-red-800' : 'text-green-800'
                            }`}>
                              {predictionResult.is_hazardous ? 'POTENTIALLY HAZARDOUS' : 'NOT HAZARDOUS'}
                            </p>
                            <p className={`text-sm ${
                              predictionResult.is_hazardous ? 'text-red-700' : 'text-green-700'
                            }`}>
                              Prediction 
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className={`p-4 rounded-xl border-2 ${
                        predictionResult.risk_level === 'CRITICAL' ? 'bg-red-50 border-red-300' :
                        predictionResult.risk_level === 'HIGH' ? 'bg-orange-50 border-orange-300' :
                        predictionResult.risk_level === 'MODERATE' ? 'bg-yellow-50 border-yellow-300' :
                        'bg-green-50 border-green-300'
                      }`}>
                        <p className="text-slate-600 text-sm mb-2">Risk Level</p>
                        <p className={`text-xl font-bold ${
                          predictionResult.risk_level === 'CRITICAL' ? 'text-red-700' :
                          predictionResult.risk_level === 'HIGH' ? 'text-orange-700' :
                          predictionResult.risk_level === 'MODERATE' ? 'text-yellow-700' :
                          'text-green-700'
                        }`}>
                          {predictionResult.risk_level}
                        </p>
                      </div>

                      <div className="p-4 bg-purple-50 rounded-xl border border-purple-200">
                        <p className="text-slate-600 text-sm mb-2 font-semibold">Interpretation</p>
                        <p className="text-slate-800 text-sm leading-relaxed">{predictionResult.interpretation}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="h-full flex items-center justify-center p-12 bg-purple-50 rounded-xl border border-purple-200">
                      <div className="text-center">
                        <Database className="w-16 h-16 text-purple-400 mx-auto mb-4 opacity-50" />
                        <p className="text-slate-800 mb-2">No prediction yet</p>
                        <p className="text-slate-600 text-sm">Enter parameters and click "Predict Hazard Status"</p>
                        {data.top_50_risks && data.top_50_risks.length > 0 && (
                          <button
                            onClick={() => loadSampleNEO(data.top_50_risks[0])}
                            className="mt-4 px-4 py-2 bg-purple-100 text-purple-700 rounded-lg text-sm hover:bg-purple-200 transition-colors"
                          >
                            Load Sample NEO Data
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

{activeTab === 'chat' && (
  <div className="space-y-6">
    <div className="bg-white rounded-2xl border border-slate-200 shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="bg-white/20 backdrop-blur rounded-xl p-3">
              <Rocket className="w-8 h-8" />
            </div>
            <div>
              <h3 className="text-2xl font-bold">NEO AI Assistant</h3>
              <p className="text-indigo-100 text-sm">Ask anything about Near-Earth Objects</p>
            </div>
          </div>
          
        </div>
        <button
          onClick={autoIndexFromNASA}
          disabled={autoIndexing}
          className={`mt-4 px-5 py-3 bg-white text-indigo-600 rounded-xl font-semibold flex items-center gap-2 hover:bg-indigo-50 transition-all ${
            autoIndexing ? 'opacity-70 cursor-not-allowed' : ''
          }`}
        >
          {autoIndexing ? (
            <>
              <Loader className="animate-spin" size={18} />
              Indexing NASA Data...
            </>
          ) : (
            <>
              <RefreshCw size={18} />
              Auto-Index from NASA
            </>
          )}
        </button>
      </div>

      {/* Messages Area */}
      <div className="h-96 overflow-y-auto p-6 bg-gradient-to-b from-slate-50 to-white">
        {chatMessages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-center">
            <div className="max-w-md">
              <div className="bg-indigo-100 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6">
                <Target className="w-10 h-10 text-indigo-600" />
              </div>
              <h4 className="text-xl font-bold text-slate-800 mb-3">
                {kbStatus?.document_count === 0
                  ? "Knowledge base is empty"
                  : "How can I help you today?"}
              </h4>
              <p className="text-slate-600 mb-6">
                {kbStatus?.document_count === 0
                  ? "Click the button above to load asteroid data first."
                  : "Ask about the most dangerous NEOs, risk analysis, impact zones, or any specific asteroid."}
              </p>

              {kbStatus?.document_count > 0 && (
                <div className="grid grid-cols-1 gap-3 max-w-lg mx-auto">
                  {[
                    "What are the most dangerous asteroids approaching Earth?",
                    "Which NEOs require immediate follow-up?",
                    "Show me the highest risk objects this month",
                    "Compare the top 3 critical risk asteroids"
                  ].map((q) => (
                    <button
                      key={q}
                      onClick={() => setChatInput(q)}
                      className="text-left p-4 bg-white border border-slate-200 rounded-xl hover:border-indigo-400 hover:shadow-md transition-all text-sm text-slate-700"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            {chatMessages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-3xl rounded-2xl px-5 py-4 shadow-sm ${
                    msg.role === 'user'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white border border-slate-200 text-slate-800'
                  }`}
                >
                  {/* Render assistant message with proper formatting */}
                  {msg.role === 'assistant' ? (
                    <div className="prose prose-sm max-w-none text-slate-800">
                      {/* Replace markdown-like formatting with proper HTML */}
                      {msg.content
                        .replace(/\*\*(.*?)\*\*/g, '<strong class="text-indigo-700 font-bold">$1</strong>')
                        .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
                        .replace(/^\d+\.\s/gm, (match) => `<strong class="text-indigo-700">${match}</strong>`)
                        .split('\n')
                        .map((line, i) => {
                          if (line.trim() === '') return <br key={i} />;
                          if (line.startsWith('•') || line.startsWith('- '))
                            return (
                              <div key={i} className="flex items-start gap-3 mt-2">
                                <span className="text-indigo-600 mt-1.5">●</span>
                                <span dangerouslySetInnerHTML={{ __html: line.slice(2) }} />
                              </div>
                            );
                          return (
                            <p key={i} className="mb-3 last:mb-0" dangerouslySetInnerHTML={{ __html: line }} />
                          );
                        })}
                    </div>
                  ) : (
                    <p className="text-sm leading-relaxed">{msg.content}</p>
                  )}

                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-slate-200">
                      <p className="text-xs font-semibold text-slate-600 mb-3">
                        📊 Sources ({msg.sources.length})
                      </p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {msg.sources.slice(0, 6).map((source, i) => (
                          <div
                            key={i}
                            className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-3 border border-indigo-100"
                          >
                            <p className="font-bold text-indigo-700 text-xs truncate">
                              {source.name || `NEO ${source.neo_id}`}
                            </p>
                            <div className="flex items-center gap-2 mt-1 text-xs text-slate-600">
                              <span>Risk: {source.risk_score?.toFixed(2)}</span>
                              <span className="px-2 py-0.5 bg-red-100 text-red-700 rounded-full font-bold">
                                {source.risk_category}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {chatLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-slate-200 rounded-2xl px-5 py-4 shadow-sm flex items-center gap-3">
                  <Loader className="animate-spin text-indigo-600" size={20} />
                  <span className="text-slate-600">Thinking...</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-5 bg-white border-t border-slate-200">
        <div className="flex gap-3">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleChatSubmit()}
            placeholder={
              kbStatus?.document_count === 0
                ? "Index data first to enable AI assistant..."
                : "Ask about asteroids, risks, impact zones..."
            }
            className="flex-1 px-5 py-4 bg-slate-50 border border-slate-300 rounded-xl text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent placeholder-slate-400"
            disabled={chatLoading || kbStatus?.document_count === 0}
          />
          <button
            onClick={handleChatSubmit}
            disabled={chatLoading || !chatInput.trim() || kbStatus?.document_count === 0}
            className={`px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold flex items-center gap-3 transition-all shadow-lg ${
              chatLoading || !chatInput.trim() || kbStatus?.document_count === 0
                ? 'opacity-50 cursor-not-allowed'
                : 'hover:from-indigo-700 hover:to-purple-700 hover:shadow-xl'
            }`}
          >
            {chatLoading ? (
              <Loader className="animate-spin" size={22} />
            ) : (
              <Zap size={22} />
            )}
            Send
          </button>
        </div>
      </div>
    </div>
  </div>
)}
      </div>
    </div>
  );
};

export default AdvancedNEODashboard;