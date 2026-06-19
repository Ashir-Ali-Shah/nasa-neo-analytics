import React from 'react';
import { Loader, Zap, AlertTriangle, Database } from 'lucide-react';

const PredictTab = ({
  predictionInput,
  handleInputChange,
  predictionError,
  handlePrediction,
  predicting,
  modelStatus,
  predictionResult,
  loadSampleNEO,
  data
}) => {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">Hazard Prediction</h3> 
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center mb-4">
              <h4 className="text-lg font-semibold text-slate-700">Input Parameters</h4>
              {data?.top_50_risks && data.top_50_risks.length > 0 && (
                <button
                  type="button"
                  onClick={() => {
                    const idx = Math.floor(Math.random() * data.top_50_risks.length);
                    loadSampleNEO(data.top_50_risks[idx]);
                  }}
                  className="px-3 py-1.5 bg-purple-50 hover:bg-purple-100 text-purple-700 border border-purple-200 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors"
                >
                  <Database size={14} />
                  Load Sample NEO
                </button>
              )}
            </div>
            
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
  );
};

export default PredictTab;
