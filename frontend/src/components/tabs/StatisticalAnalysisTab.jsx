import React, { useState } from 'react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Telescope, Building, Radio, AlertTriangle, Map, Zap, CheckCircle, Database, Eye } from 'lucide-react';
import { RISK_COLORS } from '../../utils/constants';

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

const StatisticalAnalysisTab = ({ data }) => {
  const [selectedNEO, setSelectedNEO] = useState(null);

  if (!data || !data.top_50_risks) return null;

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

  const observingTargets = data.top_50_risks.map(neo => ({
    ...neo,
    ...calculateObservingPriority(neo)
  })).sort((a, b) => b.priority - a.priority);

  return (
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
            {data.top_50_risks.slice(0, 15).map((neo) => (
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
  );
};

export default StatisticalAnalysisTab;
