import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ScatterChart, Scatter, Cell } from 'recharts';
import { RISK_COLORS } from '../../utils/constants';

const RiskTab = ({ data }) => {
  if (!data || !data.top_50_risks) return null;

  return (
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
  );
};

export default RiskTab;
