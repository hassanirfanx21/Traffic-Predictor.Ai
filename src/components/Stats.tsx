"use client";

import { useState } from "react";
import { PredictionResult, CONGESTION_COLORS } from "@/lib/types";

interface StatsProps {
  predictions: PredictionResult[];
}

export default function Stats({ predictions }: StatsProps) {
  const [showDetails, setShowDetails] = useState(false);

  const counts = {
    low: predictions.filter((p) => p.congestion === 0).length,
    moderate: predictions.filter((p) => p.congestion === 1).length,
    high: predictions.filter((p) => p.congestion === 2).length,
  };

  const total = predictions.length || 1; // Avoid div by zero
  
  const getPercentage = (count: number) => Math.round((count / total) * 100);

  return (
    <>
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="text-white/80 font-medium">Traffic Status</h3>
          <button 
            onClick={() => setShowDetails(true)}
            className="text-xs text-white/40 hover:text-white transition-colors"
          >
            See Details &gt;
          </button>
        </div>

        {/* Circular Graph (SVG Donut) */}
        <div className="relative w-40 h-40 mx-auto my-4 group">
          {/* Outer Glow Ring */}
          <div className="absolute -inset-4 rounded-full bg-gradient-to-tr from-blue-500/10 to-purple-500/10 blur-2xl opacity-50 group-hover:opacity-100 transition-opacity duration-700" />
          
          <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90 transform drop-shadow-2xl">
            <defs>
              <linearGradient id="gradLow" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#22c55e" />
                <stop offset="100%" stopColor="#4ade80" />
              </linearGradient>
              <linearGradient id="gradMod" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f59e0b" />
                <stop offset="100%" stopColor="#fbbf24" />
              </linearGradient>
              <linearGradient id="gradHigh" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ef4444" />
                <stop offset="100%" stopColor="#f87171" />
              </linearGradient>
            </defs>

            {/* Background Track */}
            <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="12" />
            
            {/* Segments */}
            {[
              { count: counts.low, color: "url(#gradLow)" },
              { count: counts.moderate, color: "url(#gradMod)" },
              { count: counts.high, color: "url(#gradHigh)" },
            ].reduce((acc, item, i) => {
              const percentage = getPercentage(item.count);
              const circumference = 2 * Math.PI * 42;
              const strokeDasharray = `${(percentage / 100) * circumference} ${circumference}`;
              const strokeDashoffset = -((acc.totalPercentage / 100) * circumference);
              
              acc.elements.push(
                <circle
                  key={i}
                  cx="50"
                  cy="50"
                  r="42"
                  fill="none"
                  stroke={item.color}
                  strokeWidth="12"
                  strokeDasharray={strokeDasharray}
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="round"
                  className="transition-all duration-1000 ease-out hover:stroke-[14] hover:opacity-100 opacity-90 cursor-pointer"
                />
              );
              acc.totalPercentage += percentage;
              return acc;
            }, { elements: [] as JSX.Element[], totalPercentage: 0 }).elements}
          </svg>

          {/* Inner Circle for Donut Chart */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
             <div className="text-center z-10">
                <div className="text-3xl font-bold text-white drop-shadow-[0_0_15px_rgba(255,255,255,0.5)] tracking-tight">{predictions.length}</div>
                <div className="text-[10px] text-white/50 uppercase tracking-[0.2em] font-medium mt-1">Locations</div>
             </div>
             {/* Inner decorative ring */}
             <div className="absolute inset-8 border border-white/10 rounded-full" />
          </div>
        </div>

        {/* Legend */}
        <div className="space-y-3 mt-2">
          {[
            { label: "Low", count: counts.low, color: CONGESTION_COLORS[0] },
            { label: "Moderate", count: counts.moderate, color: CONGESTION_COLORS[1] },
            { label: "High", count: counts.high, color: CONGESTION_COLORS[2] },
          ].map((item) => (
            <div key={item.label} className="flex items-center justify-between group">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full shadow-[0_0_8px]" style={{ backgroundColor: item.color, boxShadow: `0 0 8px ${item.color}` }} />
                <span className="text-sm text-white/60 group-hover:text-white transition-colors">{item.label}</span>
              </div>
              <span className="text-sm font-medium text-white">{item.count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Details Modal */}
      {showDetails && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" onClick={() => setShowDetails(false)}>
          <div 
            className="bg-[#0f172a]/90 backdrop-blur-xl border border-white/10 rounded-2xl w-full max-w-lg max-h-[80vh] flex flex-col shadow-2xl animate-in fade-in zoom-in duration-200" 
            onClick={e => e.stopPropagation()}
          >
            <div className="p-6 border-b border-white/10 flex justify-between items-center">
              <div>
                <h3 className="text-xl font-bold text-white">Traffic Details</h3>
                <p className="text-sm text-white/40">Real-time congestion analysis</p>
              </div>
              <button 
                onClick={() => setShowDetails(false)} 
                className="w-8 h-8 rounded-full bg-white/5 hover:bg-white/10 flex items-center justify-center text-white/60 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>
            
            <div className="p-2 overflow-y-auto custom-scrollbar">
               <table className="w-full text-left border-collapse">
                 <thead className="sticky top-0 bg-[#0f172a] z-10">
                   <tr>
                     <th className="p-3 text-xs font-medium text-white/40 uppercase tracking-wider">Location</th>
                     <th className="p-3 text-xs font-medium text-white/40 uppercase tracking-wider text-right">Status</th>
                     <th className="p-3 text-xs font-medium text-white/40 uppercase tracking-wider text-right">Conf.</th>
                   </tr>
                 </thead>
                 <tbody className="divide-y divide-white/5">
                   {predictions.length === 0 ? (
                     <tr><td colSpan={3} className="p-8 text-center text-white/30">No data available</td></tr>
                   ) : (
                     predictions
                       .sort((a, b) => b.congestion - a.congestion || b.probability - a.probability)
                       .map((p, i) => (
                         <tr key={i} className="group hover:bg-white/5 transition-colors">
                            <td className="p-3 text-sm text-white/80 font-medium">{p.location}</td>
                            <td className="p-3 text-right">
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                p.congestion === 2 ? 'bg-red-500/10 text-red-400 border border-red-500/20' :
                                p.congestion === 1 ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' :
                                'bg-green-500/10 text-green-400 border border-green-500/20'
                              }`}>
                                {p.congestionLabel}
                              </span>
                            </td>
                            <td className="p-3 text-sm text-white/40 text-right font-mono">
                              {(p.probability * 100).toFixed(0)}%
                            </td>
                         </tr>
                       ))
                   )}
                 </tbody>
               </table>
            </div>
            
            <div className="p-4 border-t border-white/10 bg-white/5 text-center text-xs text-white/30">
              Showing top {predictions.length} locations based on current filters
            </div>
          </div>
        </div>
      )}
    </>
  );
}
