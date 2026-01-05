"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { BatchPredictionResponse, PredictionResult, WeatherData } from "@/lib/types";
import { getMapCenter } from "@/lib/locations";
import TimeControls from "./TimeControls";
import WeatherPanel from "./WeatherPanel";
import Stats from "./Stats";
import Footer from "./Footer";

// Dynamic import for Map
const Map = dynamic(() => import("./Map"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center text-white/50">
      Loading map...
    </div>
  ),
});

export default function Dashboard() {
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [timestamp, setTimestamp] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Control states
  const [hour, setHour] = useState(() => new Date().getHours());
  const [day, setDay] = useState(() => {
    const d = new Date().getDay();
    return d === 0 ? 7 : d;
  });
  const [incidentFlag, setIncidentFlag] = useState(false);
  const [limit, setLimit] = useState(50);

  const mapCenter = getMapCenter();

  const fetchPredictions = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        hour: hour.toString(),
        day: day.toString(),
        incident: incidentFlag ? "1" : "0",
      });
      
      const response = await fetch(`/api/predict?${params}`);
      
      if (!response.ok) throw new Error(`API error: ${response.status}`);
      
      const data: BatchPredictionResponse = await response.json();
      setPredictions(data.predictions);
      setWeather(data.weather);
      setTimestamp(data.timestamp);
    } catch (err) {
      console.error("Failed to fetch predictions:", err);
      setError(err instanceof Error ? err.message : "Failed to load predictions");
    } finally {
      setIsLoading(false);
    }
  }, [hour, day, incidentFlag]);

  useEffect(() => {
    fetchPredictions();
  }, [fetchPredictions]);

  return (
    <div className="relative lg:h-screen min-h-screen w-full bg-gradient-to-br from-slate-900 via-slate-800 to-black text-white lg:overflow-hidden overflow-x-hidden font-sans selection:bg-blue-500/30">
      {/* Background Ambience */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-blue-600/20 rounded-full blur-[120px] animate-pulse duration-[4000ms]" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-purple-600/20 rounded-full blur-[120px] animate-pulse duration-[5000ms]" />
      <div className="absolute top-[40%] left-[40%] w-[30%] h-[30%] bg-cyan-500/10 rounded-full blur-[100px] animate-pulse duration-[7000ms]" />

      <div className="relative z-10 flex flex-col lg:flex-row lg:h-full h-auto p-4 lg:p-6 gap-6">
        
        {/* Left Sidebar (Controls & Stats) - Fixed Width */}
        <aside className="w-full lg:w-[500px] flex flex-col gap-5 shrink-0 lg:h-full h-auto overflow-y-auto no-scrollbar pb-2">
          {/* Logo / Title */}
          <div className="px-2 shrink-0 flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-light tracking-tight">Traffic<span className="font-bold text-blue-400">.AI</span></h1>
              <p className="text-white/40 text-sm mt-1">Traffic Intelligence</p>
            </div>
            <div className="lg:hidden">
              {/* Mobile Menu Button Placeholder if needed */}
            </div>
          </div>

          {/* Weather Panel */}
          <div className="shrink-0">
             <WeatherPanel weather={weather} timestamp={timestamp} />
          </div>

          {/* Stats Card */}
          <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6 shadow-2xl shrink-0">
            <Stats predictions={predictions} />
          </div>

          {/* Time Controls */}
          <div className="shrink-0">
            <TimeControls 
              hour={hour}
              day={day}
              incidentFlag={incidentFlag}
              limit={limit}
              onHourChange={setHour}
              onDayChange={setDay}
              onIncidentChange={setIncidentFlag}
              onLimitChange={setLimit}
              onRefresh={fetchPredictions}
              isLoading={isLoading}
            />
          </div>

          {/* Footer - Desktop Only */}
          <div className="hidden lg:block">
            <Footer />
          </div>
        </aside>

        {/* Main Content (Map Only) - Takes remaining space */}
        <main className="w-full lg:flex-1 h-[500px] lg:h-full rounded-[2.5rem] overflow-hidden relative shadow-2xl border border-white/10 bg-white/5 backdrop-blur-sm group">
            {error && (
              <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 bg-red-500/90 text-white px-4 py-2 rounded-lg text-sm">
                {error}
              </div>
            )}
            <Map 
              predictions={predictions.slice(0, limit)} 
              center={mapCenter} 
              isLoading={isLoading}
              weather={weather}
            />
            {/* Map Overlays */}
            <div className="absolute inset-0 pointer-events-none z-[10]">
                {/* Vignette - Reduced opacity for better readability */}
                <div className="absolute inset-0 shadow-[inset_0_0_150px_rgba(0,0,0,0.4)] rounded-[2.5rem]" />
                {/* Grid Pattern */}
                <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_at_center,black_40%,transparent_80%)]" />
            </div>
        </main>

        {/* Footer - Mobile Only */}
        <div className="lg:hidden pb-6">
          <Footer />
        </div>
      </div>
    </div>
  );
}
