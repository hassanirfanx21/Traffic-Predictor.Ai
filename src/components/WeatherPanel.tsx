"use client";

import { WeatherData } from "@/lib/types";

interface WeatherPanelProps {
  weather: WeatherData | null;
  timestamp: string | null;
}

export default function WeatherPanel({ weather }: WeatherPanelProps) {
  const date = new Date();
  const dateStr = date.toLocaleDateString("en-US", { weekday: 'long', month: 'long', day: 'numeric' });

  if (!weather) return <div className="animate-pulse h-24 w-64 bg-white/5 rounded-2xl" />;

  // Determine weather condition text based on data (simple logic for demo)
  let condition = "Clear Sky";
  if (weather.rain > 0) condition = "Rainy";
  else if (weather.precipitation > 0) condition = "Drizzle";
  else if (weather.wind_speed_10m > 20) condition = "Windy";
  else if (weather.temperature_2m < 0) condition = "Freezing";

  return (
    <div className="flex flex-col md:flex-row justify-between w-full items-end">
      <div>
        <div className="flex items-center gap-2 text-white/60 mb-1">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
          <span className="text-sm font-medium tracking-wide">Lake County, IL</span>
          <span className="text-white/20">|</span>
          <span className="text-sm text-white/40">{dateStr}</span>
        </div>
        
        <div className="flex items-start gap-4">
          <h1 className="text-7xl md:text-8xl font-thin tracking-tighter text-white">
            {Math.round(weather.temperature_2m)}°
          </h1>
          <div className="mt-4 space-y-1">
            <div className="text-2xl font-light text-white/90">{condition}</div>
            <div className="flex gap-3 text-sm font-medium text-white/50">
              <span>H: <span className="text-white/80">{Math.round(weather.temperature_2m + 5)}°</span></span>
              <span>L: <span className="text-white/80">{Math.round(weather.temperature_2m - 5)}°</span></span>
            </div>
          </div>
        </div>
      </div>

      {/* Mini Weather Cards */}
      <div className="flex gap-3 mt-4 md:mt-0">
        <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-3 min-w-[90px]">
          <div className="text-white/40 text-[10px] uppercase tracking-wider mb-1">Wind</div>
          <div className="text-lg font-medium">{weather.wind_speed_10m.toFixed(1)}</div>
          <div className="text-[10px] text-white/30">km/h</div>
        </div>
        <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-3 min-w-[90px]">
          <div className="text-white/40 text-[10px] uppercase tracking-wider mb-1">Rain</div>
          <div className="text-lg font-medium">{weather.rain.toFixed(1)}</div>
          <div className="text-[10px] text-white/30">mm</div>
        </div>
        <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-3 min-w-[90px]">
          <div className="text-white/40 text-[10px] uppercase tracking-wider mb-1">Precip</div>
          <div className="text-lg font-medium">{weather.precipitation.toFixed(1)}</div>
          <div className="text-[10px] text-white/30">mm</div>
        </div>
      </div>
    </div>
  );
}
