"use client";

import { useEffect, useState, useMemo } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import { LatLngExpression, divIcon } from "leaflet";
import { PredictionResult, CONGESTION_COLORS, CongestionLevel, WeatherData } from "@/lib/types";

interface MapProps {
  predictions: PredictionResult[];
  center: LatLngExpression;
  isLoading: boolean;
  weather?: WeatherData | null;
}

function WindOverlay({ speed }: { speed: number }) {
  // Determine wind category
  const isFast = speed >= 20;
  const isMid = speed > 8 && speed < 20;
  
  // Configuration based on speed ranges
  const config = useMemo(() => {
    let cfg = {
      baseDuration: 12,      // Slow: 12-17s duration
      opacity: "via-white/10", // Subtle
      count: 8,              // Fewer particles
      widthBase: 100,
      widthVar: 150
    };

    if (isMid) {
      cfg = {
        baseDuration: 10,     // Mid: 10-14s duration (Slower)
        opacity: "via-white/30", // Visible
        count: 15,           // Moderate particles
        widthBase: 200,
        widthVar: 200
      };
    } else if (isFast) {
      cfg = {
        baseDuration: 8,     // Fast: 8-10s duration (Much slower)
        opacity: "via-white/40", 
        count: 12,           // Fewer particles
        widthBase: 300,      // Long streaks
        widthVar: 400
      };
    }
    return cfg;
  }, [isFast, isMid]);

  const particles = useMemo(() => {
    return [...Array(config.count)].map(() => ({
      width: `${config.widthBase + Math.random() * config.widthVar}px`,
      top: `${Math.random() * 100}%`,
      animationDuration: `${config.baseDuration + Math.random() * (isFast ? 2 : 5)}s`,
      animationDelay: `${Math.random() * 5}s`
    }));
  }, [config, isFast]);

  return (
    <div className="absolute inset-0 pointer-events-none z-[20] overflow-hidden">
      {particles.map((style, i) => (
        <div
          key={i}
          className={`absolute h-[2px] md:h-[3px] bg-gradient-to-r from-transparent ${config.opacity} to-transparent rounded-full blur-[0.5px]`}
          style={{
            width: style.width,
            top: style.top,
            left: '-40%', // Start further back for long streaks
            animation: `wind ${style.animationDuration} linear infinite`,
            animationDelay: style.animationDelay,
          }}
        />
      ))}
    </div>
  );
}

function MapUpdater({ center }: { center: LatLngExpression }) {
  const map = useMap();
  
  useEffect(() => {
    map.setView(center, map.getZoom());
  }, [center, map]);

  useEffect(() => {
    // Robust resize handling
    const container = map.getContainer();
    const resizeObserver = new ResizeObserver(() => {
      map.invalidateSize();
    });
    resizeObserver.observe(container);
    
    // Initial invalidation
    setTimeout(() => map.invalidateSize(), 250);

    return () => resizeObserver.disconnect();
  }, [map]);

  return null;
}

export default function Map({ predictions, center, weather }: MapProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  if (!mounted) return null;

  const createCustomIcon = (congestion: number) => {
    const color = CONGESTION_COLORS[congestion as CongestionLevel];
    return divIcon({
      className: "bg-transparent",
      html: `
        <div class="relative flex items-center justify-center w-full h-full">
          <div class="absolute w-full h-full rounded-full opacity-20 animate-pulse" style="background-color: ${color}"></div>
          <div class="relative w-2.5 h-2.5 rounded-full border border-white/80 shadow-sm" style="background-color: ${color}"></div>
        </div>
      `,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    });
  };

  return (
    <div className="relative w-full h-full">
      <MapContainer
        center={center}
        zoom={12}
        className="w-full h-full bg-transparent z-0" // Ensure z-index is managed
        style={{ height: "100%", width: "100%", minHeight: "100%" }} // Explicit styles for Leaflet
        scrollWheelZoom={true}
        zoomControl={false} // Cleaner look
      >
        {/* Dark Matter Map Style */}
        <TileLayer
          attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          className="filter brightness-110 contrast-100" // Slightly brighter for better label readability
        />
        <MapUpdater center={center} />
        
        {predictions.map((pred, index) => (
          <Marker
            key={index}
            position={[pred.latitude, pred.longitude]}
            icon={createCustomIcon(pred.congestion)}
            eventHandlers={{
              mouseover: (e) => e.target.openPopup(),
              mouseout: (e) => e.target.closePopup(),
            }}
          >
            <Popup className="glass-popup" closeButton={false}>
              <div className="p-2 min-w-[150px]">
                <p className="font-bold text-white text-sm mb-2">{pred.location}</p>
                <div className="flex items-center gap-2 mb-1">
                  <div 
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: CONGESTION_COLORS[pred.congestion as CongestionLevel] }}
                  />
                  <span className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                    {pred.congestionLabel}
                  </span>
                </div>
                <div className="w-full bg-white/20 h-1 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-white" 
                    style={{ width: `${pred.probability * 100}%` }} 
                  />
                </div>
                <p className="text-[10px] text-slate-300 mt-1 text-right">
                  {(pred.probability * 100).toFixed(0)}% Confidence
                </p>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
      
      {/* Wind Animation Overlay */}
      {weather && <WindOverlay speed={weather.wind_speed_10m} />}
    </div>
  );
}
