// Types for SURO application

export interface WeatherData {
  temperature_2m: number;
  precipitation: number;
  rain: number;
  wind_speed_10m: number;
}

export interface PredictionInput {
  latitude: number;
  longitude: number;
  hour: number;
  day: number;
  temperature_2m: number;
  precipitation: number;
  rain: number;
  wind_speed_10m: number;
}

export interface PredictionResult {
  location: string;
  latitude: number;
  longitude: number;
  congestion: 0 | 1 | 2; // 0=Low, 1=Moderate, 2=High
  congestionLabel: "Low" | "Moderate" | "High";
  probability: number;
}

export interface BatchPredictionResponse {
  predictions: PredictionResult[];
  weather: WeatherData;
  timestamp: string;
  hour: number;
  day: number;
}

export type CongestionLevel = 0 | 1 | 2;

export const CONGESTION_LABELS: Record<CongestionLevel, string> = {
  0: "Low",
  1: "Moderate", 
  2: "High",
};

export const CONGESTION_COLORS: Record<CongestionLevel, string> = {
  0: "#22c55e", // Green
  1: "#f59e0b", // Amber
  2: "#ef4444", // Red
};
