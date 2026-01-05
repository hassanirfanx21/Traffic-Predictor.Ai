# SURO - Smart Urban Resource Optimizer

A real-time traffic congestion prediction dashboard built with Next.js, ONNX Runtime, and Leaflet.

## 🚀 Features

- **Real-time Predictions**: Predict traffic congestion levels for 49 camera locations in Lake County, IL
- **Interactive Map**: Leaflet-based dark theme map with color-coded congestion markers
- **Time Simulation**: Adjust hour and day to see predicted congestion patterns
- **Weather Integration**: Real-time weather data from Open-Meteo API
- **Incident Simulation**: Toggle incident flag to see impact on predictions

## 🧠 Model

The prediction system uses a **Hierarchical Two-Stage Classification** approach:

- **Stage 1 (Moderate Specialist)**: Detects "Moderate" traffic vs "Extreme" (Low/High)
- **Stage 2 (Generalist)**: Distinguishes between "Low" and "High" congestion
- **Threshold**: 0.54 probability for Moderate classification
- **Accuracy**: ~59% (balanced across all three classes)

### Input Features (9 total):
1. `latitude` - Location coordinate
2. `longitude` - Location coordinate  
3. `hour` - Hour of day (0-23)
4. `day` - Day of week (1-7, Monday=1)
5. `incident_flag` - Whether an incident is reported (0/1)
6. `temperature_2m` - Temperature in Celsius
7. `precipitation` - Precipitation in mm
8. `rain` - Rain in mm
9. `wind_speed_10m` - Wind speed in km/h

## 📦 Installation

```bash
# Navigate to project
cd "SURO PROJECT"

# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the dashboard.

## 🏗️ Project Structure

```
SURO PROJECT/
├── models/
│   ├── model_stage1.onnx    # Moderate vs Rest classifier
│   └── model_stage2.onnx    # Low vs High classifier
├── src/
│   ├── app/
│   │   ├── api/predict/     # ONNX inference API route
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   ├── Dashboard.tsx    # Main dashboard
│   │   ├── Map.tsx          # Leaflet map
│   │   ├── TimeControls.tsx # Hour/Day controls
│   │   ├── WeatherPanel.tsx # Weather display
│   │   └── Stats.tsx        # Congestion statistics
│   └── lib/
│       ├── locations.ts     # Camera coordinates
│       ├── types.ts         # TypeScript types
│       └── weather.ts       # Open-Meteo API
├── package.json
└── README.md
```

## 🔧 Technologies

- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **Maps**: Leaflet + React-Leaflet
- **ML Inference**: ONNX Runtime Node
- **Weather API**: Open-Meteo (free, no API key required)
- **Models**: Scikit-learn HistGradientBoostingClassifier → ONNX

## 📊 Congestion Levels

| Level | Color | Description |
|-------|-------|-------------|
| 🟢 Low | Green | Light traffic, free flow |
| 🟡 Moderate | Amber | Medium traffic, slight delays |
| 🔴 High | Red | Heavy traffic, significant delays |

## 🚀 Deployment

The app is designed to be deployed on **Vercel**:

1. Push to GitHub
2. Connect repository to Vercel
3. Deploy (ONNX models are included in the repo)

> **Note**: onnxruntime-node works in Vercel's Node.js serverless functions.

## 📄 License

MIT License - Built for SURO (Smart Urban Resource Optimizer) project.
