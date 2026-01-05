import { NextRequest, NextResponse } from "next/server";
import { cameraLocations } from "@/lib/locations";
import { getWeatherForTime, getCurrentTimeInfo } from "@/lib/weather";
import { 
  PredictionResult, 
  BatchPredictionResponse,
  CONGESTION_LABELS,
  CongestionLevel 
} from "@/lib/types";
import * as onnx from 'onnxruntime-web';
import fs from 'fs';
import path from 'path';

// Configure ONNX Runtime for serverless environment
// 1. Disable multi-threading to avoid Vercel timeouts/errors and missing threaded wasm files
onnx.env.wasm.numThreads = 1;
// 2. Disable SIMD if causing issues (optional, but safer for compatibility)
onnx.env.wasm.simd = false;
// 3. Point to the wasm files if needed (usually handled by serverExternalPackages, but good to know)
// onnx.env.wasm.wasmPaths = ...

// Global session variable for caching
let sessions: { stage1: onnx.InferenceSession | null, stage2: onnx.InferenceSession | null } = { stage1: null, stage2: null };

async function getSessions() {
  if (sessions.stage1 && sessions.stage2) return sessions;
  
  try {
    // Load both models
    const modelPath1 = path.join(process.cwd(), 'public', 'models', 'model_stage1.onnx');
    const modelPath2 = path.join(process.cwd(), 'public', 'models', 'model_stage2.onnx');
    
    const [buffer1, buffer2] = await Promise.all([
      fs.promises.readFile(modelPath1),
      fs.promises.readFile(modelPath2)
    ]);

    const [session1, session2] = await Promise.all([
      onnx.InferenceSession.create(buffer1),
      onnx.InferenceSession.create(buffer2)
    ]);

    sessions = { stage1: session1, stage2: session2 };
    return sessions;
  } catch (e) {
    console.error("Failed to load ONNX models:", e);
    throw e;
  }
}

async function runInference(session: onnx.InferenceSession, inputData: number[]): Promise<{ congestion: CongestionLevel; probability: number }> {
  try {
    const inputTensor = new onnx.Tensor('float32', Float32Array.from(inputData), [1, 8]);
    const inputName = session.inputNames[0];
    const feeds = { [inputName]: inputTensor };
    const output = await session.run(feeds);
    const outputName = session.outputNames[0];
    const outputData = output[outputName].data;
    
    // Helper to safely convert output data (which might be BigInt) to Number
    const getVal = (idx: number) => Number(outputData[idx]);

    if (outputData.length >= 3) {
        let maxIdx = 0;
        let maxVal = getVal(0);
        for (let i = 1; i < outputData.length; i++) {
            const val = getVal(i);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        return { congestion: maxIdx as CongestionLevel, probability: maxVal };
    } else {
        const val = getVal(0);
        const congestion = Math.max(0, Math.min(2, Math.round(val))) as CongestionLevel;
        return { congestion, probability: 1.0 };
    }
  } catch (e) {
    console.error("Inference failed:", e);
    return { congestion: 0, probability: 0 };
  }
}

export async function GET(request: NextRequest) {
  try {
    // Get optional parameters
    const searchParams = request.nextUrl.searchParams;
    const customHour = searchParams.get("hour");
    const customDay = searchParams.get("day");
    
    // Get current time or use custom
    const timeInfo = getCurrentTimeInfo();
    const hour = customHour ? parseInt(customHour) : timeInfo.hour;
    const day = customDay ? parseInt(customDay) : timeInfo.day;
    
    // Get weather for center of camera network
    const centerLat = cameraLocations.reduce((sum, loc) => sum + loc.latitude, 0) / cameraLocations.length;
    const centerLng = cameraLocations.reduce((sum, loc) => sum + loc.longitude, 0) / cameraLocations.length;
    const weather = await getWeatherForTime(centerLat, centerLng, hour, day);
    
    // Load the model sessions
    const { stage1, stage2 } = await getSessions();

    // Run predictions for all locations
    const predictions: PredictionResult[] = [];
    
    for (const location of cameraLocations) {
      // Input order (8 features): [lat, lon, hour, day, temp, precip, rain, wind]
      const inputData = [
        location.latitude,
        location.longitude,
        hour,
        day,
        weather.temperature_2m,
        weather.precipitation,
        weather.rain,
        weather.wind_speed_10m,
      ];
      
      // Run inference on both models
      const results = [];
      if (stage1) results.push(await runInference(stage1, inputData));
      if (stage2) results.push(await runInference(stage2, inputData));
      
      if (results.length === 0) throw new Error("No models loaded");

      // Strategy: Max Congestion (Pessimistic/Safety-first approach)
      // This ensures we see High (from Stage 2) and Moderate (from Stage 1) if they differ.
      // We take the result with the highest congestion level.
      const finalResult = results.reduce((prev, current) => {
        return (current.congestion > prev.congestion) ? current : prev;
      });

      predictions.push({
        location: location.name,
        latitude: location.latitude,
        longitude: location.longitude,
        congestion: finalResult.congestion,
        congestionLabel: CONGESTION_LABELS[finalResult.congestion] as "Low" | "Moderate" | "High",
        probability: finalResult.probability,
      });
    }
    
    const response: BatchPredictionResponse = {
      predictions,
      weather,
      timestamp: new Date().toISOString(),
      hour,
      day,
    };
    
    return NextResponse.json(response);
    
  } catch (err: any) {
    console.error("Prediction error:", err);
    return NextResponse.json(
      { 
        error: "Failed to generate predictions", 
        details: err.message || String(err),
      },
      { status: 500 }
    );
  }
}
