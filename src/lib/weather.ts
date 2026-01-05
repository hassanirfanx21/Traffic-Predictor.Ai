import { WeatherData } from "./types";

const OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast";

/**
 * Fetches current weather data from Open-Meteo API
 * @param latitude - Location latitude
 * @param longitude - Location longitude
 * @returns Weather data object
 */
export async function getCurrentWeather(
  latitude: number,
  longitude: number
): Promise<WeatherData> {
  const params = new URLSearchParams({
    latitude: latitude.toString(),
    longitude: longitude.toString(),
    current: "temperature_2m,precipitation,rain,wind_speed_10m",
    timezone: "America/Chicago", // Illinois timezone
  });

  try {
    const response = await fetch(`${OPEN_METEO_URL}?${params}`, {
      next: { revalidate: 300 }, // Cache for 5 minutes
    });

    if (!response.ok) {
      throw new Error(`Weather API error: ${response.status}`);
    }

    const data = await response.json();
    const current = data.current;

    return {
      temperature_2m: current.temperature_2m ?? 20,
      precipitation: current.precipitation ?? 0,
      rain: current.rain ?? 0,
      wind_speed_10m: current.wind_speed_10m ?? 10,
    };
  } catch (error) {
    console.error("Failed to fetch weather:", error);
    // Return default values if API fails
    return {
      temperature_2m: 20,
      precipitation: 0,
      rain: 0,
      wind_speed_10m: 10,
    };
  }
}

/**
 * Fetches forecast weather data from Open-Meteo API
 */
async function getForecastWeather(
  latitude: number,
  longitude: number,
  daysFromNow: number,
  targetHour: number
): Promise<WeatherData> {
  const params = new URLSearchParams({
    latitude: latitude.toString(),
    longitude: longitude.toString(),
    hourly: "temperature_2m,precipitation,rain,wind_speed_10m",
    timezone: "America/Chicago",
    forecast_days: "8", // Fetch enough days to cover next week
  });

  try {
    const response = await fetch(`${OPEN_METEO_URL}?${params}`, {
      next: { revalidate: 3600 }, // Cache forecast for 1 hour
    });

    if (!response.ok) {
      throw new Error(`Weather API error: ${response.status}`);
    }

    const data = await response.json();
    const hourly = data.hourly;
    
    // Calculate index: days * 24 + hour
    // Open-Meteo returns data starting from 00:00 of the current day in the requested timezone
    const index = (daysFromNow * 24) + targetHour;
    
    // Safety check
    if (index >= hourly.temperature_2m.length) {
        console.warn("Forecast index out of range, falling back to current weather");
        return getCurrentWeather(latitude, longitude);
    }

    return {
      temperature_2m: hourly.temperature_2m[index] ?? 20,
      precipitation: hourly.precipitation[index] ?? 0,
      rain: hourly.rain[index] ?? 0,
      wind_speed_10m: hourly.wind_speed_10m[index] ?? 10,
    };
  } catch (error) {
    console.error("Failed to fetch forecast:", error);
    return getCurrentWeather(latitude, longitude);
  }
}

/**
 * Get weather for a specific time (Current or Forecast)
 */
export async function getWeatherForTime(
  latitude: number,
  longitude: number,
  targetHour: number,
  targetDay: number
): Promise<WeatherData> {
  const { day: currentDay } = getCurrentTimeInfo();
  
  // Calculate days difference (0-6)
  // If target is Mon (1) and current is Wed (3): (1 - 3 + 7) % 7 = 5 days later
  // If target is Wed (3) and current is Wed (3): (3 - 3 + 7) % 7 = 0 days (Today)
  const daysDiff = (targetDay - currentDay + 7) % 7;
  
  if (daysDiff === 0) {
    return getCurrentWeather(latitude, longitude);
  }
  
  return getForecastWeather(latitude, longitude, daysDiff, targetHour);
}

/**
 * Get current time info for prediction
 * @returns Object with hour (0-23) and day (1-7, Monday=1)
 */
export function getCurrentTimeInfo(): { hour: number; day: number } {
  const now = new Date();
  // Convert to Illinois timezone
  const chicagoTime = new Date(
    now.toLocaleString("en-US", { timeZone: "America/Chicago" })
  );
  
  const hour = chicagoTime.getHours();
  // JavaScript: 0=Sunday, 1=Monday... We need: 1=Monday...7=Sunday
  const jsDay = chicagoTime.getDay();
  const day = jsDay === 0 ? 7 : jsDay;
  
  return { hour, day };
}
