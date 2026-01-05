import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        congestion: {
          low: "#22c55e",      // Green
          moderate: "#f59e0b", // Amber
          high: "#ef4444",     // Red
        },
      },
    },
  },
  plugins: [],
};
export default config;
