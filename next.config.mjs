/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  // Prevent bundling of onnxruntime-web so it can find its WASM files in node_modules
  serverExternalPackages: ['onnxruntime-web'],
  experimental: {
    outputFileTracingIncludes: {
      '/api/predict': ['./node_modules/onnxruntime-web/dist/**/*'],
    },
  },
};

export default nextConfig;
