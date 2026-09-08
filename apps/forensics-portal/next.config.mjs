/**
 * Next.js configuration for the AURA Forensics Portal.
 *
 * API_URL should point to the FastAPI backend. In production this is injected
 * at build time via the CI/CD environment. In development, it defaults to the
 * local FastAPI dev server at port 8000.
 *
 * Note: NEXT_PUBLIC_* variables are inlined at build time and visible to the
 * browser. Non-public variables (AURA_BACKEND_URL) are only accessible in
 * server-side code (API routes, Server Components).
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  env: {
    // Server-side only — used by the upload proxy API route to avoid CORS
    AURA_BACKEND_URL: process.env.AURA_BACKEND_URL ?? "http://localhost:8000",
  },

  async headers() {
    return [
      {
        // Apply strict CSP on all pages
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },

  images: {
    // Artifact images served from the FastAPI backend
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000",
        pathname: "/artifacts/**",
      },
    ],
  },
};

export default nextConfig;
