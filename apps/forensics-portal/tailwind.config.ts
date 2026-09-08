import type { Config } from "tailwindcss";

/**
 * AURA Forensics Portal — Tailwind CSS Configuration
 *
 * Color palette is deliberately chosen for:
 * - Clarity under extended forensic review sessions (low-fatigue grays)
 * - Unambiguous risk signaling (red=danger, amber=warning, green=safe)
 * - Professional appearance for legal/law enforcement contexts
 */
const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // AURA brand
        primary: {
          50: "#eff6ff",
          100: "#dbeafe",
          200: "#bfdbfe",
          300: "#93c5fd",
          400: "#60a5fa",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          800: "#1e40af",
          900: "#1e3a5f", // AURA deep blue
          950: "#172554",
          DEFAULT: "#1e3a5f",
        },
        // Risk levels — must be unambiguous even in poor lighting
        danger: {
          50: "#fef2f2",
          100: "#fee2e2",
          200: "#fecaca",
          300: "#fca5a5",
          400: "#f87171",
          500: "#ef4444",
          600: "#dc2626", // HIGH_RISK
          700: "#b91c1c",
          800: "#991b1b",
          900: "#7f1d1d",
          DEFAULT: "#dc2626",
        },
        warning: {
          50: "#fffbeb",
          100: "#fef3c7",
          200: "#fde68a",
          300: "#fcd34d",
          400: "#fbbf24",
          500: "#f59e0b", // MEDIUM_RISK
          600: "#d97706",
          700: "#b45309",
          800: "#92400e",
          900: "#78350f",
          DEFAULT: "#f59e0b",
        },
        safe: {
          50: "#f0fdf4",
          100: "#dcfce7",
          200: "#bbf7d0",
          300: "#86efac",
          400: "#4ade80",
          500: "#22c55e",
          600: "#16a34a", // LOW_RISK
          700: "#15803d",
          800: "#166534",
          900: "#14532d",
          DEFAULT: "#16a34a",
        },
        // UI neutrals — slate-based for low eye fatigue
        surface: {
          50: "#f8fafc",
          100: "#f1f5f9",
          200: "#e2e8f0",
          300: "#cbd5e1",
          400: "#94a3b8",
          500: "#64748b",
          600: "#475569",
          700: "#334155",
          800: "#1e293b",
          900: "#0f172a",
          950: "#020617",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      boxShadow: {
        "aura-sm": "0 1px 3px 0 rgba(30,58,95,0.1), 0 1px 2px -1px rgba(30,58,95,0.1)",
        "aura-md": "0 4px 6px -1px rgba(30,58,95,0.1), 0 2px 4px -2px rgba(30,58,95,0.1)",
        "aura-lg": "0 10px 15px -3px rgba(30,58,95,0.1), 0 4px 6px -4px rgba(30,58,95,0.1)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "spin-slow": "spin 3s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
