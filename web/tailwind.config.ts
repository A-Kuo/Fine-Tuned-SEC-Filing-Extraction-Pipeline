import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "hsl(var(--bg-canvas))",
        surface: "hsl(var(--bg-surface))",
        subtle: "hsl(var(--bg-subtle))",
        "border-formal": "hsl(var(--border-formal))",
        "primary-navy": "hsl(var(--primary-navy))",
        "secondary-blue": "hsl(var(--secondary-blue))",
        "accent-slate": "hsl(var(--accent-slate))",
        "text-primary": "hsl(var(--text-primary))",
        "text-secondary": "hsl(var(--text-secondary))",
        "text-muted": "hsl(var(--text-muted))",
        "market-gain": "hsl(var(--market-gain))",
        "market-gain-bg": "hsl(var(--market-gain-bg))",
        "market-loss": "hsl(var(--market-loss))",
        "market-loss-bg": "hsl(var(--market-loss-bg))",
      },
      borderRadius: {
        none: "0px",
        sm: "2px",
        DEFAULT: "2px",
        md: "4px",
        lg: "6px",
      },
      boxShadow: {
        flat: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
        "border-only": "none",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["var(--font-plex-mono)", "SFMono-Regular", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
