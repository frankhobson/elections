/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: "#0E1117",
        darkCard: "#1E222A",
        darkBorder: "#2D3748",
        brandOrange: "#E05A1A",
        brandBlue: "#33658A",
        brandCyan: "#06B6D4",
        brandEmerald: "#10B981"
      }
    },
  },
  plugins: [],
}
