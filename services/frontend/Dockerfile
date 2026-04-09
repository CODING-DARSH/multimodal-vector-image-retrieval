# services/frontend/Dockerfile
#
# MULTI-STAGE BUILD — the most important Docker pattern to understand.
#
# STAGE 1 (builder): Install Node.js, install dependencies, compile React → static HTML/JS/CSS
# STAGE 2 (runtime): Copy ONLY the compiled output into a tiny Nginx image
#
# WHY TWO STAGES:
#   node:18-alpine (with node_modules) = ~400MB
#   nginx:alpine serving static files = ~25MB
#
#   React is a build-time framework. "npm run build" compiles your JSX into
#   plain JavaScript that any browser can run. Once compiled, you don't need
#   Node.js anymore — just a web server to serve the static files.
#
#   Without multi-stage: 400MB image with Node.js sitting idle
#   With multi-stage: 25MB image, same result
#
# This is the standard production pattern for any React/Vue/Angular app.

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files first, THEN source code.
# WHY: Docker caches each layer. If you copy everything at once,
# any file change (even a .jsx edit) invalidates the npm install cache.
# By copying package.json first, npm install is only re-run when
# dependencies actually change — not on every code edit.
# This saves minutes on rebuilds.
COPY package.json package-lock.json* ./
RUN npm ci
# npm ci = "clean install" — uses exact versions from package-lock.json
# Faster and more deterministic than npm install for CI/Docker builds

COPY . .

# Build the React app
# Vite compiles JSX → JS, bundles, minifies, tree-shakes
# Output goes to /app/dist/
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# ── Stage 2: Serve ────────────────────────────────────────────────────────────
FROM nginx:alpine

# Copy compiled static files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom Nginx config to handle React Router (SPA routing)
# Without this, refreshing on /search returns 404 because Nginx
# doesn't know React handles routing — it looks for a /search folder.
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost:3000/ || exit 1

CMD ["nginx", "-g", "daemon off;"]