# Build stage: compile React app with Vite
FROM node:20-alpine AS build
WORKDIR /app

# Install deps
COPY package.json package-lock.json ./
# Use npm install to avoid lockfile mismatch errors in CI; switch back to `npm ci` once lockfile is updated
RUN npm install --no-audit --no-fund

# Copy source
COPY . .

# Inject API base URL at build time (default to localhost:8000)
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# Build static assets
RUN npm run build

# Runtime stage: serve static files via Nginx
FROM nginx:alpine

# Use custom Nginx config with SPA fallback
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
