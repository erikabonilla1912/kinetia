# Imagen del frontend. Contexto de build: frontend/. Sin variables de build (usa /api/v1 relativo).
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY angular.json tsconfig.json tsconfig.app.json .postcssrc.json ./
COPY public/ ./public/
COPY src/ ./src/
RUN npm run build

FROM nginx:1.29-alpine AS runtime
RUN rm -rf /usr/share/nginx/html/* /etc/nginx/conf.d/default.conf \
    && mkdir -p /shared-tmp \
    && chmod 1777 /shared-tmp \
    && chown -R 101:101 /var/cache/nginx
COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=build /app/dist/frontend/browser/ /usr/share/nginx/html/
USER 101:101
EXPOSE 80
# Sin el entrypoint de la imagen base: intenta reescribir conf.d y la raíz es de solo lectura.
ENTRYPOINT ["nginx", "-g", "daemon off;"]
