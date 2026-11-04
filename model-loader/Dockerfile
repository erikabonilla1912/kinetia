# Imagen model-loader. Contexto de build: raíz del repositorio (usa /.dockerignore).
# MODELS_SRC elige la carpeta de modelos. Por defecto, models/: sin el modelo real, el build falla.
FROM alpine:3.20
ARG MODELS_SRC=models
RUN addgroup -g 1000 kinetia && adduser -u 1000 -G kinetia -D -H kinetia \
    && mkdir -p /image-models /models && chown 1000:1000 /models
COPY ${MODELS_SRC}/kinetia_seg.onnx ${MODELS_SRC}/midas_small.onnx ${MODELS_SRC}/manifest.json ${MODELS_SRC}/metrics.json /image-models/
COPY model-loader/verify.sh /usr/local/bin/verify.sh
RUN chmod 0555 /usr/local/bin/verify.sh && chmod -R a+rX /image-models
USER 1000:1000
ENTRYPOINT ["/bin/sh", "/usr/local/bin/verify.sh"]
