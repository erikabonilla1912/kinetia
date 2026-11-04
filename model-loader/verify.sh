#!/bin/sh
# Copia al PVC los modelos que falten y verifica su SHA-256 contra el ConfigMap (no contra la imagen).
# docs/CONTRATOS.md, sección 7. Si un hash no coincide, termina con código 1 y el pod no arranca.
set -eu

MODELS_DIR="${MODELS_DIR:-/models}"
IMAGE_DIR=/image-models

fail() {
    echo "model-loader: ERROR: $1" >&2
    exit 1
}

[ -n "${EXPECTED_SHA256_SEG:-}" ] || fail "falta EXPECTED_SHA256_SEG"
[ -n "${EXPECTED_SHA256_DEPTH:-}" ] || fail "falta EXPECTED_SHA256_DEPTH"

for name in kinetia_seg.onnx midas_small.onnx manifest.json metrics.json; do
    if [ ! -f "$MODELS_DIR/$name" ]; then
        echo "model-loader: copiando $name al volumen"
        # Copia atómica: un corte a medias no deja un archivo truncado con el nombre final.
        cp "$IMAGE_DIR/$name" "$MODELS_DIR/.$name.tmp"
        mv "$MODELS_DIR/.$name.tmp" "$MODELS_DIR/$name"
    fi
done

check() {
    name="$1"
    expected="$2"
    actual="$(sha256sum "$MODELS_DIR/$name" | cut -d' ' -f1)"
    if [ "$actual" != "$expected" ]; then
        echo "model-loader: $name: esperado $expected, obtenido $actual" >&2
        image="$(sha256sum "$IMAGE_DIR/$name" | cut -d' ' -f1)"
        if [ "$image" = "$expected" ]; then
            echo "model-loader: la imagen trae el archivo correcto, pero el volumen conserva uno anterior." >&2
            echo "model-loader: ejecuta 'make down' y 'make demo' para recrear el volumen." >&2
        fi
        fail "SHA-256 incorrecto en $name"
    fi
    echo "model-loader: $name OK ($actual)"
}

check kinetia_seg.onnx "$EXPECTED_SHA256_SEG"
check midas_small.onnx "$EXPECTED_SHA256_DEPTH"
echo "model-loader: verificación correcta"
