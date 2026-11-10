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

# Desde el contrato v1.8 no hay modelo de profundidad: se borra el de volúmenes antiguos.
rm -f "$MODELS_DIR/midas_small.onnx"

# Copia atómica: un corte a medias no deja un archivo truncado con el nombre final.
install_from_image() {
    cp "$IMAGE_DIR/$1" "$MODELS_DIR/.$1.tmp"
    mv "$MODELS_DIR/.$1.tmp" "$MODELS_DIR/$1"
}

for name in kinetia_seg.onnx manifest.json metrics.json; do
    if [ ! -f "$MODELS_DIR/$name" ]; then
        echo "model-loader: copiando $name al volumen"
        install_from_image "$name"
    fi
done

sha_of() {
    sha256sum "$1" | cut -d' ' -f1
}

# Devuelve 0 si reemplazó un modelo antiguo del volumen por el de la imagen.
replace_stale() {
    name="$1"
    expected="$2"
    actual="$(sha_of "$MODELS_DIR/$name")"
    [ "$actual" != "$expected" ] || return 1
    [ "$(sha_of "$IMAGE_DIR/$name")" = "$expected" ] || return 1
    echo "model-loader: $name: el volumen conserva un modelo anterior ($actual); lo reemplazo por el de la imagen"
    install_from_image "$name"
}

# Las métricas y el manifiesto describen al modelo: si cambia, se reemplazan con él.
if replace_stale kinetia_seg.onnx "$EXPECTED_SHA256_SEG"; then
    install_from_image manifest.json
    install_from_image metrics.json
fi

check() {
    name="$1"
    expected="$2"
    actual="$(sha_of "$MODELS_DIR/$name")"
    if [ "$actual" != "$expected" ]; then
        fail "SHA-256 incorrecto en $name: esperado $expected, obtenido $actual"
    fi
    echo "model-loader: $name OK ($actual)"
}

check kinetia_seg.onnx "$EXPECTED_SHA256_SEG"
echo "model-loader: verificación correcta"
