import { ModelClass } from './api.models';

export interface DecodedMask {
  width: number;
  height: number;
  /** Un valor por píxel: el índice de clase del PNG. */
  indices: Uint8Array;
}

export function hexToRgb(hex: string): [number, number, number] | null {
  const match = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!match) return null;
  const value = parseInt(match[1], 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function base64ToBytes(base64: string): Uint8Array<ArrayBuffer> {
  const binary = atob(base64);
  const bytes = new Uint8Array(new ArrayBuffer(binary.length));
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/** Decodifica el PNG de 8 bits de la API. Sin conversión de color: el valor es un índice. */
export async function decodeMask(base64: string): Promise<DecodedMask> {
  const blob = new Blob([base64ToBytes(base64)], { type: 'image/png' });
  const bitmap = await createImageBitmap(blob, {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  });
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas 2D no disponible');
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const indices = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < indices.length; i++) indices[i] = rgba[i * 4];
  return { width: canvas.width, height: canvas.height, indices };
}

/**
 * Pinta la máscara con los colores de /config. Un índice que no esté en /config queda
 * transparente. Si se indica `focusId`, las demás clases quedan apagadas.
 */
export function paintMask(
  mask: DecodedMask,
  classes: readonly ModelClass[],
  focusId: number | null = null,
): ImageData {
  const palette = new Map<number, [number, number, number]>();
  for (const cls of classes) {
    const rgb = hexToRgb(cls.color);
    if (rgb) palette.set(cls.id, rgb);
  }
  const image = new ImageData(mask.width, mask.height);
  for (let i = 0; i < mask.indices.length; i++) {
    const id = mask.indices[i];
    const rgb = palette.get(id);
    if (!rgb) continue;
    const offset = i * 4;
    image.data[offset] = rgb[0];
    image.data[offset + 1] = rgb[1];
    image.data[offset + 2] = rgb[2];
    image.data[offset + 3] = focusId === null || focusId === id ? 255 : 0;
  }
  return image;
}
