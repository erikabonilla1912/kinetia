/** Pincel: geometría y serialización de `mask_strokes` (contrato, sección 4). Sin DOM. */

export const BRUSH_LIMITS = {
  maxStrokes: 200,
  maxPoints: 2000,
  minRadius: 0.005,
  maxRadius: 0.2,
} as const;

export type BrushPoint = readonly [number, number];

export interface BrushStroke {
  points: BrushPoint[];
  radius: number;
  direction_deg: number;
  strength: number;
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const round = (value: number, decimals: number) => {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
};

/** Convierte coordenadas de pantalla a la fracción de imagen (origen arriba a la izquierda). */
export function normalizePoint(x: number, y: number, width: number, height: number): BrushPoint {
  return [clamp(x / width, 0, 1), clamp(y / height, 0, 1)];
}

export function totalPoints(strokes: readonly BrushStroke[]): number {
  return strokes.reduce((sum, stroke) => sum + stroke.points.length, 0);
}

/** Distancia en fracciones del lado mayor, porque `radius` también se mide así. */
function distance(a: BrushPoint, b: BrushPoint, width: number, height: number): number {
  const longest = Math.max(width, height);
  return Math.hypot(((b[0] - a[0]) * width) / longest, ((b[1] - a[1]) * height) / longest);
}

/**
 * Agrega un punto solo si se alejó lo suficiente del anterior, para que un trazo
 * largo no agote el límite de puntos. Devuelve el mismo arreglo si lo descarta.
 */
export function appendPoint(
  points: BrushPoint[],
  point: BrushPoint,
  radius: number,
  width: number,
  height: number,
): BrushPoint[] {
  const last = points[points.length - 1];
  if (last && distance(last, point, width, height) < radius * 0.35) return points;
  return [...points, point];
}

/**
 * Dirección del trazo en grados: 0 = derecha, 90 = abajo. Se calcula en el espacio
 * de la imagen, porque en una foto no cuadrada los ejes normalizados no miden igual.
 * Devuelve null si el trazo casi no se movió (un toque).
 */
export function directionFromPoints(
  points: readonly BrushPoint[],
  width: number,
  height: number,
  minTravel = 0.005,
): number | null {
  if (points.length < 2) return null;
  const first = points[0];
  const last = points[points.length - 1];
  if (distance(first, last, width, height) < minTravel) return null;
  const degrees = (Math.atan2((last[1] - first[1]) * height, (last[0] - first[0]) * width) * 180) / Math.PI;
  return (degrees + 360) % 360;
}

export function canAddStroke(strokes: readonly BrushStroke[]): boolean {
  return strokes.length < BRUSH_LIMITS.maxStrokes && totalPoints(strokes) < BRUSH_LIMITS.maxPoints;
}

/** Puntos que aún caben, sin contar los del trazo en curso. */
export function remainingPoints(strokes: readonly BrushStroke[]): number {
  return Math.max(0, BRUSH_LIMITS.maxPoints - totalPoints(strokes));
}

/**
 * Forma exacta del campo `mask_strokes`. Sin trazos devuelve `undefined`: el campo
 * se omite. Nunca excede los límites, aunque se llame con datos fuera de rango.
 */
export function serializeStrokes(strokes: readonly BrushStroke[]): string | undefined {
  const output: BrushStroke[] = [];
  let budget: number = BRUSH_LIMITS.maxPoints;
  for (const stroke of strokes.slice(0, BRUSH_LIMITS.maxStrokes)) {
    if (budget <= 0) break;
    const points = stroke.points.slice(0, budget).map(
      ([x, y]) => [round(clamp(x, 0, 1), 4), round(clamp(y, 0, 1), 4)] as const,
    );
    if (points.length === 0) continue;
    budget -= points.length;
    output.push({
      points,
      radius: round(clamp(stroke.radius, BRUSH_LIMITS.minRadius, BRUSH_LIMITS.maxRadius), 4),
      direction_deg: round(((stroke.direction_deg % 360) + 360) % 360, 1),
      strength: round(clamp(stroke.strength, 0, 1), 2),
    });
  }
  return output.length ? JSON.stringify({ strokes: output }) : undefined;
}
