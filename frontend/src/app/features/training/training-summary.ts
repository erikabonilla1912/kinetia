import { TrainingEpoch } from '../../core/api.models';

/**
 * Mejora mínima para guardar un modelo nuevo: el valor por defecto de --min-delta en
 * ml/train.py. Con ella se reproduce qué época guardó el entrenamiento real.
 */
export const MIN_DELTA = 0.001;

export interface TrainingSummary {
  epochs: number;
  totalSeconds: number;
  /** La época que train.py guardó: la última que superó al mejor anterior en más de MIN_DELTA. */
  saved: TrainingEpoch;
  /** El máximo absoluto de mIoU de validación, que puede no ser la guardada. */
  peak: TrainingEpoch;
  minValLoss: TrainingEpoch;
  first: TrainingEpoch;
  last: TrainingEpoch;
}

export function summarize(history: readonly TrainingEpoch[], minDelta = MIN_DELTA): TrainingSummary | null {
  if (history.length === 0) return null;
  let saved = history[0];
  for (const epoch of history) {
    if (epoch.val_miou > saved.val_miou + minDelta) saved = epoch;
  }
  // La primera época siempre se guarda (el mejor inicial es -1), así que el bucle parte de ella.
  return {
    epochs: history.length,
    totalSeconds: history.reduce((sum, e) => sum + e.seconds, 0),
    saved,
    peak: history.reduce((a, e) => (e.val_miou > a.val_miou ? e : a)),
    minValLoss: history.reduce((a, e) => (e.val_loss < a.val_loss ? e : a)),
    first: history[0],
    last: history[history.length - 1],
  };
}

export interface ChartFrame {
  width: number;
  height: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export const FRAME: ChartFrame = { width: 560, height: 260, left: 46, right: 548, top: 14, bottom: 226 };

/** En un celular el SVG mide unos 300 px: un lienzo más angosto mantiene el texto legible. */
export const COMPACT_FRAME: ChartFrame = { width: 330, height: 220, left: 34, right: 312, top: 14, bottom: 186 };

export interface LineChart {
  train: string;
  val: string;
  yTicks: { y: number; label: string }[];
  xTicks: { x: number; label: string }[];
  x: (epoch: number) => number;
  y: (value: number) => number;
}

/** Trazados SVG de una métrica de entrenamiento y validación sobre el mismo eje. */
export function lineChart(
  history: readonly TrainingEpoch[],
  train: (e: TrainingEpoch) => number,
  val: (e: TrainingEpoch) => number,
  yMin: number,
  yMax: number,
  yStep: number,
  frame: ChartFrame = FRAME,
): LineChart {
  const n = Math.max(history.length, 2);
  const x = (epoch: number) => frame.left + ((epoch - 1) / (n - 1)) * (frame.right - frame.left);
  const y = (value: number) =>
    frame.bottom - ((value - yMin) / (yMax - yMin)) * (frame.bottom - frame.top);
  const path = (pick: (e: TrainingEpoch) => number) =>
    history.map((e, i) => `${i ? 'L' : 'M'}${x(e.epoch).toFixed(1)} ${y(pick(e)).toFixed(1)}`).join('');
  const yTicks: { y: number; label: string }[] = [];
  for (let v = yMin; v <= yMax + 1e-9; v += yStep) {
    yTicks.push({ y: y(v), label: formatNumber(v, 1) });
  }
  const marks = [1, ...[0.25, 0.5, 0.75].map((f) => Math.round(f * n)), n];
  const xTicks = [...new Set(marks)].map((epoch) => ({ x: x(epoch), label: String(epoch) }));
  return { train: path(train), val: path(val), yTicks, xTicks, x, y };
}

/** Coma decimal, como el resto de textos del proyecto. */
export function formatNumber(value: number, digits: number): string {
  return value.toFixed(digits).replace('.', ',');
}
