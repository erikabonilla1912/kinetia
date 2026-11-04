import {
  BRUSH_LIMITS,
  BrushPoint,
  BrushStroke,
  appendPoint,
  canAddStroke,
  directionFromPoints,
  normalizePoint,
  serializeStrokes,
} from './brush';

const stroke = (points: BrushPoint[], overrides: Partial<BrushStroke> = {}): BrushStroke => ({
  points,
  radius: 0.04,
  direction_deg: 90,
  strength: 0.7,
  ...overrides,
});

describe('serializeStrokes', () => {
  it('omite el campo cuando no hay trazos', () => {
    expect(serializeStrokes([])).toBeUndefined();
  });

  it('produce exactamente la forma del contrato', () => {
    const json = serializeStrokes([stroke([[0.12, 0.4], [0.18, 0.38]])]);
    expect(JSON.parse(json!)).toEqual({
      strokes: [
        { points: [[0.12, 0.4], [0.18, 0.38]], radius: 0.04, direction_deg: 90, strength: 0.7 },
      ],
    });
  });

  it('acota los valores a los rangos del contrato', () => {
    const json = serializeStrokes([
      stroke([[-0.5, 1.7]], { radius: 0.9, direction_deg: -90, strength: 3 }),
      stroke([[0.5, 0.5]], { radius: 0.0001, direction_deg: 450, strength: -1 }),
    ]);
    const [first, second] = JSON.parse(json!).strokes;
    expect(first.points).toEqual([[0, 1]]);
    expect(first.radius).toBe(BRUSH_LIMITS.maxRadius);
    expect(first.direction_deg).toBe(270);
    expect(first.strength).toBe(1);
    expect(second.radius).toBe(BRUSH_LIMITS.minRadius);
    expect(second.direction_deg).toBe(90);
    expect(second.strength).toBe(0);
  });

  it('no supera 200 trazos', () => {
    const many = Array.from({ length: 250 }, () => stroke([[0.5, 0.5]]));
    expect(JSON.parse(serializeStrokes(many)!).strokes.length).toBe(200);
  });

  it('no supera 2000 puntos en total', () => {
    const points: BrushPoint[] = Array.from({ length: 1500 }, () => [0.5, 0.5]);
    const parsed = JSON.parse(serializeStrokes([stroke(points), stroke(points)])!);
    const total = parsed.strokes.reduce((sum: number, s: BrushStroke) => sum + s.points.length, 0);
    expect(total).toBe(2000);
  });
});

describe('geometría del pincel', () => {
  it('normaliza contra el tamaño mostrado, con origen arriba a la izquierda', () => {
    expect(normalizePoint(50, 25, 100, 100)).toEqual([0.5, 0.25]);
    expect(normalizePoint(-10, 500, 100, 100)).toEqual([0, 1]);
  });

  it('usa la convención 0 = derecha y 90 = abajo', () => {
    expect(directionFromPoints([[0.1, 0.5], [0.9, 0.5]], 100, 100)).toBeCloseTo(0);
    expect(directionFromPoints([[0.5, 0.1], [0.5, 0.9]], 100, 100)).toBeCloseTo(90);
    expect(directionFromPoints([[0.9, 0.5], [0.1, 0.5]], 100, 100)).toBeCloseTo(180);
    expect(directionFromPoints([[0.5, 0.9], [0.5, 0.1]], 100, 100)).toBeCloseTo(270);
  });

  it('calcula en el espacio de la imagen: en 2:1 un trazo normalizado igual no da 45°', () => {
    const direction = directionFromPoints([[0, 0], [0.5, 0.5]], 2000, 1000)!;
    expect(direction).toBeCloseTo((Math.atan2(500, 1000) * 180) / Math.PI);
  });

  it('un toque no tiene dirección propia', () => {
    expect(directionFromPoints([[0.5, 0.5]], 100, 100)).toBeNull();
    expect(directionFromPoints([[0.5, 0.5], [0.5001, 0.5]], 100, 100)).toBeNull();
  });

  it('descarta puntos demasiado cercanos para ahorrar el límite', () => {
    const points: BrushPoint[] = [[0.5, 0.5]];
    expect(appendPoint(points, [0.5001, 0.5], 0.04, 100, 100)).toBe(points);
    expect(appendPoint(points, [0.6, 0.5], 0.04, 100, 100).length).toBe(2);
  });

  it('detecta cuándo ya no cabe otro trazo', () => {
    expect(canAddStroke([])).toBeTrue();
    expect(canAddStroke(Array.from({ length: 200 }, () => stroke([[0, 0]])))).toBeFalse();
    const full: BrushPoint[] = Array.from({ length: 2000 }, () => [0, 0]);
    expect(canAddStroke([stroke(full)])).toBeFalse();
  });
});
