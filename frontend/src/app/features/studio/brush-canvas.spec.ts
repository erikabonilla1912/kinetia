import { ComponentFixture, TestBed } from '@angular/core/testing';
import { BrushStroke, serializeStrokes } from '../../core/brush';
import { BrushCanvas, BrushSettings } from './brush-canvas';
import { BrushPanel } from './brush-panel';

const SETTINGS: BrushSettings = { radius: 0.05, strength: 0.6, fixedDirection: null, fallbackDirection: 45 };

describe('BrushCanvas', () => {
  let fixture: ComponentFixture<BrushCanvas>;
  let canvas: HTMLCanvasElement;
  const strokes = () => fixture.componentInstance.strokes();

  function fire(type: string, fx: number, fy: number, pointerId = 1): void {
    const rect = canvas.getBoundingClientRect();
    canvas.dispatchEvent(
      new PointerEvent(type, {
        clientX: rect.left + rect.width * fx,
        clientY: rect.top + rect.height * fy,
        bubbles: true,
        cancelable: true,
        pointerId,
      }),
    );
  }

  function drag(points: [number, number][], pointerId = 1): void {
    fire('pointerdown', points[0][0], points[0][1], pointerId);
    for (const [x, y] of points.slice(1)) fire('pointermove', x, y, pointerId);
    const end = points[points.length - 1];
    fire('pointerup', end[0], end[1], pointerId);
  }

  beforeEach(async () => {
    fixture = TestBed.createComponent(BrushCanvas);
    fixture.componentRef.setInput('strokes', []);
    fixture.componentRef.setInput('settings', SETTINGS);
    fixture.componentRef.setInput('workWidth', 1000);
    fixture.componentRef.setInput('workHeight', 500);
    const host = fixture.nativeElement as HTMLElement;
    host.style.cssText = 'display:block;position:fixed;left:0;top:0;width:400px;height:200px';
    await fixture.whenStable();
    canvas = host.querySelector('canvas')!;
    spyOn(canvas, 'setPointerCapture');
  });

  it('un trazo produce puntos normalizados con radio, fuerza y dirección del trazo', () => {
    drag([[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]]);
    expect(strokes().length).toBe(1);
    const [stroke] = strokes();
    expect(stroke.points[0][0]).toBeCloseTo(0.1, 1);
    expect(stroke.points[stroke.points.length - 1][0]).toBeCloseTo(0.9, 1);
    expect(stroke.radius).toBe(0.05);
    expect(stroke.strength).toBe(0.6);
    expect(stroke.direction_deg).toBeCloseTo(0, 0);
  });

  it('un trazo hacia abajo da 90 grados', () => {
    drag([[0.5, 0.1], [0.5, 0.9]]);
    expect(strokes()[0].direction_deg).toBeCloseTo(90, 0);
  });

  it('con dirección elegida, ignora el recorrido', async () => {
    fixture.componentRef.setInput('settings', { ...SETTINGS, fixedDirection: 270 });
    await fixture.whenStable();
    drag([[0.1, 0.5], [0.9, 0.5]]);
    expect(strokes()[0].direction_deg).toBe(270);
  });

  it('un toque usa la dirección de respaldo', () => {
    drag([[0.5, 0.5]]);
    expect(strokes()[0].points.length).toBe(1);
    expect(strokes()[0].direction_deg).toBe(45);
  });

  it('un segundo dedo no interrumpe el trazo', () => {
    fire('pointerdown', 0.1, 0.1, 1);
    fire('pointerdown', 0.9, 0.9, 2);
    fire('pointermove', 0.9, 0.9, 2);
    fire('pointerup', 0.9, 0.9, 2);
    expect(strokes().length).toBe(0);
    fire('pointerup', 0.1, 0.1, 1);
    expect(strokes().length).toBe(1);
  });

  it('pointercancel descarta el trazo', () => {
    fire('pointerdown', 0.1, 0.1);
    fire('pointercancel', 0.1, 0.1);
    expect(strokes().length).toBe(0);
  });

  it('no deja pintar más allá del límite de trazos', async () => {
    const full: BrushStroke[] = Array.from({ length: 200 }, () => ({
      points: [[0.5, 0.5]],
      radius: 0.05,
      direction_deg: 0,
      strength: 1,
    }));
    fixture.componentRef.setInput('strokes', full);
    await fixture.whenStable();
    drag([[0.1, 0.1], [0.2, 0.2]]);
    expect(strokes().length).toBe(200);
  });

  it('lo pintado se serializa con la forma del contrato', () => {
    drag([[0.1, 0.5], [0.9, 0.5]]);
    const parsed = JSON.parse(serializeStrokes(strokes())!);
    expect(Object.keys(parsed.strokes[0]).sort()).toEqual(['direction_deg', 'points', 'radius', 'strength']);
    for (const [x, y] of parsed.strokes[0].points) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(1);
    }
  });
});

describe('BrushPanel', () => {
  let fixture: ComponentFixture<BrushPanel>;
  const stroke: BrushStroke = { points: [[0.1, 0.1]], radius: 0.05, direction_deg: 0, strength: 1 };
  const q = (id: string) => (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>(`[data-testid="${id}"]`)!;

  beforeEach(async () => {
    fixture = TestBed.createComponent(BrushPanel);
    fixture.componentRef.setInput('strokes', [stroke, { ...stroke, direction_deg: 90 }]);
    fixture.componentRef.setInput('settings', SETTINGS);
    fixture.componentRef.setInput('mode', 'add');
    await fixture.whenStable();
  });

  it('deshacer quita el último trazo', async () => {
    q('brush-undo').click();
    await fixture.whenStable();
    expect(fixture.componentInstance.strokes().length).toBe(1);
    expect(fixture.componentInstance.strokes()[0].direction_deg).toBe(0);
  });

  it('borrar quita todos', async () => {
    q('brush-clear').click();
    await fixture.whenStable();
    expect(fixture.componentInstance.strokes()).toEqual([]);
    expect((q('brush-undo') as HTMLButtonElement).disabled).toBeTrue();
  });

  it('cambia entre add y only', async () => {
    q('mode-only').click();
    await fixture.whenStable();
    expect(fixture.componentInstance.mode()).toBe('only');
    q('mode-add').click();
    await fixture.whenStable();
    expect(fixture.componentInstance.mode()).toBe('add');
  });

  it('muestra el consumo frente a los límites del contrato', () => {
    expect(q('brush-usage').textContent).toContain('2/200');
    expect(q('brush-usage').textContent).toContain('2/2000');
  });

  it('el radio y la fuerza se ajustan con sus controles', async () => {
    const radius = q('brush-radius') as HTMLInputElement;
    expect(radius.min).toBe('0.005');
    expect(radius.max).toBe('0.2');
    radius.value = '0.1';
    radius.dispatchEvent(new Event('input'));
    const strength = q('brush-strength') as HTMLInputElement;
    strength.value = '0.25';
    strength.dispatchEvent(new Event('input'));
    await fixture.whenStable();
    expect(fixture.componentInstance.settings().radius).toBe(0.1);
    expect(fixture.componentInstance.settings().strength).toBe(0.25);
  });

  it('elegir dirección la fija; "del trazo" la libera', async () => {
    q('dir-fixed').click();
    await fixture.whenStable();
    expect(fixture.componentInstance.settings().fixedDirection).toBe(45);
    q('dir-from-stroke').click();
    await fixture.whenStable();
    expect(fixture.componentInstance.settings().fixedDirection).toBeNull();
  });

  it('avisa si se pide "solo lo pintado" sin trazos', async () => {
    fixture.componentRef.setInput('strokes', []);
    fixture.componentRef.setInput('mode', 'only');
    await fixture.whenStable();
    expect(q('only-needs-strokes')).not.toBeNull();
  });
});

