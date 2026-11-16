import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { TrainingEpoch } from '../../core/api.models';
import { TRAINING_HISTORY_URL } from '../../core/api.service';
import { provideLoadedConfig } from '../../testing/fixtures.spec';
import { TrainingPage } from './training.page';
import { lineChart, summarize } from './training-summary';

function epoch(n: number, valMiou: number, valLoss: number, seconds = 100): TrainingEpoch {
  return {
    epoch: n,
    train_loss: 2 - n * 0.1,
    val_loss: valLoss,
    train_miou: 0.3 + n * 0.05,
    val_miou: valMiou,
    val_pixel_accuracy: 0.8,
    val_iou_per_class: { fondo: valMiou + 0.1, cielo: valMiou - 0.1 },
    lr: 0.001,
    seconds,
    train_seconds: seconds - 10,
  };
}

// Época 4 supera a la 3 por menos de 0,001: train.py no la guarda, igual que la 89 del entrenamiento real.
const HISTORY = [epoch(1, 0.3, 1.5), epoch(2, 0.5, 1.0), epoch(3, 0.6, 0.9), epoch(4, 0.6009, 1.1)];

describe('summarize', () => {
  it('reproduce la regla de guardado de train.py con la mejora mínima', () => {
    const s = summarize(HISTORY)!;
    expect(s.saved.epoch).toBe(3);
    expect(s.peak.epoch).toBe(4);
    expect(s.minValLoss.epoch).toBe(3);
    expect(s.epochs).toBe(4);
    expect(s.totalSeconds).toBe(400);
  });

  it('sin épocas no inventa un resumen', () => {
    expect(summarize([])).toBeNull();
  });

  it('el gráfico ubica la primera y la última época en los bordes', () => {
    const chart = lineChart(HISTORY, (e) => e.train_loss, (e) => e.val_loss, 0, 2, 0.5);
    expect(chart.x(1)).toBe(46);
    expect(chart.x(4)).toBe(548);
    expect(chart.yTicks.map((t) => t.label)).toEqual(['0,0', '0,5', '1,0', '1,5', '2,0']);
    expect(chart.val.startsWith('M46.0 ')).toBeTrue();
  });
});

describe('TrainingPage', () => {
  let fixture: ComponentFixture<TrainingPage>;
  let http: HttpTestingController;
  const root = () => fixture.nativeElement as HTMLElement;
  const q = (id: string) => root().querySelector<HTMLElement>(`[data-testid="${id}"]`);

  beforeEach(async () => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    });
    http = TestBed.inject(HttpTestingController);
    provideLoadedConfig(http);
    fixture = TestBed.createComponent(TrainingPage);
    await fixture.whenStable();
  });

  it('muestra las cifras calculadas del historial y el mIoU de prueba de la API', async () => {
    expect(q('training-loading')).not.toBeNull();
    http.expectOne(TRAINING_HISTORY_URL).flush(HISTORY);
    http.expectOne('/api/v1/model/metrics').flush('{"split":"test","miou":0.6347}');
    await fixture.whenStable();
    expect(q('tile-epochs')?.textContent?.trim()).toBe('4');
    expect(q('tile-hours')?.textContent).toContain('0,1 h');
    expect(q('tile-saved')?.textContent?.trim()).toBe('3');
    expect(q('tile-test')?.textContent).toContain('0,6347');
    expect(q('chart-loss')).not.toBeNull();
    expect(q('chart-miou')?.textContent).toContain('época 3');
    expect(root().querySelectorAll('[data-testid="training-class-row"]').length).toBe(2);
  });

  it('no muestra un mIoU de prueba si metrics.json es del modelo dummy', async () => {
    http.expectOne(TRAINING_HISTORY_URL).flush(HISTORY);
    http.expectOne('/api/v1/model/metrics').flush('{"split":"dummy","dummy":true,"miou":0.9}');
    await fixture.whenStable();
    expect(q('tile-test')?.textContent).toContain('sin dato');
  });

  it('avisa si el historial no se puede cargar', async () => {
    http.expectOne(TRAINING_HISTORY_URL).flush('no', { status: 404, statusText: 'Not Found' });
    http.expectOne('/api/v1/model/metrics').flush('{"split":"test","miou":0.6}');
    await fixture.whenStable();
    expect(q('training-error')).not.toBeNull();
  });
});
