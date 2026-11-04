import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideLoadedConfig } from '../../testing/fixtures.spec';
import { MetricsPage } from './metrics.page';

const RAW = `{"model_sha256":"aa","dataset_manifest_sha256":"bb","split":"test","num_images":37,
"evaluated_at":"2026-03-04T05:06:07Z","miou":0.61230,"dice_macro":0.7071,"pixel_accuracy":0.8123,
"per_class":{"fondo":{"iou":0.11,"dice":0.22,"support_pixels":33},"cielo":{"iou":0.44,"dice":0.55,"support_pixels":66}},
"confusion_matrix":[[7,1],[2,9]],"classes":["fondo","cielo"]}`;

describe('MetricsPage', () => {
  let fixture: ComponentFixture<MetricsPage>;
  let http: HttpTestingController;
  const root = () => fixture.nativeElement as HTMLElement;
  const q = (id: string) => root().querySelector<HTMLElement>(`[data-testid="${id}"]`);

  beforeEach(async () => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    http = TestBed.inject(HttpTestingController);
    provideLoadedConfig(http);
    fixture = TestBed.createComponent(MetricsPage);
    await fixture.whenStable();
  });

  it('muestra un estado de carga mientras llega la respuesta', () => {
    expect(q('metrics-loading')).not.toBeNull();
    http.expectOne('/api/v1/model/metrics').flush(RAW);
  });

  it('muestra los valores tal como llegan', async () => {
    http.expectOne('/api/v1/model/metrics').flush(RAW);
    await fixture.whenStable();
    expect(q('metric-miou')?.textContent).toContain('0.6123');
    expect(q('metric-dice')?.textContent).toContain('0.7071');
    expect(q('metric-accuracy')?.textContent).toContain('0.8123');
    expect(q('metric-images')?.textContent).toContain('37');
    expect(root().querySelectorAll('[data-testid="metrics-row"]').length).toBe(2);
    const cells = [...root().querySelectorAll('[data-testid="confusion-cell"]')].map((c) => c.textContent?.trim());
    expect(cells).toEqual(['7', '1', '2', '9']);
  });

  it('conserva la respuesta original sin reformatear', async () => {
    http.expectOne('/api/v1/model/metrics').flush(RAW);
    await fixture.whenStable();
    expect(q('metrics-raw')?.textContent).toBe(RAW);
    expect(q('metrics-raw')?.textContent).toContain('0.61230');
  });

  it('un metrics.json real no lleva aviso de dummy', async () => {
    http.expectOne('/api/v1/model/metrics').flush(RAW);
    await fixture.whenStable();
    expect(q('metrics-dummy')).toBeNull();
    expect(root().textContent).toContain('imágenes reservadas para prueba');
  });

  for (const [name, marks] of [
    ['dummy: true', '"dummy":true,"split":"test"'],
    ['split: dummy', '"split":"dummy"'],
  ]) {
    it(`con ${name} avisa que no son resultados del modelo`, async () => {
      const dummy = RAW.replace('"split":"test"', marks)
        .replace('"miou":0.61230', '"miou":null')
        .replace('"iou":0.11', '"iou":null');
      http.expectOne('/api/v1/model/metrics').flush(dummy);
      await fixture.whenStable();
      expect(q('metrics-dummy')?.textContent).toContain('no son resultados del modelo');
      expect(q('metric-miou')?.textContent).toContain('sin dato');
      expect(root().textContent).not.toContain('imágenes reservadas para prueba');
    });
  }

  it('sin metrics.json muestra METRICS_NOT_FOUND', async () => {
    http
      .expectOne('/api/v1/model/metrics')
      .flush(JSON.stringify({ error: { code: 'METRICS_NOT_FOUND', message: 'Falta metrics.json' } }), {
        status: 404,
        statusText: 'NF',
      });
    await fixture.whenStable();
    expect(root().querySelector('[data-error-code="METRICS_NOT_FOUND"]')).not.toBeNull();
    expect(root().textContent).toContain('Falta metrics.json');
    expect(q('metrics-tiles')).toBeNull();
  });
});
