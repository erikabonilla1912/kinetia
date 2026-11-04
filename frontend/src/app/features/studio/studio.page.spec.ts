import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { LOCALE_ID } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { JobRunner } from '../../core/job-runner';
import { TEST_CONFIG, grayscalePng, provideLoadedConfig, testAnalysis, testJob, toBase64 } from '../../testing/fixtures.spec';
import { StudioPage } from './studio.page';

/** PNG real que el navegador sí decodifica, con el tamaño mínimo pedido. */
async function realPng(size: number): Promise<File> {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const blob = await new Promise<Blob>((done) => canvas.toBlob((b) => done(b!), 'image/png'));
  return new File([blob], 'foto.png', { type: 'image/png' });
}

describe('StudioPage', () => {
  let fixture: ComponentFixture<StudioPage>;
  let http: HttpTestingController;
  const root = () => fixture.nativeElement as HTMLElement;
  const q = <T extends HTMLElement = HTMLElement>(id: string) => root().querySelector<T>(`[data-testid="${id}"]`);

  const settled = async () => {
    for (let i = 0; i < 3; i++) {
      await new Promise((done) => setTimeout(done, 20));
      await fixture.whenStable();
    }
  };

  beforeEach(async () => {
    spyOn(window, 'matchMedia').and.returnValue({ matches: true } as MediaQueryList);
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting(), { provide: LOCALE_ID, useValue: 'es' }],
    });
    http = TestBed.inject(HttpTestingController);
    provideLoadedConfig(http, { ...TEST_CONFIG, limits: { ...TEST_CONFIG.limits, max_upload_bytes: 1_000_000 } });
    fixture = TestBed.createComponent(StudioPage);
    await fixture.whenStable();
  });

  afterEach(() => {
    fixture.destroy();
    http.verify();
  });

  async function load(): Promise<void> {
    await fixture.componentInstance.onFile(await realPng(400));
    await settled();
  }

  it('lee los límites de /config en la zona de carga', () => {
    expect(q('dropzone-limits')?.textContent).toContain('7 s');
    expect(q('dropzone-limits')?.textContent).toContain('300 px');
  });

  it('una subida inválida muestra el error y no llama a la API', async () => {
    await fixture.componentInstance.onFile(new File(['no soy una imagen ni un video'], 'foto.jpg'));
    await settled();
    expect(root().querySelector('[data-error-code]')?.getAttribute('data-error-code')).toBe('UNSUPPORTED_FORMAT');
    expect(fixture.componentInstance.media()).toBeNull();
    http.expectNone('/api/v1/analyze');
  });

  it('una imagen muy pequeña se rechaza con el mínimo de /config', async () => {
    await fixture.componentInstance.onFile(await realPng(100));
    await settled();
    expect(root().querySelector('[data-error-code]')?.getAttribute('data-error-code')).toBe('RESOLUTION_TOO_LOW');
  });

  it('un archivo válido se analiza y se muestran los datos de la respuesta', async () => {
    await load();
    expect(q('xray-loading')).not.toBeNull();
    const png = await grayscalePng(2, 1, [1, 3]);
    http
      .expectOne('/api/v1/analyze')
      .flush(testAnalysis({ width: 2, height: 1, mask_png_base64: toBase64(png) }));
    await settled();
    expect(root().querySelectorAll('[data-testid="class-row"]').length).toBe(2);
    expect(q('stage')?.style.aspectRatio).toBe('2 / 1');
  });

  it('un error de /analyze se muestra con su code', async () => {
    await load();
    http
      .expectOne('/api/v1/analyze')
      .flush({ error: { code: 'MODEL_NOT_READY', message: 'Cargando' } }, { status: 503, statusText: 'E' });
    await settled();
    expect(root().querySelector('[data-error-code]')?.getAttribute('data-error-code')).toBe('MODEL_NOT_READY');
    expect(root().textContent).toContain('Cargando');
  });

  it('animar envía mood, formato, duración y seed; el resultado abre el comparador', async () => {
    await load();
    http.expectOne('/api/v1/analyze').flush(testAnalysis({ mask_png_base64: toBase64(await grayscalePng(1, 1, [0])) }));
    await settled();

    q('animate')!.click();
    const post = http.expectOne('/api/v1/animate');
    const form = post.request.body as FormData;
    expect(form.get('mood')).toBe('calma');
    expect(form.get('output_format')).toBe('mp4');
    expect(form.get('duration_s')).toBe('4');
    expect(form.get('gallery_opt_in')).toBe('false');
    expect(form.has('seed')).toBeTrue();
    expect(form.has('mask_strokes')).toBeFalse();
    post.flush({ job_id: 'j', status: 'queued', queue_position: 1, status_url: '/api/v1/jobs/j' }, { status: 202, statusText: 'A' });
    await settled();

    const result = { url: '/api/v1/results/j.mp4', format: 'mp4' as const, width: 4, height: 2, duration_s: 4, fps: 24, size_bytes: 2048 };
    http.expectOne('/api/v1/jobs/j').flush(
      testJob({
        status: 'succeeded',
        stage: 'done',
        progress: 1,
        result,
        seed: 321,
        timings_ms: { segmentation: 1.5, depth: null, render: 2.5, encode: null, total: 9.5 },
      }),
    );
    await settled();

    expect(q('result-seed')?.textContent).toBe('321');
    expect(q('timing-segmentation')?.textContent).toBe('1,5 ms');
    expect(q('timing-depth')?.textContent).toBe('sin medir');
    expect(q('timing-encode')?.textContent).toBe('sin medir');
    expect(q('timing-total')?.textContent).toBe('9,5 ms');

    expect(q('comparator-handle')).not.toBeNull();
    expect(q<HTMLAnchorElement>('download')?.getAttribute('href')).toBe('/api/v1/results/j.mp4');
    expect(q('result-meta')?.textContent).toContain('4×2');
    expect(q('export-other')?.textContent).toContain('GIF');

    // Exportar al otro formato repite el mismo seed: es la misma animación.
    const firstSeed = form.get('seed');
    q('export-other')!.click();
    const second = http.expectOne('/api/v1/animate');
    const secondForm = second.request.body as FormData;
    expect(secondForm.get('output_format')).toBe('gif');
    expect(secondForm.get('seed')).toBe(firstSeed);
    fixture.debugElement.injector.get(JobRunner).reset();
  });

  it('una cola llena muestra QUEUE_FULL y conserva la foto', async () => {
    await load();
    http.expectOne('/api/v1/analyze').flush(testAnalysis({ mask_png_base64: toBase64(await grayscalePng(1, 1, [0])) }));
    await settled();
    q('animate')!.click();
    http
      .expectOne('/api/v1/animate')
      .flush({ error: { code: 'QUEUE_FULL', message: 'Hay muchos' } }, { status: 429, statusText: 'T', headers: { 'Retry-After': '30' } });
    await settled();
    expect(root().querySelector('[data-error-code="QUEUE_FULL"]')).not.toBeNull();
    expect(fixture.componentInstance.media()).not.toBeNull();
  });

  it('"Solo lo pintado" sin trazos bloquea el envío', async () => {
    await load();
    http.expectOne('/api/v1/analyze').flush(testAnalysis({ mask_png_base64: toBase64(await grayscalePng(1, 1, [0])) }));
    await settled();
    fixture.componentInstance.brushMode.set('only');
    await settled();
    expect(q<HTMLButtonElement>('animate')?.disabled).toBeTrue();
    fixture.componentInstance.strokes.set([{ points: [[0.5, 0.5]], radius: 0.05, direction_deg: 0, strength: 1 }]);
    await settled();
    expect(q<HTMLButtonElement>('animate')?.disabled).toBeFalse();
    q('animate')!.click();
    const post = http.expectOne('/api/v1/animate');
    const form = post.request.body as FormData;
    expect(form.get('brush_mode')).toBe('only');
    expect(JSON.parse(form.get('mask_strokes') as string).strokes.length).toBe(1);
    post.flush({ error: { code: 'INVALID_PARAMS', message: 'x' } }, { status: 422, statusText: 'E' });
  });
});
