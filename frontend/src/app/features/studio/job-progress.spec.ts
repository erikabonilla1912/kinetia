import { LOCALE_ID } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { JobState, STAGE_LABELS } from '../../core/job-runner';
import { JobProgress } from './job-progress';

const BASE: JobState = {
  phase: 'idle',
  uploadFraction: null,
  queuePosition: null,
  stage: null,
  progress: null,
  reconnecting: false,
  result: null,
  job: null,
  error: null,
};

describe('JobProgress', () => {
  let fixture: ComponentFixture<JobProgress>;
  const root = () => fixture.nativeElement as HTMLElement;
  const text = (id: string) => root().querySelector(`[data-testid="${id}"]`)?.textContent?.trim();
  const bar = () => root().querySelector('[role="progressbar"]')!;

  async function render(state: Partial<JobState>): Promise<void> {
    fixture.componentRef.setInput('state', { ...BASE, ...state });
    await fixture.whenStable();
  }

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [{ provide: LOCALE_ID, useValue: 'es' }] });
    spyOn(window, 'matchMedia').and.returnValue({ matches: true } as MediaQueryList);
    fixture = TestBed.createComponent(JobProgress);
  });

  it('uploading: progreso real de la subida', async () => {
    await render({ phase: 'uploading', uploadFraction: 0.4 });
    expect(text('job-title')).toBe('Subiendo tu archivo');
    expect(text('job-percent')).toBe('40 %');
    expect(root().querySelector('[data-phase]')?.getAttribute('data-phase')).toBe('uploading');
  });

  it('uploading sin total conocido: no inventa un porcentaje', async () => {
    await render({ phase: 'uploading', uploadFraction: null });
    expect(text('job-percent')).toBeUndefined();
    expect(bar().hasAttribute('aria-valuenow')).toBeFalse();
  });

  it('queued: muestra la posición informada', async () => {
    await render({ phase: 'queued', stage: 'queued', queuePosition: 4, progress: 0 });
    expect(text('job-title')).toBe(STAGE_LABELS.queued);
    expect(text('job-queue')).toContain('4 trabajos');
  });

  it('queued sin datos de cola: no los inventa', async () => {
    await render({ phase: 'queued', stage: 'queued' });
    expect(text('job-queue')).toBeUndefined();
  });

  for (const stage of ['segmenting', 'depth', 'rendering', 'encoding', 'done'] as const) {
    it(`running/${stage}: etiqueta de la etapa`, async () => {
      await render({ phase: 'running', stage, progress: 0.31 });
      expect(text('job-title')).toBe(STAGE_LABELS[stage]);
      expect(text('job-percent')).toBe('31 %');
      expect(bar().getAttribute('aria-valuenow')).toBe('31');
    });
  }

  it('succeeded', async () => {
    await render({ phase: 'succeeded', stage: 'done', progress: 1 });
    expect(text('job-title')).toBe('Listo');
    expect(text('job-percent')).toBe('100 %');
  });

  it('failed', async () => {
    await render({ phase: 'failed', stage: 'rendering', progress: 0.5 });
    expect(text('job-title')).toBe('No se pudo completar');
  });

  describe('espera viva', () => {
    const states = () =>
      [...root().querySelectorAll('[data-testid="job-step"]')].map((el) => el.getAttribute('data-state'));

    it('marca la etapa actual y las anteriores según la API', async () => {
      await render({ phase: 'running', stage: 'rendering', progress: 0.5 });
      // Cola, Lectura, Animación, Codificación: sin etapa de profundidad (contrato v1.8).
      expect(states()).toEqual(['done', 'done', 'current', 'pending']);
      expect(root().querySelector('[aria-current="step"]')?.textContent).toContain('Animación');
    });

    it('al subir, ninguna etapa ha empezado', async () => {
      await render({ phase: 'uploading', uploadFraction: 0.1 });
      expect(states().every((state) => state === 'pending')).toBeTrue();
    });

    it('con stage done, todas quedaron atrás', async () => {
      await render({ phase: 'running', stage: 'done', progress: 1 });
      expect(states().every((state) => state === 'done')).toBeTrue();
    });

    it('cuenta el tiempo real y se detiene al terminar', async () => {
      jasmine.clock().install();
      jasmine.clock().mockDate(new Date('2026-01-01T00:00:00Z'));
      try {
        await render({ phase: 'running', stage: 'rendering', progress: 0.5 });
        expect(text('job-elapsed')).toContain('0 s');
        jasmine.clock().tick(12000);
        await fixture.whenStable();
        expect(text('job-elapsed')).toContain('12 s');
        expect(root().querySelector('[data-testid="job-shimmer"]')).not.toBeNull();

        await render({ phase: 'succeeded', stage: 'done', progress: 1 });
        expect(text('job-elapsed')).toBeUndefined();
        expect(root().querySelector('[data-testid="job-shimmer"]')).toBeNull();
      } finally {
        fixture.destroy();
        jasmine.clock().uninstall();
      }
    });

    it('no estima cuánto falta', async () => {
      await render({ phase: 'running', stage: 'rendering', progress: 0.5 });
      expect(root().textContent).not.toMatch(/falta|restante/i);
    });
  });

  it('avisa si se perdió la conexión mientras consulta', async () => {
    await render({ phase: 'running', stage: 'rendering', progress: 0.5 });
    expect(text('job-reconnecting')).toBeUndefined();
    await render({ phase: 'running', stage: 'rendering', progress: 0.5, reconnecting: true });
    expect(text('job-reconnecting')).toContain('Se perdió la conexión');
  });
});
