import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { testJob } from '../testing/fixtures.spec';
import { API_BASE, buildAnimateForm } from './api.service';
import { JOB_MAX_POLL_FAILURES, JOB_POLL_MS, JobRunner } from './job-runner';

const STATUS_URL = `${API_BASE}/jobs/job-1`;
const request = { file: new File(['x'], 'a.png'), mood: 'calma', output_format: 'mp4' as const };

describe('JobRunner', () => {
  let runner: JobRunner;
  let http: HttpTestingController;

  beforeEach(() => {
    jasmine.clock().install();
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting(), JobRunner],
    });
    runner = TestBed.inject(JobRunner);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    runner.reset();
    http.verify();
    jasmine.clock().uninstall();
  });

  function accept(): void {
    runner.start(request);
    http.expectOne(`${API_BASE}/animate`).flush(
      { job_id: 'job-1', status: 'queued', queue_position: 3, status_url: STATUS_URL },
      { status: 202, statusText: 'Accepted' },
    );
    jasmine.clock().tick(0);
  }

  it('empieza en idle', () => {
    expect(runner.state().phase).toBe('idle');
    expect(runner.busy()).toBeFalse();
  });

  it('pasa por uploading al subir', () => {
    runner.start(request);
    expect(runner.state().phase).toBe('uploading');
    expect(runner.busy()).toBeTrue();
    http.expectOne(`${API_BASE}/animate`).flush({ error: { code: 'INTERNAL_ERROR', message: 'x' } }, { status: 500, statusText: 'E' });
  });

  it('queued: guarda la posición que informó el 202', () => {
    accept();
    expect(runner.state().phase).toBe('queued');
    expect(runner.state().queuePosition).toBe(3);
    http.expectOne(STATUS_URL).flush(testJob({ status: 'queued', stage: 'queued', progress: 0 }));
    expect(runner.state().phase).toBe('queued');
    jasmine.clock().tick(JOB_POLL_MS);
    http.expectOne(STATUS_URL).flush(testJob({ status: 'succeeded', stage: 'done', progress: 1 }));
  });

  it('running: refleja stage y progress recibidos y consulta cada segundo', () => {
    accept();
    http.expectOne(STATUS_URL).flush(testJob({ stage: 'segmenting', progress: 0.13 }));
    expect(runner.state().phase).toBe('running');
    expect(runner.state().stage).toBe('segmenting');
    expect(runner.state().progress).toBe(0.13);

    jasmine.clock().tick(JOB_POLL_MS - 1);
    http.expectNone(STATUS_URL);
    jasmine.clock().tick(1);
    http.expectOne(STATUS_URL).flush(testJob({ stage: 'encoding', progress: 0.87 }));
    expect(runner.state().stage).toBe('encoding');
    expect(runner.state().progress).toBe(0.87);

    runner.reset();
  });

  it('succeeded: entrega el resultado y deja de consultar', () => {
    accept();
    const result = { url: `${API_BASE}/results/job-1.gif`, format: 'gif' as const, width: 3, height: 5, duration_s: 7, fps: 11, size_bytes: 13 };
    http.expectOne(STATUS_URL).flush(testJob({ status: 'succeeded', stage: 'done', progress: 1, result }));
    expect(runner.state().phase).toBe('succeeded');
    expect(runner.state().result).toEqual(result);
    expect(runner.busy()).toBeFalse();
    jasmine.clock().tick(JOB_POLL_MS * 5);
    http.expectNone(STATUS_URL);
  });

  it('failed: toma code y message del job, incluido JOB_INTERRUPTED', () => {
    accept();
    http.expectOne(STATUS_URL).flush(
      testJob({ status: 'failed', error: { code: 'JOB_INTERRUPTED', message: 'Se reinició el pod' } }),
    );
    expect(runner.state().phase).toBe('failed');
    expect(runner.state().error?.code).toBe('JOB_INTERRUPTED');
    expect(runner.state().error?.message).toBe('Se reinició el pod');
    jasmine.clock().tick(JOB_POLL_MS * 5);
    http.expectNone(STATUS_URL);
  });

  it('429 al crear: falla con QUEUE_FULL y Retry-After', () => {
    runner.start(request);
    http.expectOne(`${API_BASE}/animate`).flush(
      { error: { code: 'QUEUE_FULL', message: 'Cola llena' } },
      { status: 429, statusText: 'Too Many', headers: { 'Retry-After': '9' } },
    );
    expect(runner.state().phase).toBe('failed');
    expect(runner.state().error?.code).toBe('QUEUE_FULL');
    expect(runner.state().error?.retryAfterSeconds).toBe(9);
  });

  it('un corte de red durante la consulta no pierde el job: sigue intentando', () => {
    accept();
    http.expectOne(STATUS_URL).error(new ProgressEvent('error'));
    expect(runner.state().phase).toBe('queued');
    expect(runner.state().reconnecting).toBeTrue();
    jasmine.clock().tick(JOB_POLL_MS);
    http.expectOne(STATUS_URL).flush(testJob({ stage: 'rendering', progress: 0.5 }));
    expect(runner.state().reconnecting).toBeFalse();
    runner.reset();
  });

  it('se rinde tras muchos fallos seguidos de red', () => {
    accept();
    for (let i = 0; i < JOB_MAX_POLL_FAILURES; i++) {
      http.expectOne(STATUS_URL).error(new ProgressEvent('error'));
      jasmine.clock().tick(JOB_POLL_MS);
    }
    expect(runner.state().phase).toBe('failed');
    expect(runner.state().error?.code).toBe('NETWORK_ERROR');
  });

  it('un 404 durante la consulta falla de inmediato', () => {
    accept();
    http.expectOne(STATUS_URL).flush(
      { error: { code: 'JOB_NOT_FOUND', message: 'No existe' } },
      { status: 404, statusText: 'NF' },
    );
    expect(runner.state().phase).toBe('failed');
    expect(runner.state().error?.code).toBe('JOB_NOT_FOUND');
  });
});

describe('buildAnimateForm', () => {
  const file = new File(['x'], 'a.png');

  it('envía los nombres de campo del contrato', () => {
    const form = buildAnimateForm({
      file,
      mood: 'epico',
      output_format: 'gif',
      duration_s: 3.5,
      mask_strokes: '{"strokes":[]}',
      brush_mode: 'only',
      gallery_opt_in: true,
      seed: 42,
    });
    expect(form.get('mood')).toBe('epico');
    expect(form.get('output_format')).toBe('gif');
    expect(form.get('duration_s')).toBe('3.5');
    expect(form.get('mask_strokes')).toBe('{"strokes":[]}');
    expect(form.get('brush_mode')).toBe('only');
    expect(form.get('gallery_opt_in')).toBe('true');
    expect(form.get('seed')).toBe('42');
    expect(form.get('file')).toBeInstanceOf(File);
  });

  it('omite lo opcional que no se usó', () => {
    const form = buildAnimateForm({ file, mood: 'calma', output_format: 'mp4', brush_mode: 'only' });
    for (const field of ['duration_s', 'mask_strokes', 'brush_mode', 'gallery_opt_in', 'seed']) {
      expect(form.has(field)).withContext(field).toBeFalse();
    }
  });
});
