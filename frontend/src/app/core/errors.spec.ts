import { HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { ComponentRef } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ErrorNotice } from '../shared/error-notice';
import { ApiError, NETWORK_ERROR, UNKNOWN_ERROR, parseRetryAfter, toApiError } from './api-error';
import { ERROR_GUIDES, describeError } from './error-catalog';

/** Sección 4 del contrato 1.2.0, incluidos los que solo aparecen en un job fallido. */
const CONTRACT_CODES = [
  'MISSING_FILE',
  'CORRUPT_FILE',
  'UNAUTHORIZED',
  'JOB_NOT_FOUND',
  'RESULT_NOT_FOUND',
  'METRICS_NOT_FOUND',
  'JOB_NOT_READY',
  'FILE_TOO_LARGE',
  'UNSUPPORTED_FORMAT',
  'RESOLUTION_TOO_LOW',
  'RESOLUTION_TOO_HIGH',
  'VIDEO_TOO_LONG',
  'INVALID_PARAMS',
  'QUEUE_FULL',
  'MODEL_NOT_READY',
  'INTERNAL_ERROR',
  'NOT_FOUND',
  'METHOD_NOT_ALLOWED',
  'RANGE_NOT_SATISFIABLE',
  'JOB_INTERRUPTED',
  'ENGINE_UNAVAILABLE',
  'ENCODER_UNAVAILABLE',
];

describe('describeError', () => {
  for (const code of CONTRACT_CODES) {
    it(`da una guía propia a ${code} y conserva el message del servidor`, () => {
      expect(ERROR_GUIDES[code]).withContext(code).toBeDefined();
      const view = describeError(new ApiError(code, `mensaje del servidor ${code}`));
      expect(view.code).toBe(code);
      expect(view.message).toBe(`mensaje del servidor ${code}`);
      expect(view.title.length).toBeGreaterThan(0);
      expect(view.hint.length).toBeGreaterThan(0);
    });
  }

  it('cada guía es distinta', () => {
    const titles = CONTRACT_CODES.map((code) => ERROR_GUIDES[code].title);
    expect(new Set(titles).size).toBe(titles.length);
  });

  it('un código desconocido cae en la guía genérica sin perder el mensaje', () => {
    const view = describeError(new ApiError('CODIGO_FUTURO', 'detalle'));
    expect(view.title).toBe(ERROR_GUIDES[UNKNOWN_ERROR].title);
    expect(view.message).toBe('detalle');
    expect(view.code).toBe('CODIGO_FUTURO');
  });
});

describe('toApiError', () => {
  it('lee el formato único de error', () => {
    const error = toApiError(
      new HttpErrorResponse({
        status: 413,
        error: { error: { code: 'FILE_TOO_LARGE', message: 'Pesa mucho', details: { max_bytes: 5 } } },
      }),
    );
    expect(error.code).toBe('FILE_TOO_LARGE');
    expect(error.message).toBe('Pesa mucho');
    expect(error.status).toBe(413);
    expect(error.details).toEqual({ max_bytes: 5 });
  });

  it('lee el error cuando la petición pidió texto', () => {
    const error = toApiError(
      new HttpErrorResponse({
        status: 404,
        error: JSON.stringify({ error: { code: 'METRICS_NOT_FOUND', message: 'Falta' } }),
      }),
    );
    expect(error.code).toBe('METRICS_NOT_FOUND');
  });

  it('lee Retry-After en un 429', () => {
    const error = toApiError(
      new HttpErrorResponse({
        status: 429,
        headers: new HttpHeaders({ 'Retry-After': '17' }),
        error: { error: { code: 'QUEUE_FULL', message: 'Llena' } },
      }),
    );
    expect(error.retryAfterSeconds).toBe(17);
  });

  it('distingue la falta de conexión', () => {
    expect(toApiError(new HttpErrorResponse({ status: 0 })).code).toBe(NETWORK_ERROR);
  });

  it('tolera una respuesta fuera de contrato, como un 502 de Nginx', () => {
    const error = toApiError(new HttpErrorResponse({ status: 502, error: '<html>Bad gateway</html>' }));
    expect(error.code).toBe(UNKNOWN_ERROR);
    expect(error.message).toContain('502');
  });
});

describe('parseRetryAfter', () => {
  it('admite segundos y fechas', () => {
    expect(parseRetryAfter('30')).toBe(30);
    expect(parseRetryAfter(null)).toBeNull();
    expect(parseRetryAfter('tonteria')).toBeNull();
    const now = Date.parse('2026-01-01T00:00:00Z');
    expect(parseRetryAfter('Thu, 01 Jan 2026 00:00:10 GMT', now)).toBe(10);
  });
});

describe('ErrorNotice', () => {
  let fixture: ComponentFixture<ErrorNotice>;
  let ref: ComponentRef<ErrorNotice>;
  const text = () => (fixture.nativeElement as HTMLElement).textContent ?? '';
  const button = (label: string) =>
    [...(fixture.nativeElement as HTMLElement).querySelectorAll('button')].find((b) =>
      b.textContent?.includes(label),
    );

  beforeEach(() => {
    jasmine.clock().install();
    fixture = TestBed.createComponent(ErrorNotice);
    ref = fixture.componentRef;
  });

  afterEach(() => {
    fixture.destroy();
    jasmine.clock().uninstall();
  });

  it('muestra guía y message, y expone el code', async () => {
    ref.setInput('error', new ApiError('INTERNAL_ERROR', 'mensaje literal'));
    await fixture.whenStable();
    expect(text()).toContain('mensaje literal');
    expect(text()).toContain(ERROR_GUIDES['INTERNAL_ERROR'].title);
    expect(fixture.nativeElement.querySelector('[data-error-code]').dataset.errorCode).toBe('INTERNAL_ERROR');
    expect(fixture.nativeElement.querySelector('[role="alert"]')).not.toBeNull();
  });

  it('en cola llena bloquea el reintento hasta que pase Retry-After', async () => {
    ref.setInput('error', new ApiError('QUEUE_FULL', 'Llena', 429, null, 2));
    await fixture.whenStable();
    expect(text()).toContain('2 s');
    expect(button('Reintentar')!.disabled).toBeTrue();
    jasmine.clock().tick(2000);
    await fixture.whenStable();
    expect(button('Reintentar')!.disabled).toBeFalse();
  });

  it('un error de archivo no ofrece reintentar lo mismo', async () => {
    ref.setInput('error', new ApiError('FILE_TOO_LARGE', 'Pesa'));
    ref.setInput('canChangeFile', true);
    await fixture.whenStable();
    expect(button('Reintentar')).toBeUndefined();
    expect(button('otro archivo')).toBeDefined();
  });
});
