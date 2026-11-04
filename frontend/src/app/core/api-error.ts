import { HttpErrorResponse } from '@angular/common/http';

/** Códigos que no vienen del servidor: los genera el navegador. */
export const NETWORK_ERROR = 'NETWORK_ERROR';
export const UNKNOWN_ERROR = 'UNKNOWN_ERROR';

/** Error normalizado: `code` decide la lógica, `message` se muestra tal cual. */
export class ApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status = 0,
    readonly details: Record<string, unknown> | null = null,
    readonly retryAfterSeconds: number | null = null,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

interface ErrorEnvelope {
  error?: { code?: unknown; message?: unknown; details?: unknown };
}

function parseBody(body: unknown): ErrorEnvelope | null {
  if (typeof body === 'string') {
    try {
      return JSON.parse(body) as ErrorEnvelope;
    } catch {
      return null;
    }
  }
  return body && typeof body === 'object' ? (body as ErrorEnvelope) : null;
}

/** `Retry-After` admite segundos o una fecha HTTP. */
export function parseRetryAfter(value: string | null, now = Date.now()): number | null {
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, Math.ceil(seconds));
  const when = Date.parse(value);
  return Number.isNaN(when) ? null : Math.max(0, Math.ceil((when - now) / 1000));
}

export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  if (err instanceof HttpErrorResponse) {
    if (err.status === 0) {
      return new ApiError(NETWORK_ERROR, 'No pudimos comunicarnos con el servidor.');
    }
    const envelope = parseBody(err.error)?.error;
    const retryAfter = parseRetryAfter(err.headers?.get('Retry-After') ?? null);
    if (envelope && typeof envelope.code === 'string') {
      return new ApiError(
        envelope.code,
        typeof envelope.message === 'string' ? envelope.message : 'Ocurrió un error.',
        err.status,
        (envelope.details as Record<string, unknown>) ?? null,
        retryAfter,
      );
    }
    // Respuesta fuera de contrato, por ejemplo un 502 de Nginx con HTML.
    return new ApiError(
      UNKNOWN_ERROR,
      `El servidor respondió con un error inesperado (HTTP ${err.status}).`,
      err.status,
      null,
      retryAfter,
    );
  }
  return new ApiError(UNKNOWN_ERROR, 'Ocurrió un error inesperado.');
}
