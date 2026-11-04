import { HttpClient, HttpEvent, HttpParams, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, catchError, of, throwError } from 'rxjs';
import {
  AnalyzeResponse,
  AnimateAccepted,
  AnimateRequest,
  AppConfig,
  GalleryResponse,
  HealthResponse,
  JobResponse,
} from './api.models';
import { toApiError } from './api-error';

/** Única raíz de la API: relativa, para que Nginx o el proxy de desarrollo la resuelvan. */
export const API_BASE = '/api/v1';

export function buildAnimateForm(request: AnimateRequest): FormData {
  const form = new FormData();
  form.append('file', request.file);
  form.append('mood', request.mood);
  form.append('output_format', request.output_format);
  if (request.duration_s !== undefined) form.append('duration_s', String(request.duration_s));
  if (request.mask_strokes) {
    form.append('mask_strokes', request.mask_strokes);
    form.append('brush_mode', request.brush_mode ?? 'add');
  }
  if (request.gallery_opt_in !== undefined) {
    form.append('gallery_opt_in', String(request.gallery_opt_in));
  }
  if (request.seed !== undefined) form.append('seed', String(request.seed));
  return form;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);

  private normalized<T>(source: Observable<T>): Observable<T> {
    return source.pipe(catchError((err) => throwError(() => toApiError(err))));
  }

  getConfig(): Observable<AppConfig> {
    return this.normalized(this.http.get<AppConfig>(`${API_BASE}/config`));
  }

  /** Un 503 también trae el estado de carga: es una respuesta válida, no un fallo. */
  getHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${API_BASE}/health`).pipe(
      catchError((err: unknown) => {
        if (err instanceof HttpErrorResponse && err.status === 503 && err.error?.status) {
          return of(err.error as HealthResponse);
        }
        return throwError(() => toApiError(err));
      }),
    );
  }

  analyze(file: File, includeDepth = false): Observable<AnalyzeResponse> {
    const form = new FormData();
    form.append('file', file);
    form.append('include_depth', String(includeDepth));
    return this.normalized(this.http.post<AnalyzeResponse>(`${API_BASE}/analyze`, form));
  }

  /** Emite eventos HTTP para poder mostrar el progreso real de la subida. */
  animate(request: AnimateRequest): Observable<HttpEvent<AnimateAccepted>> {
    return this.normalized(
      this.http.post<AnimateAccepted>(`${API_BASE}/animate`, buildAnimateForm(request), {
        observe: 'events',
        reportProgress: true,
      }),
    );
  }

  /** `statusUrl` lo entrega la API en el 202; ya es relativa. */
  getJob(statusUrl: string): Observable<JobResponse> {
    return this.normalized(this.http.get<JobResponse>(statusUrl));
  }

  getGallery(limit: number): Observable<GalleryResponse> {
    const params = new HttpParams().set('limit', limit);
    return this.normalized(this.http.get<GalleryResponse>(`${API_BASE}/gallery`, { params }));
  }

  /** Texto crudo: la pantalla de métricas muestra el JSON sin reformatear números. */
  getMetricsRaw(): Observable<string> {
    return this.normalized(
      this.http.get(`${API_BASE}/model/metrics`, { responseType: 'text' }),
    );
  }
}
