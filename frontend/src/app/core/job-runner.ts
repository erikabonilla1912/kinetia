import { HttpEventType } from '@angular/common/http';
import { DestroyRef, Injectable, computed, inject, signal } from '@angular/core';
import { Subscription } from 'rxjs';
import { AnimateRequest, JobResponse, JobResult, JobStage } from './api.models';
import { ApiError, NETWORK_ERROR, toApiError } from './api-error';
import { ApiService } from './api.service';

export const JOB_POLL_MS = 1000;
/** Tras esta cantidad de fallos seguidos de red se deja de consultar y se avisa. */
export const JOB_MAX_POLL_FAILURES = 60;

export type JobPhase = 'idle' | 'uploading' | 'queued' | 'running' | 'succeeded' | 'failed';

export interface JobState {
  phase: JobPhase;
  /** Fracción de la subida (0 a 1), si el navegador la conoce. */
  uploadFraction: number | null;
  /** Posición que informó la API en el 202. */
  queuePosition: number | null;
  stage: JobStage | null;
  progress: number | null;
  /** La última consulta falló por red; se sigue intentando. */
  reconnecting: boolean;
  result: JobResult | null;
  job: JobResponse | null;
  error: ApiError | null;
}

const IDLE: JobState = {
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

export const STAGE_LABELS: Readonly<Record<JobStage, string>> = {
  queued: 'En cola',
  segmenting: 'Leyendo la imagen con el modelo',
  depth: 'Estimando la profundidad',
  rendering: 'Animando los fotogramas',
  encoding: 'Codificando el resultado',
  done: 'Terminando',
};

/** Etapas en el orden en que ocurren, con su nombre corto para la lista de avance. */
export const STAGE_STEPS: readonly { stage: JobStage; label: string }[] = [
  { stage: 'queued', label: 'Cola' },
  { stage: 'segmenting', label: 'Lectura' },
  { stage: 'depth', label: 'Profundidad' },
  { stage: 'rendering', label: 'Animación' },
  { stage: 'encoding', label: 'Codificación' },
];

/** Máquina de estados de un job: subir, consultar cada segundo, terminar o fallar. */
@Injectable()
export class JobRunner {
  private readonly api = inject(ApiService);
  private upload: Subscription | null = null;
  private poll: Subscription | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private failures = 0;

  readonly state = signal<JobState>(IDLE);
  readonly busy = computed(() => {
    const phase = this.state().phase;
    return phase === 'uploading' || phase === 'queued' || phase === 'running';
  });

  constructor() {
    inject(DestroyRef).onDestroy(() => this.stop());
  }

  start(request: AnimateRequest): void {
    this.stop();
    this.state.set({ ...IDLE, phase: 'uploading', uploadFraction: 0 });
    this.upload = this.api.animate(request).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress) {
          const fraction = event.total ? event.loaded / event.total : null;
          this.state.update((s) => ({ ...s, uploadFraction: fraction }));
        } else if (event.type === HttpEventType.Response && event.body) {
          const accepted = event.body;
          this.state.update((s) => ({
            ...s,
            phase: 'queued',
            uploadFraction: 1,
            stage: 'queued',
            queuePosition: accepted.queue_position,
          }));
          this.schedulePoll(accepted.status_url, 0);
        }
      },
      error: (err: unknown) => this.fail(toApiError(err)),
    });
  }

  reset(): void {
    this.stop();
    this.state.set(IDLE);
  }

  private stop(): void {
    this.upload?.unsubscribe();
    this.poll?.unsubscribe();
    if (this.timer !== null) clearTimeout(this.timer);
    this.upload = this.poll = this.timer = null;
    this.failures = 0;
  }

  private schedulePoll(statusUrl: string, delay = JOB_POLL_MS): void {
    this.timer = setTimeout(() => this.fetch(statusUrl), delay);
  }

  private fetch(statusUrl: string): void {
    this.poll = this.api.getJob(statusUrl).subscribe({
      next: (job) => {
        this.failures = 0;
        this.apply(job, statusUrl);
      },
      error: (err: unknown) => {
        const error = toApiError(err);
        const transient = error.code === NETWORK_ERROR || error.status >= 500;
        if (transient && ++this.failures < JOB_MAX_POLL_FAILURES) {
          this.state.update((s) => ({ ...s, reconnecting: true }));
          this.schedulePoll(statusUrl);
        } else {
          this.fail(error);
        }
      },
    });
  }

  private apply(job: JobResponse, statusUrl: string): void {
    const base = { job, stage: job.stage, progress: job.progress, reconnecting: false };
    switch (job.status) {
      case 'queued':
        this.state.update((s) => ({ ...s, ...base, phase: 'queued' }));
        this.schedulePoll(statusUrl);
        break;
      case 'running':
        this.state.update((s) => ({ ...s, ...base, phase: 'running' }));
        this.schedulePoll(statusUrl);
        break;
      case 'succeeded':
        this.state.update((s) => ({ ...s, ...base, phase: 'succeeded', result: job.result }));
        break;
      case 'failed':
        this.state.update((s) => ({
          ...s,
          ...base,
          phase: 'failed',
          error: new ApiError(
            job.error?.code ?? 'INTERNAL_ERROR',
            job.error?.message ?? 'La animación no pudo completarse.',
          ),
        }));
        break;
    }
  }

  private fail(error: ApiError): void {
    this.state.update((s) => ({ ...s, phase: 'failed', error, reconnecting: false }));
  }
}
