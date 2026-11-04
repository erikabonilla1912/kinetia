import { DestroyRef, Injectable, computed, inject, signal } from '@angular/core';
import { AppConfig, DUMMY_MODEL_VERSION, HealthResponse, ModelClass } from './api.models';
import { ApiError, toApiError } from './api-error';
import { ApiService } from './api.service';

const CONFIG_RETRY_MS = 5000;

/** Configuración pública. Colores, clases, moods y límites solo se leen de aquí. */
@Injectable({ providedIn: 'root' })
export class ConfigStore {
  private readonly api = inject(ApiService);
  private retryTimer: ReturnType<typeof setTimeout> | null = null;

  readonly config = signal<AppConfig | null>(null);
  readonly error = signal<ApiError | null>(null);
  readonly loading = signal(false);

  readonly classesById = computed(() => {
    const map = new Map<number, ModelClass>();
    for (const cls of this.config()?.classes ?? []) map.set(cls.id, cls);
    return map;
  });

  /** Reintenta solo: en la Expo la API puede tardar más que el navegador en arrancar. */
  load(): void {
    if (this.config() || this.loading()) return;
    this.clearRetry();
    this.loading.set(true);
    this.api.getConfig().subscribe({
      next: (config) => {
        this.config.set(config);
        this.error.set(null);
        this.loading.set(false);
      },
      error: (err: unknown) => {
        this.error.set(toApiError(err));
        this.loading.set(false);
        this.retryTimer = setTimeout(() => this.load(), CONFIG_RETRY_MS);
      },
    });
  }

  private clearRetry(): void {
    if (this.retryTimer !== null) clearTimeout(this.retryTimer);
    this.retryTimer = null;
  }
}

/** Estado del modelo según GET /health. */
@Injectable({ providedIn: 'root' })
export class HealthStore {
  private readonly api = inject(ApiService);
  private timer: ReturnType<typeof setTimeout> | null = null;

  readonly health = signal<HealthResponse | null>(null);
  readonly isDummy = computed(() => this.health()?.model_version === DUMMY_MODEL_VERSION);

  load(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.api.getHealth().subscribe({
      next: (health) => {
        this.health.set(health);
        // Mientras el modelo carga, se sigue preguntando para actualizar el aviso.
        if (health.status !== 'ok') this.timer = setTimeout(() => this.load(), CONFIG_RETRY_MS);
      },
      error: () => {
        this.health.set(null);
        this.timer = setTimeout(() => this.load(), CONFIG_RETRY_MS);
      },
    });
  }
}

/** Estado de red del navegador. */
@Injectable({ providedIn: 'root' })
export class ConnectivityStore {
  readonly online = signal(typeof navigator === 'undefined' ? true : navigator.onLine);

  constructor() {
    const destroy = inject(DestroyRef);
    const update = () => this.online.set(navigator.onLine);
    window.addEventListener('online', update);
    window.addEventListener('offline', update);
    destroy.onDestroy(() => {
      window.removeEventListener('online', update);
      window.removeEventListener('offline', update);
    });
  }
}
