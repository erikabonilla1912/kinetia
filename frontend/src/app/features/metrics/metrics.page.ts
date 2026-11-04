import { DatePipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { ModelMetrics } from '../../core/api.models';
import { ApiError, toApiError } from '../../core/api-error';
import { ApiService } from '../../core/api.service';
import { ConfigStore } from '../../core/stores';
import { ErrorNotice } from '../../shared/error-notice';

/**
 * Métricas del modelo: GET /model/metrics, sin transformar. Los valores se muestran
 * tal como llegan (sin redondear); solo se calcula la intensidad de color del mapa.
 */
@Component({
  selector: 'app-metrics-page',
  imports: [DatePipe, ErrorNotice],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <main class="mx-auto max-w-[1400px] px-4 pb-16 pt-8 sm:px-6">
      <p class="eyebrow">Evaluación del modelo</p>
      <h1 class="mt-2 text-5xl font-extrabold sm:text-7xl">Lo que mide <span class="text-xray">de verdad</span></h1>
      <p class="mt-3 max-w-2xl text-mist">
        Copia literal del archivo <code class="data text-paper">metrics.json</code> que entrega la API.
        @if (!isDummy()) {
          Se calculó sobre imágenes reservadas para prueba, evaluando el ONNX que corre en producción.
        }
      </p>

      @if (error(); as err) {
        <div class="mt-8 max-w-2xl" data-testid="metrics-error">
          <app-error-notice [error]="err" (retry)="load()" />
        </div>
      } @else if (metrics(); as m) {
        @if (isDummy()) {
          <div role="alert" data-testid="metrics-dummy" class="mt-8 rounded-2xl border-2 border-warning bg-warning-bg p-4 sm:p-5">
            <p class="font-display text-2xl font-bold text-warning">Estos no son resultados del modelo</p>
            <p class="mt-1 text-paper">
              Este archivo lo generó el modelo de desarrollo (dummy-dev), con pesos aleatorios y sin evaluar con datos
              reales. No se debe citar ninguna cifra de esta pantalla.
            </p>
          </div>
        }
        <dl class="mt-8 grid grid-cols-2 gap-3 lg:grid-cols-4" data-testid="metrics-tiles">
          <div class="glass p-4">
            <dt class="eyebrow">mIoU</dt>
            <dd class="data mt-2 break-all text-2xl font-semibold sm:text-4xl" data-testid="metric-miou">{{ show(m.miou) }}</dd>
          </div>
          <div class="glass p-4">
            <dt class="eyebrow">Dice macro</dt>
            <dd class="data mt-2 break-all text-2xl font-semibold sm:text-4xl" data-testid="metric-dice">{{ show(m.dice_macro) }}</dd>
          </div>
          <div class="glass p-4">
            <dt class="eyebrow">Exactitud de píxel</dt>
            <dd class="data mt-2 break-all text-2xl font-semibold sm:text-4xl" data-testid="metric-accuracy">{{ show(m.pixel_accuracy) }}</dd>
          </div>
          <div class="glass p-4">
            <dt class="eyebrow">Imágenes ({{ m.split }})</dt>
            <dd class="data mt-2 break-all text-2xl font-semibold sm:text-4xl" data-testid="metric-images">{{ m.num_images }}</dd>
          </div>
        </dl>

        <div class="mt-4 grid gap-4 xl:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
          <section class="glass p-4 sm:p-5">
            <h2 class="text-2xl font-bold">Por clase</h2>
            <div class="mt-3 overflow-x-auto" tabindex="0" role="region" aria-label="Métricas por clase">
              <table class="data w-full min-w-[26rem] text-sm">
                <thead class="text-left text-xs text-mist">
                  <tr>
                    <th scope="col" class="py-2 pr-3 font-sans font-semibold">Clase</th>
                    <th scope="col" class="py-2 pr-3 font-medium">IoU</th>
                    <th scope="col" class="py-2 pr-3 font-medium">Dice</th>
                    <th scope="col" class="py-2 font-medium">Píxeles</th>
                  </tr>
                </thead>
                <tbody>
                  @for (row of perClass(); track row.name) {
                    <tr class="border-t border-white/10" data-testid="metrics-row">
                      <th scope="row" class="py-2 pr-3 text-left font-sans font-semibold">
                        <span class="flex items-center gap-2">
                          <span class="size-3 shrink-0 rounded-[4px] ring-1 ring-white/30" [style.background-color]="row.color"></span>
                          {{ row.label }}
                        </span>
                      </th>
                      <td class="py-2 pr-3">{{ show(row.iou) }}</td>
                      <td class="py-2 pr-3">{{ show(row.dice) }}</td>
                      <td class="py-2">{{ row.support_pixels }}</td>
                    </tr>
                  }
                </tbody>
              </table>
            </div>
          </section>

          <section class="glass min-w-0 p-4 sm:p-5">
            <h2 class="text-2xl font-bold">Matriz de confusión</h2>
            <p class="mt-1 text-sm text-mist">
              Filas: verdad. Columnas: predicción. En píxeles. Más intenso significa más píxeles dentro de esa fila.
            </p>
            <div class="mt-3 overflow-x-auto" tabindex="0" role="region" aria-label="Matriz de confusión">
              <table class="data w-full border-separate border-spacing-0.5 text-xs" data-testid="confusion-matrix">
                <thead>
                  <tr>
                    <th scope="col"><span class="sr-only">Verdad por predicción</span></th>
                    @for (name of labels(); track $index) {
                      <th scope="col" class="px-1 pb-1 text-center font-sans font-semibold text-mist">{{ name }}</th>
                    }
                  </tr>
                </thead>
                <tbody>
                  @for (row of matrix(); track row.label) {
                    <tr>
                      <th scope="row" class="whitespace-nowrap pr-2 text-right font-sans font-semibold text-mist">{{ row.label }}</th>
                      @for (cell of row.cells; track $index) {
                        <td
                          class="rounded-[4px] px-1.5 py-2 text-center"
                          data-testid="confusion-cell"
                          [style.background-color]="cell.background"
                          [style.color]="cell.ink"
                          [title]="row.label + ' → ' + labels()[$index] + ': ' + cell.value + ' píxeles'"
                        >
                          {{ cell.value }}
                        </td>
                      }
                    </tr>
                  }
                </tbody>
              </table>
            </div>
          </section>
        </div>

        <section class="glass mt-4 p-4 sm:p-5">
          <h2 class="text-2xl font-bold">Trazabilidad</h2>
          <dl class="data mt-3 grid gap-3 text-sm md:grid-cols-3">
            <div>
              <dt class="font-sans text-xs text-mist">Evaluado</dt>
              <dd class="mt-0.5">{{ m.evaluated_at | date: 'medium' : 'UTC' }} UTC</dd>
            </div>
            <div class="min-w-0">
              <dt class="font-sans text-xs text-mist">SHA-256 del modelo</dt>
              <dd class="mt-0.5 break-all">{{ m.model_sha256 }}</dd>
            </div>
            <div class="min-w-0">
              <dt class="font-sans text-xs text-mist">SHA-256 del manifiesto del dataset</dt>
              <dd class="mt-0.5 break-all">{{ m.dataset_manifest_sha256 }}</dd>
            </div>
          </dl>
          <details class="mt-4">
            <summary class="flex min-h-11 items-center font-semibold text-xray">Ver la respuesta original de la API</summary>
            <pre class="data mt-2 max-h-96 overflow-auto rounded-xl bg-ink-950/80 p-3 text-xs" tabindex="0" data-testid="metrics-raw">{{ raw() }}</pre>
          </details>
        </section>
      } @else {
        <div class="mt-8 grid grid-cols-2 gap-3 lg:grid-cols-4" aria-hidden="true" data-testid="metrics-loading">
          @for (slot of [1, 2, 3, 4]; track slot) {
            <div class="h-28 animate-pulse rounded-[var(--radius-panel)] bg-white/8"></div>
          }
        </div>
      }
    </main>
  `,
})
export class MetricsPage {
  private readonly api = inject(ApiService);
  private readonly store = inject(ConfigStore);

  protected readonly raw = signal('');
  protected readonly metrics = signal<ModelMetrics | null>(null);
  protected readonly error = signal<ApiError | null>(null);

  /** Contrato 2.3: cualquiera de las dos marcas basta para avisar. */
  protected readonly isDummy = computed(() => {
    const metrics = this.metrics();
    return metrics?.dummy === true || metrics?.split === 'dummy';
  });

  /** Sin redondear; un valor ausente se dice, no se deja en blanco ni se vuelve 0. */
  protected show(value: number | null | undefined): string {
    return value === null || value === undefined ? 'sin dato' : String(value);
  }

  private labelFor(name: string): string {
    return this.store.config()?.classes.find((cls) => cls.name === name)?.label ?? name;
  }

  protected readonly labels = computed(() =>
    (this.metrics()?.classes ?? []).map((name) => this.labelFor(name)),
  );

  protected readonly perClass = computed(() => {
    const metrics = this.metrics();
    if (!metrics) return [];
    return metrics.classes.map((name) => ({
      name,
      label: this.labelFor(name),
      color: this.store.config()?.classes.find((cls) => cls.name === name)?.color ?? 'transparent',
      ...metrics.per_class[name],
    }));
  });

  protected readonly matrix = computed(() => {
    const metrics = this.metrics();
    if (!metrics) return [];
    return metrics.confusion_matrix.map((row, rowIndex) => {
      const rowTotal = row.reduce((sum, value) => sum + value, 0);
      return {
        label: this.labelFor(metrics.classes[rowIndex]),
        cells: row.map((value) => {
          // Una sola tinta, de transparente a sólida: solo guía la vista, el dato es el texto.
          const share = rowTotal ? value / rowTotal : 0;
          return {
            value,
            background: `rgba(107, 231, 240, ${(share * 0.95).toFixed(3)})`,
            ink: share > 0.5 ? '#06070c' : '#f5f2ea',
          };
        }),
      };
    });
  });

  constructor() {
    this.load();
  }

  protected load(): void {
    this.error.set(null);
    this.metrics.set(null);
    this.api.getMetricsRaw().subscribe({
      next: (text) => {
        this.raw.set(text);
        try {
          this.metrics.set(JSON.parse(text) as ModelMetrics);
        } catch {
          this.error.set(toApiError(null));
        }
      },
      error: (err: unknown) => this.error.set(toApiError(err)),
    });
  }
}
