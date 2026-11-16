import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ModelMetrics, TrainingEpoch } from '../../core/api.models';
import { ApiError, toApiError } from '../../core/api-error';
import { ApiService, TRAINING_HISTORY_URL } from '../../core/api.service';
import { ConfigStore } from '../../core/stores';
import { ErrorNotice } from '../../shared/error-notice';
import { Faq } from './faq';
import { COMPACT_FRAME, FRAME, MIN_DELTA, formatNumber, lineChart, summarize } from './training-summary';

/**
 * Cómo se entrenó el modelo desplegado. Todas las cifras salen del history.json real de
 * Colab (frontend/public/datos) o de GET /model/metrics; nada está escrito a mano.
 */
@Component({
  selector: 'app-training-page',
  imports: [ErrorNotice, Faq, RouterLink],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <main class="mx-auto max-w-[1400px] px-4 pb-16 pt-8 sm:px-6">
      <p class="eyebrow">Entrenamiento</p>
      <h1 class="mt-2 text-5xl font-extrabold sm:text-7xl">Lo entrenamos <span class="text-xray">desde cero</span></h1>
      <p class="mt-3 max-w-2xl text-mist">
        Ningún peso preentrenado: la red empezó con valores aleatorios y aprendió solo de nuestras fotos.
        Esta página dibuja el registro real del entrenamiento en Google Colab, época por época.
      </p>

      @if (error(); as err) {
        <div class="mt-8 max-w-2xl" data-testid="training-error">
          <app-error-notice [error]="err" (retry)="load()" />
        </div>
      } @else if (summary(); as s) {
        <dl class="mt-8 grid grid-cols-2 gap-3 lg:grid-cols-4" data-testid="training-tiles">
          <div class="glass p-4">
            <dt class="eyebrow">Épocas</dt>
            <dd class="data mt-2 text-2xl font-semibold sm:text-4xl" data-testid="tile-epochs">{{ s.epochs }}</dd>
          </div>
          <div class="glass p-4">
            <dt class="eyebrow">Tiempo en GPU T4</dt>
            <dd class="data mt-2 text-2xl font-semibold sm:text-4xl" data-testid="tile-hours">{{ hours() }} h</dd>
          </div>
          <div class="glass p-4">
            <dt class="eyebrow">Época guardada</dt>
            <dd class="data mt-2 text-2xl font-semibold sm:text-4xl" data-testid="tile-saved">{{ s.saved.epoch }}</dd>
            <dd class="mt-1 text-sm text-mist">mIoU de validación {{ fmt(s.saved.val_miou, 4) }}</dd>
          </div>
          <div class="glass p-4">
            <dt class="eyebrow">mIoU en prueba</dt>
            <dd class="data mt-2 text-2xl font-semibold sm:text-4xl" data-testid="tile-test">
              {{ testMiou() === null ? 'sin dato' : fmt(testMiou()!, 4) }}
            </dd>
            <dd class="mt-1 text-sm text-mist">fotos que nunca vio al entrenar</dd>
          </div>
        </dl>

        <div class="mt-4 grid gap-4 lg:grid-cols-2">
          @for (chart of charts(); track chart.title) {
            <section class="glass min-w-0 p-4 sm:p-5" [attr.data-testid]="chart.testid">
              <h2 class="text-2xl font-bold">{{ chart.title }}</h2>
              <p class="mt-1 text-sm text-mist">{{ chart.note }}</p>
              <div class="mt-2 flex gap-4 text-xs text-mist" aria-hidden="true">
                <span class="flex items-center gap-1.5"><span class="h-0.5 w-5 bg-fog"></span>entrenamiento</span>
                <span class="flex items-center gap-1.5"><span class="h-0.5 w-5 bg-xray"></span>validación</span>
              </div>
              <svg
                class="mt-2 w-full"
                [attr.viewBox]="'0 0 ' + frame.width + ' ' + (frame.height + 22)"
                role="img"
                [attr.aria-label]="chart.title + ': ' + chart.note"
              >
                @for (t of chart.data.yTicks; track t.label) {
                  <line [attr.x1]="frame.left" [attr.x2]="frame.right" [attr.y1]="t.y" [attr.y2]="t.y" stroke="rgb(255 255 255 / 0.08)" />
                  <text [attr.x]="frame.left - 8" [attr.y]="t.y + 4" text-anchor="end" font-size="12" fill="#8f95a8">{{ t.label }}</text>
                }
                @for (t of chart.data.xTicks; track t.label) {
                  <text [attr.x]="t.x" [attr.y]="frame.bottom + 18" text-anchor="middle" font-size="12" fill="#8f95a8">{{ t.label }}</text>
                }
                <text [attr.x]="(frame.left + frame.right) / 2" [attr.y]="frame.height + 18" text-anchor="middle" font-size="12" fill="#8f95a8">época</text>
                <path [attr.d]="chart.data.train" fill="none" stroke="#8f95a8" stroke-width="2" />
                <path [attr.d]="chart.data.val" fill="none" stroke="#6be7f0" stroke-width="2" />
                <circle [attr.cx]="chart.mark.x" [attr.cy]="chart.mark.y" r="5" fill="none" stroke="#c8ff3e" stroke-width="2" />
                <text
                  [attr.x]="chart.mark.x + chart.mark.dx"
                  [attr.y]="chart.mark.y + chart.mark.dy"
                  [attr.text-anchor]="chart.mark.anchor"
                  font-size="12"
                  fill="#f5f2ea"
                >{{ chart.mark.label }}</text>
              </svg>
            </section>
          }
        </div>

        <section class="glass mt-4 p-4 sm:p-5" data-testid="training-classes">
          <h2 class="text-2xl font-bold">Lo que aprendió cada clase</h2>
          <p class="mt-1 text-sm text-mist">
            IoU de validación en la primera época y en la época guardada ({{ s.saved.epoch }}). 1 es coincidencia perfecta.
          </p>
          <div class="mt-3 overflow-x-auto" tabindex="0" role="region" aria-label="IoU por clase durante el entrenamiento">
            <table class="data w-full min-w-[28rem] text-sm">
              <thead class="text-left text-xs text-mist">
                <tr>
                  <th scope="col" class="py-2 pr-3 font-sans font-semibold">Clase</th>
                  <th scope="col" class="py-2 pr-3 font-medium">Época 1</th>
                  <th scope="col" class="py-2 pr-3 font-medium">Época {{ s.saved.epoch }}</th>
                  <th scope="col" class="w-1/2 py-2 font-medium"><span class="sr-only">Barra</span></th>
                </tr>
              </thead>
              <tbody>
                @for (row of classRows(); track row.name) {
                  <tr class="border-t border-white/10" data-testid="training-class-row">
                    <th scope="row" class="py-2 pr-3 text-left font-sans font-semibold">
                      <span class="flex items-center gap-2">
                        <span class="size-3 shrink-0 rounded-[4px] ring-1 ring-white/30" [style.background-color]="row.color"></span>
                        {{ row.label }}
                      </span>
                    </th>
                    <td class="py-2 pr-3 text-mist">{{ fmt(row.start, 3) }}</td>
                    <td class="py-2 pr-3">{{ fmt(row.end, 3) }}</td>
                    <td class="py-2" aria-hidden="true">
                      <div class="relative h-3 rounded-full bg-white/8">
                        <div class="absolute inset-y-0 left-0 rounded-full bg-white/25" [style.width.%]="row.start * 100"></div>
                        <div class="absolute inset-y-0 left-0 rounded-full bg-xray/70" [style.width.%]="row.end * 100" [style.opacity]="0.9"></div>
                      </div>
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        </section>

        <div class="mt-4 grid gap-4 lg:grid-cols-2">
          <section class="glass p-4 sm:p-5">
            <h2 class="text-2xl font-bold">Cómo se entrenó</h2>
            <dl class="mt-3 grid gap-3 text-sm">
              @for (fact of facts; track fact.term) {
                <div>
                  <dt class="font-semibold text-paper">{{ fact.term }}</dt>
                  <dd class="mt-0.5 text-mist">{{ fact.detail }}</dd>
                </div>
              }
            </dl>
          </section>

          <section class="glass p-4 sm:p-5" data-testid="training-verify">
            <h2 class="text-2xl font-bold">Compruébalo</h2>
            <ol class="mt-3 grid list-decimal gap-3 pl-5 text-sm text-mist">
              <li>
                <span class="font-semibold text-paper">Entrenar en vivo.</span>
                Entrena una copia nueva, desde cero, con el mismo código de Colab, pocas fotos y la CPU de este portátil.
                No reemplaza el modelo de la app.
                <code class="data mt-1 block break-all rounded-lg bg-ink-950/80 p-2 text-xs text-paper">ml\\.venv\\Scripts\\python.exe ml\\demo_entrenamiento.py</code>
              </li>
              <li>
                <span class="font-semibold text-paper">Reevaluar el modelo real.</span>
                Vuelve a medir las fotos de prueba con el mismo ONNX de la app y compara con
                <a routerLink="/modelo" class="text-xray underline">las métricas publicadas</a>.
                <code class="data mt-1 block break-all rounded-lg bg-ink-950/80 p-2 text-xs text-paper">ml\\.venv\\Scripts\\python.exe ml\\evaluate.py --verify models\\metrics.json</code>
              </li>
              <li>
                <span class="font-semibold text-paper">Ver el registro original.</span>{{ ' ' }}
                <a [href]="historyUrl" class="text-xray underline" download>Descargar history.json</a>,
                el archivo que escribió el entrenamiento en Colab y del que sale esta página.
              </li>
            </ol>
          </section>
        </div>

        <app-faq [testMiou]="testMiou()" [epochs]="s.epochs" />
      } @else {
        <div class="mt-8 grid grid-cols-2 gap-3 lg:grid-cols-4" aria-hidden="true" data-testid="training-loading">
          @for (slot of [1, 2, 3, 4]; track slot) {
            <div class="h-28 animate-pulse rounded-[var(--radius-panel)] bg-white/8"></div>
          }
        </div>
      }
    </main>
  `,
})
export class TrainingPage {
  private readonly api = inject(ApiService);
  private readonly store = inject(ConfigStore);

  /** Se elige al abrir la página; girar el celular después no cambia el lienzo. */
  protected readonly frame =
    typeof matchMedia === 'function' && matchMedia('(max-width: 639px)').matches ? COMPACT_FRAME : FRAME;
  protected readonly historyUrl = TRAINING_HISTORY_URL;
  protected readonly fmt = formatNumber;

  protected readonly history = signal<TrainingEpoch[] | null>(null);
  protected readonly metrics = signal<ModelMetrics | null>(null);
  protected readonly error = signal<ApiError | null>(null);

  protected readonly summary = computed(() => {
    const history = this.history();
    return history ? summarize(history) : null;
  });

  protected readonly hours = computed(() => formatNumber((this.summary()?.totalSeconds ?? 0) / 3600, 1));

  /** El mIoU de prueba solo se muestra si metrics.json es del modelo real. */
  protected readonly testMiou = computed(() => {
    const metrics = this.metrics();
    if (!metrics || metrics.dummy === true || metrics.split !== 'test') return null;
    return metrics.miou;
  });

  protected readonly charts = computed(() => {
    const history = this.history();
    const s = this.summary();
    if (!history || !s) return [];
    const loss = lineChart(history, (e) => e.train_loss, (e) => e.val_loss, 0, 2, 0.5, this.frame);
    const miou = lineChart(history, (e) => e.train_miou, (e) => e.val_miou, 0.2, 0.9, 0.1, this.frame);
    return [
      {
        title: 'Pérdida',
        testid: 'chart-loss',
        note: `Menor es mejor. La de validación toca su mínimo en la época ${s.minValLoss.epoch} (${formatNumber(s.minValLoss.val_loss, 4)}).`,
        data: loss,
        mark: {
          x: loss.x(s.minValLoss.epoch),
          y: loss.y(s.minValLoss.val_loss),
          dx: 0,
          dy: -14,
          anchor: 'middle',
          label: `mínimo, época ${s.minValLoss.epoch}`,
        },
      },
      {
        title: 'mIoU',
        testid: 'chart-miou',
        note: `Mayor es mejor. Se guardó la época ${s.saved.epoch}: después ninguna mejoró en más de ${formatNumber(MIN_DELTA, 3)}.`,
        data: miou,
        mark: {
          x: miou.x(s.saved.epoch),
          y: miou.y(s.saved.val_miou),
          dx: -8,
          dy: 22,
          anchor: 'end',
          label: `guardada, época ${s.saved.epoch}`,
        },
      },
    ];
  });

  protected readonly classRows = computed(() => {
    const s = this.summary();
    if (!s) return [];
    const classes = this.store.config()?.classes ?? [];
    return Object.keys(s.saved.val_iou_per_class).map((name) => {
      const cls = classes.find((c) => c.name === name);
      return {
        name,
        label: cls?.label ?? name,
        color: cls?.color ?? 'transparent',
        start: s.first.val_iou_per_class[name] ?? 0,
        end: s.saved.val_iou_per_class[name] ?? 0,
      };
    });
  });

  /** Hechos del código de entrenamiento (ml/train.py, ml/model.py, ml/dataset.py). */
  protected readonly facts = [
    { term: 'Red', detail: 'U-Net con encoder MobileNetV3 Large, 6,69 millones de parámetros, entrada de 320×320.' },
    { term: 'Inicialización', detail: 'Pesos aleatorios en toda la red. Ningún modelo preentrenado ni pesos de ImageNet.' },
    { term: 'Datos', detail: '10 583 fotos de entrenamiento y 2268 de validación (COCO-Stuff y BoWFire). Las 2268 de prueba nunca se abren al entrenar.' },
    { term: 'Normalización', detail: 'Media y desviación calculadas con nuestras propias fotos de entrenamiento.' },
    { term: 'Pérdida', detail: 'Entropía cruzada ponderada por clase más Dice: el fuego, que tiene pocos píxeles, pesa más.' },
    { term: 'Optimizador', detail: 'AdamW con tasa 1e-3 que baja en coseno hasta 1e-6; lotes de 32 fotos.' },
    { term: 'Guardado y parada', detail: 'Se guarda la época con mejor mIoU de validación; se para tras 15 épocas sin mejora o al llegar a 100.' },
  ];

  constructor() {
    this.load();
  }

  protected load(): void {
    this.error.set(null);
    this.history.set(null);
    this.api.getTrainingHistory().subscribe({
      next: (history) => {
        if (!Array.isArray(history) || history.length === 0) {
          this.error.set(toApiError(null));
          return;
        }
        this.history.set(history);
      },
      error: (err: unknown) => this.error.set(toApiError(err)),
    });
    // Opcional: si /model/metrics falla, la página se muestra igual sin el mIoU de prueba.
    this.api.getMetricsRaw().subscribe({
      next: (text) => {
        try {
          this.metrics.set(JSON.parse(text) as ModelMetrics);
        } catch {
          this.metrics.set(null);
        }
      },
      error: () => this.metrics.set(null),
    });
  }
}
