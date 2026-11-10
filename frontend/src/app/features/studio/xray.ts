import { DecimalPipe } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  computed,
  effect,
  inject,
  input,
  model,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import gsap from 'gsap';
import { AnalyzedClass, AnalyzeResponse, DUMMY_MODEL_VERSION } from '../../core/api.models';
import { DecodedMask, decodeMask, paintMask } from '../../core/mask';
import { prefersReducedMotion } from '../../core/motion';
import { formatMs } from '../../core/presentation';
import { ConfigStore } from '../../core/stores';

/** Capa sobre la imagen: pinta la máscara real con los colores de /config. */
@Component({
  selector: 'app-xray-overlay',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'pointer-events-none absolute inset-0 block overflow-hidden' },
  template: `
    <canvas #canvas class="absolute inset-0 h-full w-full" [style.opacity]="opacity()"></canvas>
    @if (decodeFailed()) {
      <p role="alert" class="absolute inset-x-3 bottom-3 rounded-xl bg-danger-bg px-3 py-2 text-sm font-semibold text-paper">
        No pudimos dibujar la máscara en este navegador. Los datos del panel siguen siendo válidos.
      </p>
    }
    <div
      #scan
      class="absolute inset-y-0 left-0 w-1 bg-xray opacity-0 shadow-[0_0_40px_12px_rgba(107,231,240,.7)]"
    ></div>
  `,
})
export class XrayOverlay implements OnDestroy {
  readonly maskBase64 = input.required<string>();
  readonly opacity = input(0.6);
  readonly focusId = input<number | null>(null);

  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
  private readonly scan = viewChild.required<ElementRef<HTMLElement>>('scan');
  private readonly config = inject(ConfigStore);
  private readonly mask = signal<DecodedMask | null>(null);
  private reveal: gsap.core.Timeline | null = null;
  private revealed = false;

  readonly decodeFailed = signal(false);

  constructor() {
    effect(() => {
      const base64 = this.maskBase64();
      this.revealed = false;
      this.decodeFailed.set(false);
      decodeMask(base64).then(
        (mask) => {
          // Si mientras tanto llegó otra máscara, esta ya no aplica.
          if (untracked(this.maskBase64) === base64) this.mask.set(mask);
        },
        () => this.decodeFailed.set(true),
      );
    });
    effect(() => {
      const mask = this.mask();
      if (!mask) return;
      const canvas = this.canvas().nativeElement;
      canvas.width = mask.width;
      canvas.height = mask.height;
      const image = paintMask(mask, this.config.config()?.classes ?? [], this.focusId());
      canvas.getContext('2d')?.putImageData(image, 0, 0);
      if (!this.revealed) {
        this.revealed = true;
        this.playReveal(canvas);
      }
    });
  }

  ngOnDestroy(): void {
    this.reveal?.kill();
  }

  /** La máscara aparece detrás de una línea de escaneo, como una placa de rayos X. */
  private playReveal(canvas: HTMLCanvasElement): void {
    if (prefersReducedMotion()) return;
    const scan = this.scan().nativeElement;
    this.reveal?.kill();
    const state = { x: 0 };
    this.reveal = gsap
      .timeline({
        onComplete: () => {
          canvas.style.clipPath = '';
        },
      })
      .set(scan, { opacity: 1 })
      .to(state, {
        x: 100,
        duration: 1.3,
        ease: 'power2.inOut',
        onUpdate: () => {
          canvas.style.clipPath = `inset(0 ${100 - state.x}% 0 0)`;
          scan.style.left = `${state.x}%`;
        },
      })
      .to(scan, { opacity: 0, duration: 0.25 });
  }
}

/** Panel de Rayos X. Todas las cifras salen de la respuesta de /analyze. */
@Component({
  selector: 'app-xray-panel',
  imports: [DecimalPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @let data = analysis();
    @if (isDummy()) {
      <p
        role="alert"
        data-testid="dummy-warning"
        class="mb-4 rounded-xl border border-warning/40 bg-warning-bg px-3 py-2.5 text-sm font-semibold text-warning"
      >
        Este análisis viene del modelo de desarrollo (dummy-dev), con pesos aleatorios. Sus clases y
        cifras no significan nada.
      </p>
    }

    <label class="block">
      <span class="flex items-baseline justify-between text-sm font-semibold">
        Opacidad de la máscara
        <span class="data text-mist">{{ opacity() * 100 | number: '1.0-0' }} %</span>
      </span>
      <input
        type="range"
        class="range"
        min="0"
        max="1"
        step="0.01"
        data-testid="xray-opacity"
        [style.--fill.%]="opacity() * 100"
        [value]="opacity()"
        (input)="opacity.set(+$any($event.target).value)"
      />
    </label>

    @if (rows().length === 0) {
      <p class="mt-3 rounded-xl bg-white/5 p-3 text-sm text-mist" data-testid="xray-empty">
        El modelo no devolvió ninguna clase para esta imagen.
      </p>
    }

    <ul class="mt-2 space-y-1" aria-label="Clases detectadas">
      @for (row of rows(); track row.id) {
        <li>
          <button
            type="button"
            data-testid="class-row"
            class="w-full rounded-xl px-2 py-2 text-left transition-colors hover:bg-white/8"
            [class.bg-white/10]="focusId() === row.id"
            [attr.aria-pressed]="focusId() === row.id"
            (click)="focusId.set(focusId() === row.id ? null : row.id)"
          >
            <span class="flex items-center gap-2.5">
              <span
                class="size-3.5 shrink-0 rounded-[5px] ring-1 ring-white/30"
                data-testid="class-swatch"
                [style.background-color]="row.color"
              ></span>
              <span class="min-w-0 flex-1 truncate font-semibold" data-testid="class-label">{{ row.label }}</span>
              <span class="data text-sm" data-testid="class-pct">{{ row.pixel_pct | number: '1.1-1' }} %</span>
            </span>
            <span class="mt-1.5 block h-1 overflow-hidden rounded-full bg-white/10" aria-hidden="true">
              <span
                class="block h-full rounded-full"
                [style.width.%]="row.pixel_pct"
                [style.background-color]="row.color"
              ></span>
            </span>
            <span class="mt-1 block text-xs text-mist">
              Confianza media
              <span class="data text-paper" data-testid="class-confidence">{{ row.confidence | number: '1.2-2' }}</span>
            </span>
          </button>
        </li>
      }
    </ul>

    <p class="mt-1 text-xs text-mist">Toca una clase para verla sola en la imagen.</p>

    <details class="mt-3 rounded-xl border border-xray/25 bg-xray/5 px-3 py-2 text-sm" open>
      <summary class="flex min-h-11 items-center font-semibold text-xray">Qué significa la confianza</summary>
      <p class="pb-1 pt-1 text-mist" data-testid="confidence-help">
        Es el promedio de la probabilidad máxima que el modelo dio a cada píxel asignado a esa clase.
        Indica qué tan seguro estuvo en esos píxeles. <strong class="text-paper">No es la probabilidad
        de que la clase esté en la imagen.</strong>
        Los píxeles con probabilidad máxima menor que
        <span class="data text-paper" data-testid="conf-threshold">{{ data.conf_threshold }}</span>
        pasan a {{ fallbackLabel() }}.
      </p>
    </details>

    <dl class="mt-3 grid grid-cols-2 gap-2 text-sm">
      <div class="rounded-xl bg-white/5 p-3">
        <dt class="text-xs text-mist">Inferencia total</dt>
        <dd class="data mt-0.5 text-lg font-semibold" data-testid="inference-total">
          {{ ms(data.inference_ms.total) }}
        </dd>
      </div>
      <div class="rounded-xl bg-white/5 p-3">
        <dt class="text-xs text-mist">Segmentación</dt>
        <dd class="data mt-0.5 text-lg font-semibold" data-testid="inference-seg">
          {{ ms(data.inference_ms.segmentation) }}
        </dd>
      </div>
      <div class="col-span-2 rounded-xl bg-white/5 p-3">
        <dt class="text-xs text-mist">Modelo</dt>
        <dd class="data mt-0.5 break-all" data-testid="model-version">{{ data.model_version }}</dd>
        <dd class="data mt-0.5 text-xs text-mist" [title]="data.model_sha256">
          sha256 {{ data.model_sha256.slice(0, 16) }}…
        </dd>
      </div>
    </dl>
  `,
})
export class XrayPanel {
  readonly analysis = input.required<AnalyzeResponse>();
  readonly opacity = model(0.6);
  readonly focusId = model<number | null>(null);

  private readonly config = inject(ConfigStore);
  protected readonly ms = formatMs;

  protected readonly isDummy = computed(
    () => this.analysis().model_version === DUMMY_MODEL_VERSION,
  );

  protected readonly rows = computed(() =>
    this.analysis().classes.map((cls: AnalyzedClass) => {
      const known = this.config.classesById().get(cls.id);
      return {
        ...cls,
        // Una clase que /config no conozca se muestra con su nombre crudo, sin inventar texto.
        label: known?.label ?? cls.name,
        color: known?.color ?? 'transparent',
      };
    }),
  );

  /** La clase 0 del contrato, leída de /config. */
  protected readonly fallbackLabel = computed(() => {
    const zero = this.config.classesById().get(0);
    return zero?.label ?? 'la clase de fondo';
  });
}
