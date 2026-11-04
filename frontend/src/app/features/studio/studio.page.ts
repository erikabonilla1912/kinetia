import { DecimalPipe } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  computed,
  effect,
  inject,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import gsap from 'gsap';
import { Subscription } from 'rxjs';
import { AnalyzeResponse, BrushMode, OutputFormat } from '../../core/api.models';
import { ApiError, toApiError } from '../../core/api-error';
import { ApiService } from '../../core/api.service';
import { BrushStroke, serializeStrokes } from '../../core/brush';
import { JobRunner } from '../../core/job-runner';
import { prefersReducedMotion } from '../../core/motion';
import { formatMs } from '../../core/presentation';
import { ConfigStore, HealthStore } from '../../core/stores';
import { ValidatedMedia, formatBytes, validateUpload } from '../../core/upload-validator';
import { Aurora } from '../../shared/aurora';
import { ErrorNotice } from '../../shared/error-notice';
import { WorkingSweep } from '../../shared/working-sweep';
import { BrushCanvas, BrushSettings } from './brush-canvas';
import { BrushPanel } from './brush-panel';
import { Comparator } from './comparator';
import { Dropzone } from './dropzone';
import { HeroTitle } from './hero-title';
import { JobProgress } from './job-progress';
import { MoodPicker } from './mood-picker';
import { XrayOverlay, XrayPanel } from './xray';

type StudioTab = 'xray' | 'brush' | 'result';

const DEFAULT_DURATION_S = 4;
const DEFAULT_RADIUS = 0.04;
const DEFAULT_STRENGTH = 0.7;
/** Con el mismo seed, exportar al otro formato repite la misma animación. */
const MAX_SEED = 2 ** 31 - 1;

@Component({
  selector: 'app-studio-page',
  imports: [
    Aurora,
    DecimalPipe,
    BrushCanvas,
    BrushPanel,
    Comparator,
    Dropzone,
    ErrorNotice,
    HeroTitle,
    JobProgress,
    MoodPicker,
    WorkingSweep,
    XrayOverlay,
    XrayPanel,
  ],
  providers: [JobRunner],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './studio.page.html',
})
export class StudioPage {
  private readonly api = inject(ApiService);
  private readonly configStore = inject(ConfigStore);
  private readonly destroyRef = inject(DestroyRef);
  protected readonly runner = inject(JobRunner);
  protected readonly health = inject(HealthStore);

  private readonly workspace = viewChild<ElementRef<HTMLElement>>('workspace');
  private readonly stageSection = viewChild<ElementRef<HTMLElement>>('stageSection');
  private analyzeSub: Subscription | null = null;

  protected readonly config = this.configStore.config;
  protected readonly job = this.runner.state;

  readonly media = signal<ValidatedMedia | null>(null);
  readonly mediaUrl = signal<string | null>(null);
  protected readonly previewBroken = signal(false);
  protected readonly validating = signal(false);
  protected readonly uploadError = signal<ApiError | null>(null);

  readonly analysis = signal<AnalyzeResponse | null>(null);
  protected readonly analyzing = signal(false);
  protected readonly analysisError = signal<ApiError | null>(null);

  protected readonly tab = signal<StudioTab>('xray');
  protected readonly opacity = signal(0.6);
  protected readonly focusId = signal<number | null>(null);

  readonly mood = signal('');
  readonly outputFormat = signal<OutputFormat>('mp4');
  readonly duration = signal(DEFAULT_DURATION_S);

  readonly strokes = signal<BrushStroke[]>([]);
  readonly brushMode = signal<BrushMode>('add');
  readonly brushSettings = signal<BrushSettings>({
    radius: DEFAULT_RADIUS,
    strength: DEFAULT_STRENGTH,
    fixedDirection: null,
    fallbackDirection: 0,
  });

  private seed = 0;

  protected readonly formatBytes = formatBytes;
  protected readonly ms = formatMs;

  protected readonly moods = computed(() => this.config()?.moods ?? []);

  /** Las clases de /config, sin la de fondo (id 0), en texto corrido. */
  protected readonly classNames = computed(() => {
    const names = (this.config()?.classes ?? [])
      .filter((cls) => cls.id !== 0)
      .map((cls) => cls.label.toLowerCase());
    return new Intl.ListFormat('es', { type: 'conjunction' }).format(names);
  });

  /** Tamaño del escenario. Después de /analyze se usa el tamaño de trabajo de la API. */
  protected readonly stageWidth = computed(
    () => this.analysis()?.width ?? this.media()?.width ?? null,
  );
  protected readonly stageHeight = computed(
    () => this.analysis()?.height ?? this.media()?.height ?? null,
  );
  protected readonly stageRatio = computed(() => {
    const width = this.stageWidth();
    const height = this.stageHeight();
    return width && height ? `${width} / ${height}` : '16 / 9';
  });
  /** Evita que una foto vertical crezca más que la ventana. */
  protected readonly stageMaxWidth = computed(() => {
    const width = this.stageWidth();
    const height = this.stageHeight();
    const ratio = width && height ? width / height : 16 / 9;
    return `calc(72dvh * ${ratio.toFixed(4)})`;
  });

  protected readonly result = computed(() => this.job().result);

  protected readonly brushBlocksAnimate = computed(
    () => this.brushMode() === 'only' && this.strokes().length === 0,
  );
  protected readonly canAnimate = computed(
    () => !!this.media() && !!this.mood() && !this.runner.busy() && !this.brushBlocksAnimate(),
  );
  protected readonly otherFormat = computed<OutputFormat>(() =>
    this.result()?.format === 'gif' ? 'mp4' : 'gif',
  );

  constructor() {
    this.destroyRef.onDestroy(() => {
      this.analyzeSub?.unsubscribe();
      this.revokeUrl();
    });

    effect(() => {
      const moods = this.moods();
      untracked(() => {
        if (moods.length && !moods.includes(this.mood())) this.mood.set(moods[0]);
      });
    });

    // Al terminar el job, el escenario pasa al comparador.
    effect(() => {
      if (this.job().phase === 'succeeded') untracked(() => this.tab.set('result'));
    });

    // Sin resultado no hay nada que comparar: por ejemplo, al reintentar y fallar.
    effect(() => {
      if (this.tab() === 'result' && !this.result()) this.tab.set('xray');
    });

    effect(() => {
      const element = this.workspace()?.nativeElement;
      if (element && !prefersReducedMotion()) {
        gsap.from(element.querySelectorAll('[data-reveal]'), {
          y: 24,
          opacity: 0,
          duration: 0.7,
          stagger: 0.07,
          ease: 'power3.out',
          clearProps: 'all',
        });
      }
    });
  }

  private revokeUrl(): void {
    const url = this.mediaUrl();
    if (url) URL.revokeObjectURL(url);
  }

  async onFile(file: File): Promise<void> {
    const limits = this.config()?.limits;
    if (!limits || this.validating()) return;
    this.validating.set(true);
    this.uploadError.set(null);
    const validation = await validateUpload(file, limits);
    this.validating.set(false);
    if (!validation.ok) {
      this.uploadError.set(validation.error);
      return;
    }
    this.startSession(validation.media);
  }

  private startSession(media: ValidatedMedia): void {
    this.revokeUrl();
    this.runner.reset();
    this.media.set(media);
    this.mediaUrl.set(URL.createObjectURL(media.file));
    this.previewBroken.set(false);
    this.strokes.set([]);
    this.brushMode.set('add');
    this.focusId.set(null);
    this.tab.set('xray');
    this.analyze();
  }

  protected analyze(): void {
    const media = this.media();
    if (!media) return;
    this.analyzeSub?.unsubscribe();
    this.analysis.set(null);
    this.analysisError.set(null);
    this.analyzing.set(true);
    this.analyzeSub = this.api.analyze(media.file).subscribe({
      next: (analysis) => {
        this.analysis.set(analysis);
        this.analyzing.set(false);
      },
      error: (err: unknown) => {
        this.analysisError.set(toApiError(err));
        this.analyzing.set(false);
      },
    });
  }

  animate(format: OutputFormat = this.outputFormat(), reuseSeed = false): void {
    const media = this.media();
    if (!media || !this.mood()) return;
    if (!reuseSeed) this.seed = Math.floor(Math.random() * MAX_SEED);
    this.outputFormat.set(format);
    const maskStrokes = serializeStrokes(this.strokes());
    this.runner.start({
      file: media.file,
      mood: this.mood(),
      output_format: format,
      duration_s: media.kind === 'image' ? this.duration() : undefined,
      mask_strokes: maskStrokes,
      brush_mode: maskStrokes ? this.brushMode() : undefined,
      gallery_opt_in: false,
      seed: this.seed,
    });
    this.revealStageOnMobile();
  }

  /** En una columna, el botón queda lejos del escenario, que es donde se ve el progreso. */
  private revealStageOnMobile(): void {
    if (window.innerWidth >= 1024) return;
    this.stageSection()?.nativeElement.scrollIntoView({
      behavior: prefersReducedMotion() ? 'auto' : 'smooth',
      block: 'start',
    });
  }

  protected reset(): void {
    this.analyzeSub?.unsubscribe();
    this.runner.reset();
    this.revokeUrl();
    this.media.set(null);
    this.mediaUrl.set(null);
    this.analysis.set(null);
    this.analysisError.set(null);
    this.analyzing.set(false);
    this.uploadError.set(null);
    this.strokes.set([]);
  }

  protected downloadName(format: string): string {
    return `kinetia-${this.mood()}.${format}`;
  }
}
