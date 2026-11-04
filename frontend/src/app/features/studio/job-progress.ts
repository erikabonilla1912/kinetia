import { DecimalPipe } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  computed,
  effect,
  input,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import gsap from 'gsap';
import { JobPhase, JobState, STAGE_LABELS, STAGE_STEPS } from '../../core/job-runner';
import { prefersReducedMotion } from '../../core/motion';

const ACTIVE_PHASES: readonly JobPhase[] = ['uploading', 'queued', 'running'];

/**
 * Avance de un job. Un render puede tardar decenas de segundos, así que la espera se
 * mantiene viva con datos ciertos: etapa, porcentaje de la API y tiempo transcurrido
 * medido aquí. Nunca estima cuánto falta ni inventa avance.
 */
@Component({
  selector: 'app-job-progress',
  imports: [DecimalPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @let s = state();
    <div data-testid="job-progress" [attr.data-phase]="s.phase" role="status" aria-live="polite">
      <div class="flex items-baseline justify-between gap-3">
        <p class="font-display text-xl font-bold" data-testid="job-title">{{ title() }}</p>
        @if (percent() !== null) {
          <p class="data text-xl font-semibold text-pulse" data-testid="job-percent">
            {{ percent() | number: '1.0-0' }} %
          </p>
        }
      </div>
      <div
        class="relative mt-3 h-2 overflow-hidden rounded-full bg-white/15"
        role="progressbar"
        aria-label="Progreso"
        aria-valuemin="0"
        aria-valuemax="100"
        [attr.aria-valuenow]="percent()"
      >
        <div
          #bar
          class="h-full w-full origin-left rounded-full bg-gradient-to-r from-xray via-pulse to-pulse"
          style="transform: scaleX(0)"
        ></div>
        @if (active()) {
          <div class="shimmer absolute inset-0" data-testid="job-shimmer" aria-hidden="true"></div>
        }
      </div>

      @if (active()) {
        <ol class="mt-3 flex flex-wrap gap-1.5" aria-label="Etapas">
          @for (step of steps(); track step.stage) {
            <li
              data-testid="job-step"
              class="data flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs"
              [class]="step.state === 'current' ? 'bg-pulse font-semibold text-on-pulse' : step.state === 'done' ? 'bg-white/10 text-paper' : 'text-mist'"
              [attr.data-state]="step.state"
              [attr.aria-current]="step.state === 'current' ? 'step' : null"
            >
              <span aria-hidden="true">{{ step.state === 'done' ? '✓' : step.state === 'current' ? '●' : '○' }}</span>
              {{ step.label }}
            </li>
          }
        </ol>
        <p class="data mt-2 text-sm text-mist" data-testid="job-elapsed" aria-live="off">
          Tiempo transcurrido: <span class="text-paper">{{ elapsed() }} s</span>
        </p>
      }

      @if (s.phase === 'queued' && s.queuePosition !== null) {
        <p class="mt-2 text-sm text-mist" data-testid="job-queue">
          Al entrar había {{ s.queuePosition }} {{ s.queuePosition === 1 ? 'trabajo' : 'trabajos' }} en la
          cola.
        </p>
      }
      @if (s.reconnecting) {
        <p class="mt-2 text-sm font-semibold text-warning" data-testid="job-reconnecting">
          Se perdió la conexión. Tu animación sigue en el servidor; reintentando…
        </p>
      }
    </div>
  `,
  styles: `
    .shimmer {
      background: linear-gradient(90deg, transparent, rgb(255 255 255 / 0.55), transparent);
      transform: translateX(-100%);
      animation: job-shimmer 1.8s ease-in-out infinite;
    }
    @keyframes job-shimmer {
      to {
        transform: translateX(100%);
      }
    }
    @media (prefers-reduced-motion: reduce) {
      .shimmer {
        display: none;
      }
    }
  `,
})
export class JobProgress implements OnDestroy {
  readonly state = input.required<JobState>();

  private readonly bar = viewChild.required<ElementRef<HTMLElement>>('bar');
  private ticker: ReturnType<typeof setInterval> | null = null;

  protected readonly elapsed = signal(0);
  protected readonly active = computed(() => ACTIVE_PHASES.includes(this.state().phase));

  protected readonly percent = computed(() => {
    const s = this.state();
    if (s.phase === 'uploading') return s.uploadFraction === null ? null : s.uploadFraction * 100;
    return s.progress === null ? null : s.progress * 100;
  });

  protected readonly title = computed(() => {
    const s = this.state();
    if (s.phase === 'uploading') return 'Subiendo tu archivo';
    if (s.phase === 'succeeded') return 'Listo';
    if (s.phase === 'failed') return 'No se pudo completar';
    return s.stage ? STAGE_LABELS[s.stage] : 'En cola';
  });

  protected readonly steps = computed(() => {
    const s = this.state();
    const current =
      s.phase === 'uploading' ? -1 : STAGE_STEPS.findIndex((step) => step.stage === s.stage);
    // `done` no está en la lista: con ella, todas las etapas quedaron atrás.
    const reached = s.stage === 'done' ? STAGE_STEPS.length : current;
    return STAGE_STEPS.map((step, index) => ({
      ...step,
      state: index < reached ? 'done' : index === reached ? 'current' : 'pending',
    }));
  });

  constructor() {
    effect(() => {
      const value = (this.percent() ?? 0) / 100;
      const bar = this.bar().nativeElement;
      if (prefersReducedMotion()) {
        gsap.set(bar, { scaleX: value });
      } else {
        gsap.to(bar, { scaleX: value, duration: 0.6, ease: 'power2.out', overwrite: true });
      }
    });

    // Cronómetro real: arranca al entrar en actividad y se detiene al terminar.
    effect(() => {
      const active = this.active();
      untracked(() => {
        if (active && this.ticker === null) {
          const startedAt = Date.now();
          this.elapsed.set(0);
          this.ticker = setInterval(
            () => this.elapsed.set(Math.floor((Date.now() - startedAt) / 1000)),
            1000,
          );
        } else if (!active) {
          this.stopTicker();
        }
      });
    });
  }

  private stopTicker(): void {
    if (this.ticker !== null) clearInterval(this.ticker);
    this.ticker = null;
  }

  ngOnDestroy(): void {
    this.stopTicker();
    gsap.killTweensOf(this.bar().nativeElement);
  }
}
