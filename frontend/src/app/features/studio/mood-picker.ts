import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  computed,
  input,
  model,
  viewChild,
} from '@angular/core';
import gsap from 'gsap';
import { prefersReducedMotion } from '../../core/motion';
import { MoodPresentation, moodPresentation } from '../../core/presentation';

interface Particle {
  x: number;
  y: number;
  size: number;
  speed: number;
  phase: number;
}

/** Partículas decorativas de la tarjeta. Un solo ciclo por tarjeta, vía el ticker de GSAP. */
@Component({
  selector: 'app-mood-particles',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<canvas #canvas class="absolute inset-0 h-full w-full" aria-hidden="true"></canvas>`,
  host: { class: 'pointer-events-none absolute inset-0 block' },
})
export class MoodParticles implements AfterViewInit, OnDestroy {
  readonly presentation = input.required<MoodPresentation>();

  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
  private particles: Particle[] = [];
  private readonly loop = (time: number) => this.render(time);

  ngAfterViewInit(): void {
    const style = this.presentation().particle;
    if (style === 'none') return;
    const count = style === 'chispas' ? 22 : style === 'brasas' ? 18 : 14;
    this.particles = Array.from({ length: count }, () => ({
      x: Math.random(),
      y: Math.random(),
      size: 0.6 + Math.random() * 1.8,
      speed: 0.03 + Math.random() * 0.06,
      phase: Math.random() * Math.PI * 2,
    }));
    if (prefersReducedMotion()) {
      this.render(0);
    } else {
      gsap.ticker.add(this.loop);
    }
  }

  ngOnDestroy(): void {
    gsap.ticker.remove(this.loop);
  }

  private render(time: number): void {
    const canvas = this.canvas().nativeElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    const context = canvas.getContext('2d');
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const { particle, particleColor } = this.presentation();
    context.fillStyle = particleColor;
    for (const p of this.particles) {
      const rises = particle === 'brasas' || particle === 'chispas';
      const travel = (p.y + (rises ? -1 : 0.4) * p.speed * time) % 1;
      const y = (travel + 1) % 1;
      const x = (p.x + Math.sin(time * 0.7 + p.phase) * 0.03) * width;
      const twinkle =
        particle === 'chispas' ? Math.abs(Math.sin(time * 3 + p.phase)) : 0.45 + 0.4 * Math.sin(time + p.phase);
      context.globalAlpha = Math.max(0.1, twinkle);
      context.beginPath();
      context.arc(x, y * height, p.size * (particle === 'brasas' ? 1.2 : 1), 0, Math.PI * 2);
      context.fill();
    }
  }
}

/** Selector de mood. Los moods llegan de /config. Son radios nativos: teclado incluido. */
@Component({
  selector: 'app-mood-picker',
  imports: [MoodParticles],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <fieldset>
      <legend class="sr-only">Mood de la animación</legend>
      <div class="grid grid-cols-2 gap-2.5">
        @for (card of cards(); track card.name) {
          <label
            data-testid="mood-card"
            class="group relative block aspect-[5/4] cursor-pointer overflow-hidden rounded-2xl border bg-ink-800 transition-[border-color,box-shadow]"
            [class]="
              selected() === card.name
                ? 'border-pulse shadow-glow'
                : 'border-white/15 hover:border-white/40'
            "
          >
            <input
              type="radio"
              class="sr-only-input peer"
              name="mood"
              [value]="card.name"
              [checked]="selected() === card.name"
              (change)="selected.set(card.name)"
            />
            @if (previewUrl(); as url) {
              <img
                alt=""
                class="absolute inset-0 h-full w-full object-cover transition-transform duration-700 group-hover:scale-105"
                [src]="url"
                [style.filter]="card.view.filter"
              />
            } @else {
              <div class="thumb-fallback absolute inset-0" [style.filter]="card.view.filter"></div>
            }
            <div class="absolute inset-0" [style.background]="card.view.wash"></div>
            <app-mood-particles [presentation]="card.view" />
            <div class="absolute inset-0 bg-gradient-to-t from-ink-950/95 via-ink-950/40 to-transparent"></div>
            <div class="absolute inset-x-0 bottom-0 p-2.5 sm:p-3">
              <p class="font-display text-lg font-bold leading-none sm:text-xl">{{ card.view.label }}</p>
              <p class="mt-1 text-xs leading-snug text-paper/90">{{ card.view.tagline }}</p>
            </div>
            @if (selected() === card.name) {
              <span
                class="absolute right-2 top-2 grid size-6 place-items-center rounded-full bg-pulse text-sm font-bold text-on-pulse"
                aria-hidden="true"
                >✓</span
              >
            }
          </label>
        }
      </div>
    </fieldset>
  `,
  styles: `
    .thumb-fallback {
      background: linear-gradient(180deg, #314a78 0%, #d98a6a 52%, #27343a 52%, #0f1a22 100%);
    }
    label:has(input:focus-visible) {
      outline: 3px solid var(--color-pulse);
      outline-offset: 2px;
    }
  `,
})
export class MoodPicker {
  readonly moods = input.required<readonly string[]>();
  readonly previewUrl = input<string | null>(null);
  readonly selected = model.required<string>();

  protected readonly cards = computed(() =>
    this.moods().map((name) => ({ name, view: moodPresentation(name) })),
  );
}
