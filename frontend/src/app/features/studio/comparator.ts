import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  computed,
  inject,
  input,
  signal,
  viewChild,
} from '@angular/core';
import gsap from 'gsap';
import { prefersReducedMotion } from '../../core/motion';
import { ResultMedia } from '../../shared/result-media';

const KEY_STEP = 2;
const PAGE_STEP = 10;

/** Valor siguiente del deslizador para una tecla, o null si la tecla no le corresponde. */
export function nextSliderValue(current: number, key: string): number | null {
  const clamp = (value: number) => Math.min(100, Math.max(0, value));
  switch (key) {
    case 'ArrowLeft':
    case 'ArrowDown':
      return clamp(current - KEY_STEP);
    case 'ArrowRight':
    case 'ArrowUp':
      return clamp(current + KEY_STEP);
    case 'PageDown':
      return clamp(current - PAGE_STEP);
    case 'PageUp':
      return clamp(current + PAGE_STEP);
    case 'Home':
      return 0;
    case 'End':
      return 100;
    default:
      return null;
  }
}

/** Antes a la izquierda, después a la derecha. Arrastrable, táctil y con teclado. */
@Component({
  selector: 'app-comparator',
  imports: [ResultMedia],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'absolute inset-0 block overflow-hidden' },
  template: `
    <div
      #area
      data-testid="comparator-area"
      class="absolute inset-0 cursor-ew-resize select-none"
      style="touch-action: pan-y"
      (pointerdown)="onPointerDown($event)"
      (pointermove)="onPointerMove($event)"
      (pointerup)="dragging = false"
      (pointercancel)="dragging = false"
    >
      <div class="absolute inset-0" data-testid="after-layer">
        <app-result-media [url]="afterUrl()" alt="Después: la imagen animada" />
      </div>
      <div class="absolute inset-0" data-testid="before-layer" [style.clip-path]="beforeClip()">
        @if (beforeKind() === 'video') {
          <video
            class="block h-full w-full object-cover"
            [src]="beforeUrl()"
            [muted]="true"
            autoplay
            loop
            playsinline
            aria-label="Antes: el video original"
          ></video>
        } @else {
          <img
            class="block h-full w-full object-cover"
            draggable="false"
            alt="Antes: la imagen original"
            [src]="beforeUrl()"
          />
        }
      </div>

      <span class="eyebrow pointer-events-none absolute left-3 top-3 rounded-full bg-ink-950/75 px-3 py-1.5 !text-paper">
        Antes
      </span>
      <span class="eyebrow pointer-events-none absolute right-3 top-3 rounded-full bg-pulse px-3 py-1.5 !text-on-pulse">
        Después
      </span>

      <div
        class="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-1/2 bg-paper shadow-[0_0_18px_rgba(0,0,0,.7)]"
        [style.left.%]="position()"
      ></div>
      <div
        #handle
        data-testid="comparator-handle"
        role="slider"
        tabindex="0"
        aria-label="Comparar antes y después"
        aria-orientation="horizontal"
        aria-valuemin="0"
        aria-valuemax="100"
        [attr.aria-valuenow]="rounded()"
        [attr.aria-valuetext]="valueText()"
        class="absolute top-1/2 grid size-14 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border-2 border-ink-900 bg-paper text-ink-900 shadow-[0_8px_30px_rgba(0,0,0,.6)]"
        [style.left.%]="position()"
        (keydown)="onKeydown($event)"
      >
        <svg width="28" height="16" viewBox="0 0 28 16" fill="none" aria-hidden="true">
          <path
            d="M9 2 3 8l6 6M19 2l6 6-6 6"
            stroke="currentColor"
            stroke-width="2.6"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </div>
    </div>
  `,
})
export class Comparator implements AfterViewInit, OnDestroy {
  readonly beforeUrl = input.required<string>();
  readonly beforeKind = input<'image' | 'video'>('image');
  readonly afterUrl = input.required<string>();

  private readonly area = viewChild.required<ElementRef<HTMLElement>>('area');
  private readonly handle = viewChild.required<ElementRef<HTMLElement>>('handle');
  private readonly host = inject(ElementRef);
  private intro: gsap.core.Timeline | null = null;
  protected dragging = false;

  readonly position = signal(50);
  protected readonly rounded = computed(() => Math.round(this.position()));
  protected readonly beforeClip = computed(() => `inset(0 ${100 - this.position()}% 0 0)`);
  protected readonly valueText = computed(
    () => `${this.rounded()} % de la imagen original visible`,
  );

  ngAfterViewInit(): void {
    if (prefersReducedMotion() || !this.host.nativeElement.isConnected) return;
    // Barrido inicial: enseña que la línea se arrastra.
    const state = { value: 50 };
    const apply = () => this.position.set(state.value);
    this.intro = gsap
      .timeline({ delay: 0.4 })
      .to(state, { value: 22, duration: 0.7, ease: 'power2.inOut', onUpdate: apply })
      .to(state, { value: 50, duration: 0.8, ease: 'power3.out', onUpdate: apply });
  }

  ngOnDestroy(): void {
    this.intro?.kill();
  }

  private moveTo(clientX: number): void {
    const rect = this.area().nativeElement.getBoundingClientRect();
    if (rect.width === 0) return;
    this.position.set(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
  }

  protected onPointerDown(event: PointerEvent): void {
    this.intro?.kill();
    this.dragging = true;
    (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
    this.handle().nativeElement.focus({ preventScroll: true });
    this.moveTo(event.clientX);
  }

  protected onPointerMove(event: PointerEvent): void {
    if (this.dragging) this.moveTo(event.clientX);
  }

  protected onKeydown(event: KeyboardEvent): void {
    const next = nextSliderValue(this.position(), event.key);
    if (next === null) return;
    event.preventDefault();
    this.intro?.kill();
    this.position.set(next);
  }
}
