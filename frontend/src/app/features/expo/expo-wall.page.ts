import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  computed,
  effect,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import gsap from 'gsap';
import { GalleryItem } from '../../core/api.models';
import { ApiService } from '../../core/api.service';
import { prefersReducedMotion } from '../../core/motion';
import { moodPresentation } from '../../core/presentation';
import { ConfigStore, HealthStore } from '../../core/stores';
import { Aurora } from '../../shared/aurora';
import { QrCode } from '../../shared/qr-code';
import { ResultMedia } from '../../shared/result-media';

export const WALL_POLL_MS = 4000;
/** Más videos a la vez saturan al portátil de la Expo. */
export const WALL_MAX_TILES = 9;

export interface WallLayout {
  cols: number;
  rows: number;
  heroCols: number;
  heroRows: number;
}

/** Reparte N tiles sin huecos. Si no es un rectángulo exacto, el más reciente crece. */
export function wallLayout(count: number): WallLayout {
  switch (count) {
    case 0:
    case 1:
      return { cols: 1, rows: 1, heroCols: 1, heroRows: 1 };
    case 2:
      return { cols: 2, rows: 1, heroCols: 1, heroRows: 1 };
    case 3:
      return { cols: 2, rows: 2, heroCols: 1, heroRows: 2 };
    case 4:
      return { cols: 2, rows: 2, heroCols: 1, heroRows: 1 };
    case 5:
      return { cols: 3, rows: 2, heroCols: 2, heroRows: 1 };
    case 6:
      return { cols: 3, rows: 2, heroCols: 1, heroRows: 1 };
    case 7:
      return { cols: 4, rows: 2, heroCols: 2, heroRows: 1 };
    case 8:
      return { cols: 3, rows: 3, heroCols: 2, heroRows: 1 };
    default:
      return { cols: 3, rows: 3, heroCols: 1, heroRows: 1 };
  }
}

export function joinUrl(base: string, path: string): string {
  return `${base.replace(/\/+$/, '')}${path}`;
}

/**
 * Muro para TV. Se consulta la lista completa en cada ciclo (y no solo lo nuevo con
 * `after`): así una animación que el equipo modera desaparece también de la pantalla.
 */
@Component({
  selector: 'app-expo-wall-page',
  imports: [Aurora, QrCode, ResultMedia],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <main
      #root
      class="wall-grid min-h-dvh gap-4 bg-ink-950/40 p-4 lg:h-dvh lg:gap-6 lg:p-6"
      [class.cursor-none]="idle()"
      (mousemove)="wakeCursor()"
    >
      <aside class="glass flex min-h-0 flex-col justify-between gap-4 p-5 lg:p-[clamp(1.25rem,1.8vw,2.25rem)]">
        <div>
          <p class="font-display text-3xl font-extrabold tracking-tight xl:text-5xl">
            KINETIA<span class="text-pulse">.</span>
          </p>
          <h1 class="mt-4 text-5xl font-extrabold leading-[.92] xl:text-7xl min-[1800px]:text-[6.5rem]">
            Dale <span class="text-pulse">vida</span> a tu foto
          </h1>
        </div>
        <div>
          <div class="mx-auto w-full max-w-72 overflow-hidden rounded-2xl lg:max-w-none" data-testid="wall-qr">
            <app-qr-code [value]="uploadUrl()" />
          </div>
          <p class="mt-3 text-center font-display text-2xl font-bold xl:text-4xl">Escanea con tu celular</p>
          <p class="data mt-1 break-all text-center text-sm text-mist xl:text-base" data-testid="wall-url">{{ uploadUrl() }}</p>
          @if (health.isDummy()) {
            <p class="mt-2 rounded-lg bg-warning-bg px-3 py-1.5 text-center text-sm font-semibold text-warning">
              Modelo de desarrollo (dummy-dev)
            </p>
          }
        </div>
      </aside>

      <section class="relative min-h-[60dvh] min-w-0 lg:min-h-0" aria-label="Animaciones de los visitantes">
        @if (tiles().length) {
          <div
            class="wall-tiles h-full gap-3 lg:gap-4"
            [style.--cols]="layout().cols"
            [style.--rows]="layout().rows"
            data-testid="wall-tiles"
          >
            @for (item of tiles(); track item.job_id; let first = $first) {
              <article
                class="wall-tile relative min-h-48 overflow-hidden rounded-[var(--radius-stage)] border border-white/15 bg-ink-900 lg:min-h-0"
                data-testid="wall-tile"
                [attr.data-job]="item.job_id"
                [style.--span-cols]="first ? layout().heroCols : 1"
                [style.--span-rows]="first ? layout().heroRows : 1"
              >
                <app-result-media [url]="item.result_url" alt="Animación de un visitante" (failed)="drop(item.job_id)" />
                <span class="absolute bottom-3 left-3 rounded-full bg-ink-950/80 px-4 py-1.5 font-display text-lg font-bold xl:text-2xl">
                  {{ moodLabel(item.mood) }}
                </span>
              </article>
            }
          </div>
        } @else {
          <app-aurora />
          <div class="grid h-full min-h-[60dvh] place-items-center p-6 text-center lg:min-h-0" data-testid="wall-empty">
            <div>
              <p class="font-display text-5xl font-extrabold leading-none xl:text-8xl">
                @if (loaded()) {
                  El muro está vacío
                } @else {
                  Preparando el muro…
                }
              </p>
              @if (loaded()) {
                <p class="mt-4 text-xl text-mist xl:text-3xl">La primera animación puede ser la tuya.</p>
              }
            </div>
          </div>
        }

        @if (disconnected()) {
          <p
            role="status"
            data-testid="wall-offline"
            class="absolute right-3 top-3 rounded-full bg-warning-bg px-4 py-2 font-semibold text-warning"
          >
            Sin conexión con el servidor; reintentando…
          </p>
        }
      </section>

      <button
        type="button"
        class="btn fixed bottom-4 right-4 z-10 transition-opacity"
        data-testid="wall-fullscreen"
        [class.opacity-0]="idle()"
        [class.hidden]="fullscreen()"
        (click)="toggleFullscreen()"
      >
        Pantalla completa
      </button>
    </main>
  `,
  styles: `
    .wall-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
    }
    .wall-tiles {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
    }
    @media (min-width: 1024px) {
      .wall-grid {
        /* El QR debe poder leerse a varios metros: columna ancha, no un adorno. */
        grid-template-columns: clamp(320px, 25vw, 520px) minmax(0, 1fr);
      }
      .wall-tiles {
        grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
        grid-template-rows: repeat(var(--rows), minmax(0, 1fr));
      }
      .wall-tile {
        grid-column: span var(--span-cols, 1);
        grid-row: span var(--span-rows, 1);
      }
    }
  `,
})
export class ExpoWallPage {
  private readonly api = inject(ApiService);
  private readonly store = inject(ConfigStore);
  protected readonly health = inject(HealthStore);
  private readonly root = viewChild.required<ElementRef<HTMLElement>>('root');

  private timer: ReturnType<typeof setTimeout> | null = null;
  private cursorTimer: ReturnType<typeof setTimeout> | null = null;
  private seen = new Set<string>();
  private readonly hidden = signal<ReadonlySet<string>>(new Set());

  private readonly items = signal<GalleryItem[]>([]);
  protected readonly loaded = signal(false);
  protected readonly disconnected = signal(false);
  protected readonly idle = signal(false);
  protected readonly fullscreen = signal(false);

  protected readonly tiles = computed(() =>
    this.items()
      .filter((item) => !this.hidden().has(item.job_id))
      .slice(0, WALL_MAX_TILES),
  );
  protected readonly layout = computed(() => wallLayout(this.tiles().length));
  protected readonly uploadUrl = computed(() =>
    joinUrl(this.store.config()?.public_base_url ?? '', '/expo/subir'),
  );

  constructor() {
    const destroy = inject(DestroyRef);
    const onFullscreen = () => this.fullscreen.set(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFullscreen);
    destroy.onDestroy(() => {
      if (this.timer !== null) clearTimeout(this.timer);
      if (this.cursorTimer !== null) clearTimeout(this.cursorTimer);
      document.removeEventListener('fullscreenchange', onFullscreen);
    });

    this.poll();
    this.wakeCursor();

    // Cada animación nueva entra con un destello; las que ya estaban no se tocan.
    effect(() => {
      const tiles = this.tiles();
      const root = this.root().nativeElement;
      queueMicrotask(() => this.announce(tiles, root));
    });
  }

  private announce(tiles: GalleryItem[], root: HTMLElement): void {
    const firstRun = this.seen.size === 0;
    for (const tile of tiles) {
      if (this.seen.has(tile.job_id)) continue;
      this.seen.add(tile.job_id);
      if (firstRun || prefersReducedMotion()) continue;
      const element = root.querySelector(`[data-job="${CSS.escape(tile.job_id)}"]`);
      if (element) {
        gsap.from(element, { scale: 0.86, opacity: 0, duration: 1.1, ease: 'expo.out', clearProps: 'all' });
      }
    }
  }

  private poll(): void {
    this.api.getGallery(WALL_MAX_TILES).subscribe({
      next: (response) => {
        this.items.set(response.items);
        this.loaded.set(true);
        this.disconnected.set(false);
        this.scheduleNext();
      },
      error: () => {
        // Se conserva lo que ya está en pantalla: un corte breve no debe vaciar el muro.
        this.disconnected.set(true);
        this.scheduleNext();
      },
    });
  }

  private scheduleNext(): void {
    this.timer = setTimeout(() => this.poll(), WALL_POLL_MS);
  }

  /** Un resultado que ya no carga (expiró o se borró) sale del muro. */
  protected drop(jobId: string): void {
    this.hidden.update((current) => new Set(current).add(jobId));
  }

  protected moodLabel(mood: string): string {
    return moodPresentation(mood).label;
  }

  protected wakeCursor(): void {
    this.idle.set(false);
    if (this.cursorTimer !== null) clearTimeout(this.cursorTimer);
    this.cursorTimer = setTimeout(() => this.idle.set(true), 3000);
  }

  protected toggleFullscreen(): void {
    document.documentElement.requestFullscreen?.().catch(() => undefined);
  }
}
