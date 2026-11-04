import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, input, model } from '@angular/core';
import { BrushMode } from '../../core/api.models';
import { BRUSH_LIMITS, BrushStroke, canAddStroke, totalPoints } from '../../core/brush';
import { BrushSettings } from './brush-canvas';

/** Rosa de direcciones. 0 = derecha, 90 = abajo, como en el contrato. */
export const DIRECTIONS = [
  { deg: 270, label: 'Arriba', arrow: '↑' },
  { deg: 315, label: 'Arriba a la derecha', arrow: '↗' },
  { deg: 0, label: 'Derecha', arrow: '→' },
  { deg: 45, label: 'Abajo a la derecha', arrow: '↘' },
  { deg: 90, label: 'Abajo', arrow: '↓' },
  { deg: 135, label: 'Abajo a la izquierda', arrow: '↙' },
  { deg: 180, label: 'Izquierda', arrow: '←' },
  { deg: 225, label: 'Arriba a la izquierda', arrow: '↖' },
] as const;

@Component({
  selector: 'app-brush-panel',
  imports: [DecimalPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <p class="text-sm text-mist">
      Pinta sobre la imagen lo que quieres mover. Cada trazo guarda su zona, su fuerza y su dirección.
    </p>

    <fieldset class="mt-4">
      <legend class="mb-1.5 text-sm font-semibold">Qué se anima</legend>
      <div class="segmented">
        <button type="button" data-testid="mode-add" [attr.aria-pressed]="mode() === 'add'" (click)="mode.set('add')">
          Sumar al modelo
        </button>
        <button type="button" data-testid="mode-only" [attr.aria-pressed]="mode() === 'only'" (click)="mode.set('only')">
          Solo lo pintado
        </button>
      </div>
    </fieldset>

    <label class="mt-4 block">
      <span class="flex items-baseline justify-between text-sm font-semibold">
        Radio
        <span class="data text-mist">{{ settings().radius * 100 | number: '1.1-1' }} %</span>
      </span>
      <input
        type="range"
        class="range"
        data-testid="brush-radius"
        [min]="limits.minRadius"
        [max]="limits.maxRadius"
        step="0.005"
        [style.--fill.%]="radiusFill()"
        [value]="settings().radius"
        (input)="patch({ radius: +$any($event.target).value })"
      />
    </label>

    <label class="block">
      <span class="flex items-baseline justify-between text-sm font-semibold">
        Fuerza
        <span class="data text-mist">{{ settings().strength | number: '1.2-2' }}</span>
      </span>
      <input
        type="range"
        class="range"
        data-testid="brush-strength"
        min="0"
        max="1"
        step="0.05"
        [style.--fill.%]="settings().strength * 100"
        [value]="settings().strength"
        (input)="patch({ strength: +$any($event.target).value })"
      />
    </label>

    <fieldset class="mt-2">
      <legend class="mb-1.5 text-sm font-semibold">Dirección</legend>
      <div class="segmented">
        <button
          type="button"
          data-testid="dir-from-stroke"
          [attr.aria-pressed]="settings().fixedDirection === null"
          (click)="patch({ fixedDirection: null })"
        >
          Del trazo
        </button>
        <button
          type="button"
          data-testid="dir-fixed"
          [attr.aria-pressed]="settings().fixedDirection !== null"
          (click)="patch({ fixedDirection: settings().fallbackDirection })"
        >
          Elegida
        </button>
      </div>
      @if (settings().fixedDirection !== null) {
        <div class="mt-3 grid grid-cols-4 gap-1.5 sm:grid-cols-8 lg:grid-cols-4 xl:grid-cols-8">
          @for (dir of directions; track dir.deg) {
            <button
              type="button"
              class="btn !rounded-xl !px-0 text-lg"
              [class.!bg-paper]="settings().fixedDirection === dir.deg"
              [class.!text-ink-900]="settings().fixedDirection === dir.deg"
              [attr.aria-label]="dir.label"
              [attr.aria-pressed]="settings().fixedDirection === dir.deg"
              (click)="patch({ fixedDirection: dir.deg, fallbackDirection: dir.deg })"
            >
              {{ dir.arrow }}
            </button>
          }
        </div>
      }
    </fieldset>

    <div class="mt-4 flex items-center gap-2">
      <button
        type="button"
        class="btn flex-1"
        data-testid="brush-undo"
        [disabled]="strokes().length === 0"
        (click)="undo()"
      >
        Deshacer
      </button>
      <button
        type="button"
        class="btn flex-1"
        data-testid="brush-clear"
        [disabled]="strokes().length === 0"
        (click)="strokes.set([])"
      >
        Borrar todo
      </button>
    </div>

    <p class="data mt-3 text-xs text-mist" data-testid="brush-usage" aria-live="polite">
      {{ strokes().length }}/{{ limits.maxStrokes }} trazos · {{ points() }}/{{ limits.maxPoints }} puntos
    </p>
    @if (full()) {
      <p class="mt-1 text-sm font-semibold text-warning" role="status">
        Llegaste al límite del pincel. Deshaz algún trazo para seguir pintando.
      </p>
    }
    @if (mode() === 'only' && strokes().length === 0) {
      <p class="mt-1 text-sm font-semibold text-warning" role="status" data-testid="only-needs-strokes">
        En «Solo lo pintado» hace falta al menos un trazo para animar.
      </p>
    }
  `,
})
export class BrushPanel {
  readonly strokes = model.required<BrushStroke[]>();
  readonly settings = model.required<BrushSettings>();
  readonly mode = model.required<BrushMode>();
  readonly disabled = input(false);

  protected readonly limits = BRUSH_LIMITS;
  protected readonly directions = DIRECTIONS;
  protected readonly points = computed(() => totalPoints(this.strokes()));
  protected readonly full = computed(() => !canAddStroke(this.strokes()));
  protected readonly radiusFill = computed(
    () =>
      ((this.settings().radius - BRUSH_LIMITS.minRadius) /
        (BRUSH_LIMITS.maxRadius - BRUSH_LIMITS.minRadius)) *
      100,
  );

  protected patch(change: Partial<BrushSettings>): void {
    this.settings.update((current) => ({ ...current, ...change }));
  }

  protected undo(): void {
    this.strokes.update((strokes) => strokes.slice(0, -1));
  }
}
