import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  computed,
  effect,
  input,
  model,
  signal,
  viewChild,
} from '@angular/core';
import {
  BrushPoint,
  BrushStroke,
  appendPoint,
  canAddStroke,
  directionFromPoints,
  normalizePoint,
  remainingPoints,
} from '../../core/brush';

export interface BrushSettings {
  radius: number;
  strength: number;
  /** null: se toma del trazo. */
  fixedDirection: number | null;
  /** Dirección para un toque sin movimiento cuando se toma del trazo. */
  fallbackDirection: number;
}

const INK = '200, 255, 62';

/**
 * Lienzo del pincel. Ocupa exactamente la imagen, así las coordenadas normalizadas
 * corresponden a la imagen. Las medidas `workWidth/Height` dan la proporción real.
 */
@Component({
  selector: 'app-brush-canvas',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'absolute inset-0 block' },
  template: `
    <canvas
      #canvas
      data-testid="brush-canvas"
      class="absolute inset-0 h-full w-full cursor-crosshair"
      style="touch-action: none"
      aria-label="Lienzo del pincel: arrastra sobre la imagen para indicar qué se mueve y hacia dónde"
      (pointerdown)="onDown($event)"
      (pointermove)="onMove($event)"
      (pointerup)="onUp($event)"
      (pointercancel)="onUp($event)"
    ></canvas>
  `,
})
export class BrushCanvas implements AfterViewInit, OnDestroy {
  readonly strokes = model.required<BrushStroke[]>();
  readonly settings = input.required<BrushSettings>();
  readonly workWidth = input.required<number>();
  readonly workHeight = input.required<number>();

  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
  private readonly current = signal<BrushPoint[] | null>(null);
  private activePointer: number | null = null;
  private resizer: ResizeObserver | null = null;
  private readonly tick = signal(0);

  readonly limitReached = computed(() => !canAddStroke(this.strokes()));

  constructor() {
    effect(() => {
      this.tick();
      this.draw(this.strokes(), this.current(), this.settings());
    });
  }

  ngAfterViewInit(): void {
    const canvas = this.canvasRef().nativeElement;
    this.resizer = new ResizeObserver(() => {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      this.tick.update((value) => value + 1);
    });
    this.resizer.observe(canvas);
  }

  ngOnDestroy(): void {
    this.resizer?.disconnect();
  }

  private toPoint(event: PointerEvent): BrushPoint {
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    return normalizePoint(event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height);
  }

  protected onDown(event: PointerEvent): void {
    if (this.activePointer !== null || !canAddStroke(this.strokes())) return;
    event.preventDefault();
    this.activePointer = event.pointerId;
    (event.target as Element).setPointerCapture?.(event.pointerId);
    this.current.set([this.toPoint(event)]);
  }

  protected onMove(event: PointerEvent): void {
    const points = this.current();
    if (event.pointerId !== this.activePointer || !points) return;
    if (points.length >= remainingPoints(this.strokes())) return;
    this.current.set(
      appendPoint(points, this.toPoint(event), this.settings().radius, this.workWidth(), this.workHeight()),
    );
  }

  protected onUp(event: PointerEvent): void {
    if (event.pointerId !== this.activePointer) return;
    this.activePointer = null;
    const points = this.current();
    this.current.set(null);
    if (!points || event.type === 'pointercancel') return;
    const { radius, strength } = this.settings();
    this.strokes.update((strokes) => [
      ...strokes,
      { points, radius, strength, direction_deg: this.resolveDirection(points) },
    ]);
  }

  private resolveDirection(points: BrushPoint[]): number {
    const { fixedDirection, fallbackDirection } = this.settings();
    if (fixedDirection !== null) return fixedDirection;
    return directionFromPoints(points, this.workWidth(), this.workHeight()) ?? fallbackDirection;
  }

  private draw(strokes: BrushStroke[], current: BrushPoint[] | null, settings: BrushSettings): void {
    const canvas = this.canvasRef().nativeElement;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (const stroke of strokes) this.drawStroke(context, stroke);
    if (current) {
      this.drawStroke(context, {
        points: current,
        radius: settings.radius,
        strength: settings.strength,
        direction_deg: this.resolveDirection(current),
      });
    }
  }

  private drawStroke(context: CanvasRenderingContext2D, stroke: BrushStroke): void {
    const { width, height } = context.canvas;
    // El radio es una fracción del lado mayor, según el contrato.
    const radius = stroke.radius * Math.max(width, height);
    const opacity = 0.22 + stroke.strength * 0.38;
    const first = stroke.points[0];
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.strokeStyle = context.fillStyle = `rgba(${INK}, ${opacity})`;
    if (stroke.points.length === 1) {
      context.beginPath();
      context.arc(first[0] * width, first[1] * height, radius, 0, Math.PI * 2);
      context.fill();
    } else {
      context.lineWidth = radius * 2;
      context.beginPath();
      stroke.points.forEach(([x, y], index) => {
        if (index === 0) context.moveTo(x * width, y * height);
        else context.lineTo(x * width, y * height);
      });
      context.stroke();
    }
    this.drawArrow(context, stroke, radius);
  }

  /** Flecha en el centro del trazo: muestra la dirección que se enviará. */
  private drawArrow(context: CanvasRenderingContext2D, stroke: BrushStroke, radius: number): void {
    const { width, height } = context.canvas;
    const middle = stroke.points[Math.floor(stroke.points.length / 2)];
    const unit = Math.max(width, height) / 100;
    const length = Math.max(radius * 1.4, unit * 2.4);
    const angle = (stroke.direction_deg * Math.PI) / 180;
    context.save();
    context.translate(middle[0] * width, middle[1] * height);
    context.rotate(angle);
    context.lineWidth = Math.max(unit * 0.5, 2);
    context.strokeStyle = 'rgba(16, 20, 10, .95)';
    context.beginPath();
    context.moveTo(-length / 2, 0);
    context.lineTo(length / 2, 0);
    context.moveTo(length / 2 - length * 0.35, -length * 0.3);
    context.lineTo(length / 2, 0);
    context.lineTo(length / 2 - length * 0.35, length * 0.3);
    context.stroke();
    context.restore();
  }
}


