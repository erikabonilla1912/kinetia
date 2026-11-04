import { ChangeDetectionStrategy, Component, computed, input, output, signal } from '@angular/core';
import { AppLimits } from '../../core/api.models';
import { acceptFromFormats, formatBytes, formatList } from '../../core/upload-validator';

@Component({
  selector: 'app-dropzone',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <label
      data-testid="dropzone"
      class="glass group relative flex min-h-56 cursor-pointer flex-col items-center justify-center gap-3 overflow-hidden !rounded-[var(--radius-stage)] border-dashed p-6 text-center transition-all duration-300 sm:min-h-64"
      [class.!border-pulse]="dragging()"
      [class.scale-[1.01]]="dragging()"
      (dragenter)="onDrag($event, true)"
      (dragover)="onDrag($event, true)"
      (dragleave)="onDrag($event, false)"
      (drop)="onDrop($event)"
    >
      <input
        #input
        type="file"
        class="sr-only-input"
        data-testid="file-input"
        [accept]="accept()"
        [disabled]="disabled()"
        (change)="onPick(input)"
      />
      <span
        class="grid size-16 place-items-center rounded-full bg-pulse text-on-pulse transition-transform duration-300 group-hover:scale-110"
        aria-hidden="true"
      >
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
          <path
            d="M12 19V6m0 0-5.5 5.5M12 6l5.5 5.5"
            stroke="currentColor"
            stroke-width="2.4"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </span>
      <span class="font-display text-2xl font-bold sm:text-3xl">
        @if (disabled()) {
          Revisando tu archivo…
        } @else {
          <span class="hidden sm:inline">Arrastra una foto o un video</span>
          <span class="sm:hidden">Toca para elegir una foto o un video</span>
        }
      </span>
      <span class="hidden text-mist sm:inline">o haz clic para buscarlo</span>
      <span class="data text-xs text-mist" data-testid="dropzone-limits">{{ limitsText() }}</span>
    </label>
  `,
})
export class Dropzone {
  readonly limits = input.required<AppLimits>();
  readonly disabled = input(false);
  readonly fileChosen = output<File>();

  protected readonly dragging = signal(false);
  protected readonly accept = computed(() => acceptFromFormats(this.limits().formats));
  protected readonly limitsText = computed(() => {
    const limits = this.limits();
    return (
      `${formatList(limits.formats)} · hasta ${formatBytes(limits.max_upload_bytes)} · ` +
      `video de ${limits.max_video_seconds} s máx. · lado menor ≥ ${limits.min_side_px} px`
    );
  });

  protected onDrag(event: DragEvent, active: boolean): void {
    event.preventDefault();
    this.dragging.set(active);
  }

  protected onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(false);
    const file = event.dataTransfer?.files?.[0];
    if (file && !this.disabled()) this.fileChosen.emit(file);
  }

  protected onPick(element: HTMLInputElement): void {
    const file = element.files?.[0];
    // Se limpia para que elegir el mismo archivo otra vez vuelva a disparar el evento.
    element.value = '';
    if (file) this.fileChosen.emit(file);
  }
}
