import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  computed,
  effect,
  input,
  output,
  viewChild,
} from '@angular/core';

/**
 * Reproduce un resultado de la API: GIF como imagen, MP4 como video en bucle y silencioso.
 * `muted` se fija por código porque Angular no refleja el atributo y sin él no hay autoplay.
 */
@Component({
  selector: 'app-result-media',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block h-full w-full' },
  template: `
    @if (isGif()) {
      <img
        class="block h-full w-full object-cover"
        draggable="false"
        [alt]="alt()"
        [src]="url()"
        (error)="failed.emit()"
      />
    } @else {
      <video
        #video
        class="block h-full w-full object-cover"
        autoplay
        loop
        playsinline
        disablepictureinpicture
        preload="auto"
        [attr.aria-label]="alt()"
        (error)="failed.emit()"
      ></video>
    }
  `,
})
export class ResultMedia {
  readonly url = input.required<string>();
  readonly alt = input('Animación generada por KINETIA');
  readonly failed = output<void>();

  private readonly video = viewChild<ElementRef<HTMLVideoElement>>('video');
  protected readonly isGif = computed(() => /\.gif($|\?)/i.test(this.url()));

  constructor() {
    effect(() => {
      const element = this.video()?.nativeElement;
      if (!element) return;
      element.muted = true;
      element.defaultMuted = true;
      element.src = this.url();
      element.play()?.catch(() => undefined);
    });
  }
}
