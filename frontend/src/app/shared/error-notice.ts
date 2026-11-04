import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import { ApiError } from '../core/api-error';
import { describeError } from '../core/error-catalog';

/** Muestra un error: guía según `code`, `message` literal y la salida que corresponde. */
@Component({
  selector: 'app-error-notice',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @let err = view();
    <div
      role="alert"
      class="rounded-2xl border border-danger/40 bg-danger-bg/90 p-4 sm:p-5"
      [attr.data-error-code]="err.code"
    >
      <p class="font-display text-lg font-bold text-paper">{{ err.title }}</p>
      <p class="mt-1 text-paper">{{ err.message }}</p>
      <p class="mt-1 text-sm text-danger">{{ err.hint }}</p>
      @if (countdown() > 0) {
        <p class="data mt-2 text-sm text-paper" aria-live="polite">
          Podrás reintentar en {{ countdown() }} s
        </p>
      }
      <div class="mt-3 flex flex-wrap gap-2">
        @if (err.recovery === 'retry' && canRetry()) {
          <button type="button" class="btn" [disabled]="countdown() > 0" (click)="retry.emit()">
            Reintentar
          </button>
        }
        @if (canChangeFile()) {
          <button type="button" class="btn" (click)="changeFile.emit()">Elegir otro archivo</button>
        }
      </div>
    </div>
  `,
})
export class ErrorNotice {
  readonly error = input.required<ApiError>();
  readonly canRetry = input(true);
  readonly canChangeFile = input(false);
  readonly retry = output<void>();
  readonly changeFile = output<void>();

  protected readonly view = computed(() => describeError(this.error()));
  protected readonly countdown = signal(0);

  constructor() {
    let timer: ReturnType<typeof setInterval> | null = null;
    const stop = () => {
      if (timer !== null) clearInterval(timer);
      timer = null;
    };
    inject(DestroyRef).onDestroy(stop);
    effect(() => {
      stop();
      const seconds = this.error().retryAfterSeconds ?? 0;
      this.countdown.set(seconds);
      if (seconds > 0) {
        timer = setInterval(() => {
          this.countdown.update((value) => value - 1);
          if (this.countdown() <= 0) stop();
        }, 1000);
      }
    });
  }
}
