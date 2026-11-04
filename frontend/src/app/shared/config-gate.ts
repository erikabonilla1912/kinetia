import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { ConfigStore } from '../core/stores';
import { describeError } from '../core/error-catalog';

/** Nada se dibuja sin /config: sin ella no hay clases, colores ni límites que mostrar. */
@Component({
  selector: 'app-config-gate',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (store.config()) {
      <ng-content />
    } @else {
      <main class="grid min-h-dvh place-items-center p-6">
        <div class="glass w-full max-w-md p-8 text-center" role="status" aria-live="polite">
          <p class="font-display text-4xl font-extrabold tracking-tight">
            KINETIA<span class="text-pulse">.</span>
          </p>
          @if (view(); as err) {
            <h1 class="mt-6 text-2xl font-bold">{{ err.title }}</h1>
            <p class="mt-2 text-mist">{{ err.message }}</p>
            <p class="mt-4 text-sm text-mist">Reintentando automáticamente…</p>
          } @else {
            <p class="mt-6 text-mist">Conectando con el servidor…</p>
          }
        </div>
      </main>
    }
  `,
})
export class ConfigGate {
  protected readonly store = inject(ConfigStore);
  protected readonly view = computed(() => {
    const error = this.store.error();
    return error ? describeError(error) : null;
  });
}
