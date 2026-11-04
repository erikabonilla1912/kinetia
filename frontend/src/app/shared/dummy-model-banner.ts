import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { HealthStore } from '../core/stores';

/** Aviso global según /health: aplica también a vistas que no llaman a /analyze. */
@Component({
  selector: 'app-dummy-model-banner',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (health.isDummy()) {
      <p
        role="alert"
        class="mx-auto max-w-[1720px] rounded-xl border border-warning/40 bg-warning-bg px-4 py-2.5 text-sm font-semibold text-warning"
      >
        Modelo de desarrollo (dummy-dev): sus pesos son aleatorios y los resultados no significan nada.
      </p>
    } @else if (health.health()?.status === 'loading') {
      <p
        role="status"
        class="mx-auto max-w-[1720px] rounded-xl border border-xray/30 bg-ink-800 px-4 py-2.5 text-sm font-semibold text-xray"
      >
        El modelo se está cargando. Podrás animar en cuanto termine.
      </p>
    }
  `,
})
export class DummyModelBanner {
  protected readonly health = inject(HealthStore);
}
