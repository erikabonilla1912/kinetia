import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { filter, map } from 'rxjs';
import { ConnectivityStore, ConfigStore, HealthStore } from './core/stores';
import { ConfigGate } from './shared/config-gate';
import { DummyModelBanner } from './shared/dummy-model-banner';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, ConfigGate, DummyModelBanner],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (!online()) {
      <div
        role="alert"
        class="sticky top-0 z-50 bg-warning-bg px-4 py-2 text-center text-sm font-semibold text-warning"
      >
        Sin conexión. Lo que ya subiste sigue en el servidor; reconéctate para continuar.
      </div>
    }

    @if (showChrome()) {
      <header class="relative z-40 px-3 pt-3 sm:px-6 lg:sticky lg:top-0 lg:bg-gradient-to-b lg:from-ink-950 lg:via-ink-950/70 lg:to-transparent lg:pb-3">
        <nav
          aria-label="Principal"
          class="glass mx-auto flex max-w-[1720px] items-center justify-between gap-2 px-3 py-1.5 sm:px-5"
        >
          <a
            routerLink="/"
            class="flex min-h-11 items-center font-display text-xl font-extrabold tracking-tight"
            aria-label="KINETIA, ir al estudio"
          >
            KINETIA<span class="text-pulse" aria-hidden="true">.</span>
          </a>
          <div class="flex items-center gap-1 text-sm font-semibold">
            <a
              routerLink="/"
              [routerLinkActiveOptions]="{ exact: true }"
              routerLinkActive="!text-paper bg-white/10"
              class="flex min-h-11 items-center rounded-full px-3 text-mist hover:text-paper"
              >Estudio</a
            >
            <a
              routerLink="/modelo"
              routerLinkActive="!text-paper bg-white/10"
              class="flex min-h-11 items-center rounded-full px-3 text-mist hover:text-paper"
              >Modelo</a
            >
            <a
              routerLink="/expo/pantalla"
              class="flex min-h-11 items-center rounded-full px-3 text-mist hover:text-paper"
              >Muro</a
            >
          </div>
        </nav>
      </header>
      <app-dummy-model-banner class="mt-3 block px-3 sm:px-6" />
    }

    <app-config-gate>
      <router-outlet />
    </app-config-gate>
  `,
})
export class App {
  private readonly router = inject(Router);
  private readonly connectivity = inject(ConnectivityStore);

  protected readonly online = this.connectivity.online;

  /** Las vistas de la Expo son a pantalla completa, sin navegación. */
  private readonly chromeFlag = toSignal(
    this.router.events.pipe(
      filter((event) => event instanceof NavigationEnd),
      map(() => {
        let route = this.router.routerState.snapshot.root;
        while (route.firstChild) route = route.firstChild;
        return route.data['chrome'] !== false;
      }),
    ),
    { initialValue: null },
  );
  protected readonly showChrome = computed(() => this.chromeFlag() === true);

  constructor() {
    inject(ConfigStore).load();
    inject(HealthStore).load();
  }
}
