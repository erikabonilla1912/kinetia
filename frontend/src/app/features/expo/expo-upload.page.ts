import { ChangeDetectionStrategy, Component, computed, effect, inject, signal, untracked } from '@angular/core';
import { ApiError } from '../../core/api-error';
import { JobRunner } from '../../core/job-runner';
import { moodPresentation } from '../../core/presentation';
import { ConfigStore } from '../../core/stores';
import { ValidatedMedia, validateUpload } from '../../core/upload-validator';
import { DummyModelBanner } from '../../shared/dummy-model-banner';
import { ErrorNotice } from '../../shared/error-notice';
import { ResultMedia } from '../../shared/result-media';
import { WorkingSweep } from '../../shared/working-sweep';
import { JobProgress } from '../studio/job-progress';

/** Vista móvil a la que lleva el QR. Una columna, un paso a la vez. */
@Component({
  selector: 'app-expo-upload-page',
  imports: [DummyModelBanner, ErrorNotice, JobProgress, ResultMedia, WorkingSweep],
  providers: [JobRunner],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <main class="mx-auto flex min-h-dvh w-full max-w-lg flex-col gap-4 px-4 pb-10 pt-6">
      <header>
        <p class="font-display text-xl font-extrabold tracking-tight">KINETIA<span class="text-pulse">.</span></p>
        <h1 class="mt-3 text-[2.75rem] font-extrabold leading-[.95]">
          Dale <span class="text-pulse">vida</span> a tu foto
        </h1>
      </header>
      <app-dummy-model-banner />

      @if (job().phase === 'idle' || job().phase === 'failed') {
        <section class="glass p-4" aria-labelledby="step-file">
          <h2 id="step-file" class="text-lg font-bold"><span class="data mr-2 text-sm text-pulse">1</span>Tu foto o video</h2>

          @if (media(); as selected) {
            <div class="mt-3 overflow-hidden rounded-xl border border-white/15 bg-ink-900">
              @if (selected.kind === 'image') {
                <img class="max-h-72 w-full object-cover" alt="La foto que elegiste" data-testid="expo-preview" [src]="previewUrl()" />
              } @else {
                <video
                  class="max-h-72 w-full object-cover"
                  preload="metadata"
                  playsinline
                  aria-label="El video que elegiste"
                  data-testid="expo-preview"
                  [muted]="true"
                  [src]="previewUrl() + '#t=0.001'"
                ></video>
              }
            </div>
          }

          <div class="mt-3 grid gap-2">
            <label class="btn" [class.btn-primary]="!media()" [class.btn-lg]="!media()">
              <!-- Contrato, sección 9: la cámara se abre así, sin getUserMedia. -->
              <input
                #camera
                type="file"
                accept="image/*,video/*"
                capture
                class="sr-only-input"
                data-testid="expo-camera"
                (change)="onPick(camera)"
              />
              {{ media() ? 'Tomar otra' : 'Abrir la cámara' }}
            </label>
            <label class="btn">
              <input
                #gallery
                type="file"
                accept="image/*,video/*"
                class="sr-only-input"
                data-testid="expo-gallery"
                (change)="onPick(gallery)"
              />
              Elegir de mi galería
            </label>
          </div>
          @if (limitsHint(); as hint) {
            <p class="data mt-2 text-xs text-mist">{{ hint }}</p>
          }
          @if (validating()) {
            <p class="mt-2 text-sm text-mist" role="status">Revisando tu archivo…</p>
          }
          @if (uploadError(); as error) {
            <div class="mt-3"><app-error-notice [error]="error" [canRetry]="false" /></div>
          }
        </section>

        <section class="glass p-4" aria-labelledby="step-mood">
          <h2 id="step-mood" class="text-lg font-bold"><span class="data mr-2 text-sm text-pulse">2</span>Elige un mood</h2>
          <fieldset class="mt-3">
            <legend class="sr-only">Mood de la animación</legend>
            <div class="grid grid-cols-2 gap-2">
              @for (option of moodOptions(); track option.name) {
                <label
                  class="flex min-h-14 cursor-pointer items-center justify-center rounded-xl border px-2 text-center font-display text-lg font-bold transition-colors has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-pulse"
                  [class]="mood() === option.name ? 'border-pulse bg-pulse text-on-pulse' : 'border-white/20 bg-white/5'"
                >
                  <input
                    type="radio"
                    name="expo-mood"
                    class="sr-only-input"
                    [value]="option.name"
                    [checked]="mood() === option.name"
                    (change)="mood.set(option.name)"
                  />
                  {{ option.label }}
                </label>
              }
            </div>
          </fieldset>
        </section>

        <section class="glass p-4" aria-labelledby="step-consent">
          <h2 id="step-consent" class="text-lg font-bold"><span class="data mr-2 text-sm text-pulse">3</span>¿Va al muro?</h2>
          <p class="mt-2 text-paper" data-testid="public-notice">
            Si marcas la casilla, tu animación <strong>aparecerá en una pantalla pública</strong> del stand,
            a la vista de cualquier visitante.
          </p>
          <label class="mt-3 flex min-h-11 cursor-pointer items-start gap-3 rounded-xl border border-white/20 bg-white/5 p-3">
            <input
              type="checkbox"
              class="mt-0.5 size-6 shrink-0 accent-[var(--color-pulse)]"
              data-testid="consent"
              [checked]="consent()"
              (change)="consent.set($any($event.target).checked)"
            />
            <span class="font-semibold">Acepto que se muestre en la pantalla pública. Si aparecen otras personas, ellas también están de acuerdo; si son menores de edad, tengo el permiso de su acudiente.</span>
          </label>
          <p class="mt-2 text-sm text-mist" data-testid="consent-state">
            @if (consent()) {
              Se mostrará en el muro. Para retirarla, habla con el equipo del stand.
            } @else {
              Sin marcar, la animación no aparece en el muro.
            }
          </p>
        </section>

        @if (job().phase === 'failed' && job().error; as error) {
          <div data-testid="job-error">
            <app-error-notice [error]="error" (retry)="animate()" />
          </div>
        }

        <button
          type="button"
          class="btn btn-primary btn-lg w-full"
          data-testid="expo-animate"
          [disabled]="!media() || !mood()"
          (click)="animate()"
        >
          Dale vida
        </button>
      } @else if (runner.busy()) {
        <section class="glass overflow-hidden">
          @if (media()?.kind === 'image') {
            <div class="relative">
              <img class="block max-h-80 w-full object-cover" alt="" [src]="previewUrl()" />
              <app-working-sweep />
            </div>
          }
          <div class="p-5">
          <app-job-progress [state]="job()" />
          <p class="mt-3 text-sm text-mist">Deja esta página abierta mientras termina.</p>
          </div>
        </section>
      } @else if (job().result; as res) {
        <section class="glass overflow-hidden" data-testid="expo-result">
          <div class="bg-ink-900" [style.aspect-ratio]="res.width + ' / ' + res.height">
            <app-result-media [url]="res.url" />
          </div>
          <div class="p-4">
            <h2 class="text-2xl font-bold">¡Cobró vida!</h2>
            <p class="mt-1 text-mist" data-testid="expo-outcome">
              @if (consent()) {
                Mira la pantalla grande: tu animación aparecerá en el muro.
              } @else {
                No se publicó en el muro.
              }
            </p>
            <div class="mt-3 grid gap-2">
              <a class="btn btn-primary btn-lg" data-testid="expo-download" [href]="res.url" [download]="'kinetia.' + res.format">
                Guardar en mi celular
              </a>
              <button type="button" class="btn" data-testid="expo-again" (click)="startOver()">Animar otra</button>
            </div>
          </div>
        </section>
      }
    </main>
  `,
})
export class ExpoUploadPage {
  private readonly store = inject(ConfigStore);
  protected readonly runner = inject(JobRunner);
  protected readonly job = this.runner.state;

  readonly media = signal<ValidatedMedia | null>(null);
  protected readonly previewUrl = signal('');
  protected readonly validating = signal(false);
  protected readonly uploadError = signal<ApiError | null>(null);
  readonly mood = signal('');
  /** Sin marcar por defecto: el consentimiento es explícito. */
  readonly consent = signal(false);

  protected readonly moodOptions = computed(() =>
    (this.store.config()?.moods ?? []).map((name) => ({ name, label: moodPresentation(name).label })),
  );
  protected readonly limitsHint = computed(() => {
    const limits = this.store.config()?.limits;
    return limits ? `Video de ${limits.max_video_seconds} s como máximo.` : null;
  });

  constructor() {
    effect(() => {
      const options = this.moodOptions();
      untracked(() => {
        if (options.length && !this.mood()) this.mood.set(options[0].name);
      });
    });
  }

  protected async onPick(element: HTMLInputElement): Promise<void> {
    const file = element.files?.[0];
    element.value = '';
    const limits = this.store.config()?.limits;
    if (!file || !limits) return;
    this.validating.set(true);
    this.uploadError.set(null);
    const validation = await validateUpload(file, limits);
    this.validating.set(false);
    if (!validation.ok) {
      this.uploadError.set(validation.error);
      return;
    }
    if (this.previewUrl()) URL.revokeObjectURL(this.previewUrl());
    this.previewUrl.set(URL.createObjectURL(file));
    this.media.set(validation.media);
    this.runner.reset();
  }

  animate(): void {
    const media = this.media();
    if (!media || !this.mood()) return;
    this.runner.start({
      file: media.file,
      mood: this.mood(),
      output_format: 'mp4',
      gallery_opt_in: this.consent(),
    });
  }

  protected startOver(): void {
    this.runner.reset();
    if (this.previewUrl()) URL.revokeObjectURL(this.previewUrl());
    this.previewUrl.set('');
    this.media.set(null);
    this.consent.set(false);
  }
}
