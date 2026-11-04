import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  viewChild,
} from '@angular/core';
import gsap from 'gsap';
import { prefersReducedMotion } from '../core/motion';

/**
 * Barrido de luz sobre la imagen mientras el motor trabaja. Es solo una señal de
 * actividad: no representa avance. El avance real lo muestra JobProgress.
 */
@Component({
  selector: 'app-working-sweep',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'pointer-events-none absolute inset-0 block overflow-hidden', 'aria-hidden': 'true' },
  template: `
    <div class="absolute inset-0 bg-ink-950/35"></div>
    <div #band class="band absolute inset-y-0 left-0 w-1/3 opacity-0"></div>
  `,
  styles: `
    .band {
      background: linear-gradient(
        90deg,
        transparent,
        rgb(107 231 240 / 0.08) 45%,
        rgb(200 255 62 / 0.4) 92%,
        rgb(245 242 234 / 0.9) 100%
      );
    }
  `,
})
export class WorkingSweep implements AfterViewInit, OnDestroy {
  private readonly band = viewChild.required<ElementRef<HTMLElement>>('band');
  private tween: gsap.core.Tween | null = null;

  ngAfterViewInit(): void {
    if (prefersReducedMotion()) return;
    this.tween = gsap.fromTo(
      this.band().nativeElement,
      { xPercent: -100, opacity: 1 },
      { xPercent: 300, duration: 2.6, ease: 'sine.inOut', repeat: -1, repeatDelay: 0.35 },
    );
  }

  ngOnDestroy(): void {
    this.tween?.kill();
  }
}
