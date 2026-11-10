import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { LOCALE_ID } from '@angular/core';
import { registerLocaleData } from '@angular/common';
import localeEs from '@angular/common/locales/es';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { AnalyzeResponse } from '../../core/api.models';
import { provideLoadedConfig, testAnalysis } from '../../testing/fixtures.spec';
import { XrayPanel } from './xray';

registerLocaleData(localeEs);

describe('XrayPanel', () => {
  let fixture: ComponentFixture<XrayPanel>;
  const root = () => fixture.nativeElement as HTMLElement;
  const all = (id: string) => [...root().querySelectorAll<HTMLElement>(`[data-testid="${id}"]`)];
  const one = (id: string) => root().querySelector<HTMLElement>(`[data-testid="${id}"]`);

  async function render(analysis: AnalyzeResponse): Promise<void> {
    fixture.componentRef.setInput('analysis', analysis);
    await fixture.whenStable();
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting(), { provide: LOCALE_ID, useValue: 'es' }],
    });
    provideLoadedConfig(TestBed.inject(HttpTestingController));
    fixture = TestBed.createComponent(XrayPanel);
  });

  it('dibuja solo las clases recibidas, en el orden recibido', async () => {
    await render(testAnalysis());
    expect(all('class-row').length).toBe(2);
    expect(all('class-label').map((el) => el.textContent?.trim())).toEqual(['Etiqueta 9', 'Etiqueta 4']);
  });

  it('una clase que /config no conoce se muestra con su nombre crudo', async () => {
    await render(testAnalysis({ classes: [{ id: 50, name: 'otra_clase', pixel_pct: 1, confidence: 0.5 }] }));
    expect(one('class-label')?.textContent?.trim()).toBe('otra_clase');
  });

  it('muestra pixel_pct y confianza recibidos', async () => {
    await render(testAnalysis());
    expect(all('class-pct')[0].textContent).toContain('73,4');
    expect(all('class-confidence')[0].textContent).toContain('0,81');
    expect(all('class-pct')[1].textContent).toContain('11,5');
    expect(all('class-confidence')[1].textContent).toContain('0,38');
  });

  it('cambia con otros datos: nada está fijo', async () => {
    await render(
      testAnalysis({
        classes: [{ id: 1, name: 'cielo', pixel_pct: 5.25, confidence: 0.11 }],
        inference_ms: { segmentation: 1.5, depth: null, total: 2.5 },
        conf_threshold: 0.77,
        model_version: 'otra-version',
      }),
    );
    expect(all('class-label').map((el) => el.textContent?.trim())).toEqual(['Etiqueta 1']);
    expect(one('class-pct')?.textContent).toContain('5,3');
    expect(one('class-confidence')?.textContent).toContain('0,11');
    expect(one('inference-total')?.textContent).toContain('2,5');
    expect(one('inference-seg')?.textContent).toContain('1,5');
    expect(one('conf-threshold')?.textContent).toContain('0.77');
    expect(one('model-version')?.textContent).toContain('otra-version');
  });

  it('toma los colores de /config', async () => {
    await render(testAnalysis());
    const swatches = all('class-swatch').map((el) => el.style.backgroundColor);
    expect(swatches).toEqual(['rgb(171, 205, 239)', 'rgb(119, 136, 153)']);
  });

  it('un tiempo null se muestra como "sin medir", nunca como 0', async () => {
    await render(testAnalysis({ inference_ms: { segmentation: null, depth: null, total: 0 } }));
    expect(one('inference-seg')?.textContent).toContain('sin medir');
    // Un cero medido sí es un dato.
    expect(one('inference-total')?.textContent).toContain('0,0 ms');
  });

  it('no muestra profundidad: ya no hay modelo de profundidad', async () => {
    await render(testAnalysis());
    expect(one('inference-depth')).toBeNull();
  });

  it('muestra el tiempo de inferencia recibido', async () => {
    await render(testAnalysis());
    expect(one('inference-total')?.textContent).toContain('987,7');
    expect(one('inference-seg')?.textContent).toContain('321,4');
  });

  it('explica la confianza según el contrato, aclarando lo que no es', async () => {
    await render(testAnalysis());
    const help = one('confidence-help')?.textContent ?? '';
    expect(help).toContain('promedio');
    expect(help).toContain('No es la probabilidad');
    expect(help).toContain('0.42');
    // La confianza no se muestra como porcentaje: se leería como probabilidad de presencia.
    expect(all('class-confidence')[0].textContent).not.toContain('%');
  });

  it('avisa con model_version dummy-dev', async () => {
    await render(testAnalysis());
    expect(one('dummy-warning')).toBeNull();
    await render(testAnalysis({ model_version: 'dummy-dev' }));
    expect(one('dummy-warning')?.textContent).toContain('dummy-dev');
  });

  it('muestra el estado vacío si no llegan clases', async () => {
    await render(testAnalysis({ classes: [] }));
    expect(one('xray-empty')).not.toBeNull();
    expect(all('class-row').length).toBe(0);
  });

  it('tocar una clase la enfoca y volver a tocarla la libera', async () => {
    await render(testAnalysis());
    all('class-row')[1].click();
    await fixture.whenStable();
    expect(fixture.componentInstance.focusId()).toBe(4);
    expect(all('class-row')[1].getAttribute('aria-pressed')).toBe('true');
    all('class-row')[1].click();
    await fixture.whenStable();
    expect(fixture.componentInstance.focusId()).toBeNull();
  });
});
