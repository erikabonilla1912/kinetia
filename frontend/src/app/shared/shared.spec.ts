import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { formatMs, moodPresentation } from '../core/presentation';
import { ConfigStore, HealthStore } from '../core/stores';
import { API_BASE } from '../core/api.service';
import { TEST_CONFIG } from '../testing/fixtures.spec';
import { ConfigGate } from './config-gate';
import { DummyModelBanner } from './dummy-model-banner';
import { QrCode } from './qr-code';
import { ResultMedia } from './result-media';

describe('presentation', () => {
  it('formatMs distingue null de 0', () => {
    expect(formatMs(null)).toBe('sin medir');
    expect(formatMs(undefined)).toBe('sin medir');
    expect(formatMs(0)).toBe('0,0 ms');
    expect(formatMs(12.34)).toBe('12,3 ms');
    expect(formatMs(76084)).toBe('76,1 s');
  });

  it('un mood desconocido se muestra igual', () => {
    expect(moodPresentation('otro_mood').label).toBe('Otro mood');
    expect(moodPresentation('epico').label).toBe('Épico');
  });
});

describe('ConfigStore', () => {
  let http: HttpTestingController;
  beforeEach(() => {
    jasmine.clock().install();
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    http = TestBed.inject(HttpTestingController);
  });
  afterEach(() => jasmine.clock().uninstall());

  it('reintenta solo cuando falla y luego expone los datos', () => {
    const store = TestBed.inject(ConfigStore);
    store.load();
    http.expectOne(`${API_BASE}/config`).error(new ProgressEvent('error'));
    expect(store.error()?.code).toBe('NETWORK_ERROR');
    jasmine.clock().tick(5000);
    http.expectOne(`${API_BASE}/config`).flush(TEST_CONFIG);
    expect(store.error()).toBeNull();
    expect(store.classesById().get(9)?.color).toBe('#ABCDEF');
  });
});

describe('HealthStore y DummyModelBanner', () => {
  let http: HttpTestingController;
  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    http = TestBed.inject(HttpTestingController);
  });

  it('avisa con dummy-dev', async () => {
    TestBed.inject(HealthStore).load();
    http.expectOne(`${API_BASE}/health`).flush({ status: 'ok', model_loaded: true, depth_loaded: true, model_version: 'dummy-dev' });
    const fixture = TestBed.createComponent(DummyModelBanner);
    await fixture.whenStable();
    expect(fixture.nativeElement.textContent).toContain('dummy-dev');
  });

  it('no avisa con un modelo real', async () => {
    TestBed.inject(HealthStore).load();
    http.expectOne(`${API_BASE}/health`).flush({ status: 'ok', model_loaded: true, depth_loaded: true, model_version: 'prueba-1' });
    const fixture = TestBed.createComponent(DummyModelBanner);
    await fixture.whenStable();
    expect(fixture.nativeElement.textContent.trim()).toBe('');
  });

  it('un 503 de carga es un estado, no un fallo', async () => {
    TestBed.inject(HealthStore).load();
    http
      .expectOne(`${API_BASE}/health`)
      .flush({ status: 'loading', model_loaded: false, depth_loaded: false }, { status: 503, statusText: 'L' });
    const fixture = TestBed.createComponent(DummyModelBanner);
    await fixture.whenStable();
    expect(fixture.nativeElement.textContent).toContain('cargando');
    fixture.destroy();
  });
});

describe('ConfigGate', () => {
  it('no deja pasar sin /config y muestra el motivo', async () => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    const http = TestBed.inject(HttpTestingController);
    jasmine.clock().install();
    const store = TestBed.inject(ConfigStore);
    store.load();
    http.expectOne(`${API_BASE}/config`).error(new ProgressEvent('error'));
    const fixture = TestBed.createComponent(ConfigGate);
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Sin conexión con el servidor');
    jasmine.clock().uninstall();
  });
});

describe('QrCode', () => {
  it('cambia con el valor y conserva la zona de silencio', async () => {
    const fixture = TestBed.createComponent(QrCode);
    fixture.componentRef.setInput('value', 'http://a/expo/subir');
    await fixture.whenStable();
    const svg: SVGElement = fixture.nativeElement.querySelector('svg');
    const before = svg.querySelector('path')!.getAttribute('d');
    const size = Number(svg.getAttribute('viewBox')!.split(' ')[2]);
    // Ningún módulo oscuro entra en los 4 módulos del borde.
    const xs = [...before!.matchAll(/M(\d+) (\d+)/g)].flatMap((m) => [Number(m[1]), Number(m[2])]);
    expect(Math.min(...xs)).toBeGreaterThanOrEqual(4);
    expect(Math.max(...xs)).toBeLessThan(size - 4);

    fixture.componentRef.setInput('value', 'http://b/expo/subir');
    await fixture.whenStable();
    expect(svg.querySelector('path')!.getAttribute('d')).not.toBe(before);
  });
});

describe('ResultMedia', () => {
  it('un MP4 va en video en bucle, silencioso e inline', async () => {
    const fixture = TestBed.createComponent(ResultMedia);
    fixture.componentRef.setInput('url', '/api/v1/results/a.mp4');
    await fixture.whenStable();
    const video: HTMLVideoElement = fixture.nativeElement.querySelector('video');
    expect(video.muted).toBeTrue();
    expect(video.loop).toBeTrue();
    expect(video.hasAttribute('playsinline')).toBeTrue();
    expect(video.getAttribute('src')).toBe('/api/v1/results/a.mp4');
    video.removeAttribute('src');
  });

  it('un GIF va en imagen', async () => {
    const fixture = TestBed.createComponent(ResultMedia);
    fixture.componentRef.setInput('url', '/api/v1/results/a.gif');
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelector('img').getAttribute('src')).toBe('/api/v1/results/a.gif');
    expect(fixture.nativeElement.querySelector('video')).toBeNull();
  });
});
