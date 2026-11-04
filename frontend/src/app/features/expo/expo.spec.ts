import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ConfigStore } from '../../core/stores';
import { provideLoadedConfig } from '../../testing/fixtures.spec';
import { ExpoUploadPage } from './expo-upload.page';
import { ExpoWallPage, WALL_MAX_TILES, WALL_POLL_MS, joinUrl, wallLayout } from './expo-wall.page';

function setup(): HttpTestingController {
  TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
  const http = TestBed.inject(HttpTestingController);
  provideLoadedConfig(http);
  return http;
}

describe('ExpoUploadPage', () => {
  let fixture: ComponentFixture<ExpoUploadPage>;
  let http: HttpTestingController;
  const root = () => fixture.nativeElement as HTMLElement;
  const q = <T extends HTMLElement = HTMLElement>(id: string) => root().querySelector<T>(`[data-testid="${id}"]`);

  beforeEach(async () => {
    http = setup();
    fixture = TestBed.createComponent(ExpoUploadPage);
    await fixture.whenStable();
  });

  afterEach(() => fixture.destroy());

  it('usa el input de cámara exacto del contrato', () => {
    const input = q<HTMLInputElement>('expo-camera')!;
    expect(input.type).toBe('file');
    expect(input.accept).toBe('image/*,video/*');
    expect(input.hasAttribute('capture')).toBeTrue();
  });

  it('el consentimiento empieza sin marcar y hay aviso de pantalla pública', () => {
    expect(q<HTMLInputElement>('consent')!.checked).toBeFalse();
    expect(q('public-notice')?.textContent).toContain('pantalla pública');
    expect(q('consent-state')?.textContent).toContain('no aparece en el muro');
  });

  it('los moods vienen de /config, con nombre derivado si son nuevos', () => {
    const text = root().textContent ?? '';
    expect(text).toContain('Calma');
    expect(text).toContain('Épico');
    expect(text).toContain('Mood nuevo');
    expect(text).not.toContain('Nostálgico');
  });

  function submit(consent: boolean): FormData {
    const instance = fixture.componentInstance;
    instance.media.set({
      file: new File(['x'], 'a.png'),
      kind: 'image',
      format: 'png',
      width: 400,
      height: 400,
      durationSeconds: null,
    });
    instance.consent.set(consent);
    instance.animate();
    const post = http.expectOne('/api/v1/animate');
    const form = post.request.body as FormData;
    post.flush({ error: { code: 'INTERNAL_ERROR', message: 'x' } }, { status: 500, statusText: 'E' });
    return form;
  }

  it('sin marcar envía gallery_opt_in=false', () => {
    expect(submit(false).get('gallery_opt_in')).toBe('false');
  });

  it('marcado envía gallery_opt_in=true', () => {
    expect(submit(true).get('gallery_opt_in')).toBe('true');
  });
});

describe('ExpoWallPage', () => {
  let fixture: ComponentFixture<ExpoWallPage>;
  let http: HttpTestingController;
  const root = () => fixture.nativeElement as HTMLElement;
  const q = (id: string) => root().querySelector<HTMLElement>(`[data-testid="${id}"]`);
  const galleryUrl = `/api/v1/gallery?limit=${WALL_MAX_TILES}`;
  const items = (...ids: string[]) => ({
    items: ids.map((job_id) => ({
      job_id,
      mood: 'calma',
      result_url: `/api/v1/results/${job_id}.gif`,
      created_at: '2026-01-01T00:00:00Z',
    })),
  });

  beforeEach(async () => {
    jasmine.clock().install();
    http = setup();
    fixture = TestBed.createComponent(ExpoWallPage);
    fixture.autoDetectChanges();
  });

  afterEach(() => {
    fixture.destroy();
    http.verify();
    jasmine.clock().uninstall();
  });

  const tiles = () => root().querySelectorAll('[data-testid="wall-tile"]').length;
  const render = async () => {
    jasmine.clock().tick(1);
    await fixture.whenStable();
  };

  it('el QR y el texto apuntan a public_base_url + /expo/subir, sin doble barra', async () => {
    http.expectOne(galleryUrl).flush(items());
    await render();
    expect(q('wall-url')?.textContent).toContain('http://10.1.2.3:8080/expo/subir');
    expect(root().querySelector('svg[role="img"]')?.getAttribute('aria-label')).toContain(
      'http://10.1.2.3:8080/expo/subir',
    );
    expect(root().querySelector('svg path')?.getAttribute('d')?.length).toBeGreaterThan(100);
    expect(TestBed.inject(ConfigStore).config()?.public_base_url).toBe('http://10.1.2.3:8080/');
  });

  it('galería vacía: estado vacío con el QR', async () => {
    http.expectOne(galleryUrl).flush(items());
    await render();
    expect(q('wall-empty')?.textContent).toContain('vacío');
    expect(q('wall-qr')).not.toBeNull();
  });

  it('consulta periódicamente y refleja lo nuevo', async () => {
    http.expectOne(galleryUrl).flush(items('a'));
    await render();
    expect(tiles()).toBe(1);
    jasmine.clock().tick(WALL_POLL_MS);
    http.expectOne(galleryUrl).flush(items('b', 'a'));
    await render();
    expect(tiles()).toBe(2);
    fixture.destroy();
  });

  it('lo que el equipo modera desaparece del muro', async () => {
    http.expectOne(galleryUrl).flush(items('a', 'b'));
    await render();
    jasmine.clock().tick(WALL_POLL_MS);
    http.expectOne(galleryUrl).flush(items('b'));
    await render();
    expect(tiles()).toBe(1);
    expect(root().querySelector('[data-job="a"]')).toBeNull();
    fixture.destroy();
  });

  it('un corte de conexión avisa pero conserva el muro', async () => {
    http.expectOne(galleryUrl).flush(items('a'));
    await render();
    jasmine.clock().tick(WALL_POLL_MS);
    http.expectOne(galleryUrl).error(new ProgressEvent('error'));
    await render();
    expect(q('wall-offline')).not.toBeNull();
    expect(tiles()).toBe(1);
    jasmine.clock().tick(WALL_POLL_MS);
    http.expectOne(galleryUrl).flush(items('a'));
    await render();
    expect(q('wall-offline')).toBeNull();
    fixture.destroy();
  });

  it('un resultado que ya no carga sale del muro', async () => {
    http.expectOne(galleryUrl).flush(items('a', 'b'));
    await render();
    root().querySelector('[data-job="a"] img')!.dispatchEvent(new Event('error'));
    await render();
    expect(root().querySelector('[data-job="a"]')).toBeNull();
    fixture.destroy();
  });
});

describe('wallLayout', () => {
  it('no deja huecos con ninguna cantidad de tiles', () => {
    for (let count = 1; count <= WALL_MAX_TILES; count++) {
      const { cols, rows, heroCols, heroRows } = wallLayout(count);
      const cells = count - 1 + heroCols * heroRows;
      expect(cells).withContext(`${count} tiles`).toBe(cols * rows);
    }
  });
});

describe('joinUrl', () => {
  it('une con o sin barra final', () => {
    expect(joinUrl('http://a:1/', '/x')).toBe('http://a:1/x');
    expect(joinUrl('http://a:1', '/x')).toBe('http://a:1/x');
  });
});
