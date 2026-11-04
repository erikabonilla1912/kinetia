"""Validación de subidas: cada código de error, orden de comprobación y EXIF."""
from __future__ import annotations

import io

import numpy as np
import pytest
from api_fakes import API, assert_error, image_bytes, make_settings, png_header_only, post_file, running_client
from PIL import Image

from app.services import encoder
from app.services.validation import apply_exif_orientation, decode_image, sniff_format

UPLOAD_ROUTES = ["/analyze", "/animate"]


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_missing_file(client, route):
    assert_error(client.post(f"{API}{route}", data={"mood": "calma"}), 400, "MISSING_FILE")


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_file_sent_as_text_field_is_missing_file(client, route):
    assert_error(client.post(f"{API}{route}", data={"file": "no soy un archivo"}), 400, "MISSING_FILE")


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_empty_file_is_missing_file(client, route):
    assert_error(post_file(client, route, b""), 400, "MISSING_FILE")


def test_json_body_is_missing_file(client):
    assert_error(client.post(f"{API}/analyze", json={"a": 1}), 400, "MISSING_FILE")


def test_malformed_multipart_is_format_error(client):
    response = client.post(
        f"{API}/analyze", content=b"xxx", headers={"Content-Type": "multipart/form-data"}
    )
    assert_error(response, 400, "MISSING_FILE")


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\xd8\xff\xe0" + b"\x00" * 200,
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 200,
        b"RIFF\x10\x00\x00\x00WEBP" + b"\x00" * 200,
    ],
    ids=["jpg", "png", "webp"],
)
@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_valid_magic_with_garbage_is_corrupt(client, route, payload):
    assert_error(post_file(client, route, payload), 400, "CORRUPT_FILE")


def test_truncated_jpeg_is_corrupt(client):
    data = image_bytes("JPEG", 640, 480)
    assert_error(post_file(client, "/analyze", data[: len(data) // 2]), 400, "CORRUPT_FILE")


def test_truncated_png_is_corrupt(client):
    data = image_bytes("PNG", 640, 480)
    assert_error(post_file(client, "/analyze", data[: len(data) // 2]), 400, "CORRUPT_FILE")


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_fake_extension_text_named_jpg(client, route):
    response = post_file(client, route, b"Esto es texto plano, no una imagen.", filename="foto.jpg")
    assert_error(response, 415, "UNSUPPORTED_FORMAT")


def test_fake_extension_exe_with_mp4_name(client):
    assert_error(post_file(client, "/analyze", b"MZ" + b"\x00" * 100, filename="a.mp4", content_type="video/mp4"), 415, "UNSUPPORTED_FORMAT")


def test_real_png_with_wrong_name_and_type_is_accepted(client):
    response = post_file(client, "/analyze", image_bytes("PNG"), filename="nota.txt", content_type="text/plain")
    assert response.status_code == 200


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_other_real_images_are_unsupported(client, fmt):
    assert_error(post_file(client, "/analyze", image_bytes(fmt), filename="a.jpg"), 415, "UNSUPPORTED_FORMAT")


@pytest.mark.parametrize("brand", [b"heic", b"mif1", b"avif", b"M4A "])
def test_iso_container_that_is_not_video_is_unsupported(client, brand):
    data = b"\x00\x00\x00\x18ftyp" + brand + b"\x00" * 100
    assert_error(post_file(client, "/analyze", data, filename="a.mp4"), 415, "UNSUPPORTED_FORMAT")


def test_sniff_format_table():
    assert sniff_format(b"\xff\xd8\xff\xe0abc") == "jpg"
    assert sniff_format(b"\x89PNG\r\n\x1a\nabc") == "png"
    assert sniff_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert sniff_format(b"\x00\x00\x00\x18ftypisom") == "mp4"
    assert sniff_format(b"\x00\x00\x00\x14ftypqt  ") == "mov"
    assert sniff_format(b"\x00\x00\x00\x08wide....") == "mov"
    assert sniff_format(b"GIF89a") is None
    assert sniff_format(b"") is None


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_too_small_resolution(client, route):
    error = assert_error(post_file(client, route, image_bytes("PNG", 200, 500)), 422, "RESOLUTION_TOO_LOW")
    assert error["details"]["min_side_px"] == 256


def test_exact_min_side_is_accepted(client):
    assert post_file(client, "/analyze", image_bytes("PNG", 256, 256)).status_code == 200


def test_too_many_pixels_via_setting(tmp_path, dummy_models, engine):
    settings = make_settings(tmp_path, dummy_models, max_pixels=640 * 480 - 1)
    with running_client(settings, lambda: engine) as client:
        assert_error(post_file(client, "/analyze", image_bytes()), 422, "RESOLUTION_TOO_HIGH")


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_decompression_bomb_header_is_rejected_without_decoding(client, route):
    response = post_file(client, route, png_header_only(60000, 60000), filename="a.png")
    error = assert_error(response, 422, "RESOLUTION_TOO_HIGH")
    assert error["details"]["max_pixels"] == 40000000


def test_real_highly_compressed_bomb(client):
    buffer = io.BytesIO()
    Image.new("1", (7000, 7000)).save(buffer, format="PNG")
    assert len(buffer.getvalue()) < 1024 * 1024
    assert_error(post_file(client, "/analyze", buffer.getvalue()), 422, "RESOLUTION_TOO_HIGH")


def test_file_too_large_exact_check(tmp_path, dummy_models, engine):
    data = image_bytes("PNG")
    settings = make_settings(tmp_path, dummy_models, max_upload_bytes=len(data) - 1)
    with running_client(settings, lambda: engine) as client:
        error = assert_error(post_file(client, "/analyze", data), 413, "FILE_TOO_LARGE")
        assert error["details"]["max_bytes"] == len(data) - 1


def test_file_exactly_at_limit_is_accepted(tmp_path, dummy_models, engine):
    data = image_bytes("PNG")
    settings = make_settings(tmp_path, dummy_models, max_upload_bytes=len(data))
    with running_client(settings, lambda: engine) as client:
        assert post_file(client, "/analyze", data).status_code == 200


def test_file_too_large_by_content_length_uses_contract_message(client):
    size = 22 * 1024 * 1024
    error = assert_error(post_file(client, "/analyze", b"\xff\xd8\xff" + b"\x00" * size), 413, "FILE_TOO_LARGE")
    assert error["details"] == {"max_bytes": 15728640}
    assert "22" in error["message"] and "15" in error["message"]


def test_chunked_upload_over_limit_is_cut(client):
    def body():
        yield b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.jpg\"\r\n\r\n"
        for _ in range(20):
            yield b"\x00" * (1024 * 1024)
        yield b"\r\n--b--\r\n"

    response = client.post(
        f"{API}/analyze", content=body(), headers={"Content-Type": "multipart/form-data; boundary=b"}
    )
    assert_error(response, 413, "FILE_TOO_LARGE")


def test_order_size_before_format(tmp_path, dummy_models, engine):
    settings = make_settings(tmp_path, dummy_models, max_upload_bytes=10)
    with running_client(settings, lambda: engine) as client:
        assert_error(post_file(client, "/analyze", b"texto plano, demasiado largo"), 413, "FILE_TOO_LARGE")


def test_order_format_before_decoding(client):
    assert_error(post_file(client, "/analyze", b"\x00" * 100), 415, "UNSUPPORTED_FORMAT")


def test_order_decoding_before_low_resolution(client):
    data = image_bytes("PNG", 64, 64)
    assert_error(post_file(client, "/analyze", data[: len(data) // 2]), 400, "CORRUPT_FILE")


def _orientation_jpeg(orientation: int, width: int = 400, height: int = 300) -> bytes:
    # Cuadrante superior izquierdo rojo; el resto, azul.
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[...] = (0, 0, 255)
    pixels[: height // 2, : width // 2] = (255, 0, 0)
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[0x0112] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=100, subsampling=0, exif=exif)
    return buffer.getvalue()


# Dónde queda el cuadrante rojo tras aplicar la orientación: (arriba?, izquierda?), dimensiones giradas?
ORIENTATIONS = {
    1: (True, True, False),
    2: (True, False, False),
    3: (False, False, False),
    4: (False, True, False),
    5: (True, True, True),
    6: (True, False, True),
    7: (False, False, True),
    8: (False, True, True),
}


@pytest.mark.parametrize("orientation", sorted(ORIENTATIONS))
def test_exif_orientation_pixels(settings, orientation):
    top, left, swapped = ORIENTATIONS[orientation]
    image = decode_image(_orientation_jpeg(orientation), "jpg", settings)
    height, width = image.shape[:2]
    assert (width, height) == ((300, 400) if swapped else (400, 300))
    row = height // 4 if top else 3 * height // 4
    column = width // 4 if left else 3 * width // 4
    blue, green, red = image[row, column]
    assert red > 200 and blue < 60, f"Orientación {orientation}: {image[row, column]}"
    # El cuadrante opuesto no es rojo.
    other = image[3 * height // 4 if top else height // 4, 3 * width // 4 if left else width // 4]
    assert other[2] < 60


def test_exif_matches_pillow_transpose(settings):
    for orientation in range(1, 9):
        data = _orientation_jpeg(orientation)
        ours = decode_image(data, "jpg", settings)
        from PIL import ImageOps

        pillow = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
        assert ours.shape == pillow.shape
        # Mismo contenido salvo diferencias mínimas entre decodificadores de JPEG.
        assert np.abs(ours[..., ::-1].astype(int) - pillow.astype(int)).mean() < 3


def test_analyze_uses_rotated_dimensions(client):
    body = post_file(client, "/analyze", _orientation_jpeg(6)).json()
    assert (body["width"], body["height"]) == (300, 400)


def test_low_resolution_is_checked_after_rotation(client):
    # 200x400 con cualquier rotación sigue teniendo lado menor 200.
    assert_error(post_file(client, "/analyze", _orientation_jpeg(6, 400, 200)), 422, "RESOLUTION_TOO_LOW")


def test_apply_orientation_identity():
    array = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert apply_exif_orientation(array, 1) is array


@pytest.mark.parametrize(
    "form, field",
    [
        ({"mood": "furioso"}, "mood"),
        ({"output_format": "avi"}, "output_format"),
        ({"duration_s": "1.9"}, "duration_s"),
        ({"duration_s": "6.1"}, "duration_s"),
        ({"duration_s": "abc"}, "duration_s"),
        ({"duration_s": "nan"}, "duration_s"),
        ({"brush_mode": "todo"}, "brush_mode"),
        ({"gallery_opt_in": "quizas"}, "gallery_opt_in"),
        ({"seed": "uno"}, "seed"),
        ({"seed": "-1"}, "seed"),
        ({"mask_strokes": "{no es json"}, "mask_strokes"),
        ({"brush_mode": "only"}, "mask_strokes"),
    ],
)
def test_invalid_params_use_single_format(client, form, field):
    error = assert_error(post_file(client, "/animate", image_bytes(), **form), 422, "INVALID_PARAMS")
    assert field in {entry["field"] for entry in error["details"]["fields"]}


def test_invalid_query_param_uses_single_format(client):
    assert_error(client.get(f"{API}/gallery", params={"limit": "x"}), 422, "INVALID_PARAMS")
    assert_error(client.get(f"{API}/gallery", params={"limit": 0}), 422, "INVALID_PARAMS")
    assert_error(client.get(f"{API}/gallery", params={"after": "ayer"}), 422, "INVALID_PARAMS")


def test_invalid_bool_on_analyze(client):
    assert_error(post_file(client, "/analyze", image_bytes(), include_depth="tal vez"), 422, "INVALID_PARAMS")


def test_unexpected_exception_is_internal_error_without_trace(client, monkeypatch):
    services = client.app.state.services

    def explode(_):
        raise ValueError("detalle secreto C:\\ruta\\interna")

    monkeypatch.setattr(services.models, "segment", explode)
    response = post_file(client, "/analyze", image_bytes())
    error = assert_error(response, 500, "INTERNAL_ERROR")
    assert "secreto" not in response.text and "ruta" not in response.text
    assert error["details"] == {}


def test_unknown_route_and_method_use_single_format(client):
    assert_error(client.get(f"{API}/no-existe"), 404, "NOT_FOUND")
    assert_error(client.get(f"{API}/analyze"), 405, "METHOD_NOT_ALLOWED")


def _stub_video(monkeypatch, width, height, duration_s):
    """Sustituye a ffprobe y FFmpeg para probar las reglas sin ellos. La versión real está en test_api_ffmpeg.py."""
    info = encoder.VideoInfo(width=width, height=height, duration_s=duration_s, fps=24.0)
    monkeypatch.setattr(encoder, "probe_video", lambda path: info)
    monkeypatch.setattr(encoder, "read_first_frame", lambda path: np.zeros((height, width, 3), dtype=np.uint8))


MP4_HEAD = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100


def test_video_duration_rule(client, monkeypatch):
    _stub_video(monkeypatch, 640, 480, 10.5)
    error = assert_error(post_file(client, "/analyze", MP4_HEAD, filename="v.mp4"), 422, "VIDEO_TOO_LONG")
    assert error["details"]["max_video_seconds"] == 10
    assert_error(post_file(client, "/animate", MP4_HEAD, filename="v.mp4"), 422, "VIDEO_TOO_LONG")


def test_video_exactly_at_max_duration_is_accepted(client, monkeypatch):
    _stub_video(monkeypatch, 640, 480, 10.0)
    assert post_file(client, "/analyze", MP4_HEAD, filename="v.mp4").status_code == 200


def test_video_resolution_is_checked_before_duration(client, monkeypatch):
    _stub_video(monkeypatch, 100, 100, 99.0)
    assert_error(post_file(client, "/analyze", MP4_HEAD, filename="v.mp4"), 422, "RESOLUTION_TOO_LOW")
    _stub_video(monkeypatch, 9000, 9000, 99.0)
    assert_error(post_file(client, "/analyze", MP4_HEAD, filename="v.mp4"), 422, "RESOLUTION_TOO_HIGH")


def test_video_unreadable_by_ffprobe_is_corrupt(client, monkeypatch):
    def unreadable(path):
        raise encoder.ProbeError("no se pudo leer")

    monkeypatch.setattr(encoder, "probe_video", unreadable)
    assert_error(post_file(client, "/analyze", MP4_HEAD, filename="v.mp4"), 400, "CORRUPT_FILE")


def test_video_magic_without_ffmpeg_is_clear_internal_error(client, monkeypatch):
    def missing(_):
        raise encoder.FFmpegMissing("no hay")

    monkeypatch.setattr(encoder, "find_tool", missing)
    data = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100
    assert_error(post_file(client, "/analyze", data, filename="a.mp4"), 500, "INTERNAL_ERROR")
