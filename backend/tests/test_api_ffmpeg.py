"""Pruebas que necesitan FFmpeg real: video validado de verdad y parámetros de salida.

Si FFmpeg no está instalado se saltan de forma explícita y visible (`-ra` lo muestra).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from api_fakes import API, FakeEngine, assert_error, image_bytes, make_settings, post_file, running_client, wait_job

from app.services import encoder

pytestmark = pytest.mark.skipif(
    not encoder.ffmpeg_available(), reason="FFmpeg no está instalado: esta prueba necesita ffmpeg y ffprobe reales"
)


def make_video(path: Path, seconds: float = 1.0, size: str = "320x256", fps: int = 24, extra: list[str] | None = None) -> bytes:
    command = [
        encoder.find_tool("ffmpeg"), "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={size}:rate={fps}",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
        *(extra or []),
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return path.read_bytes()


def ffprobe(path: Path) -> dict:
    output = subprocess.run(
        [encoder.find_tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
    ).stdout
    return json.loads(output)


def box_order(path: Path) -> list[bytes]:
    data = path.read_bytes()
    position, boxes = 0, []
    while position + 8 <= len(data):
        size = int.from_bytes(data[position : position + 4], "big")
        boxes.append(data[position + 4 : position + 8])
        if size < 8:
            break
        position += size
    return boxes


@pytest.mark.parametrize("extension, brand", [("mp4", b"isom"), ("mov", b"qt  ")])
def test_real_video_is_analyzed_by_first_frame(client, tmp_path, extension, brand):
    data = make_video(tmp_path / f"v.{extension}")
    assert data[8:12] == brand
    response = post_file(client, "/analyze", data, filename=f"v.{extension}", content_type="video/mp4")
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["width"], body["height"]) == (320, 256)


def test_mov_with_mp4_name_is_validated_by_content(client, tmp_path):
    data = make_video(tmp_path / "v.mov")
    assert post_file(client, "/analyze", data, filename="foto.jpg", content_type="image/jpeg").status_code == 200


def test_video_too_long(client, tmp_path):
    data = make_video(tmp_path / "v.mp4", seconds=11.0, fps=5)
    error = assert_error(post_file(client, "/analyze", data, filename="v.mp4"), 422, "VIDEO_TOO_LONG")
    assert error["details"]["max_video_seconds"] == 10
    assert error["details"]["duration_s"] > 10
    assert_error(post_file(client, "/animate", data, filename="v.mp4"), 422, "VIDEO_TOO_LONG")


def test_video_low_resolution_before_duration(client, tmp_path):
    data = make_video(tmp_path / "v.mp4", seconds=11.0, size="128x128", fps=5)
    assert_error(post_file(client, "/analyze", data, filename="v.mp4"), 422, "RESOLUTION_TOO_LOW")


def test_video_too_many_pixels(tmp_path, dummy_models, engine):
    data = make_video(tmp_path / "v.mp4", size="640x480")
    settings = make_settings(tmp_path, dummy_models, max_pixels=1000)
    with running_client(settings, lambda: engine) as client:
        assert_error(post_file(client, "/analyze", data, filename="v.mp4"), 422, "RESOLUTION_TOO_HIGH")


@pytest.mark.parametrize("route", ["/analyze", "/animate"])
def test_mp4_magic_with_garbage_is_corrupt(client, route):
    data = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\xab" * 2000
    assert_error(post_file(client, route, data, filename="v.mp4"), 400, "CORRUPT_FILE")


def test_truncated_video_is_corrupt(client, tmp_path):
    data = make_video(tmp_path / "v.mp4", seconds=1.0)
    assert_error(post_file(client, "/analyze", data[: len(data) // 3], filename="v.mp4"), 400, "CORRUPT_FILE")


def test_audio_only_mp4_is_corrupt(client, tmp_path):
    path = tmp_path / "a.mp4"
    subprocess.run(
        [encoder.find_tool("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac", str(path)],
        check=True,
        capture_output=True,
    )
    assert_error(post_file(client, "/analyze", path.read_bytes(), filename="a.mp4"), 400, "CORRUPT_FILE")


def test_rotated_video_reports_rotated_dimensions(client, tmp_path):
    source = make_video(tmp_path / "v.mp4", size="400x300")
    assert source
    rotated = tmp_path / "r.mp4"
    subprocess.run(
        [encoder.find_tool("ffmpeg"), "-y", "-v", "error", "-display_rotation", "90", "-i", str(tmp_path / "v.mp4"), "-c", "copy", str(rotated)],
        check=True,
        capture_output=True,
    )
    body = post_file(client, "/analyze", rotated.read_bytes(), filename="r.mp4").json()
    assert (body["width"], body["height"]) == (300, 400)


def test_encoder_mp4_parameters(client, settings, tmp_path, engine):
    # Dimensiones impares a propósito: el contrato exige salida par.
    job_id = post_file(client, "/animate", image_bytes("PNG", 641, 481), duration_s="2").json()["job_id"]
    body = wait_job(client, job_id, 120)
    assert body["status"] == "succeeded", body
    path = settings.results_dir / f"{job_id}.mp4"
    info = ffprobe(path)
    stream = info["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["width"] % 2 == 0 and stream["height"] % 2 == 0
    assert (stream["width"], stream["height"]) == (640, 480)
    assert (body["result"]["width"], body["result"]["height"]) == (640, 480)
    assert int(stream["nb_frames"]) == 48
    assert stream["avg_frame_rate"] == "24/1"
    assert abs(float(info["format"]["duration"]) - 2.0) < 0.1
    boxes = box_order(path)
    assert boxes.index(b"moov") < boxes.index(b"mdat"), "+faststart: moov debe ir antes de mdat"
    assert body["result"]["size_bytes"] == path.stat().st_size
    assert not list(settings.results_dir.glob(".*")), "No deben quedar archivos parciales"
    assert client.get(body["result"]["url"], headers={"Range": "bytes=0-1"}).status_code == 206


def test_encoder_gif_parameters(client, settings):
    job_id = post_file(client, "/animate", image_bytes("PNG", 1000, 500), duration_s="2", output_format="gif").json()["job_id"]
    body = wait_job(client, job_id, 120)
    assert body["status"] == "succeeded", body
    path = settings.results_dir / f"{job_id}.gif"
    assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    stream = ffprobe(path)["streams"][0]
    assert stream["codec_name"] == "gif"
    assert max(stream["width"], stream["height"]) == 480
    assert stream["r_frame_rate"] == "12/1" or stream["avg_frame_rate"] == "12/1"
    result = body["result"]
    assert result["format"] == "gif" and result["fps"] == 12
    assert max(result["width"], result["height"]) == 480
    assert result["duration_s"] == 2.0
    response = client.get(result["url"])
    assert response.headers["content-type"] == "image/gif"


def test_encoder_failure_is_reported(tmp_path):
    sink = encoder.FrameEncoder(tmp_path / "no-existe" / "x.mp4", "mp4", 64, 64, 24)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(encoder.EncoderError):
        for _ in range(2000):
            sink.write(frame)
        sink.close()
    sink.abort()


def test_encoder_rejects_wrong_frame_shape(tmp_path):
    sink = encoder.FrameEncoder(tmp_path / "x.mp4", "mp4", 64, 64, 24)
    with pytest.raises(encoder.EncoderError):
        sink.write(np.zeros((10, 10, 3), dtype=np.uint8))
    sink.abort()


def test_video_job_cycle(settings):
    engine = FakeEngine()
    with running_client(settings, lambda: engine) as client:
        data = make_video(settings.tmp_dir.parent / "v.mp4", seconds=2.0, size="320x256", fps=24)
        # duration_s se ignora en video, incluso fuera de rango.
        job_id = post_file(client, "/animate", data, filename="v.mp4", duration_s="50").json()["job_id"]
        body = wait_job(client, job_id, 120)
        assert body["status"] == "succeeded", body
        call = engine.video_calls[0]
        assert call["frames"] == 48
        assert call["segments"] == 10  # VIDEO_INFER_EVERY_N = 5
        assert call["depth"] is None  # contrato v1.8: sin modelo de profundidad
        result = body["result"]
        assert result["fps"] == 24 and result["duration_s"] == 2.0
        assert body["timings_ms"]["segmentation"] > 0
        path = settings.results_dir / f"{job_id}.mp4"
        assert int(ffprobe(path)["streams"][0]["nb_frames"]) == 48


def test_video_is_resampled_to_max_frames(tmp_path, dummy_models):
    engine = FakeEngine()
    settings = make_settings(tmp_path, dummy_models, max_frames=30)
    with running_client(settings, lambda: engine) as client:
        data = make_video(tmp_path / "v.mp4", seconds=2.0, size="320x256", fps=30)
        body = wait_job(client, post_file(client, "/animate", data, filename="v.mp4").json()["job_id"], 120)
        assert body["status"] == "succeeded", body
        assert engine.video_calls[0]["frames"] <= 30
        assert body["result"]["fps"] == 15


def test_video_is_reduced_to_work_size(tmp_path, dummy_models):
    engine = FakeEngine()
    settings = make_settings(tmp_path, dummy_models, work_max_side=320)
    with running_client(settings, lambda: engine) as client:
        data = make_video(tmp_path / "v.mp4", seconds=1.0, size="640x480", fps=10)
        body = wait_job(client, post_file(client, "/animate", data, filename="v.mp4").json()["job_id"], 120)
        assert body["status"] == "succeeded", body
        assert (body["result"]["width"], body["result"]["height"]) == (320, 240)


def test_real_engine_real_ffmpeg_image(settings):
    with running_client(settings) as client:
        job_id = post_file(client, "/animate", image_bytes("PNG", 320, 256), duration_s="2", seed="5").json()["job_id"]
        body = wait_job(client, job_id, 240)
    assert body["status"] == "succeeded", body
    stream = ffprobe(settings.results_dir / f"{job_id}.mp4")["streams"][0]
    assert stream["codec_name"] == "h264" and int(stream["nb_frames"]) == 48
    assert body["seed"] == 5


def test_returned_random_seed_reproduces_the_animation(settings):
    """Prueba de que `seed` es el valor usado de verdad: reenviarlo da el mismo video byte a byte."""
    data = image_bytes("PNG", 320, 256)
    with running_client(settings) as client:
        first = wait_job(client, post_file(client, "/animate", data, duration_s="2").json()["job_id"], 240)
        assert first["status"] == "succeeded", first
        second = wait_job(
            client, post_file(client, "/animate", data, duration_s="2", seed=str(first["seed"])).json()["job_id"], 240
        )
        assert second["status"] == "succeeded", second
    assert second["seed"] == first["seed"]
    one = (settings.results_dir / f"{first['job_id']}.mp4").read_bytes()
    two = (settings.results_dir / f"{second['job_id']}.mp4").read_bytes()
    assert one == two


def test_real_engine_real_ffmpeg_video(settings):
    with running_client(settings) as client:
        data = make_video(settings.tmp_dir.parent / "v.mp4", seconds=1.0, size="320x256", fps=12)
        job_id = post_file(client, "/animate", data, filename="v.mp4").json()["job_id"]
        body = wait_job(client, job_id, 240)
    assert body["status"] == "succeeded", body
    assert int(ffprobe(settings.results_dir / f"{job_id}.mp4")["streams"][0]["nb_frames"]) == 12
