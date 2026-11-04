"""Descarga con Range, galería y moderación."""
from __future__ import annotations

import json
import time
import uuid

import pytest
from api_fakes import API, assert_error, make_settings, running_client

from app.routers.results import parse_range
from app.services import job_manager as jm

CONTENT = bytes(range(256)) * 8  # 2048 bytes


def add_job(client, status="succeeded", output_format="mp4", opt_in=False, content=CONTENT, created_at=None):
    services = client.app.state.services
    job_id = str(uuid.uuid4())
    now = created_at or jm.to_iso(jm.utc_now())
    services.store.insert(
        job_id=job_id,
        status=status,
        stage="done" if status == "succeeded" else "rendering",
        progress=1.0,
        media="image",
        mood="magico",
        output_format=output_format,
        duration_s=4.0,
        seed=1,
        brush_mode="add",
        gallery_opt_in=int(opt_in),
        created_at=now,
        updated_at=now,
        result_json=json.dumps({"url": jm.result_url(job_id, output_format)}) if status == "succeeded" else None,
    )
    if status == "succeeded":
        services.settings.results_dir.mkdir(parents=True, exist_ok=True)
        (services.settings.results_dir / f"{job_id}.{output_format}").write_bytes(content)
    return job_id


def test_full_download_advertises_ranges(client):
    job_id = add_job(client)
    response = client.get(f"{API}/results/{job_id}.mp4")
    assert response.status_code == 200
    assert response.content == CONTENT
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-length"] == str(len(CONTENT))


def test_gif_media_type(client):
    job_id = add_job(client, output_format="gif")
    response = client.get(f"{API}/results/{job_id}.gif")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/gif"


@pytest.mark.parametrize(
    "header, start, end",
    [
        ("bytes=0-1", 0, 1),  # la sonda que usa Safari
        ("bytes=100-199", 100, 199),
        ("bytes=2000-", 2000, 2047),
        ("bytes=-48", 2000, 2047),
        ("bytes=2040-99999", 2040, 2047),
        ("bytes=-99999", 0, 2047),
    ],
)
def test_range_requests(client, header, start, end):
    job_id = add_job(client)
    response = client.get(f"{API}/results/{job_id}.mp4", headers={"Range": header})
    assert response.status_code == 206
    assert response.content == CONTENT[start : end + 1]
    assert response.headers["content-range"] == f"bytes {start}-{end}/{len(CONTENT)}"
    assert response.headers["content-length"] == str(end - start + 1)
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-type"] == "video/mp4"


@pytest.mark.parametrize("header", ["bytes=2048-", "bytes=99999-100000", "bytes=-0"])
def test_unsatisfiable_range(client, header):
    job_id = add_job(client)
    response = client.get(f"{API}/results/{job_id}.mp4", headers={"Range": header})
    assert_error(response, 416, "RANGE_NOT_SATISFIABLE")
    assert response.headers["content-range"] == f"bytes */{len(CONTENT)}"


@pytest.mark.parametrize("header", ["items=0-1", "bytes=abc", "bytes=0-1,5-6", "bytes=5-2"])
def test_invalid_or_multiple_range_sends_full_file(client, header):
    job_id = add_job(client)
    response = client.get(f"{API}/results/{job_id}.mp4", headers={"Range": header})
    assert response.status_code == 200
    assert response.content == CONTENT


def test_if_range_with_old_etag_sends_full_file(client):
    job_id = add_job(client)
    url = f"{API}/results/{job_id}.mp4"
    etag = client.get(url).headers["etag"]
    assert client.get(url, headers={"Range": "bytes=0-1", "If-Range": etag}).status_code == 206
    assert client.get(url, headers={"Range": "bytes=0-1", "If-Range": '"viejo"'}).status_code == 200


def test_head_reports_size_without_body(client):
    job_id = add_job(client)
    response = client.head(f"{API}/results/{job_id}.mp4")
    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["content-length"] == str(len(CONTENT))
    assert response.headers["accept-ranges"] == "bytes"


def test_result_not_found_cases(client):
    job_id = add_job(client)
    assert_error(client.get(f"{API}/results/{uuid.uuid4()}.mp4"), 404, "RESULT_NOT_FOUND")
    assert_error(client.get(f"{API}/results/{job_id}.gif"), 404, "RESULT_NOT_FOUND")
    assert_error(client.get(f"{API}/results/{job_id}.exe"), 404, "RESULT_NOT_FOUND")
    assert_error(client.get(f"{API}/results/no-es-uuid.mp4"), 404, "RESULT_NOT_FOUND")
    assert_error(client.get(f"{API}/results/{job_id}"), 404, "RESULT_NOT_FOUND")
    assert_error(client.get(f"{API}/results/..%2F..%2Fkinetia.db.mp4"), 404, "NOT_FOUND")


def test_result_file_removed_from_disk(client, settings):
    job_id = add_job(client)
    (settings.results_dir / f"{job_id}.mp4").unlink()
    assert_error(client.get(f"{API}/results/{job_id}.mp4"), 404, "RESULT_NOT_FOUND")


def test_job_not_ready_for_running_job(client):
    job_id = add_job(client, status="running")
    assert_error(client.get(f"{API}/results/{job_id}.mp4"), 409, "JOB_NOT_READY")


def test_parse_range_empty():
    assert parse_range(None, 10) is None
    assert parse_range("", 10) is None
    assert parse_range("bytes=-", 10) is None


def test_gallery_format_and_order(client):
    old = add_job(client, opt_in=True, created_at="2026-11-25T10:00:00.000Z")
    new = add_job(client, opt_in=True, created_at="2026-11-25T12:00:00.000Z")
    add_job(client, opt_in=False, created_at="2026-11-25T13:00:00.000Z")
    add_job(client, opt_in=True, status="running", created_at="2026-11-25T14:00:00.000Z")
    body = client.get(f"{API}/gallery").json()
    assert set(body) == {"items"}
    assert [item["job_id"] for item in body["items"]] == [new, old]
    item = body["items"][0]
    assert set(item) == {"job_id", "mood", "result_url", "created_at"}
    assert item["mood"] == "magico"
    assert item["result_url"] == f"{API}/results/{new}.mp4"
    assert item["created_at"] == "2026-11-25T12:00:00.000Z"


def test_gallery_limit_and_after(client):
    ids = [add_job(client, opt_in=True, created_at=f"2026-11-25T1{i}:00:00.000Z") for i in range(4)]
    assert [i["job_id"] for i in client.get(f"{API}/gallery", params={"limit": 2}).json()["items"]] == [ids[3], ids[2]]
    after = client.get(f"{API}/gallery", params={"after": "2026-11-25T11:00:00Z"}).json()["items"]
    assert [i["job_id"] for i in after] == [ids[3], ids[2]]
    # Con desplazamiento horario y con el "+" convertido en espacio por la URL.
    offset = client.get(f"{API}/gallery?after=2026-11-25T07:00:00-05:00").json()["items"]
    assert [i["job_id"] for i in offset] == [ids[3]]
    spaced = client.get(f"{API}/gallery?after=2026-11-25T13:00:00+01:00").json()["items"]
    assert [i["job_id"] for i in spaced] == [ids[3]]


def test_moderation_requires_token(client):
    job_id = add_job(client, opt_in=True)
    url = f"{API}/gallery/{job_id}"
    assert_error(client.delete(url), 401, "UNAUTHORIZED")
    assert_error(client.delete(url, headers={"X-Admin-Token": "incorrecto"}), 401, "UNAUTHORIZED")
    assert len(client.get(f"{API}/gallery").json()["items"]) == 1


def test_moderation_removes_item_and_file(client, settings):
    job_id = add_job(client, opt_in=True)
    headers = {"X-Admin-Token": settings.admin_token}
    response = client.delete(f"{API}/gallery/{job_id}", headers=headers)
    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"{API}/gallery").json()["items"] == []
    assert not (settings.results_dir / f"{job_id}.mp4").exists()
    assert_error(client.get(f"{API}/results/{job_id}.mp4"), 404, "RESULT_NOT_FOUND")
    # Repetirlo no es un error.
    assert client.delete(f"{API}/gallery/{job_id}", headers=headers).status_code == 204


def test_moderation_unknown_or_not_in_gallery(client, settings):
    headers = {"X-Admin-Token": settings.admin_token}
    assert_error(client.delete(f"{API}/gallery/{uuid.uuid4()}", headers=headers), 404, "JOB_NOT_FOUND")
    private = add_job(client, opt_in=False)
    assert_error(client.delete(f"{API}/gallery/{private}", headers=headers), 404, "JOB_NOT_FOUND")


def test_moderation_rejects_everything_without_configured_token(tmp_path, dummy_models, engine):
    settings = make_settings(tmp_path, dummy_models, admin_token="")
    with running_client(settings, lambda: engine) as client:
        job_id = add_job(client, opt_in=True)
        assert_error(client.delete(f"{API}/gallery/{job_id}", headers={"X-Admin-Token": ""}), 401, "UNAUTHORIZED")


def test_gallery_end_to_end_with_consent(client, fake_encoder):
    from api_fakes import image_bytes, post_file, wait_job

    public = post_file(client, "/animate", image_bytes(), duration_s="2", gallery_opt_in="true").json()["job_id"]
    private = post_file(client, "/animate", image_bytes(), duration_s="2").json()["job_id"]
    wait_job(client, public)
    wait_job(client, private)
    time.sleep(0.01)
    assert [item["job_id"] for item in client.get(f"{API}/gallery").json()["items"]] == [public]
