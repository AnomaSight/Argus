import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.main import app
from app.schemas import InspectResponse

client = TestClient(app, raise_server_exceptions=False)
MODEL = "mvtec_ad/hazelnut/m100/seed20260801"


def _png(size=(300, 260), seed=0, fmt="PNG"):
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1], size[0], 3)) * 60 + 100).astype(np.uint8)
    arr[100:140, 120:170] = 255  # a bright "defect"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def _post(data, name="part.png", ctype="image/png", model=MODEL):
    return client.post("/inspect", files={"image": (name, data, ctype)}, data={"model_id": model})


def _assert_error(r, status, code):
    assert r.status_code == status, r.text
    assert set(r.json()) == {"error"} and r.json()["error"]["code"] == code


def test_models_lists_banks_sorted():
    ids = [m["model_id"] for m in client.get("/models").json()]
    assert MODEL in ids and ids == sorted(ids) and len(ids) == 8


def test_valid_upload_returns_every_c5_field():
    r = _post(_png())
    assert r.status_code == 200, r.text
    body = InspectResponse(**r.json())  # extra="forbid": exactly the C5 fields
    assert body.model_id == MODEL
    assert body.verdict == ("defect" if body.score > body.tau else "good")
    view = Image.open(io.BytesIO(base64.b64decode(body.input_png_base64)))
    heat = Image.open(io.BytesIO(base64.b64decode(body.heatmap_png_base64)))
    assert view.size == heat.size == (224, 224) and heat.mode == "RGBA"
    assert body.heatmap_range == (1.5, 3.0) and body.latency_ms >= 0


def test_same_image_same_score():
    a, b = _post(_png(seed=3)).json(), _post(_png(seed=3)).json()
    assert a["score"] == b["score"]


@pytest.mark.parametrize(
    "fmt, ctype", [("JPEG", "image/jpeg"), ("BMP", "image/bmp"), ("TIFF", "image/tiff")]
)
def test_other_allowed_formats(fmt, ctype):
    assert _post(_png(fmt=fmt), name=f"x.{fmt.lower()}", ctype=ctype).status_code == 200


def test_unknown_model_is_404():
    _assert_error(_post(_png(), model="../../etc/passwd"), 404, "model_not_found")


def test_not_an_image_type_is_415():
    _assert_error(
        _post(b"%PDF-1.4", name="doc.pdf", ctype="application/pdf"), 415, "unsupported_media_type"
    )


def test_gif_is_415_even_when_it_decodes():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="GIF")
    _assert_error(
        _post(buf.getvalue(), name="x.png", ctype="image/png"), 415, "unsupported_media_type"
    )


def test_corrupt_png_is_422():
    _assert_error(_post(b"\x89PNG\r\n\x1a\n garbage"), 422, "invalid_image")


def test_over_10_mb_is_413():
    _assert_error(_post(b"\0" * (10 * 1024 * 1024 + 1)), 413, "file_too_large")


def test_missing_field_is_400():
    r = client.post("/inspect", data={"model_id": MODEL})
    _assert_error(r, 400, "bad_request")


def test_results_reads_csv_tables(tmp_path, monkeypatch):
    table = tmp_path / "results/mvtec_ad/hazelnut/memory_sweep_alpha0.05.csv"
    table.parent.mkdir(parents=True)
    table.write_text("setting,fpr,n,note\nm100,0.048,131,\nm10,nan,131,flag\n")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    body = client.get("/results").json()
    assert body == [
        {
            "name": "mvtec_ad/hazelnut/memory_sweep_alpha0.05",
            "columns": ["setting", "fpr", "n", "note"],
            "rows": [["m100", 0.048, 131, None], ["m10", None, 131, "flag"]],
        }
    ]


def test_results_empty_without_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    assert client.get("/results").json() == []
