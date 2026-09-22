from fastapi.testclient import TestClient

from neuroplex.server import create_app

HEADERS = {"X-Neuroplex-Client": "dashboard"}


def test_dashboard_control_validation_websocket_and_shutdown_save(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/health").json()["status"] == "ok"
        assert client.post("/api/control", json={"action": "pause"}).status_code == 403
        assert client.post("/api/control", headers={**HEADERS, "Origin": "http://elsewhere"},
                           json={"action": "pause"}).status_code == 403
        assert client.post("/api/control", headers=HEADERS, json={"action": "speed", "value": 100}).status_code == 422
        assert client.post("/api/control", headers=HEADERS, json={"action": "new_life"}).status_code == 409
        assert client.post("/api/control", headers=HEADERS, json={"action": "pause"}).json()["paused"]
        frozen = client.post("/api/control", headers=HEADERS, json={"action": "learning", "value": False}).json()
        assert frozen["brain"]["learning"] is False
        with client.websocket_connect("/ws") as ws:
            state = ws.receive_json()
            assert state["paused"] and state["brain"]["neurons"] == 500
        client.post("/api/control", headers=HEADERS, json={"action": "save"})
    assert (tmp_path / "checkpoint.npz").is_file()
    with TestClient(create_app(tmp_path)) as client:
        restored = client.get("/api/state").json()
        assert restored["world"]["time"] == frozen["world"]["time"]
        assert restored["brain"]["learning"] is False
