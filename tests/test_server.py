from fastapi.testclient import TestClient

from neuroplex.server import create_app
from neuroplex.simulation import Simulation

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
            frame = ws.receive_json()
            assert frame["kind"] == "frame" and "brain" not in frame
            assert frame["world"]["x"] == state["world"]["x"]
        client.post("/api/control", headers=HEADERS, json={"action": "save"})
    assert (tmp_path / "checkpoint.npz").is_file()
    with TestClient(create_app(tmp_path)) as client:
        restored = client.get("/api/state").json()
        assert restored["world"]["time"] == frozen["world"]["time"]
        assert restored["brain"]["learning"] is False


def test_automatic_life_controls_validation_and_immediate_persistence(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        def control(action, value=None):
            return client.post("/api/control", headers=HEADERS, json={"action": action, "value": value})
        assert control("auto_life", False).json()["auto_life"] is False
        assert control("auto_life_delay", 23).json()["auto_life_delay"] == 23
        saved = Simulation.load(tmp_path / "checkpoint.npz")
        assert not saved.auto_life and saved.auto_life_delay == 23
        for invalid in (0, 301, -1, True, 1.5, "10", None):
            assert control("auto_life_delay", invalid).status_code == 422
        for invalid in (0, 1, "true", None):
            assert control("auto_life", invalid).status_code == 422
        assert control("auto_life", True).json()["auto_life"] is True
        assert client.get("/static/motion.js").status_code == 200


def test_dead_paused_world_can_resume_countdown_and_restart_without_browser(tmp_path):
    sim = Simulation()
    sim.world.regrow_at.fill(1e9)
    sim.world.energy = .001
    sim.tick()
    sim.paused = True
    sim.auto_life_delay = 1
    sim.save(tmp_path / "checkpoint.npz")
    with TestClient(create_app(tmp_path)) as client:
        state = client.get("/api/state").json()
        assert not state["world"]["alive"] and state["auto_life_status"]["remaining"] == 1
        state = client.post("/api/control", headers=HEADERS, json={"action": "resume"}).json()
        assert not state["auto_life_status"]["paused"]
        # No WebSocket/browser connection is responsible for the restart.
        import time
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = client.get("/api/state").json()
            if state["life"] == 2:
                break
            time.sleep(.05)
        assert state["life"] == 2 and state["world"]["alive"]
