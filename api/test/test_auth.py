"""Tests for the /auth routes (api.routes.auth)."""

from common.config import SEED_PASSWORD, SEED_USER


def _login(client) -> dict:
    return client.post("/auth/token",
                       data={"username": SEED_USER, "password": SEED_PASSWORD}).json()


def test_login_returns_token_pair(client):
    resp = client.post("/auth/token",
                       data={"username": SEED_USER, "password": SEED_PASSWORD})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["access_token"] and body["refresh_token"]
    assert body["token_type"] == "bearer"


def test_login_bad_password_401(client):
    resp = client.post("/auth/token",
                       data={"username": SEED_USER, "password": "wrong"})
    assert resp.status_code == 401


def test_me_returns_current_user(client, auth_headers):
    resp = client.get("/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["username"] == SEED_USER


def test_me_requires_auth(client):
    assert client.get("/auth/me").status_code == 401


def test_refresh_rotates_session(client):
    pair = _login(client)
    resp = client.post("/auth/refresh", json={"refresh_token": pair["refresh_token"]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["access_token"] != pair["access_token"]

    # The rotated refresh token is revoked and can't be reused.
    assert client.post("/auth/refresh",
                       json={"refresh_token": pair["refresh_token"]}).status_code == 401


def test_logout_revokes_session(client):
    pair = _login(client)
    headers = {"Authorization": f"Bearer {pair['access_token']}"}
    assert client.get("/auth/me", headers=headers).status_code == 200
    assert client.post("/auth/logout", headers=headers).status_code == 204
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_logout_all_revokes_every_session(client):
    first = _login(client)
    second = _login(client)
    client.post("/auth/logout-all",
                headers={"Authorization": f"Bearer {first['access_token']}"})
    for pair in (first, second):
        headers = {"Authorization": f"Bearer {pair['access_token']}"}
        assert client.get("/auth/me", headers=headers).status_code == 401
