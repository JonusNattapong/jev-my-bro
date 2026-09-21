from pathlib import Path


def test_go_gateway_contains_namespaced_jev_routes() -> None:
    source = Path("server/go/main.go").read_text(encoding="utf-8")
    for route in ("/v1/jev-my-bro/health", "/v1/jev-my-bro/decide", "/v1/jev-my-bro/predict"):
        assert route in source
