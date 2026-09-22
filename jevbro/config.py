"""Small config helpers shared by the reproducible training entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config without embedding secrets or environment state."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by install failures
        raise RuntimeError("PyYAML is required when --config is used") from exc

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{source}: config root must be a mapping")
    return values


def apply_config_defaults(parser, argv: list[str] | None) -> None:
    """Apply flat YAML values as argparse defaults; explicit CLI flags win."""
    probe = parser.parse_known_args(argv)[0]
    if not probe.config:
        return
    values = load_config(probe.config)
    unknown = sorted(set(values) - {action.dest for action in parser._actions})
    if unknown:
        raise ValueError(f"{probe.config}: unknown config keys: {', '.join(unknown)}")
    parser.set_defaults(**values)
