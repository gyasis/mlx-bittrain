"""envcheck — print the platform and the mlx / mlx-lm versions as JSON, exit 1 on a mismatch."""

import json
import platform
import sys
from importlib.metadata import version

PINNED = {"mlx": "0.32.4", "mlx-lm": "0.32.2"}


def main() -> int:
    import mlx.core as mx

    found = {pkg: version(pkg) for pkg in PINNED}
    report = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "metal": mx.metal.is_available(),
        "default_device": str(mx.default_device()),
        "versions": found,
        "pinned_ok": found == PINNED,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pinned_ok"] and report["metal"] else 1


if __name__ == "__main__":
    sys.exit(main())
