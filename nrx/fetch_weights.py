"""Download the Sionna tutorial's pre-trained NRX weights and verify the checksum.

    uv run python -m nrx.fetch_weights
"""

import hashlib
import sys

from nrx.config import WEIGHTS_GDRIVE_ID, WEIGHTS_PATH, WEIGHTS_SHA256


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not (WEIGHTS_PATH.exists() and sha256(WEIGHTS_PATH) == WEIGHTS_SHA256):
        import gdown

        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=WEIGHTS_GDRIVE_ID, output=str(WEIGHTS_PATH), quiet=False)
    digest = sha256(WEIGHTS_PATH)
    if digest != WEIGHTS_SHA256:
        sys.exit(f"checksum mismatch for {WEIGHTS_PATH}: {digest} != {WEIGHTS_SHA256}")
    print(f"ok {WEIGHTS_PATH} sha256={digest[:12]}…")


if __name__ == "__main__":
    main()
