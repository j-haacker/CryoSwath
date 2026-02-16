#!/usr/bin/env python3
"""Probe anonymous FTP login behavior without failing CI shells.

This script always exits with status 0 and prints whether anonymous login
worked against the ESA CryoSat FTP endpoint.
"""

from __future__ import annotations

import datetime as _dt
import ftplib
import random
import socket
import string

HOST = "science-pds.cryosat.esa.int"
TIMEOUT_SECONDS = 30


def _random_email() -> str:
    rng = random.SystemRandom()
    local = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(16))
    suffix = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(10))
    return f"{local}@{suffix}.example.org"


def main() -> int:
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    email = _random_email()

    print(f"[{ts}] Anonymous FTP login probe")
    print(f"Host: {HOST}")
    print(f"Anonymous password used: {email}")

    worked = False
    response = ""
    error_name = ""
    error_message = ""

    try:
        with ftplib.FTP(HOST, timeout=TIMEOUT_SECONDS) as ftp:
            response = ftp.login(user="anonymous", passwd=email)
            worked = True
    except Exception as exc:  # noqa: BLE001
        error_name = type(exc).__name__
        error_message = str(exc)

    print("---")
    print(f"ANON_LOGIN_WORKED={worked}")
    if worked:
        print(f"SERVER_RESPONSE={response}")
    else:
        print(f"ERROR_TYPE={error_name}")
        print(f"ERROR_MESSAGE={error_message}")
        print("NOTE=Script exits 0 intentionally for reporting-only behavior.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
