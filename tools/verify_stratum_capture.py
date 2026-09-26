"""Verify accepted Kryptex shares with this repository's Pearl proof verifier.

The capture files contain wallet data. Keep them outside the repository.
"""

import argparse
import base64
import ctypes
import gzip
import json
import struct
from pathlib import Path


def messages(path):
    for line in Path(path).read_bytes().splitlines():
        try:
            yield json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-prefix", default="/tmp/stratum")
    parser.add_argument("--library", required=True)
    args = parser.parse_args()

    lib = ctypes.CDLL(args.library)
    verify = lib.cp_proof_verify
    verify.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
                       ctypes.c_size_t, ctypes.c_char_p, ctypes.c_uint32,
                       ctypes.c_void_p, ctypes.c_size_t]
    verify.restype = ctypes.c_int

    jobs = {}
    accepted = set()
    for message in messages(args.capture_prefix + ".server.capture"):
        if message.get("method") == "mining.notify":
            params = message["params"]
            jobs[params["job_id"]] = params
        if message.get("result") is True and isinstance(message.get("id"), int):
            accepted.add(message["id"])

    checked = 0
    for message in messages(args.capture_prefix + ".client.capture"):
        if message.get("method") != "mining.submit":
            continue
        params = message["params"]
        job = jobs.get(params["job_id"])
        if job is None:
            raise SystemExit(f"Missing job for submit {params['job_id']}")
        if message["id"] not in accepted:
            raise SystemExit(f"Share {message['id']} was not accepted by pool")
        proof = gzip.decompress(base64.b64decode(params["plain_proof"]))
        raw_b64 = base64.b64encode(proof)
        header = bytes.fromhex(job["header"])
        target = bytes.fromhex(job["target"])
        error = ctypes.create_string_buffer(1024)
        result = verify(header, len(header), raw_b64, len(raw_b64), target,
                        job["cert_version"], error, len(error))
        dims = struct.unpack_from("<4Q", proof)
        print(f"share {message['id']}: pool accepted, dimensions={dims}, "
              f"local verify={'OK' if result == 0 else error.value.decode(errors='replace')}")
        if result != 0:
            raise SystemExit(1)
        checked += 1
    if not checked:
        raise SystemExit("No captured shares")


if __name__ == "__main__":
    main()
