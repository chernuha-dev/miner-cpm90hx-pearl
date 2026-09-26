"""Capture a plaintext Stratum session for protocol development.

Capture files contain the miner's wallet and submitted proofs. Keep them private.
"""

import argparse
import asyncio
import os
from pathlib import Path


async def relay(reader, writer, capture):
    try:
        while data := await reader.read(65536):
            capture.write(data)
            capture.flush()
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


async def main(args):
    os.umask(0o077)
    prefix = Path(args.output)

    async def connection(client_reader, client_writer):
        upstream_reader, upstream_writer = await asyncio.open_connection(
            args.host, args.port
        )
        with prefix.with_suffix(".client.capture").open("ab") as sent, \
                prefix.with_suffix(".server.capture").open("ab") as received:
            await asyncio.gather(
                relay(client_reader, upstream_writer, sent),
                relay(upstream_reader, client_writer, received),
            )

    server = await asyncio.start_server(connection, args.listen_host, args.listen_port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=17048)
    parser.add_argument("--output", default="/tmp/stratum")
    asyncio.run(main(parser.parse_args()))
