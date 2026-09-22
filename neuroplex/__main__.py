import argparse
from pathlib import Path

import uvicorn

from .server import create_app


def main():
    parser = argparse.ArgumentParser(description="Neuroplex: a persistent, CPU-only artificial life experiment")
    parser.add_argument("--host", default="127.0.0.1", help="default: loopback; use an SSH tunnel")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=7, help="used only for a NEW checkpoint")
    parser.add_argument("--untrained", action="store_true", help="new worlds only: start without the bundled learned motor values")
    parser.add_argument("--stage", type=int, choices=range(5), default=0,
                        help="new worlds only: 0 food, 1 scarcity, 2 water, 3 predator, 4 wild habitat")
    args = parser.parse_args()
    uvicorn.run(create_app(args.data_dir, args.seed, args.untrained, args.stage), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
