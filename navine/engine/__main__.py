from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    from navine.engine.runtime import get_engine

    parser = argparse.ArgumentParser(prog="navine.engine", description="Navine Engine")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sub.add_parser("status")
    sub.add_parser("load")
    sub.add_parser("gguf")
    run_p = sub.add_parser("run")
    run_p.add_argument("name")
    run_p.add_argument("prompt", nargs="+")
    run_p.add_argument("--output", default=None)
    run_p.add_argument("--max-tokens", type=int, default=None)
    run_p.add_argument("--temperature", type=float, default=None)
    run_p.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    engine = get_engine()
    if args.cmd == "probe":
        print(json.dumps(engine.probe, indent=2))
        print(engine.write_manifest())
        return 0
    if args.cmd == "status":
        print(json.dumps(engine.status(), indent=2))
        return 0
    if args.cmd == "load":
        print(json.dumps(engine.load_all(), indent=2))
        print(engine.write_manifest())
        return 0
    if args.cmd == "gguf":
        from navine.gguf_export import export_all_gguf

        print(json.dumps(export_all_gguf(), indent=2))
        print(engine.write_manifest())
        return 0
    prompt = " ".join(args.prompt).strip()
    kwargs = {}
    if args.max_tokens is not None:
        kwargs["max_new_tokens"] = args.max_tokens
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.seed is not None:
        kwargs["seed"] = args.seed
    result = engine.run(args.name, prompt, output=args.output, **kwargs)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
