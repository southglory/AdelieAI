"""Small operational CLI for validating, importing, and running personas."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.personas.packs import PackValidationError, PersonaImportService, load_persona_pack


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adelie", description="Run portable personas locally")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate an unpacked .adelie directory")
    validate.add_argument("path", type=Path)

    run = sub.add_parser("run", help="optionally import a character and start the web console")
    run.add_argument("source", type=Path, nargs="?")
    run.add_argument("--packs-dir", type=Path, default=Path("packs"))
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8770)
    run.add_argument("--no-open", action="store_true", help="do not open the browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            loaded = load_persona_pack(args.path)
            print(f"ok: {loaded.persona.persona_id} ({loaded.source_format})")
            return 0

        os.environ["ADELIE_PACKS_DIR"] = str(args.packs_dir)
        imported_id = None
        if args.source is not None:
            payload = args.source.read_bytes()
            imported = PersonaImportService(args.packs_dir).install(args.source.name, payload)
            imported_id = imported.persona.persona_id

        import uvicorn

        if not args.no_open:
            import threading
            import webbrowser

            path = f"/web/chat/{imported_id}" if imported_id else "/web/personas/import"
            threading.Timer(0.8, webbrowser.open, args=(f"http://{args.host}:{args.port}{path}",)).start()
        uvicorn.run("core.api.app:app", host=args.host, port=args.port, reload=False)
        return 0
    except (OSError, PackValidationError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
