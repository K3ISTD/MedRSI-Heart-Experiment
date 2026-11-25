"""Apply the mandatory heart_disease_predictor call to medrsi/agent.py once."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path.cwd()
TARGET = ROOT / "medrsi" / "agent.py"
BACKUP = ROOT / "medrsi" / "agent.py.bundle-backup"
OLD = '''        calls, obj, reply = [], {}, ""\n        for step in range(self.max_steps):\n'''
NEW = '''        calls, obj, reply = [], {}, ""\n\n        # Force the Logistic Regression tool when it is registered.\n        required_tool = "heart_disease_predictor"\n        if required_tool in self.tools:\n            try:\n                output = to_jsonable(runner.call(required_tool))\n            except Exception as e:\n                output = {"error": f"{type(e).__name__}: {e}"}\n\n            calls.append({\n                "tool": self.tools[required_tool].id,\n                "output": output,\n            })\n\n            tool_result = json.dumps({required_tool: output}, indent=1)\n            messages.append({\n                "role": "user",\n                "content": (\n                    "MANDATORY TOOL RESULT: The required tool has already "\n                    "been executed on this patient's data. Use its output "\n                    "as evidence when making your diagnosis. Do not claim "\n                    "that the tool was not run.\\n"\n                    + tool_result\n                ),\n            })\n\n        for step in range(self.max_steps):\n'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="Restore agent.py from agent.py.bundle-backup")
    args = parser.parse_args()
    if not TARGET.is_file():
        raise FileNotFoundError(f"Cannot find {TARGET}. Run from the MedRSI project root.")

    source = TARGET.read_text(encoding="utf-8")
    if args.restore:
        if not BACKUP.is_file():
            raise FileNotFoundError(f"No bundle backup found: {BACKUP}")
        shutil.copy2(BACKUP, TARGET)
        print(f"Restored {TARGET} from {BACKUP}")
        return 0

    if 'required_tool = "heart_disease_predictor"' in source:
        print("Mandatory tool-call patch is already present; no changes made.")
        return 0
    if source.count(OLD) != 1:
        raise RuntimeError("Could not find exactly one expected insertion point; file left unchanged.")
    if BACKUP.exists():
        raise FileExistsError(f"Backup already exists at {BACKUP}; refusing to overwrite it.")

    shutil.copy2(TARGET, BACKUP)
    TARGET.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Patch applied. Backup saved to {BACKUP}")
    print("Next, validate with: python -m py_compile .\\medrsi\\agent.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
