#!/usr/bin/env python3
import os, re, sys

xml = sys.argv[1] if len(sys.argv) > 1 else "scene_swapped.xml"
base = os.path.dirname(os.path.abspath(xml))

missing = []
for i, line in enumerate(open(xml, "r", encoding="utf-8"), 1):
    for m in re.finditer(r'file="([^"]+)"', line):
        path = m.group(1)
        # ignore mujoco builtins or empty
        if path.startswith("builtin:") or path.strip()=="":
            continue
        # relative to xml directory
        abs_path = os.path.normpath(os.path.join(base, path))
        if not os.path.exists(abs_path):
            missing.append((i, path, abs_path))

if missing:
    print("Missing files:")
    for i, p, ap in missing:
        print(f"  L{i}: {p} -> {ap}")
    sys.exit(1)
else:
    print("All file paths exist!")