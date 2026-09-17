"""Minimal dependency-free test harness (no pytest required)."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))


def close(name: str, got: float, want: float, rel: float = 1e-6, abs_: float = 0.0) -> None:
    ok = abs(got - want) <= max(rel * abs(want), abs_)
    check(name, ok, f"got {got:.6g}, want {want:.6g} (rel tol {rel:g})")


def raises(name: str, fn, exc=Exception) -> None:
    try:
        fn()
    except exc:
        check(name, True)
        return
    except Exception as e:  # wrong exception type
        check(name, False, f"raised {type(e).__name__}: {e}")
        return
    check(name, False, "did not raise")


def run(title: str, tests: list) -> int:
    for t in tests:
        try:
            t()
        except Exception:
            check(t.__name__, False, "EXCEPTION\n" + traceback.format_exc())
    failed = [r for r in _results if not r[1]]
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    print(f"{title}: {len(_results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0
