#!/usr/bin/env python3
"""
Debug harness for check_updates_cmd (VS Code-friendly).
Run via launch.json: Passes --test-path /path --use-cache True
Coerces use_cache to bool; adds prints for step-debug.
"""

import argparse
import click
import traceback
from pathlib import Path
from click.testing import CliRunner
from depman.cli import cli  # CLI group for ctx sim
from depman.commands.checker import check_updates_cmd  # Your callback

def create_mock_ctx(root: str = ".") -> click.Context:
    """Mock ctx with obj['root'] set."""
    ctx = click.Context(cli, info_name="depman")
    ctx.ensure_object(dict)
    ctx.obj["root"] = Path(root).resolve()
    return ctx

def test_check_updates_command():
    parser = argparse.ArgumentParser(description="Debug Checker Functions")
    parser.add_argument("--test-path", type=str, default=".", help="Git root path")
    parser.add_argument("--use-cache", type=str, default=None, help="Cache mode (any value → True)")
    
    args = parser.parse_args()
    
    # Fix: Coerce to bool (handles "True"/"true"/any str → True; None → False)
    use_cache_bool = bool(args.use_cache) if args.use_cache else False
    
    root = args.test_path
    print(f"🔍 Debug Start: Root={root}, use_cache={use_cache_bool} (type: {type(use_cache_bool)})")
    
    # ctx = create_mock_ctx(root)
    
    # cli(root=root)  # Ensure CLI context setup if needed
    # cli({},root) # Explicitly pass an initial object for ctx.obj
    try:
        # Direct call (ctx first, then kwargs matching sig)
        check_updates_cmd.make_context("check_updates", [root])
        check_updates_cmd.invoke(ctx)  # Add ,recursive=False if needed
        print("✅ Debug: Command ran (check terminal for CLI output).")
    except Exception as e:
        print(f"❌ Debug error: {e}")
        traceback.print_exc()
        # Optional: VS Code breakpoint here ^ for inspect

if __name__ == "__main__":
    # test_check_updates_command()
    root = r'C:\Users\ivanm\Documents\MATLAB\EKL\deps_tests\gitman_proj_test'
    cli(['--root', root, 'check-updates', '--use-cache'])
    # runner = CliRunner()
    # result = runner.invoke(cli, ['--root', root, 'check_updates', '--use-cache'])