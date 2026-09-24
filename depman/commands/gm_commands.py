"""
Gm subcommand: gitman wrapper via API.
"""

from typing import List, Optional

import typer
from gitman import install, update, list as list_deps, lock, uninstall, init

gm_app = typer.Typer(name="gm", help="Gitman command wrapper (short: gm)")


@gm_app.command("init")
def gm_init():
    """Init a new .gitman.yml."""
    init()


@gm_app.command("install")
def gm_install(
    ctx: typer.Context,
    names: Optional[List[str]] = typer.Argument(None),
    depth: Optional[int] = typer.Option(None, "--depth"),
    force: bool = typer.Option(False, "--force"),
):
    root = ctx.obj["root"]
    install(*(names or []), root=root, depth=depth, force=force)


@gm_app.command("update")
def gm_update(
    ctx: typer.Context,
    names: Optional[List[str]] = typer.Argument(None),
    recurse: bool = typer.Option(False, "--recurse"),
    lock: bool = typer.Option(False, "--lock"),
):
    root = ctx.obj["root"]
    update(*(names or []), root=root, recurse=recurse, lock=lock if recurse else None)


@gm_app.command("list")
def gm_list(
    ctx: typer.Context,
    depth: Optional[int] = typer.Option(None, "--depth"),
):
    root = ctx.obj["root"]
    list_deps(root=root, depth=depth)


@gm_app.command("lock")
def gm_lock(
    ctx: typer.Context,
    names: Optional[List[str]] = typer.Argument(None),
):
    root = ctx.obj["root"]
    lock(*(names or []), root=root)


@gm_app.command("uninstall")
def gm_uninstall(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force"),
    keep_location: bool = typer.Option(False, "--keep-location"),
):
    root = ctx.obj["root"]
    uninstall(root=root, force=force, keep_location=keep_location)
