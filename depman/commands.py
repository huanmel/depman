"""
Gm subcommand: gitman wrapper via API.
"""

import inspect
import click
from gitman import install, update, list as list_deps, lock, uninstall, init


@click.group(name="gm", help="Gitman command wrapper (short: gm)")
@click.pass_context
def gm(ctx: click.Context):
    pass


@gm.command("init")
def gm_init():
    """Init a new .gitman.yml."""
    init()


@gm.command("install")
@click.argument("names", nargs=-1)
@click.option("--depth", type=int)
@click.option("--force", is_flag=True)
# ... (add other opts as needed; map to gitman.install args)
@click.pass_context
def gm_install(ctx: click.Context, names, depth, force):
    root = ctx.obj["root"]
    install(*names, root=root, depth=depth, force=force)


@gm.command("update")
@click.argument("names", nargs=-1)
@click.option("--recurse", is_flag=True)
@click.option("--lock", is_flag=True)
# ... (similar for other opts)
@click.pass_context
def gm_update(ctx: click.Context, names, recurse, lock):
    root = ctx.obj["root"]
    update(*names, root=root, recurse=recurse, lock=lock if recurse else None)


@gm.command("list")
@click.option("--depth", type=int)
@click.pass_context
def gm_list(ctx: click.Context, depth):
    root = ctx.obj["root"]
    list_deps(root=root, depth=depth)


# Add more: lock, uninstall, etc.
# For full arg mapping, inspect gitman funcs dynamically if needed.