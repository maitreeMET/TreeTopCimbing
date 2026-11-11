"""TreeTopClimbing: Isaac Lab tasks for humanoids on tree-top rope courses."""


def register_tasks():
    """Register all TreeTop gym tasks. Pass to Isaac Lab as `--external_callback treetop.register_tasks`."""
    from . import tasks  # noqa: F401
    from .tasks import plank_bridge  # noqa: F401

    return None
