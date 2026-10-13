import uuid


def new_id() -> str:
    """Return a new collision-resistant identifier.

    Returns:
        A hex string unique across creations.
    """
    return uuid.uuid4().hex
