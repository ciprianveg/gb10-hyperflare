"""Test-local cleanup sequencing around Heliopause's existing native owner lease.

The injected drain callable must implement export_synchronize(context, owners):
on failed drain its native OwnerLease permanently retains the supplied owner.
No Python global-list lifetime or implicit GPU context-manager unwind is used.
"""
class FixtureOwners:
    def __init__(self, drain):
        self.drain = drain
        self.owners = []
        self.contexts = []
        self.cleanup = []
        self.final_cleanup = []
        self.error = None
        self.drained = False
        self.finished = False

    def hold(self, owner):
        self.owners.append(owner)
        return owner

    def enter(self, context):
        # Retain even an owner whose __enter__ partially constructs then fails.
        self.hold(context)
        result = context.__enter__()
        self.contexts.append(context)
        return self.hold(result)

    def after_drain(self, callback):
        self.cleanup.append(callback)

    def after_contexts(self, callback):
        self.final_cleanup.append(callback)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.finished:
            return False
        self.finished = True
        # On uncertainty the leaked native lease retains this exception and its
        # traceback too, including partially constructed owners in body frames.
        self.error = value
        try:
            self.drain(self)
        except BaseException as error:
            self.error = value if value is not None else error
            self.error.add_note(
                f"fixture drain unconfirmed; native OwnerLease retains all owners; "
                f"no graph/session/runtime/storage/context cleanup attempted: {error!r}")
            if value is None:
                raise
            return False
        self.drained = True
        errors = []
        # Only a confirmed drain permits any release callback. Cleanup failures
        # cannot replace an exception from the fixture body.
        actions = [*self.cleanup,
                   *(lambda context=context: context.__exit__(kind, value, traceback)
                     for context in reversed(self.contexts)),
                   *self.final_cleanup]
        for callback in actions:
            try:
                callback()
            except BaseException as error:
                errors.append(error)
        if errors:
            primary = value if value is not None else errors[0]
            for error in errors:
                primary.add_note(f"fixture cleanup failed after confirmed drain: {error!r}")
            if value is None:
                raise primary
        self.owners.clear()
        self.contexts.clear()
        self.cleanup.clear()
        self.final_cleanup.clear()
        return False
