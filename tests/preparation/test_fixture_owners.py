"""Control-flow tests of the real fixture guard, without CUDA.

The injected native-drain boundary models the already-tested OwnerLease
contract: failed native drain retains its owner. GPU owners are recording
objects; no alternate cleanup implementation is tested.
"""
import gc
import unittest
import weakref

from fixture_owners import FixtureOwners


class Owner:
    def __init__(self, name, events, enter_error=None):
        self.name, self.events, self.enter_error = name, events, enter_error

    def __enter__(self):
        self.events.append(self.name+".enter")
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, *args):
        self.events.append(self.name+".exit")

    def reset(self):
        self.events.append(self.name+".reset")

    def close(self):
        self.events.append(self.name+".close")


class NativeDrain:
    """External GPU synchronization/lease boundary, not fixture cleanup logic."""
    def __init__(self, events, error=None):
        self.events, self.error = events, error
        self.leases = []

    def __call__(self, owner):
        self.events.append("synchronize")
        if self.error is not None:
            self.leases.append(owner)
            raise self.error


class FixtureOwnerTests(unittest.TestCase):
    def test_confirmed_drain_cleanup_is_ordered_and_exactly_once(self):
        events = []
        drain = NativeDrain(events)
        with FixtureOwners(drain) as guard:
            guard.enter(Owner("images",events))
            guard.enter(Owner("storage",events))
            for name, method in (("graph","reset"),("session","close"),("runtime","close")):
                resource = guard.hold(Owner(name,events))
                guard.after_drain(getattr(resource,method))
            guard.after_contexts(lambda:events.append("groups.close"))
        self.assertTrue(guard.drained)
        self.assertEqual(events,["images.enter","storage.enter","synchronize","graph.reset",
                                 "session.close","runtime.close","storage.exit","images.exit","groups.close"])
        before = list(events)
        guard.__exit__(None,None,None)
        self.assertEqual(events,before)
        self.assertEqual(drain.leases,[])
        self.assertEqual(guard.owners,[])

    def test_failed_drain_keeps_every_owner_and_never_calls_cleanup(self):
        events = []
        original = ValueError("original operation failed")
        identity = id(original)
        drain = NativeDrain(events,RuntimeError("device drain failed"))
        references = []
        try:
            with FixtureOwners(drain) as guard:
                for name in ("images","storage"):
                    references.append(weakref.ref(guard.enter(Owner(name,events))))
                for name, method in (("graph","reset"),("session","close"),("runtime","close")):
                    resource = guard.hold(Owner(name,events))
                    references.append(weakref.ref(resource))
                    guard.after_drain(getattr(resource,method))
                for name in ("tensors","stream"):
                    references.append(weakref.ref(guard.hold(Owner(name,events))))
                def must_not_run():
                    raise AssertionError("cleanup must not run after uncertain drain")
                guard.after_contexts(must_not_run)
                raise original
        except ValueError as caught:
            self.assertIs(caught,original)
        else:
            self.fail("original exception was suppressed")
        self.assertIs(drain.leases[0],guard)
        del resource, guard, original
        gc.collect()
        retained = drain.leases[0]
        self.assertEqual(id(retained.error),identity)
        self.assertFalse(retained.drained)
        self.assertTrue(all(reference() is not None for reference in references))
        self.assertEqual(events,["images.enter","storage.enter","synchronize"])
        self.assertIn("no graph/session/runtime/storage/context cleanup",retained.error.__notes__[0])

    def test_sync_failure_without_body_error_preserves_sync_exception(self):
        events = []
        error = RuntimeError("drain failed")
        drain = NativeDrain(events,error)
        try:
            with FixtureOwners(drain) as guard:
                guard.enter(Owner("images",events))
        except RuntimeError as caught:
            self.assertIs(caught,error)
        else:
            self.fail("drain failure was suppressed")
        self.assertIs(drain.leases[0],guard)
        self.assertEqual(events,["images.enter","synchronize"])

    def test_partial_context_entry_is_retained_without_outer_unwind(self):
        events = []
        original = RuntimeError("storage enter failed")
        drain = NativeDrain(events,RuntimeError("drain also failed"))
        partial = Owner("storage",events,original)
        try:
            with FixtureOwners(drain) as guard:
                guard.enter(Owner("images",events))
                guard.enter(partial)
        except RuntimeError as caught:
            self.assertIs(caught,original)
        else:
            self.fail("entry failure was suppressed")
        self.assertIn(partial,drain.leases[0].owners)
        self.assertEqual(events,["images.enter","storage.enter","synchronize"])

    def test_partial_constructor_survives_via_retained_original_traceback(self):
        events, references = [], []
        drain = NativeDrain(events,RuntimeError("drain failed"))
        class Partial:
            def __init__(self):
                references.append(weakref.ref(self))
                self.tensor = Owner("partial tensor",events)
                references.append(weakref.ref(self.tensor))
                raise ValueError("constructor failed")
        try:
            with FixtureOwners(drain) as guard:
                guard.enter(Owner("images",events))
                guard.enter(Partial())
        except ValueError:
            pass
        else:
            self.fail("constructor failure was suppressed")
        del guard
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in references))
        self.assertEqual(events,["images.enter","synchronize"])

    def test_cleanup_failure_cannot_replace_original_after_confirmed_drain(self):
        events = []
        original = ValueError("original body error")
        def cleanup_error():
            events.append("cleanup.error")
            raise RuntimeError("close failed")
        try:
            with FixtureOwners(NativeDrain(events)) as guard:
                guard.enter(Owner("images",events))
                guard.after_drain(cleanup_error)
                guard.after_contexts(lambda:events.append("groups.close"))
                raise original
        except ValueError as caught:
            self.assertIs(caught,original)
        else:
            self.fail("original exception was suppressed")
        self.assertTrue(guard.drained)
        self.assertEqual(events,["images.enter","synchronize","cleanup.error","images.exit","groups.close"])
        self.assertIn("cleanup failed after confirmed drain",original.__notes__[0])

    def test_partial_construction_with_confirmed_drain_releases_entered_contexts(self):
        events = []
        original = RuntimeError("storage enter failed")
        try:
            with FixtureOwners(NativeDrain(events)) as guard:
                guard.enter(Owner("images",events))
                guard.enter(Owner("storage",events,original))
        except RuntimeError as caught:
            self.assertIs(caught,original)
        self.assertEqual(events,["images.enter","storage.enter","synchronize","images.exit"])


if __name__ == "__main__":
    unittest.main()
