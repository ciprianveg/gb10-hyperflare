"""CPU fixture binding checks; no real Gloo group or network traffic."""
import contextlib
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ignition.preparation import host_ranks
from multi_roce_fixture import initialize_gloo
from multi_roce_reference import HOSTS


class GlooBindingTests(unittest.TestCase):
    def test_each_rank_selects_its_local_address_before_group_creation(self):
        for rank in range(16):
            with self.subTest(rank=rank):
                events = []
                output = io.StringIO()
                def resolve(address):
                    events.append(("resolve", address))
                    self.assertEqual(address, HOSTS[rank])
                    return f"test-fabric-{rank}"
                def create(backend, **kwargs):
                    events.append(("group", backend))
                    self.assertEqual(os.environ["GLOO_SOCKET_IFNAME"], f"test-fabric-{rank}")
                    self.assertEqual(events[0], ("resolve", HOSTS[rank]))
                    logged = json.loads(output.getvalue())
                    self.assertEqual(logged, dict(gloo_rank=rank, local_address=HOSTS[rank],
                                                 local_interface=f"test-fabric-{rank}"))
                    self.assertEqual(kwargs["rank"], rank)
                    self.assertEqual(kwargs["world_size"], 16)
                    self.assertEqual(kwargs["init_method"], "tcp://192.168.177.111:29101")
                    self.assertEqual(kwargs["timeout"].total_seconds(), 300)
                with patch.dict(os.environ, {"GLOO_SOCKET_IFNAME": "stale-interface"}), \
                     patch.object(host_ranks, "interface_of", side_effect=resolve), \
                     contextlib.redirect_stdout(output):
                    initialize_gloo(SimpleNamespace(rank=rank, rendezvous="192.168.177.111:29101"),
                                    SimpleNamespace(init_process_group=create))
                self.assertEqual(events, [("resolve", HOSTS[rank]), ("group", "gloo")])

    def test_unmapped_address_does_not_create_group_or_fall_back(self):
        def no_interface(address):
            raise RuntimeError(f"no local interface holds {address}")
        def forbidden(*args, **kwargs):
            self.fail("Gloo must not start after interface resolution fails")
        with patch.dict(os.environ, {"GLOO_SOCKET_IFNAME": "stale-interface"}), \
             patch.object(host_ranks, "interface_of", side_effect=no_interface):
            with self.assertRaisesRegex(RuntimeError, "no local interface holds"):
                initialize_gloo(SimpleNamespace(rank=15, rendezvous="192.168.177.111:29101"),
                                SimpleNamespace(init_process_group=forbidden))

    def test_loopback_resolution_is_rejected_before_group_creation(self):
        def forbidden(*args, **kwargs):
            self.fail("Gloo must not start on loopback")
        with patch.dict(os.environ), patch.object(host_ranks, "interface_of", return_value="lo"):
            with self.assertRaisesRegex(RuntimeError, "must not be loopback"):
                initialize_gloo(SimpleNamespace(rank=0, rendezvous="192.168.177.111:29101"),
                                SimpleNamespace(init_process_group=forbidden))

    def test_invalid_rank_cannot_select_another_hosts_address(self):
        with patch.object(host_ranks, "interface_of") as resolve:
            for rank in (-1, 16, True):
                with self.assertRaises(ValueError):
                    initialize_gloo(SimpleNamespace(rank=rank), SimpleNamespace())
            resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
