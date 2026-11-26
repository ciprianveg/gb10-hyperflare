"""CPU-only tests of the actual preparation descriptor functions; no Torch import."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[2]/"ignition/preparation/roce_transport.py"
spec=importlib.util.spec_from_file_location("roce_transport_metadata",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CommunicatorMetadataTests(unittest.TestCase):
    def test_actual_description_keeps_native_geometry_and_order(self):
        native=dict(world_size=8,rank=4,proxy_abi=14,region=dict(address=0x10000,nbytes=8192),
                    counters=dict(address=0x14000,nbytes=4096),slot_bytes=16,pipe_bytes=16,
                    ctrl_offset=4096,varlen_bytes=0)
        value=module.description(native,"dcp-b",list(range(8,16)),world=16,rank=12)
        self.assertEqual(value["members"],list(range(8,16)))
        self.assertEqual({k:v for k,v in value.items() if k not in ("members","communicator_id")},native)
        self.assertNotIn("communicator_id",native)
        other=module.description(native,"independent",list(range(8,16)),world=16,rank=12)
        self.assertNotEqual(value["communicator_id"],other["communicator_id"])

    def test_invalid_membership_or_global_identity_fails(self):
        for name,members,world,rank,local_world,local_rank in [
            ("",list(range(8)),16,0,8,0),
            ("x",[0,0],16,0,2,0),
            ("x",[0,16],16,0,2,0),
            ("x",[1,0],16,0,2,0),
            ("x",list(range(8)),16,0,16,0),
            ("x",list(range(8)),16,0,8,8),
        ]:
            with self.assertRaises(ValueError):
                module.membership(name,members,world=world,rank=rank,local_world=local_world,local_rank=local_rank)

    def test_native_description_cannot_override_membership(self):
        with self.assertRaises(ValueError):
            module.description(dict(world_size=8,rank=0,members=list(range(8))),
                               "x",list(range(8)),world=16,rank=0)


if __name__=="__main__":
    unittest.main()
