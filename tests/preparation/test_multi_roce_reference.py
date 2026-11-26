"""Independent CPU checks of fixed schedule references, not GPU qualification."""
import struct
import unittest
from multi_roce_reference import (
    GROUPS, HOSTS, SCHEDULE, VALID_BYTES, CHUNK_BYTES,
    gather_reference, history_reference, history_output_reference, logical_accounting, members, reduce_reference, source,
    protocol_accounting, communicator_accounting,
)


def floats(data):
    return [struct.unpack("<f", struct.pack("<I", word << 16))[0]
            for word, in struct.iter_unpack("<H", data)]


class MultiRoceReferences(unittest.TestCase):
    def test_fleet_mapping_and_mixed_graph_order(self):
        self.assertEqual(HOSTS[0], "192.168.177.111")
        self.assertEqual(HOSTS[7], "192.168.177.118")
        self.assertEqual(HOSTS[8], "192.168.177.11")
        self.assertEqual(HOSTS[15], "192.168.177.18")
        self.assertEqual([op.communicator for op in SCHEDULE],
                         ["tp", "dcp", "dcp", "dcp", "tp", "dcp", "tp"])
        for rank in range(16):
            self.assertEqual(members(SCHEDULE[1], rank), GROUPS[rank // 8])

    def test_reduce_matches_decoded_inputs_without_changing_sources(self):
        for index in (0, 2, 3, 6):
            op = SCHEDULE[index]
            for rank in (0, 8):
                originals = tuple(source(op, peer) for peer in members(op, rank))
                decoded = [floats(data) for data in originals]
                independent = [sum(values) for values in zip(*decoded, strict=True)]
                self.assertEqual(floats(reduce_reference(op, rank)), independent)
                self.assertEqual(originals, tuple(source(op, peer) for peer in members(op, rank)))

    def test_gather_preserves_rows_and_distinguishes_subgroups(self):
        for index in (1, 4):
            op = SCHEDULE[index]
            for rank in (0, 8):
                result = floats(gather_reference(op, rank))
                independent = []
                for row in range(op.rows):
                    for peer in members(op, rank):
                        values = floats(source(op, peer))
                        independent.extend(values[row*op.columns:(row+1)*op.columns])
                self.assertEqual(result, independent)
        self.assertNotEqual(gather_reference(SCHEDULE[1], 0), gather_reference(SCHEDULE[1], 8))
        self.assertEqual(gather_reference(SCHEDULE[4], 0), gather_reference(SCHEDULE[4], 8))

    def test_history_valid_prefixes_inactive_entries_and_chunk_tails(self):
        op = SCHEDULE[5]
        first, second = history_reference(op, 0), history_reference(op, 8)
        self.assertEqual(tuple(map(len, first)), VALID_BYTES)
        self.assertEqual(tuple(map(len, second)), VALID_BYTES)
        self.assertEqual(first[0], b"")
        self.assertNotEqual(first[1:], second[1:])
        for rank, count in enumerate(VALID_BYTES):
            account = logical_accounting(op, rank)
            self.assertEqual(account["world"], 8)
            self.assertEqual(account["local_payload_bytes"], count)
            self.assertEqual(account["local_payload_chunks"], (count + CHUNK_BYTES - 1)//CHUNK_BYTES)
            self.assertEqual(account["all_member_payload_bytes"], sum(VALID_BYTES))
        self.assertEqual(logical_accounting(op, 4)["local_payload_chunks"], 2)
        self.assertEqual(logical_accounting(SCHEDULE[0], 0)["world"], 16)

    def test_history_physical_rows_keep_inactive_and_tail_bytes_zero(self):
        op = SCHEDULE[5]
        capacity = op.columns * 2
        for rank in (0, 8):
            expected = history_output_reference(op, rank)
            self.assertEqual(len(expected), 8 * capacity)
            for local, peer in enumerate(members(op, rank)):
                count = VALID_BYTES[local]
                row = expected[local*capacity:(local+1)*capacity]
                self.assertEqual(row[:count], source(op, peer)[:count])
                self.assertEqual(row[count:], bytes(capacity-count))

    def test_native_completion_counts_are_not_total_posted_wrs(self):
        op = SCHEDULE[5]
        empty = protocol_accounting(op,0)
        self.assertEqual(empty["writes_completed"],14)
        self.assertEqual(empty["derived_total_wrs"],14)
        self.assertEqual(empty["payload_bytes_posted"],0)
        one_pack = protocol_accounting(op,1)
        self.assertEqual(one_pack["writes_completed"],14)
        self.assertEqual(one_pack["derived_total_wrs"],21)
        self.assertEqual(one_pack["payload_bytes_posted"],16*7)
        tail = protocol_accounting(op,4)
        self.assertEqual(tail["protocol_chunks"],2)
        self.assertEqual(tail["writes_completed"],28)
        self.assertEqual(tail["derived_total_wrs"],49)
        self.assertEqual(tail["payload_bytes_posted"],(CHUNK_BYTES+16)*7)
        for rank in range(16):
            account = communicator_accounting(rank)
            self.assertEqual(account["tp"]["ops_posted"],3)
            self.assertEqual(account["dcp"]["ops_posted"],4)
            self.assertEqual(account["tp"]["writes_completed"],90)
            self.assertEqual(account["tp"]["derived_total_wrs"],180)
            self.assertEqual(account["tp"]["dispatch_bytes"],18944)
            self.assertEqual(account["tp"]["payload_bytes_posted"],18944*15)
            self.assertEqual(account["dcp"]["dispatch_bytes"],18944+VALID_BYTES[rank%8])


if __name__ == "__main__":
    unittest.main()
