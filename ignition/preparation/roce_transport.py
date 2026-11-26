# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU metadata for independently owned RoCE communicators."""


def membership(communicator_id, members, *, world, rank, local_world, local_rank):
    if (not isinstance(communicator_id, str) or not communicator_id.strip()
            or not isinstance(members, (list, tuple))
            or any(type(r) is not int or not 0 <= r < world for r in members)
            or not 2 <= len(members) <= 16 or len(set(members)) != len(members)
            or local_world != len(members) or not 0 <= local_rank < len(members)
            or members[local_rank] != rank):
        raise ValueError("communicator ID/membership must match global and group-local rank identity")
    return tuple(members)


def description(native, communicator_id, members, *, world, rank):
    members = membership(communicator_id, members, world=world, rank=rank,
                         local_world=native["world_size"], local_rank=native["rank"])
    if "communicator_id" in native or "members" in native:
        raise ValueError("native transport already carries communicator metadata")
    return {**native, "communicator_id": communicator_id, "members": list(members)}


class RegisteredTransport:
    """Retain the existing runtime owner and attach only its export identity."""

    def __init__(self, runtime, communicator_id, members, *, world, rank):
        self.runtime, self.communicator_id, self.members = runtime, communicator_id, tuple(members)
        self.world, self.rank = world, rank

    def native_description(self):
        return description(self.runtime.native_description(), self.communicator_id,
                           self.members, world=self.world, rank=self.rank)

    def close(self):
        return self.runtime.close()
