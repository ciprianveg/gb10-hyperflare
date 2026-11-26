"""Test-local mixed TP16/DCP8 capture; imported only by released preparation.

Both preparation sessions must finish their own group-specific jobs first.
This module does not initialize CUDA or import preparation dependencies at load.
"""


def capture_mixed(forward, *, tp_runtime, dcp_runtime, tp_session, dcp_session,
                  stream, storage, owners):
    """Capture one graph while retaining both independent transport/session owners."""
    import torch
    from ignition.preparation.runtime_core import record_export_graph

    if tp_runtime is dcp_runtime or tp_session is dcp_session:
        raise ValueError("TP and DCP require distinct runtime and session owners")
    for session in (tp_session, dcp_session):
        if session._tune_status != "ready":
            raise RuntimeError("both group-specific preparation jobs must finish before capture")
        if getattr(session, "_startup_collectives", None) is not None:
            raise RuntimeError("pure RoCE fixture unexpectedly registered PCIe startup tuning")
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    stream.wait_stream(torch.cuda.current_stream(stream.device))
    retained = (tp_runtime, dcp_runtime, tp_session, dcp_session, forward, *owners)
    # These are two instance-local runtime guards and two session guards around
    # ONE CUDA graph. No coordinator receives another group's declarations.
    with tp_runtime.capture(stream=stream), dcp_runtime.capture(stream=stream), \
            tp_session.capture(), dcp_session.capture(), \
            torch.cuda.graph(graph, stream=stream, pool=storage.graph_pool.id):
        forward()
    return record_export_graph(graph, storage, owners=retained)
