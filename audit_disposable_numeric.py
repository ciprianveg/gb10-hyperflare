#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Report-only integrity audit: numeric tensor bytes vs declared-disposable storage.

Concept from third-party flare fork PR (g0san/flare PR #11, voipmonitor lineage):
an export can declare a buffer disposable (the cold-started core restores it
zero-filled) while the live preparing process still READS those bytes as
numeric data. Long-context retrieval then silently collapses in the exported
core while the preparing process behaves correctly.

This script intersects, per rank, from the export's own recorded metadata:

  1. NUMERIC tensor bytes  -- typed tensors the process reads as numeric
     (``BundleIndex.execution.buffers`` and ``BundleIndex.rebuild_weights``
     tensors, decoded with shape/stride element layout exactly like
     ``ignition/preparation/tensor_metadata.py:46 tensor_ranges``).
  2. DECLARED-DISPOSABLE ranges -- bytes of each recorded ``Allocation`` that
     the export did NOT store (``Allocation.data_spans`` complement). Bytes no
     span covers stay absent from the payload: the cold-started core restores
     them zero-filled (helios/src/core/export.rs:153).

REPORT-ONLY: prints findings, exits 0 either way. Never writes, locks, or
modifies anything it inspects.

Recorded-metadata APIs used (read-only, same as the export pipeline):
  - staged rank source (in-flight export): flare_heliopause.export_read_sources
    / ignition/preparation/core_file.py:93 read_sources_record; the on-disk
    rank-*.json is itself JSON, so stdlib json works without flare_heliopause.
  - finished core: flare_heliopause.export_read_index
    / ignition/preparation/core_file.py:51 read_index, decoded with
    ignition/preparation/runtime_core_pb2.ModelIndex.

Usage:
  python3 audit_disposable_numeric.py --cache /home/ciprian/models/models2b/flare-cores
  python3 audit_disposable_numeric.py --staging DIR/rank-N.json [--staging ...]
  python3 audit_disposable_numeric.py --core PATH/TO/CORE      (finished core)

Notes:
  - Against an in-flight export, staged rank-*.json files appear as each rank
    finishes native capture; incomplete state is tolerated and reported.
  - A finished-core audit needs the export to have completed; run it
    post-export against the cache (or a specific core path).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# This module must stay import-side-effect free: only the stdlib imports above
# run at import time. flare_heliopause and protobuf are loaded lazily inside
# functions so importing this file never touches the export's runtime identity.

DEFAULT_CACHE = "/home/ciprian/models/models2b/flare-cores"

WARNING_HEADER = (
    "WARNING: a native process restores these zero-filled - numeric reads "
    "from these ranges are silently corrupted in the cold-started core"
)

# Element sizes for the dtype strings recorded in wire::Buffer records.
DTYPE_BYTES = {
    "bool": 1, "i8": 1, "u8": 1, "int8": 1, "uint8": 1, "byte": 1,
    "i16": 2, "u16": 2, "int16": 2, "uint16": 2, "fp16": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
    "i32": 4, "u32": 4, "int32": 4, "uint32": 4, "fp32": 4, "float32": 4, "f32": 4,
    "i64": 8, "u64": 8, "int64": 8, "uint64": 8, "fp64": 8, "float64": 8, "f64": 8,
}

MEMORY_KINDS = {0: "device", 1: "mapped_host", 2: "peer"}

# Mirrors tensor_metadata.py tensor_ranges: outer-layout expansion is capped to
# avoid unbounded interval catalogs; beyond the cap the record is reported over
# its whole declared numeric extent instead (conservative, report-only).
MAX_OUTER_ELEMENTS = 65536


def merge_ranges(ranges):
    """Sorted disjoint half-open byte intervals (tensor_metadata.py:16)."""
    result = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def subtract_ranges(ranges, exclusions):
    """Half-open ranges minus sorted merged exclusions (tensor_metadata.py:29)."""
    result, index = [], 0
    for start, end in merge_ranges(ranges):
        while index < len(exclusions) and exclusions[index][1] <= start:
            index += 1
        cursor, other = start, index
        while other < len(exclusions) and exclusions[other][0] < end:
            left, right = exclusions[other]
            if cursor < left:
                result.append((cursor, left))
            cursor = max(cursor, right)
            other += 1
        if cursor < end:
            result.append((cursor, end))
    return result


def intersect_ranges(a, b):
    """Intersection of two merged interval lists."""
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start < end:
            result.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def element_spans(shape, strides, element_size, fallback_nbytes):
    """Covered byte spans of a strided element layout, allocation-relative.

    Replicates ignition/preparation/tensor_metadata.py:46 tensor_ranges:
    contiguous leading dimensions fold into one span; strided outer
    dimensions expand as repeated intervals. Returns None when the layout is
    too fragmented to classify (the caller then uses the whole extent).
    """
    if not shape:
        return [(0, element_size)]
    width = element_size
    outer = []
    for size, stride in sorted(((size, stride) for size, stride in zip(shape, strides)
                                if size > 1 and stride), key=lambda item: item[1]):
        if stride < 0:
            return None
        if not outer and stride <= width:
            width += (size - 1) * stride
        else:
            outer.append((stride, size))
    if math.prod(size for _, size in outer) > MAX_OUTER_ELEMENTS:
        return None
    spans = [(0, width)]
    for stride, size in outer:
        spans = [(left + index * stride, right + index * stride)
                 for index in range(size) for left, right in spans]
    return merge_ranges(spans)


def field(record, name, default=None):
    """Read a field from either a JSON dict or a protobuf message."""
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def numeric_records(index):
    """Typed numeric tensors recorded for one rank, normalized.

    Yields (name, kind, dtype, element_size, nbytes, allocation, offset,
    shape, strides). These are the tensors the serving process reads as
    numeric: execution buffers (bound into graphs via input bindings and
    state accesses) and rebuild-weight tensors.
    """
    execution = field(index, "execution")
    for buffer in field(execution, "buffers", []) or []:
        reference = field(buffer, "reference")
        dtype = field(buffer, "dtype", "")
        yield {
            "name": field(buffer, "name", "?"),
            "kind": "execution-buffer",
            "dtype": dtype,
            "element_size": DTYPE_BYTES.get(dtype),
            "nbytes": int(field(buffer, "nbytes", 0)),
            "allocation": field(reference, "allocation"),
            "offset": int(field(reference, "offset", 0)),
            "shape": list(field(buffer, "shape", []) or []),
            "strides": list(field(buffer, "strides", []) or []),
        }
    rebuild = field(index, "rebuild_weights")
    for tensor in field(rebuild, "tensors", []) or []:
        storage = field(tensor, "storage")
        element_bytes = field(tensor, "element_bytes")
        yield {
            "name": field(tensor, "name", "?"),
            "kind": "rebuild-weight",
            "dtype": field(tensor, "dtype", ""),
            "element_size": int(element_bytes) if element_bytes else None,
            "nbytes": int(field(tensor, "element_bytes", 0) or 0)
            * (1 + sum((int(size) - 1) * int(stride)
                       for size, stride in zip(field(tensor, "shape", []) or [],
                                               field(tensor, "stride", []) or []))),
            "allocation": field(storage, "allocation"),
            "offset": int(field(storage, "offset", 0)),
            "shape": list(field(tensor, "shape", []) or []),
            "strides": list(field(tensor, "stride", []) or []),
        }


def disposable_by_allocation(index):
    """Per allocation: (name, memory, nbytes, unstored ranges) allocation-relative.

    Bytes not covered by data_spans stay absent from the payload and are
    restored zero-filled on load (helios/src/core/export.rs:153).
    """
    result = {}
    for allocation in field(index, "allocations", []) or []:
        nbytes = int(field(allocation, "nbytes", 0))
        stored = merge_ranges(
            (int(field(span, "offset", 0)), int(field(span, "offset", 0)) + int(field(span, "nbytes", 0)))
            for span in field(allocation, "data_spans", []) or [])
        memory = field(allocation, "memory", 0)
        memory = int(memory) if isinstance(memory, (int, float)) else memory
        result[field(allocation, "name", "?")] = {
            "name": field(allocation, "name", "?"),
            "memory": MEMORY_KINDS.get(memory, memory),
            "nbytes": nbytes,
            "unstored": subtract_ranges([(0, nbytes)], stored),
        }
    return result


def rank_findings(bundle):
    """Intersect recorded numeric tensor bytes with declared-disposable ranges."""
    disposable = disposable_by_allocation(bundle)
    findings = []
    for record in numeric_records(bundle):
        spans = None
        if record["element_size"]:
            spans = element_spans(record["shape"], record["strides"],
                                  record["element_size"], record["nbytes"])
        if spans is None:
            # Unclassified layout: report the whole declared numeric extent.
            spans = [(0, record["nbytes"])]
        absolute = merge_ranges((record["offset"] + left, record["offset"] + right)
                                for left, right in spans)
        allocation = disposable.get(record["allocation"])
        if allocation is None:
            if record["allocation"] is not None:
                findings.append({"record": record, "allocation": None,
                                 "overlap": [], "note": "references an unrecorded allocation"})
            continue
        overlap = intersect_ranges(absolute, allocation["unstored"])
        if overlap:
            findings.append({"record": record, "allocation": allocation,
                             "overlap": overlap, "note": ""})
    return findings


def rank_label(source_path, bundle, position):
    roce = field(bundle, "roce")
    rank = field(roce, "rank")
    if rank is not None:
        return f"rank-{int(rank)}"
    name = Path(source_path).name
    if name.startswith("rank-") and name.endswith(".json"):
        return f"{name[:-5]}#{position}"
    return f"rank@position-{position} ({name})"


def audit_index(source_path, index):
    """Audit every recorded rank (BundleIndex) of one ModelIndex source."""
    results = []
    for position, bundle in enumerate(field(index, "ranks", []) or [None]):
        findings = rank_findings(bundle)
        label = rank_label(source_path, bundle, position)
        allocations = disposable_by_allocation(bundle)
        unstored_bytes = sum(end - start
                             for allocation in allocations.values()
                             for start, end in allocation["unstored"])
        total_allocations = len(allocations)
        stored_bytes = sum(int(field(a, "nbytes", 0)) for a in
                           field(bundle, "allocations", []) or []) - unstored_bytes
        overlap_bytes = sum(right - left for finding in findings
                            for left, right in finding["overlap"])
        results.append({
            "label": label,
            "source": str(source_path),
            "allocations": total_allocations,
            "stored_bytes": stored_bytes,
            "unstored_bytes": unstored_bytes,
            "overlap_bytes": overlap_bytes,
            "findings": findings,
        })
    return results


def print_report(result, out=sys.stdout):
    print(f"\n=== {result['label']}  ({result['source']}) ===", file=out)
    print(f"  allocations: {result['allocations']}  "
          f"stored: {result['stored_bytes']:,} B  "
          f"zero-filled-on-load: {result['unstored_bytes']:,} B", file=out)
    if not result["findings"]:
        print("  no numeric tensor bytes fall inside declared-disposable ranges", file=out)
        return
    print(f"\n!!! {WARNING_HEADER}", file=out)
    for finding in result["findings"]:
        record, allocation = finding["record"], finding["allocation"]
        print(f"  - {record['kind']} {record['name']!r} dtype={record['dtype']!r} "
              f"allocation={record['allocation']!r}", file=out)
        if allocation is None:
            print(f"      {finding['note']}", file=out)
            continue
        print(f"      memory={allocation['memory']} "
              f"allocation extent={allocation['nbytes']:,} B", file=out)
        shown = finding["overlap"][:8]
        for left, right in shown:
            print(f"      overlap: allocation+{left:,} .. allocation+{right:,} "
                  f"({right - left:,} B)", file=out)
        if len(finding["overlap"]) > len(shown):
            print(f"      ... and {len(finding['overlap']) - len(shown)} more ranges", file=out)
        print(f"      total overlap: {sum(right - left for left, right in finding['overlap']):,} B", file=out)


def audit_staged_rank(path):
    """Read one staged rank source the way the pipeline does (core_file.py:93).

    Prefers flare_heliopause.export_read_sources (native decode + payload
    location); falls back to plain JSON when the native module is unavailable.
    """
    import flare_heliopause  # lazy: zero import-time side effects

    record = json.loads(flare_heliopause.export_read_sources(str(path)))
    proto = load_runtime_proto()
    index = proto.ModelIndex.FromString(bytes(record["index"]))
    return index


def load_runtime_proto():
    """Load runtime_core_pb2 standalone (no ignition package import)."""
    import importlib.util

    here = Path(__file__).resolve().parent
    module_path = here / "ignition" / "preparation" / "runtime_core_pb2.py"
    if not module_path.exists():
        raise ImportError(f"runtime_core_pb2 not found at {module_path}")
    spec = importlib.util.spec_from_file_location("audit_runtime_core_pb2", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_staged_rank_json(path):
    """Read one staged rank source with stdlib only (no native module)."""
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    return record["index"]


def audit_finished_core(path):
    """Read one finished core's ModelIndex (core_file.py:51 read_index)."""
    import flare_heliopause  # lazy: zero import-time side effects

    encoded, _base = flare_heliopause.export_read_index(str(path))
    proto = load_runtime_proto()
    return proto.ModelIndex.FromString(encoded)


def discover(cache):
    """Find finished cores and staged rank sources under a cache directory.

    Read-only discovery; tolerates a cache with an export still in flight.
    """
    finished, staged = [], []
    cache = Path(cache)
    models = cache / "models"
    if models.is_dir():
        for entry in sorted(models.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                finished.append(entry)
    sources = cache / "sources"
    if sources.is_dir():
        for entry in sorted(sources.iterdir()):
            if entry.is_dir() and (entry / "rank-0.json").exists():
                finished.append(entry)
    for directory in sorted(cache.glob(".flare-export-*")):
        if not directory.is_dir():
            continue
        for rank_file in sorted(directory.glob("rank-*.json")):
            staged.append(rank_file)
    return finished, staged


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Report-only audit: numeric tensor bytes vs declared-disposable "
                    "(zero-filled-on-load) core ranges. Exits 0 either way.")
    parser.add_argument("--cache", default=None,
                        help=f"core cache directory (default: {DEFAULT_CACHE}); "
                             "audits finished cores and in-flight staged rank sources")
    parser.add_argument("--core", action="append", default=[],
                        help="finished core path to audit (repeatable)")
    parser.add_argument("--staging", action="append", default=[],
                        help="staged rank-*.json source to audit (repeatable)")
    parser.add_argument("--json", action="store_true",
                        help="print a machine-readable summary at the end")
    args = parser.parse_args(argv)

    cache = args.cache or DEFAULT_CACHE
    if not args.core and not args.staging and not Path(cache).exists():
        print(f"note: cache {cache} does not exist yet; nothing to audit", file=sys.stderr)
        return 0

    finished, staged = [], []
    if not args.core and not args.staging:
        finished, staged = discover(cache)
        print(f"cache: {cache}")
        print(f"finished cores found: {len(finished)}  staged rank sources found: {len(staged)}")
        if not finished and not staged:
            print("note: export appears not to have produced any recorded metadata yet "
                  "(in-flight or empty cache); run again post-export")

    targets = []  # (kind, path)
    for path in args.core:
        targets.append(("core", path))
    for path in args.staging:
        targets.append(("staging", path))
    for path in finished:
        targets.append(("core", path))
    for path in staged:
        targets.append(("staging", path))

    native_available = True
    try:
        import flare_heliopause  # noqa: F401
    except ImportError:
        native_available = False
        if any(kind == "core" for kind, _ in targets):
            print("note: flare_heliopause unavailable; finished cores are skipped "
                  "(staged rank-*.json still audited via stdlib JSON)", file=sys.stderr)

    results, problems = [], []
    for kind, path in targets:
        try:
            if kind == "core":
                if not native_available:
                    continue
                index = audit_finished_core(path)
            else:
                index = (audit_staged_rank(path) if native_available
                         else audit_staged_rank_json(path))
            for result in audit_index(path, index):
                results.append(result)
        except Exception as error:  # tolerate incomplete / concurrently written state
            problems.append((str(path), f"{type(error).__name__}: {error}"))

    total_overlap = 0
    for result in results:
        print_report(result)
        total_overlap += result["overlap_bytes"]

    print(f"\n=== summary ===")
    print(f"ranks audited: {len(results)}  read errors (tolerated): {len(problems)}")
    for path, error in problems:
        print(f"  unreadable (likely in-flight): {path}: {error}", file=sys.stderr)
    if total_overlap:
        print(f"\n!!! {WARNING_HEADER}")
        print(f"!!! total overlapping numeric bytes across ranks: {total_overlap:,}")
        print("!!! report-only: the export pipeline was not modified")
    else:
        print("no numeric-vs-disposable overlap found in the recorded metadata")

    if args.json:
        summary = {
            "report_only": True,
            "cache": cache,
            "ranks": [{
                "label": result["label"], "source": result["source"],
                "allocations": result["allocations"],
                "stored_bytes": result["stored_bytes"],
                "zero_filled_bytes": result["unstored_bytes"],
                "overlap_bytes": result["overlap_bytes"],
                "findings": [{
                    "kind": f["record"]["kind"], "name": f["record"]["name"],
                    "dtype": f["record"]["dtype"], "allocation": f["record"]["allocation"],
                    "overlap": [[left, right] for left, right in f["overlap"]],
                } for f in result["findings"]],
            } for result in results],
            "total_overlap_bytes": total_overlap,
            "unreadable": [{"path": path, "error": error} for path, error in problems],
        }
        print(json.dumps(summary, indent=1))
    return 0  # report-only: exit 0 whether or not overlap was found


if __name__ == "__main__":
    sys.exit(main())
