# Deterministic generated-program promotion

`tools/import-generated-program-promotion` installs generated CUDA or cubin
artifacts without modifying their bytes. The producer supplies a JSON manifest;
the importer verifies every source before writing, copies through same-filesystem
temporary files, atomically replaces each destination, then rehashes both sides.

The schema is deliberately workload-neutral:

```json
{
  "artifacts": [
    {
      "destination": "csrc/example/kernel.cu",
      "executable": false,
      "sha256": "<64 lowercase hexadecimal characters>",
      "size_bytes": 1234,
      "source": "cuda/kernel.cu"
    }
  ],
  "kind": "flashinfer.generated_program_promotion",
  "mode": "cuda",
  "name": "example-program",
  "schema_version": 1
}
```

`mode` is either `cuda` or `cubin`. Paths are normalized POSIX-relative paths,
artifacts are sorted by destination, and no source or destination may traverse
a symlink. Duplicate JSON keys, unknown fields, repeated paths, parent/child
destination collisions, undeclared payload files, byte-count mismatches, and
digest mismatches are fatal. The payload directory is a dedicated exact
inventory; unrelated files already present in the destination checkout are not
part of that producer inventory.

Import into a checkout:

```bash
tools/import-generated-program-promotion \
  --manifest /path/to/manifest.json \
  --payload-root /path/to/payload \
  --output-root "$PWD" \
  --mode cuda
```

Existing identical files are accepted. Updating an existing artifact requires
the explicit `--replace` flag. CI and reviewers can perform a read-only
source-to-destination identity check with `--check`:

```bash
tools/import-generated-program-promotion \
  --manifest /path/to/manifest.json \
  --payload-root /path/to/payload \
  --output-root "$PWD" \
  --mode cuda \
  --check
```

Performance and correctness receipts remain separate inputs to promotion.
They decide whether an artifact is acceptable; this importer only proves that
the accepted artifact is the one copied into the public tree.

Runtime adapters are workload-specific and are themselves eligible manifest
artifacts. For FP32 indexed recurrent-KDA prefill, the stable dispatcher looks
for `flashinfer.jit.kda_fp32_indexed_promotion`. That module must expose
`is_available(compute_capability=...)` and `run(**kwargs)`. Until such a
verified adapter and its CUDA or cubin payload are installed, FP32 indexed
state-pool calls fail closed and existing BF16 dispatch is unchanged.
