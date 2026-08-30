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
artifacts. For FP32 indexed recurrent-KDA prefill, the stable
`flashinfer.jit.kda_fp32_indexed_promotion` adapter validates
`csrc/kda/kda_fp32_indexed_promotion_manifest.json`. The checked-in manifest is
pending, so FP32 indexed calls fail closed until a complete promoted payload is
installed; existing BF16 dispatch does not probe this adapter.

A complete runtime manifest fixes the public operation contract, one ordered
FFI argument table, the entry-point and module identifiers, and exact SM100a
and SM103a file closures. It also selects exactly one representation:

- `cuda` declares hashed generated sources and the translation units compiled
  with FlashInfer's fixed target-specific JIT recipe.
- `cubin` declares hashed target cubins and hashed C++ host bindings. The
  adapter embeds the verified cubin bytes in the binding; it does not compile
  or rewrite device code.

`is_available(compute_capability=...)` only reports true for a complete,
integrity-checked contract and matching architecture. `load` additionally
requires the caller to name `mode="cuda"` or `mode="cubin"`, and rejects a
mode different from the installed manifest. `run` resolves only generic output
and scale defaults before invoking the manifest's ordered argument table. A
missing artifact, single-byte mutation, unsupported architecture, contract
change, absent symbol, or non-enqueueing return fails closed.
