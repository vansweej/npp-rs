# Plan: cargo-tarpaulin in the Nix shell + full non-GPU coverage workflow

## Goal

Make `cargo tarpaulin` reproducible inside the Nix dev shell and stand up a
real coverage workflow that **hard-gates at ≥ 90 % on the non-GPU surface**,
using the **default ptrace engine**, with all GPU/CUDA/FFI code excluded from
the denominator.

## Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Scope | Full workflow: tool + config + exclusions + 2 tests + docs + CI |
| 2 | Engine | Default **ptrace** — no toolchain change, preserves shellHook `RUSTFLAGS` |
| 3 | Coverage scope | **Non-GPU surface only** (mirrors `cargo test --no-default-features`) |
| 4 | Threshold | **`fail-under = 90`**, hard gate in both `tarpaulin.toml` and CI |
| 5 | CI lane | **Included** — new `coverage` job appended to existing `build.yml` |

---

## Phase 1 — Make the tool available in the dev shell

**1.1** `flake.nix`: add `cargo-tarpaulin` to `buildInputs` (after `pkg-config`).

**1.2** Smoke test:

```bash
nix develop . --command cargo tarpaulin --workspace --engine ptrace --skip-clean --out Stdout
```

Must compile the FFI crate (no link error) and print a coverage table.

---

## Phase 2 — Close the two pure-logic test gaps

**2.1** `npp/src/error.rs`: add unit test for `check_status`.

Test cases:
- `Ok(())` on status `0`
- `Ok(())` on a positive warning code (e.g. `1`)
- `Err(NppError::Npp(-22))` on a negative code

**2.2** `npp/src/resize_ops.rs`: add unit test for `interpolation_mode`.

Test cases — all 5 `ResizeInterpolation` variants map to the expected `npp_sys`
constant:
- `NearestNeighbor` → `NppiInterpolationMode_NPPI_INTER_NN`
- `Linear` → `NppiInterpolationMode_NPPI_INTER_LINEAR`
- `Cubic` → `NppiInterpolationMode_NPPI_INTER_CUBIC`
- `Super` → `NppiInterpolationMode_NPPI_INTER_SUPER`
- `Lanczos` → `NppiInterpolationMode_NPPI_INTER_LANCZOS`

---

## Phase 3 — Exclude the GPU/FFI surface

**3.1** New `tarpaulin.toml` at repo root:

```toml
[tarpaulin]
engine = "ptrace"
workspace = true
fail-under = 90
exclude-files = [
    "npp-sys/*",
    "*_generated.rs",
    "*_macros.rs",
    "npp/src/test_helpers.rs",
    "*/build.rs",
    "npp-codegen/examples/*",
    "npp/examples/*",
]
```

No `features` key → non-GPU surface.

**3.2** Add `#[cfg(not(tarpaulin_include))]` to hand-written GPU-only items in
mixed source files:

| File | Items to annotate |
|------|-------------------|
| `npp/src/cuda.rs` | `initialize_cuda_device`, `default_cuda_device` |
| `npp/src/image.rs` | `new`, `from_host`, `sub_image`, `sub_image_mut`, `CudaImageView::device_ptr`, `CudaImageViewMut::device_ptr_mut`, `TryFrom<&CudaImage<T>> for Vec<T>` |

Generated files (e.g. `*_generated.rs`) and macro files (`*_macros.rs`) are
already excluded by `tarpaulin.toml` globs.

**3.3** Re-run tarpaulin; confirm the pure-logic surface reports **≥ 90 %**.
Iterate annotations only if a stray GPU line is still counted.

---

## Phase 4 — Documentation

**4.1** `docs/getting-started.md`: add a **Coverage** subsection.

```markdown
### Coverage

```bash
nix develop . --command cargo tarpaulin
```

This runs `cargo tarpaulin` on the **non-GPU surface** only (pure-logic layout,
error handling, codegen tests). GPU/CUDA/FFI code is excluded via
`tarpaulin.toml` globs and `#[cfg(not(tarpaulin_include))]` annotations.
The gate is set at 90 % — anything below that fails the run.
```

**4.2** `README.md`: add the coverage command to Quick start (after lint,
before documentation).

```bash
# Coverage (non-GPU surface)
cargo tarpaulin
```

**4.3** `AGENTS.md`: add to **Build / test** section:

```bash
nix develop . --command cargo tarpaulin    # coverage, non-GPU only, must be ≥ 90 %
```

And add to **Conventions**:

```markdown
- Coverage: GPU/CUDA/FFI functions are excluded from `cargo tarpaulin` with
  `#[cfg(not(tarpaulin_include))]`. When adding new GPU-only code in a file
  that also contains pure-logic code, add this annotation to the function.
```

---

## Phase 5 — CI coverage lane

**5.1** Append a `coverage` job to `.github/workflows/build.yml`:

```yaml
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cachix/install-nix-action@v27
      - run: nix develop . --impure --command cargo tarpaulin --engine ptrace --out Xml --fail-under 90
```

- Same `nix develop . --impure --command` form as every other job.
- Scope/excludes come from `tarpaulin.toml`.
- `--fail-under 90` makes the gate explicit at the job level too.
- `--out Xml` emits `cobertura.xml` (no upload step — reserved for future Codecov integration).

---

## Verification checklist

- [ ] `cargo tarpaulin` → ≥ 90 % non-GPU, exits 0 against `fail-under = 90`
- [ ] `cargo test --no-default-features` → green (incl. the 2 new tests)
- [ ] `cargo test --features gpu` → green (annotations are inert in normal builds)
- [ ] `cargo clippy -- -D warnings` + `cargo fmt --check` → clean
- [ ] `nix develop .` re-enters cleanly with the new `buildInputs` entry

---

## Files touched (summary)

| File | Change |
|------|--------|
| `flake.nix` | +1 `buildInputs` line (`cargo-tarpaulin`) |
| `tarpaulin.toml` | **new** — ptrace engine, workspace, excludes, `fail-under = 90` |
| `npp/src/error.rs` | +1 unit test (`check_status`) |
| `npp/src/resize_ops.rs` | +1 unit test (`interpolation_mode`) |
| `npp/src/cuda.rs` | `#[cfg(not(tarpaulin_include))]` ×2 |
| `npp/src/image.rs` | `#[cfg(not(tarpaulin_include))]` on device-touching methods |
| `docs/getting-started.md` | Coverage section |
| `README.md` | Coverage command in Quick start |
| `AGENTS.md` | tarpaulin command + exclusion convention |
| `.github/workflows/build.yml` | +`coverage` job |

---

## Suggested commit sequence

1. `chore(nix): add cargo-tarpaulin to dev shell` (Phase 1)
2. `test(npp-rs): cover check_status and interpolation_mode` (Phase 2)
3. `chore(coverage): add tarpaulin.toml and exclude GPU/FFI surface` (Phase 3)
4. `ci: add tarpaulin coverage gate at 90%` (Phase 5)
5. `docs: document coverage workflow` (Phase 4)
