# `seismic-cli` — Complete Module and Flag Reference

**Every command, every flag, every module in `seismic_cli/`, verified against the
code rather than against prior documentation.**

Where a `--help` string and the implementation disagreed, the implementation is
authoritative here and the help string was corrected in `seismic_cli/cli.py`.
Where a runtime print is still wrong, it is listed in
[§9 Known defects](#9-known-defects-and-traps) rather than papered over.

---

## Contents

1. [Scope and invocation](#1-scope-and-invocation)
2. [Command index](#2-command-index)
3. [Shared machinery](#3-shared-machinery)
4. [Command reference](#4-command-reference)
5. [Module reference](#5-module-reference)
6. [Manifest schemas](#6-manifest-schemas)
7. [On-disk output formats](#7-on-disk-output-formats)
8. [Non-CLI entry points](#8-non-cli-entry-points)
9. [Known defects and traps](#9-known-defects-and-traps)
10. [End-to-end recipes](#10-end-to-end-recipes)

---

## 1. Scope and invocation

This document covers the `seismic_cli` package only. **No module in
`seismic_cli/` imports anything from `src/`** (verified by grep), so `src/` is
outside this reference. `src/extract.py` and `src/download.py` remain the tools
that produce the mseed corpus the CLI consumes — see `README.md` for those; the
rest of `src/` is superseded history.

### Installing and running

`pyproject.toml` declares the console script:

```toml
[project.scripts]
seismic-cli = "seismic_cli.cli:app"
```

```bash
uv sync                       # or: pip install -e .
seismic-cli --help            # after activating .venv
uv run seismic-cli --help     # without activating
python -m seismic_cli.cli --help   # equivalent, no install needed
```

Requires Python ≥ 3.12. Runtime deps: obspy, pandas, scikit-learn, torch,
torchaudio, typer, numpy, scipy, pillow. `torch`/`torchaudio` are imported
**lazily inside workers**, so the pure-RAM `.png` path runs without touching
them.

### Global options

The root app takes no options of its own beyond Typer's built-ins:
`--install-completion`, `--show-completion`, `--help`. Everything else is
per-command.

---

## 2. Command index

Fourteen commands. Ten of them are the *same* orchestrator with a different
encoder; the remaining four are structurally distinct.

| Command | Orchestrator | Encoder | Writes | Split discipline |
|---|---|---|---|---|
| `anchor-windows` | `anchor.run_anchor_windows` | — | anchored `.mseed` | — |
| `generate-dataset` | `core.run_balanced_preprocessing` | `RamImageEncoder` | `.png` | station-disjoint |
| `generate-ram-aux-dataset` | same | `RamAuxEncoder` / `V2` | `.pt` `{img, aux}` | station-disjoint |
| `generate-spectrogram-dataset` | same | `SpectrogramEncoder` | `.pt` bare tensor | station-disjoint |
| `generate-dual-dataset` | same | `RamDualEncoder` | `.pt` `{seq, img}` | station-disjoint |
| `generate-dual-aux-dataset` | same | `RamDualAuxEncoder` / `V2` | `.pt` `{seq, img, aux}` | station-disjoint |
| `generate-spec-dual-dataset` | same | `SpectrogramDualEncoder` | `.pt` `{seq, img}` | station-disjoint |
| `generate-spec-dual-aux-dataset` | same | `SpectrogramDualAuxEncoder` / `V2` | `.pt` `{seq, img, aux}` | station-disjoint |
| `generate-regression-dataset` | `regression.run_regression_preprocessing` | any of the above | `.pt`/`.png` + magnitude | **event**-disjoint (default) |
| `generate-riskclass-dataset` | `riskclass.run_riskclass_preprocessing` | `SpectrogramEncoder` or `RamImageEncoder` | `.pt`/`.png` + 3 classes | station-disjoint, all 3 classes |
| `generate-catalog-dataset` | `catalog.run_catalog_dataset` | *(writes directly)* | `.pt` `{seq, img, aux}` | chronological / random / LOEO |
| `generate-catalog-forecast-dataset` | `catalog.run_catalog_forecast_dataset` | *(writes directly)* | `.pt` `{seq, img, aux}` | chronological, horizon embargo |
| `generate-groundmotion-dataset` | `groundmotion.run_groundmotion_preprocessing` | *(writes directly)* | `.pt` bare tensor + PGA/PGV | **event**-disjoint (forced) |
| `eval-sta-lta` | `eval_baseline.run_eval_sta_lta` | — | metrics to stdout | — |

Only `generate-dataset` writes PNGs by default. Every other generator writes
`.pt` — including `generate-ram-aux-dataset`, whose RAM image lives inside a
tensor dict rather than an image file.

---

## 3. Shared machinery

Everything in this section is inherited by the seven station-split commands and
(except where noted) by `generate-regression-dataset` and
`generate-riskclass-dataset`. `generate-catalog-*` reads a CSV and shares none of
it; `generate-groundmotion-dataset` shares only the trace-grouping and component
selection.

### 3.1 Per-window signal chain — `core.clean_and_filter_1d`

Applied to each component separately, on each extracted window, immediately
before the encoder:

1. `scipy.signal.detrend(x, type='linear')`
2. `scipy.signal.detrend(x, type='constant')`
3. Hann taper over the leading and trailing `int(n * 0.05)` samples
4. 4th-order Butterworth bandpass `[freqmin, freqmax]` via `filtfilt`
   (zero-phase, so the effective response is 8th-order)

Two edge behaviours matter:

- If `fs/2 <= freqmax`, the high corner silently becomes `fs/2 - 1.0` Hz. At the
  default `--freqmax 45` this only bites below 92 Hz sampling.
- If the resulting high corner is **not** greater than `freqmin`, **no filter is
  applied at all** — the data passes through detrended and tapered only.

### 3.2 Windowing and gap rejection — `core.window_array_indexed`

```
target_samples = int(fs_station * window_seconds)
step_samples   = int(target_samples * (1 - overlap))      # must be >= 1
tolerance      = int(target_samples * 0.05)
n_windows      = ceil((n_samples - target_samples) / step_samples) + 1
```

- `overlap` high enough to make `step_samples == 0` raises
  `ValueError: Overlap fraction too high; step size must be at least 1 sample.`
- A trace shorter than `target_samples - tolerance` yields **no** windows.
- A window shorter than `target_samples - tolerance` is skipped; one between that
  and full length is zero-padded up to `target_samples`.
- A window whose **worst** channel has more than `max_gap_fraction` (**0.05**,
  hard-coded, not a flag) interpolated samples is rejected and counted.
- Each kept window carries its **original** index `i`. That index goes into the
  filename, so `start_sample = i * step_samples` stays recoverable even after
  caps or `--max` trimming subsample the list. `eval-sta-lta` relies on this.

**Gap tracking.** In the generation path traces are merged with
`st.merge(method=1)` and *no* `fill_value`, so gaps stay masked.
`core._masked_to_filled` then linearly interpolates them (so filtering sees
contiguous data) while returning a boolean mask of exactly which samples are
synthetic. That mask is what drives rejection. Fewer than two real samples ⇒
gaps become `0.0`.

`anchor.py`, `eval_baseline.py`, `core._scan_noise_file`,
`spectrogram.compute_station_spectral_baselines` and
`groundmotion.extract_event_file` instead merge with
`fill_value='interpolate'`, which produces unmasked arrays.

### 3.3 Channel selection — `core.select_components`

Role-ordered, never alphabetical:

```python
_COMPONENT_ROLES = (('Z',), ('N', '1'), ('E', '2'))
```

Returns `(z, n, e)` or `None`. A station with no usable vertical is **skipped**,
not given a horizontal in the Z slot. Column order is fixed, so the RGB mapping
(R = Z, G = N-ish, B = E-ish) and every per-component aux slot mean the same
thing at every station.

If the three selected components do not share one sampling rate, the station is
skipped for that file. Where a component appears more than once (a second
location code, say), the **longest** trace wins.

### 3.4 Station noise baselines — `core.compute_station_noise_baselines`

Scans every `*.mseed` under `noise_dir` (recursively), applies the *same*
cleaning as training windows, and accumulates `(sum, sum-of-squares, n)` per
`(station_key, component)`. Fans out one task per file across a
`ProcessPoolExecutor`; the accumulation is associative, so results are identical
to a sequential run.

A pair qualifies only with at least `min_baseline_seconds * fs` samples **and**
non-zero variance; otherwise it is dropped and its consumers fall back to
per-window self-standardization (or `log_snr = 0.0`). Traces shorter than 10 s
are skipped outright.

Consumers, which are independent of each other:

| Consumer | Needs baselines | Controlled by |
|---|---|---|
| `--baseline` standardization | yes | the `--baseline` flag |
| `log_snr` in aux vectors / manifests | yes | always on where the command emits it |
| Hard-negative amplitude ranking | *effectively* yes | see §9.3 |
| Spectrogram `--normalize station` | **no** — it uses a *separate* spectral baseline | `--normalize` |

### 3.5 Spectral baselines — `spectrogram.compute_station_spectral_baselines`

Separate machinery for `--normalize station`. Per `(station, component)`, the
**median** dB-per-frequency-bin profile over noise frames (median, not mean, so
one leaked event cannot lift a station's floor).

Non-obvious properties, none of them exposed as flags:

- `max_files_per_station = 20` — only the first 20 files per component are read.
- `min_seconds = 60.0` — a profile needs `60 * nominal_fs / hop_length` frames.
  `--min-baseline-seconds` does **not** reach here.
- It runs **single-threaded in the main process** and imports torch there. On a
  large noise corpus this is often the slowest part of a spectrogram run.
- Files are visited in `sorted()` order, so it is deterministic.

### 3.6 The encoder protocol

One call signature, implemented by every window encoder:

```python
encoder(cleaned_win,        # (samples, 3) float64, already cleaned/filtered
        fs_station,         # this station's real sampling rate
        sta_key,            # "NET.STA"
        selection,          # (z, n, e) component letters
        station_baselines,  # {(sta_key, comp): (mu, sigma)}; {} unless --baseline
        out_dir, stem) -> filename
```

Plus optional attributes: `ext` (`".png"` or `".pt"`), `requires_spawn` (bool),
`target_samples()`.

| Encoder | Module | `ext` | Payload | Resamples to `--fs`? |
|---|---|---|---|---|
| `RamImageEncoder` | `core` | `.png` | RGB image | no |
| `RamAuxEncoder` | `ram_aux` | `.pt` | `{img, aux(2,)}` | no |
| `RamAuxEncoderV2` | `ram_aux` | `.pt` | `{img, aux(6,)}` | no |
| `SpectrogramEncoder` | `spectrogram` | `.pt` | bare `(3, F, T)` tensor | yes |
| `SpectrogramDualEncoder` | `spectrogram` | `.pt` | `{seq, img}` | yes |
| `SpectrogramDualAuxEncoder` | `spectrogram` | `.pt` | `{seq, img, aux(2,)}` | yes |
| `SpectrogramDualAuxEncoderV2` | `spectrogram` | `.pt` | `{seq, img, aux(6,)}` | yes |
| `RamDualEncoder` | `ram_dual` | `.pt` | `{seq, img}` | yes |
| `RamDualAuxEncoder` | `ram_dual` | `.pt` | `{seq, img, aux(2,)}` | yes |
| `RamDualAuxEncoderV2` | `ram_dual` | `.pt` | `{seq, img, aux(6,)}` | yes |

Every encoder except `RamImageEncoder` sets `requires_spawn = True`.

### 3.7 The RAM transform — `core.ram_matrix`

```
x_std   = standardize(x, mu, sigma)          # window's own stats if mu/sigma are None
d       = max(2, ceil(len(x) / target_n))
M       = pad-or-truncate(x_std, d * target_n).reshape(target_n, d).T   # (d, target_n)
Xbar    = mean(M, axis=1)                                                # (d,)
beta_i  = arccos( <M[:,i], Xbar> / (|M[:,i]| |Xbar|) )                   # target_n angles
RAM     = beta[None, :] - beta[:, None]                                  # (target_n, target_n)
```

Output is always `target_n × target_n`, in radians on `[-π, π]` after clipping.
`RamImageEncoder` maps that to `uint8` via `core.to_uint8`; every other encoder
maps it to float32 on `[0, 1]` via `(clip(R, -π, π) + π) / 2π`. `catalog.py` uses
the `to_uint8` path and then divides by 255, so its images are 8-bit-quantized
where the waveform dual encoders' are not.

**RAM is exactly scale-invariant**: `RAM(c·x) == RAM(x)` for any `c > 0`. That is
why `--baseline` cannot change a RAM image's content, and why the `aux` variants
exist at all.

### 3.8 Amplitude scalars

Computed from the *cleaned* window, per component, then averaged over whichever
components produced a finite value:

```
log_rms = log(std(window_component))
log_snr = log(std(window_component) / sigma_station_noise_component)
```

`aux(2,)` = `[log_snr, log_rms]` (Z/N/E-averaged; `0.0` if no component
contributed). `aux(6,)` = `[log_snr_Z, log_snr_N, log_snr_E, log_rms_Z,
log_rms_N, log_rms_E]`, each slot defaulting to `0.0` independently.

`regression.py` writes `log_snr` into the **manifest**, not the tensor, and uses
`NaN` rather than `0.0` for "unavailable".

### 3.9 Execution model and determinism

- **Header-only pre-scan** (`core.scan_single_mseed`) counts extractable windows
  per `(file, station)` without reading sample data. Split allocation runs on
  these *estimates*; realised counts differ slightly because the scan's formula
  (`(min_len - target + tolerance) // step + 1`) is not identical to
  `window_array_indexed`'s, and because gap rejection only happens later.
- **One task per file.** Each file is read once and only its assigned stations
  are written.
- **`spawn` workers for torch encoders.** Under the default `fork`, torch's
  threading/OpenMP state does not survive and the workers deadlock *silently* at
  0 % CPU. Encoders with `requires_spawn = True` get a spawn context, announced
  as `(using 'spawn' workers -- required for torch-based encoders)`.
- **Determinism.** `run_balanced_preprocessing` calls `random.seed(42)` before
  shuffling stations and uses `random.Random(123)` for the per-station cap — it
  has **no `--seed` flag**. `regression`, `riskclass` and `groundmotion` do have
  `--seed`. `catalog` uses `--seed` only for `--split-mode random`.
  Hard-negative amplitude matching uses `np.random.default_rng(0)`.

### 3.10 Split ratios

`--train-ratio` / `--val-ratio` / `--test-ratio` carry **no help text** in
`--help`; they render as bare defaults. They are also **not validated or
normalised** — nothing checks that they sum to 1.0. How they are used differs:

| Path | Use |
|---|---|
| `run_balanced_preprocessing`, non-`--max` | `train = int(target·r0)`, `val = int(target·r1)`, **`test = target - train - val`** (the remainder) |
| `run_balanced_preprocessing`, `--max` | per-class targets `r_i · class_total`, used as deficits |
| `regression` / `riskclass` / `groundmotion` | targets `r_i · total_windows`, largest-relative-deficit assignment |
| `catalog` | *time* cuts at `t0 + span·r0` and `t0 + span·(r0+r1)`; test is everything after |

### 3.11 Station-disjoint allocation, in detail

Two different algorithms live in `run_balanced_preprocessing`:

**Default (no `--max`)** — stations are shuffled, then for each station the
**first** split in `train → val → test` order that is still below target for any
class this station carries is chosen. Stations are dropped once every split they
could help is full. This is first-fit, not best-fit.

**`--max`** — *every* usable station is assigned, to the split with the largest
summed relative deficit across the classes it carries. Then, per split, the
surplus class is trimmed down to the smaller class by distributing per-`(station,
file)` quotas with largest-remainder proportional rounding. Quotas are enforced
at generation time by evenly-spaced window subsampling (`np.linspace`).

In both cases a station occupies **exactly one** split across **both** classes.
That is printed as `[INFO] Every station occupies exactly one split across BOTH
classes.`

`--max-windows-per-station` is applied first, as a per-`(station, file)` **quota**
rather than by dropping files — a file-granularity cap cannot go below one file's
window count.

---

## 4. Command reference

### 4.1 `anchor-windows`

Re-derives arrival-anchored short windows from already-downloaded long records
using a coarse STA/LTA pick. No network access.

```
seismic-cli anchor-windows [OPTIONS]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--source-dir` | str | **required** | Directory of long-window mseed (searched recursively for `*.mseed`). |
| `--output-base-dir` | str | **required** | Parent directory; one subfolder per target length is created under it. |
| `--target-seconds`, `-t` | float | **required**, repeatable | Short window length(s). `-t 3 -t 6 -t 10`. |
| `--pick-sta-seconds` | float | `1.0` | STA length for the pick. `nsta = max(1, int(sta·fs))`. |
| `--pick-lta-seconds` | float | `10.0` | LTA length for the pick. `nlta = max(nsta+1, int(lta·fs))`. |
| `--trigger-on` | float | `3.5` | STA/LTA ratio that declares an arrival. |
| `--trigger-off` | float | `1.0` | Ratio that ends the trigger. |
| `--pre-arrival-fraction` | float | `0.2` | Fraction of the output window placed *before* the arrival. |
| `--limit-files` | int | none | Process only the first N source files. |

**Output naming.** `{output_base_dir}/window_post_{S}s_anchored/{source_filename}`,
where `S` is `int(s)` when `s` is integral and the float otherwise —
`-t 3` → `window_post_3s_anchored/`, `-t 3.4` → `window_post_3.4s_anchored/`.

**Per-station algorithm.**

1. Traces are grouped by `NET.STA`; a station with **fewer than 3 traces** is
   skipped (counted as `skipped (<3 channels)`).
2. Pick candidates are ordered: all `*Z` channels first, then the rest sorted by
   full channel code. The first channel that triggers wins.
3. A **linearly detrended copy** is used for the pick only — raw MiniSEED counts
   carry DC offsets that pin `classic_sta_lta` near 1. The written data stays raw.
4. Arrival sample `a` = first `trigger_onset` crossing. Slice
   `[a − f·T, a − f·T + T)` with `f = --pre-arrival-fraction`, `T = int(fs·target)`.
   - `a − f·T < 0` ⇒ this station contributes **nothing** for that target length.
   - The slice overrunning the record end ⇒ start is clamped to `n − T`, so the
     pre-arrival fraction is **not** honoured for that window.
5. **Every** trace of the station is sliced, not just the Z/N/E selection.
   `stats.mseed` is dropped and data is cast to float32.

Output is written as MSEED with `encoding='FLOAT32'`, one file per
`(target length, source file)`, containing every station that picked.

**Diagnostics.** A `[PICK DIAGNOSTICS]` block reports stations seen, skipped for
too few channels, picked on Z, picked on a fallback channel, and unpicked — plus
the median and max STA/LTA ratio the failures reached, so "no pick" can be told
apart from "nearly picked".

Single-process; progress prints every 200 files.

```bash
seismic-cli anchor-windows \
    --source-dir raw/data/batched_waveforms/window_post_60s \
    --output-base-dir raw/data/batched_waveforms \
    -t 3 -t 6 -t 10

# smoke test with a looser trigger
seismic-cli anchor-windows \
    --source-dir raw/data/batched_waveforms/window_post_60s \
    --output-base-dir /tmp/anchor_test \
    -t 6 --trigger-on 2.5 --limit-files 50
```

> **Caveat that propagates downstream:** `anchor.py` replaces `tr.data` but never
> advances `tr.stats.starttime`. Anchored files therefore carry their parent
> record's start time, and the absolute arrival time is not recoverable from
> them. `groundmotion.py` works around this by replaying the pick against the
> 60 s record.

---

### 4.2 `generate-dataset`

Balanced, station-disjoint RAM-image dataset.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--eq-dir` | str | **required** | Earthquake mseed directory (recursive `*.mseed`). |
| `--noise-dir` | str | **required** | Noise mseed directory. |
| `--output-dir` | str | **required** | Dataset root; `train/`, `val/`, `test/` each with a `<class>/` subdirectory, plus `manifest.csv`. |
| `--window-seconds` | float | `60.0` | Window length. |
| `--overlap` | float | `0.5` | Sliding-window overlap fraction. |
| `--target-n` | int | `64` | RAM image side length; output is `target_n × target_n`. |
| `--train-ratio` | float | `0.7` | See §3.10. |
| `--val-ratio` | float | `0.15` | See §3.10. |
| `--test-ratio` | float | `0.15` | Remainder in this path; see §3.10. |
| `--limit-pictures` | int | none | Cap on total images across both classes (`target_per_class = limit // 2`). Mutually exclusive with `--max`. |
| `--max` | flag | off | Maximum balanced dataset; see §3.11. Mutually exclusive with `--limit-pictures`. |
| `--max-windows-per-station` | int | none | Per-**window** cap on one station's total contribution. |
| `--baseline` / `--no-baseline` | flag | `--no-baseline` | Standardize each channel against that station's long-term noise `(mu, sigma)` instead of the window's own. Per-channel fallback where no baseline exists. |
| `--freqmin` | float | `1.0` | Bandpass low corner (Hz). |
| `--freqmax` | float | `45.0` | Bandpass high corner (Hz). |
| `--min-baseline-seconds` | float | `60.0` | Minimum usable noise per `(station, component)` before its baseline is trusted. Only consulted when `--baseline` is set. |
| `--num-cores` | int | `cpu_count() - 1` | Worker processes. |

**No `--fs` flag.** `run_balanced_preprocessing` is called with `fs=100.0`
hard-coded. That value is used **only** to convert `--min-baseline-seconds` into
a sample count; all windowing uses each station's own header sampling rate.
`RamImageEncoder` does not resample, so a 50 Hz station's window has half the
samples and therefore half the RAM reshape depth `d` — the image is still
`target_n × target_n`.

**`--overlap` in practice.** It applies to both classes. On the standard corpus
the earthquake files are exactly one window long (a 60 s record at
`--window-seconds 60`, a 3 s anchored record at `--window-seconds 3`), so only
the 300 s noise files produce more than one window and overlap only moves the
noise count. That stops being true the moment `--window-seconds` is shorter than
the event records.

Because RAM is scale-invariant, `--baseline` changes nothing about the emitted
image content here; it is retained for parity with the encoders where it does
matter (`seq` branches).

```bash
# 60 s, defaults
seismic-cli generate-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_60s \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_60s \
    --window-seconds 60 --overlap 0.25

# 6 s anchored, everything the data supports, station cap
seismic-cli generate-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_anchored \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_6s_max \
    --window-seconds 6 --overlap 0.5 \
    --max --max-windows-per-station 20
```

---

### 4.3 `generate-ram-aux-dataset`

RAM image plus the amplitude scalars RAM's scale-invariance discards.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--eq-dir` | str | **required** | Earthquake mseed directory. |
| `--noise-dir` | str | **required** | Noise mseed directory; **also** the source of the baselines `log_snr` is measured against. |
| `--output-dir` | str | **required** | Dataset root. |
| `--window-seconds` | float | `60.0` | Window length. |
| `--overlap` | float | `0.5` | Overlap fraction. |
| `--target-n` | int | `64` | RAM image side length. |
| `--train-ratio` | float | `0.7` | |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | |
| `--limit-pictures` | int | none | Cap on total tensors. Mutually exclusive with `--max`. |
| `--max` | flag | off | Maximum balanced dataset. |
| `--max-windows-per-station` | int | none | Per-window station cap. |
| `--freqmin` | float | `1.0` | Bandpass low corner. |
| `--freqmax` | float | `45.0` | Bandpass high corner. |
| `--fs` | float | `100.0` | **Not a resample.** Only converts `--min-baseline-seconds` to samples (see below). |
| `--min-baseline-seconds` | float | `60.0` | Minimum usable noise before a station's baseline is trusted for `log_snr`; otherwise that component contributes nothing and `log_snr` falls back to `0.0`. |
| `--per-component-aux` | flag | off | Emit `aux` as 6 per-component scalars instead of 2 Z/N/E-averaged ones. |
| `--num-cores` | int | `cpu_count() - 1` | Worker processes. |

**There is no `--baseline` flag here, by design.** Baselines are computed
unconditionally for `log_snr`, and the image is *always* built with plain
per-window self-standardization (`RamAuxEncoder` passes no `mu`/`sigma` to
`ram_matrix`). Because RAM is scale-invariant this is immaterial to the image.

`--fs` reaches `compute_station_noise_baselines(fs=...)` and
`run_balanced_preprocessing(fs=...)`, and in both places it is used only for the
baseline sample-count threshold. No resampling happens on this path.

If no baselines are built at all, the run prints
`[WARN] No station noise baselines built; log_snr will default to 0.0 for every window.`

```bash
seismic-cli generate-ram-aux-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_anchored \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_ramaux_6s \
    --window-seconds 6 --overlap 0.5 --max --max-windows-per-station 20

# 6 per-component aux slots instead of 2 averaged ones
seismic-cli generate-ram-aux-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_anchored \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_ramaux_6s_pc \
    --window-seconds 6 --max --per-component-aux
```

---

### 4.4 `generate-spectrogram-dataset`

3-channel log-power spectrogram tensors.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--eq-dir` | str | **required** | Earthquake mseed directory. |
| `--noise-dir` | str | **required** | Noise mseed directory (also the spectral-baseline source). |
| `--output-dir` | str | **required** | Dataset root. |
| `--window-seconds` | float | `60.0` | Window length. |
| `--overlap` | float | `0.5` | Overlap fraction. |
| `--n-fft` | int | `256` | FFT size. Frequency bins = `n_fft // 2 + 1`. |
| `--hop-length` | int | `n_fft // 4` | STFT hop. Time frames = `n_samples // hop + 1`. |
| `--top-db` | float | `80.0` | Dynamic-range clamp for `AmplitudeToDB`. |
| `--normalize` | str | `station` | `station` / `per_window` / `none`. Validated; anything else is a clean `Invalid value` error. |
| `--train-ratio` | float | `0.7` | |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | |
| `--limit-pictures` | int | none | Cap on total tensors. Mutually exclusive with `--max`. |
| `--max` | flag | off | Maximum balanced dataset. |
| `--max-windows-per-station` | int | none | Per-window station cap. |
| `--freqmin` | float | `1.0` | Bandpass low corner. |
| `--freqmax` | float | `45.0` | Bandpass high corner. |
| `--fs` | float | `100.0` | Nominal rate. **Every window is polyphase-resampled to this**, so all tensors share one shape. |
| `--num-cores` | int | `cpu_count() - 1` | Worker processes. |

**Normalization modes.**

- `station` — subtract that station's median noise dB profile per frequency bin.
  The result reads as *dB above this station's own noise floor*: instrument gain
  cancels, genuine amplitude-above-background survives. A station with no usable
  profile silently falls back to `per_window` for that window.
- `per_window` — z-score the whole `(3, F, T)` tensor. Removes gain **and**
  absolute amplitude.
- `none` — raw dB. Keeps amplitude, leaks instrument gain.

There is **no `--baseline` flag** on this command; `station_baselines` stays empty
and the amplitude baseline machinery is never invoked. The spectral profiles
(§3.5) are a separate computation.

**STFT geometry is checked, not assumed.** `SpectrogramEncoder.__init__` calls
`check_stft_resolution` once and prints a `[WARN]`/`[SEVERE]` block when `n_fft`
is badly matched to the window, with a concrete suggestion. Measured values at
100 Hz:

| Window | Default `n_fft 256` | Suggested `--n-fft / --hop-length` | Suggested shape |
|---|---|---|---|
| 3.0 s | 129 × 5 frames, one frame spans 85 % of the window | `64 / 16` | 33 × 19 |
| 3.4 s | 129 × 6 | `64 / 16` | 33 × 22 |
| 6.0 s | 129 × 10 | `128 / 32` | 65 × 19 |
| 60.0 s | 129 × 94 | `1024 / 256` | 513 × 24 |

Keeping a mismatched value is only correct when you are deliberately matching an
existing dataset — a model must see the geometry it was trained on.

```bash
seismic-cli generate-spectrogram-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_60s \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_spec_60s \
    --window-seconds 60 --overlap 0.25 --max --normalize station

# 3 s, STFT geometry matched to the window
seismic-cli generate-spectrogram-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_3s_anchored \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_spec_3s \
    --window-seconds 3 --n-fft 64 --hop-length 16 \
    --max --max-windows-per-station 20
```

---

### 4.5 `generate-dual-dataset`

`{seq, img}` for the dual-channel CNN+LSTM, with a RAM image as the 2D channel.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--eq-dir` | str | **required** | Earthquake mseed directory. |
| `--noise-dir` | str | **required** | Noise mseed directory. |
| `--output-dir` | str | **required** | Dataset root. |
| `--window-seconds` | float | `60.0` | Window length. |
| `--overlap` | float | `0.5` | Overlap fraction. |
| `--target-n` | int | `64` | RAM image side length. |
| `--fs` | float | `100.0` | Nominal rate; **windows are resampled to it**, and it fixes `seq`'s length `m = round(fs · window_seconds)`. |
| `--train-ratio` | float | `0.7` | |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | |
| `--limit-pictures` | int | none | Cap on total tensors. Mutually exclusive with `--max`. |
| `--max` | flag | off | Maximum balanced dataset. |
| `--max-windows-per-station` | int | none | Per-window station cap. |
| `--baseline` / `--no-baseline` | flag | `--no-baseline` | Standardize against the station's noise baseline. Applied **identically to both branches** — `seq` and the RAM reshape read the same `(mu, sigma)`. |
| `--freqmin` | float | `1.0` | Bandpass low corner. |
| `--freqmax` | float | `45.0` | Bandpass high corner. |
| `--min-baseline-seconds` | float | `60.0` | Baseline trust threshold. |
| `--num-cores` | int | `cpu_count() - 1` | Worker processes. |

`seq` is `(m, 3)` float32 — the raw standardized Z/N/E waveform, one timestep per
sample. `img` is `(3, target_n, target_n)` float32 on `[0, 1]`.

**Attention cost.** The 1D branch's self-attention is `O(m²)`. At 100 Hz a 6 s
window is `m = 600`; a 60 s window is `m = 6000` (36 M entries per head). Long
windows at high `--fs` may need a smaller training batch size.

Because `--baseline` here only changes the `(mu, sigma)` handed to `standardize`
and `ram_matrix`, and RAM is scale-invariant, it is the **`seq` branch alone**
that the flag actually affects.

```bash
seismic-cli generate-dual-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_anchored \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_dual_6s \
    --window-seconds 6 --overlap 0.5 --fs 100 \
    --max --max-windows-per-station 20 --baseline
```

---

### 4.6 `generate-dual-aux-dataset`

`generate-dual-dataset` plus the `[log_snr, log_rms]` aux vector.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--eq-dir` | str | **required** | Earthquake mseed directory. |
| `--noise-dir` | str | **required** | Noise mseed directory; **also** the source of the aux baselines `log_snr` is measured against. |
| `--output-dir` | str | **required** | Dataset root. |
| `--window-seconds` | float | `60.0` | Window length. |
| `--overlap` | float | `0.5` | Overlap fraction. |
| `--target-n` | int | `64` | RAM image side length. |
| `--fs` | float | `100.0` | Nominal rate; **windows are resampled to it**, and it fixes `seq`'s length `m = round(fs · window_seconds)`. |
| `--train-ratio` | float | `0.7` | |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | |
| `--limit-pictures` | int | none | Cap on total tensors. Mutually exclusive with `--max`. |
| `--max` | flag | off | Maximum balanced dataset. |
| `--max-windows-per-station` | int | none | Per-window station cap. |
| `--baseline` / `--no-baseline` | flag | `--no-baseline` | Standardizes the `seq` and `img` branches. **Independent of the aux branch's `log_snr`, which always needs a station baseline and is computed regardless of this flag.** |
| `--freqmin` | float | `1.0` | Bandpass low corner. |
| `--freqmax` | float | `45.0` | Bandpass high corner. |
| `--min-baseline-seconds` | float | `60.0` | Baseline trust threshold, for both the aux baselines and the `--baseline` pass. |
| `--per-component-aux` | flag | off | 6 per-component aux scalars instead of 2 Z/N/E-averaged ones. |
| `--num-cores` | int | `cpu_count() - 1` | Worker processes. |

`seq` is `(m, 3)`, `img` is `(3, target_n, target_n)`, `aux` is `(2,)` or `(6,)`.
Attention cost scales as `O(m²)` exactly as in §4.5.

**Double scan warning.** The command calls `compute_station_noise_baselines`
itself for the aux vector, and then passes `use_baseline_standardization=baseline`
to `run_balanced_preprocessing`, which computes them **again**. With `--baseline`
the whole noise corpus is scanned twice.

```bash
seismic-cli generate-dual-aux-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_anchored \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_dualaux_6s \
    --window-seconds 6 --fs 100 --max --per-component-aux
```

---

### 4.7 `generate-spec-dual-dataset`

`{seq, img}` with a log-power spectrogram as the 2D channel. This is the command
with the hard-negative mining machinery.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--eq-dir` | str | **required** | Earthquake mseed directory. |
| `--noise-dir` | str | **required** | Noise mseed directory. |
| `--output-dir` | str | **required** | Dataset root. |
| `--window-seconds` | float | `60.0` | Window length. |
| `--overlap` | float | `0.5` | Overlap fraction. |
| `--n-fft` | int | `256` | FFT size. |
| `--hop-length` | int | `n_fft // 4` | STFT hop. |
| `--top-db` | float | `80.0` | dB dynamic-range clamp. |
| `--normalize` | str | `station` | 2D-channel normalization; see §4.4. Validated. |
| `--train-ratio` | float | `0.7` | |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | |
| `--limit-pictures` | int | none | Cap on total tensors. Mutually exclusive with `--max`. |
| `--max` | flag | off | Maximum balanced dataset. |
| `--max-windows-per-station` | int | none | Per-window station cap. |
| `--baseline` / `--no-baseline` | flag | `--no-baseline` | Standardizes the **1D (`seq`) channel only**. Independent of `--normalize`, which controls the 2D channel. |
| `--freqmin` | float | `1.0` | Bandpass low corner. |
| `--freqmax` | float | `45.0` | Bandpass high corner. |
| `--fs` | float | `100.0` | Nominal rate; resample target and `seq` length. |
| `--min-baseline-seconds` | float | `60.0` | Amplitude-baseline trust threshold. **Does not affect the spectral profiles** (§3.5). |
| `--hard-negatives` / `--no-hard-negatives` | flag | `--no-hard-negatives` | Mine noise windows by amplitude instead of sampling them evenly. |
| `--hard-negative-band` | 2 floats | `0.75 0.99` | Percentile band of the amplitude ranking to mine, as `LOW HIGH` (two space-separated values). |
| `--match-negative-amplitude` / `--no-match-negative-amplitude` | flag | off | **Requires `--hard-negatives`.** Ignores the band; matches the noise amplitude *distribution* to the events' instead. |
| `--num-cores` | int | `cpu_count() - 1` | Worker processes. |

**How hard-negative mining works** (`core._mine_hard_negatives_globally`):

1. One extra pass over the mined class's files scores every candidate window:
   the loudest component's standard deviation, divided by that
   `(station, component)`'s long-term noise sigma **when a baseline exists**, and
   left in raw counts otherwise.
2. Scoring is **global per split**, not per file. Almost all amplitude variance
   lives between stations and times; a first per-file version moved the amplitude
   floor only 0.9535 → 0.9312 where a global one predicted ≈ 0.86.
3. Each split's quota (the sum of the counts the balancing stage already planned)
   is filled from the ranking, so class balance is preserved exactly.
4. The selection is turned into per-`(file, station)` window **whitelists**. An
   empty whitelist is honoured as "write nothing" rather than falling back to a
   quota.

`--hard-negative-band`'s **upper** bound is deliberate: the loudest tail of a
screened noise archive is where a catalog-missed earthquake hides, and mining it
would inject positives into the negative class. Selection is spread evenly across
the band rather than taken from its top, and the band widens (downward first)
rather than under-filling a quota.

`--match-negative-amplitude` exists because the band puts a hard amplitude
**floor** under every negative while the positives have none. On a P-only window,
where the loud phases are cut away, that makes `P(event | amplitude)` U-shaped and
lets a model learn "very quiet ⇒ event" — an artifact of the mining. Matching
quantile-bins the event distribution and fills each bin from the noise pool,
redistributing what the (mostly loud) empty bins cannot supply to the nearest
bins that can. The run reports the residual, e.g.
`21.4% of events are louder than any candidate noise window and cannot be matched`.

> **Trap.** Amplitude scores are only comparable *across* stations when they are
> divided by each station's own noise sigma, and `station_baselines` is empty
> unless `--baseline` is passed. Without it the global ranking runs on raw
> counts, where "loudest" mostly means "highest-gain instrument". Pass
> `--baseline` with `--hard-negatives`. See §9.3.

```bash
# plain
seismic-cli generate-spec-dual-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_catalog \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_specdual_6s \
    --window-seconds 6 --n-fft 128 --hop-length 32 --fs 100 \
    --max --max-windows-per-station 20

# hard negatives from the 75-99th percentile band
seismic-cli generate-spec-dual-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_catalog \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_specdual_6s_hard \
    --window-seconds 6 --n-fft 128 --hop-length 32 \
    --max --baseline --hard-negatives --hard-negative-band 0.75 0.99

# amplitude-matched negatives on a P-only window
seismic-cli generate-spec-dual-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_3.4s_ponly \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_specdual_ponly_3p4s_matched \
    --window-seconds 3.4 --n-fft 64 --hop-length 16 \
    --max --baseline --hard-negatives --match-negative-amplitude
```

---

### 4.8 `generate-spec-dual-aux-dataset`

`generate-spec-dual-dataset` plus the aux vector — but **without** the
hard-negative flags.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--eq-dir` | str | **required** | Earthquake mseed directory. |
| `--noise-dir` | str | **required** | Noise mseed directory; also the aux-baseline source. |
| `--output-dir` | str | **required** | Dataset root. |
| `--window-seconds` | float | `60.0` | Window length. |
| `--overlap` | float | `0.5` | Overlap fraction. |
| `--n-fft` | int | `256` | FFT size. |
| `--hop-length` | int | `n_fft // 4` | STFT hop. |
| `--top-db` | float | `80.0` | dB clamp. |
| `--normalize` | str | `station` | 2D-channel normalization. Validated. |
| `--train-ratio` | float | `0.7` | |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | |
| `--limit-pictures` | int | none | Cap. Mutually exclusive with `--max`. |
| `--max` | flag | off | Maximum balanced dataset. |
| `--max-windows-per-station` | int | none | Per-window station cap. |
| `--baseline` / `--no-baseline` | flag | `--no-baseline` | 1D channel only. Independent of `--normalize` **and** of the aux `log_snr`, which is always computed. |
| `--freqmin` | float | `1.0` | Bandpass low corner. |
| `--freqmax` | float | `45.0` | Bandpass high corner. |
| `--fs` | float | `100.0` | Nominal rate; resample target and `seq` length. |
| `--min-baseline-seconds` | float | `60.0` | Amplitude-baseline threshold (aux and `--baseline`); not the spectral profiles. |
| `--per-component-aux` | flag | off | 6 aux scalars instead of 2. |
| `--num-cores` | int | `cpu_count() - 1` | Worker processes. |

**Up to three separate noise scans.** Spectral profiles (if
`--normalize station`), aux amplitude baselines (always), and the `--baseline`
pass inside `run_balanced_preprocessing` (if `--baseline`). This is the slowest
generator to start.

```bash
seismic-cli generate-spec-dual-aux-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_catalog \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_specdualaux_6s \
    --window-seconds 6 --n-fft 128 --hop-length 32 --max
```

---

### 4.9 `generate-regression-dataset`

Magnitude-labelled windows from earthquake mseed only. No noise class — the noise
directory is used solely as the amplitude reference.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--eq-dir` | str | **required** | Earthquake mseed directory. |
| `--noise-dir` | str | **required** | Noise directory, used **only** for the `log_snr` reference baselines. |
| `--catalog-path` | str | **required** | Event catalog CSV. Needs a magnitude column; an ID column and lat/lon are strongly recommended. |
| `--output-dir` | str | **required** | Dataset root; the `train/`, `val/`, `test/` directories are flat (no class subdirectories). |
| `--station-catalog` | str | none | Station CSV enabling `distance_km`. Without it that column is `NaN`. |
| `--encoding` | str | `spectrogram` | `spectrogram` → `.pt`, `ram` → `.png`. Validated (clean error). |
| `--dual` | flag | off | Also write a `seq` channel. **Forces `.pt` output for both encodings**, since RAM's single-channel `.png` has no `seq` slot. |
| `--window-seconds` | float | `60.0` | Window length. |
| `--overlap` | float | `0.5` | Overlap fraction. |
| `--split-by` | str | `event` | `event` keeps whole events together (the label is per-event). `station` is station-disjoint instead. **Not validated by Typer** — a bad value raises `ValueError` as a traceback, before any work. |
| `--n-fft` | int | `256` | FFT size (spectrogram encoding). |
| `--hop-length` | int | `n_fft // 4` | STFT hop. The default is 64, giving only ~5 frames from a 3 s window; pass e.g. 16 for finer time resolution at the same frequency resolution. |
| `--top-db` | float | `80.0` | dB clamp. |
| `--normalize` | str | `station` | Spectrogram normalization. **Not validated by Typer** — a bad value raises `ValueError` from `SpectrogramEncoder.__init__`. |
| `--target-n` | int | `64` | RAM image side length (`ram` encoding). |
| `--train-ratio` | float | `0.7` | |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | |
| `--max-windows-per-station` | int | none | Per-window station cap. |
| `--freqmin` | float | `1.0` | Bandpass low corner. |
| `--freqmax` | float | `45.0` | Bandpass high corner. |
| `--fs` | float | `100.0` | Nominal rate; also the resample target when the encoder resamples. |
| `--seed` | int | `42` | Seeds the split-key shuffle (`seed`) and the station-cap RNG (`seed + 1`). |
| `--per-component-aux` | flag | off | Write 3 manifest columns `log_snr_0/1/2` (one per Z/N/E) instead of one averaged `log_snr`. **Manifest columns, not tensor contents.** |
| `--num-cores` | int | `cpu_count() - 1` | Worker processes. |

**There is no `--max`, no `--limit-pictures`, and no `--baseline`.**
`min_baseline_seconds` is fixed at `60.0` internally.

**Event ID parsing.** `regression.parse_event_id` matches
`^(?:noise_)?event_(.+?)_raw$` against the file stem, so files must be named
`event_<EventID>_raw.mseed`. A file whose ID is absent from the catalog is
dropped and counted.

**Catalog column auto-detection** (case-insensitive, first match wins):

| Field | Accepted headers |
|---|---|
| event id | `eventid`, `event_id`, `id` — falls back to `idx_<row>`, which will never match a filename |
| magnitude | `magnitude`, `mag`, `ml`, `m` — **required**, else `ValueError` |
| latitude | `latitude`, `lat` |
| longitude | `longitude`, `lon`, `long` |

**Station catalog columns:** network ∈ {`network`, `net`, `network_code`},
station ∈ {`station`, `sta`, `station_code`, `code`}, plus lat/lon as above.
Coordinates are keyed under both `NET.STA` and the bare station code, so a
network-code mismatch still resolves.

**Encoder selection:**

| `--encoding` | `--dual` | Encoder | Extension |
|---|---|---|---|
| `spectrogram` | off | `SpectrogramEncoder` | `.pt` |
| `spectrogram` | on | `SpectrogramDualEncoder` | `.pt` |
| `ram` | off | `RamImageEncoder` | `.png` |
| `ram` | on | `RamDualEncoder` | `.pt` |

**Splitting.** Windows are grouped by event id (or station key), the group keys
are shuffled with `--seed`, and each group goes to the split with the largest
relative deficit. Station caps are applied afterwards as per-`(station, file)`
quotas. Whichever axis was *not* split on is measured and reported:

```
[leakage] 1183/1421 stations appear in more than one split.
          Events are disjoint (the per-event magnitude label cannot leak);
          shared stations mean site response is seen across splits.
```

**Worker initialisation.** `event_meta` (≈ 22 MB pickled on the full catalog) is
shipped to each worker once via `ProcessPoolExecutor(initializer=...)` rather than
inside every one of ~31 k tasks.

```bash
# magnitude regression on 6 s catalog windows, spectrogram 2D + seq 1D
seismic-cli generate-regression-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_catalog \
    --noise-dir raw/data/batched_noise_waveforms \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --station-catalog catalogs/stations.csv \
    --output-dir dataset_magreg_catalog_6s \
    --encoding spectrogram --dual \
    --window-seconds 6 --n-fft 256 --split-by event

# 3 s, finer STFT hop, per-component log_snr columns
seismic-cli generate-regression-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_3s_anchored \
    --noise-dir raw/data/batched_noise_waveforms \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir dataset_magreg_3s_hop16 \
    --window-seconds 3 --n-fft 64 --hop-length 16 \
    --dual --per-component-aux
```

---

### 4.10 `generate-riskclass-dataset`

Three-class dataset: `00_noise` / `01_low_risk` / `02_high_risk`.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--eq-dir` | str | **required** | Earthquake mseed directory. |
| `--noise-dir` | str | **required** | Noise directory — here a **class of its own**, not just a reference. |
| `--catalog-path` | str | **required** | Event catalog CSV (same auto-detection as §4.9). |
| `--output-dir` | str | **required** | Dataset root; the `train/`, `val/`, `test/` directories are flat. |
| `--station-catalog` | str | none | Enables `distance_km` for the earthquake rows. |
| `--encoding` | str | `spectrogram` | `spectrogram` → `.pt`, `ram` → `.png`. Validated. |
| `--mag-threshold` | float | `4.0` | `magnitude >= threshold` ⇒ `02_high_risk`, else `01_low_risk`. |
| `--balance-ratio` | float | `4.0` | Per split, caps `01_low_risk` and `00_noise` at `round(ratio × count(02_high_risk))`. Pass a very large value to disable. |
| `--min-log-snr` | float | `-3.0` | Drop windows whose `log_snr` falls below this. `exp(-3) ≈ 5 %` of the station's own noise floor. Pass e.g. `-99` to disable. |
| `--window-seconds` | float | **`3.0`** | Note: differs from every other generator's `60.0`. |
| `--overlap` | float | `0.5` | Overlap fraction. |
| `--n-fft` | int | `256` | FFT size. |
| `--top-db` | float | `80.0` | dB clamp. |
| `--normalize` | str | `station` | Spectrogram normalization. **Not validated by Typer** (see §4.9). |
| `--target-n` | int | `64` | RAM image side length. |
| `--train-ratio` | float | `0.7` | |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | |
| `--max-windows-per-station` | int | none | Per-window station cap. |
| `--freqmin` | float | `1.0` | Bandpass low corner. |
| `--freqmax` | float | `45.0` | Bandpass high corner. |
| `--fs` | float | `100.0` | Nominal rate; also the resample target for the spectrogram encoder. |
| `--seed` | int | `42` | Station shuffle (`seed`), balance-cap RNG (`seed + 7`), station-cap RNG (`seed + 11`). |
| `--num-cores` | int | `cpu_count() - 1` | Worker processes. |

**There is no `--hop-length` and no `--dual`.** The spectrogram hop is always
`n_fft // 4`, both for the encoder and for the spectral baselines.

**Splitting.** Stations from *both* directories are pooled and assigned together
by total window count, largest relative deficit first. Event-level overlap across
splits is **measured and printed**, not enforced — forcing both station- and
event-disjointness simultaneously would push every station that recorded a given
event into one split.

**`--balance-ratio` mechanics.** The cap is computed per split from **header-scan
estimates** of the `02_high_risk` count, then applied by shuffling that
`(split, class)`'s files and admitting whole files until the running window total
reaches the cap. If a split has zero `02_high_risk` windows the cap is `None`
(uncapped) for that split.

**`--min-log-snr` mechanics.** `log_snr` is the mean over components of
`log(sigma_window / sigma_station_noise)`. The window is dropped only when at
least one component had a usable baseline; a window with **no** baseline at all
has `log_snr = NaN` and is **kept**. The threshold is applied uniformly to every
class and every split.

The rationale is instrument-fault rejection, not outlier trimming: one station
contributed 199 noise windows whose raw traces spanned ~58 counts on a
~5.38-million-count DC offset with ~50 unique values across 30 001 samples — a
stuck digitizer, verified by reading the MiniSEED. Its `log_snr` sits at ≈ −6
while the pooled noise 5th percentile is −2.67, a clean gap.

```bash
seismic-cli generate-riskclass-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_3s_anchored \
    --noise-dir raw/data/batched_noise_waveforms \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --station-catalog catalogs/stations.csv \
    --output-dir dataset_risk_3s \
    --window-seconds 3 --n-fft 64 --mag-threshold 4.0 --balance-ratio 4

# no balancing, no dead-instrument filter (raw natural distribution)
seismic-cli generate-riskclass-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_3s_anchored \
    --noise-dir raw/data/batched_noise_waveforms \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir dataset_risk_3s_natural \
    --balance-ratio 1e9 --min-log-snr -99
```

---

### 4.11 `generate-catalog-dataset`

Sliding windows over an earthquake **catalog**. No waveforms are read.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--catalog-path` | str | **required** | AFAD/Kandilli-style catalog CSV. |
| `--output-dir` | str | **required** | Dataset root. |
| `--window-events` | int | `64` | Events per window (fixed count, not fixed duration). |
| `--stride-events` | int | `8` | Events advanced between windows. |
| `--major-magnitude` | float | `6.0` | Magnitude that defines a prediction target. |
| `--min-magnitude` | float | `2.0` | Drop catalog events below this before anything else. |
| `--target-n` | int | `32` | RAM image side length for the 2D channel. |
| `--lat-min` | float | none | Bounding box; **all four** must be given for the bbox to apply. |
| `--lat-max` | float | none | |
| `--lon-min` | float | none | |
| `--lon-max` | float | none | |
| `--center-lat` | float | none | Circular selection; **both** must be given, together with `--radius-km`. |
| `--center-lon` | float | none | |
| `--radius-km` | float | none | Alternative to a bbox. |
| `--region`, `-r` | str | none, repeatable | `'lat_min,lat_max,lon_min,lon_max'`. **Overrides the bbox/center flags entirely when given.** Windows are built independently per region, then pooled. |
| `--split-mode` | str | `chronological` | `chronological` / `random` / `loeo`. Validated (clean error). |
| `--embargo-days` | float | none | **Additional** hard gap at each split boundary. Not the primary leak defence — see below. |
| `--max-horizon-days` | float | `3650.0` | Discard windows whose next target event is further out than this. |
| `--class-lo-days` | float | none | Lower risk-class boundary. **Both** `--class-lo-days` and `--class-hi-days` must be given, or both are ignored and terciles are derived. |
| `--class-hi-days` | float | none | Upper risk-class boundary. |
| `--train-ratio` | float | `0.7` | Fraction of the *time span*. |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | Implicit: everything after the val cut. |
| `--seed` | int | `42` | Used **only** by `--split-mode random`. |
| `--decluster` / `--no-decluster` | flag | `--decluster` | Restrict prediction *targets* to independent Gardner–Knopoff mainshocks. |

**Catalog parsing** (`catalog.load_catalog`) auto-detects columns including
Turkish headers:

| Field | Accepted headers |
|---|---|
| datetime | `datetime`, `date_time`, `origin_time`, `time`, `tarih`, `olus zamani`, `oluş zamanı`, `date`, `tarih_saat` |
| date + time | `date`/`tarih` joined with `time`/`saat` when both exist and differ |
| latitude | `latitude`, `lat`, `enlem` |
| longitude | `longitude`, `lon`, `long`, `boylam` |
| depth | `depth`, `derinlik`, `depth_km` |
| magnitude | `magnitude`, `mag`, `ml`, `mw`, `buyukluk`, `büyüklük` — **required** |

Dates are parsed with `dayfirst=True`. Rows without a parseable time or magnitude
are dropped. Missing depths are filled with the catalog median (or `10.0`).

**Region selection precedence:** `--region` (one or more) → bbox (all four
present) → `center + radius_km` → whole catalog. Regions supplied via `-r` are
labelled `region1`, `region2`, … in output and in the manifest's `region` column;
the CLI does not expose custom names.

Each region needs at least `2 × window_events` events, else it is skipped.

**Declustering.** `catalog.decluster_gardner_knopoff` flags independent
mainshocks, largest magnitude first. `L_km = 10^(0.1238·M + 0.983)` and
`T_days = 10^(0.032·M + 2.7389)` for `M ≥ 6.5`, `10^(0.5409·M − 0.547)` otherwise.
Two deliberate deviations from the strict 1974 formulation, both driven by real
data:

- The window is applied **symmetrically in time** (`|Δt| ≤ T`), so a foreshock is
  absorbed too, not just aftershocks.
- Only events with `mag <= mag[i]` can be claimed, so a small event cannot demote
  the larger mainshock next to it.

Declustering affects **target selection only**. Dependent events stay in the
window feature sequence.

**Window construction** (`catalog.build_windows`). A window is dropped when:
it contains any raw `M ≥ major_magnitude` event (it would describe the aftermath,
not a precursor state); no target event follows it; or the wait exceeds
`--max-horizon-days`.

**Tensor contents.**

- `seq` `(window_events, 6)` — `magnitude`, `log_dt`, `depth`, `log_energy`,
  `cum_energy_frac`, `dist_km`
- `img` `(3, target_n, target_n)` — RAM of `magnitude`, `log_dt`, `log_energy`
- `aux` `(9,)` — `n_events`, `log_duration_days`, `log_rate`, `mean_mag`,
  `max_mag`, `log_total_energy`, `b_value`, `lyapunov`, `mag_std`

`b_value` is Aki (1965) MLE, `NaN` for fewer than 10 events. `lyapunov` is the
largest Lyapunov exponent by Rosenstein et al. (1993) (embedding dim 4, delay 1,
Theiler window 3), `NaN` when the window is too short to embed.

**Splitting.**

- `chronological` — cuts the time span at `r0` and `r0 + r1`. The primary leak
  defence is **label-aware**: a window whose target event falls beyond its own
  split's boundary is dropped, which is far cheaper than a blanket gap.
  `--embargo-days` adds an *extra* hard gap on top and defaults to none.
- `random` — deliberately leaky; provided only to quantify the leak.
- `loeo` — no split at all; every window goes into a single flat `all/` bucket for
  leave-one-event-out CV at training time. Requires **at least 5** distinct target
  events, else nothing is written. Class boundaries are then derived from the
  whole pool rather than from a train split.

**Risk classes.** Default boundaries are the **train terciles** of
`days_to_major`, and the class *names* are generated from the boundaries in force
via `class_names_for` (e.g. `lt_26d` / `26d_71d` / `gt_71d`), because the fixed
`lt_1y / 1_5y / gt_5y` names were wrong by more than an order of magnitude on the
pooled catalog.

**The diagnostic that matters most** is the count of distinct target events, not
windows — a region with one qualifying event cannot support a split however many
thousands of windows slide out of it. `report_major_events` lists every target
with its date, magnitude and location, reports inter-event gaps, and prints
concrete remediation when there are fewer than four.

```bash
# single region, chronological
seismic-cli generate-catalog-dataset \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir dataset_catalog_marmara \
    --lat-min 40.0 --lat-max 41.5 --lon-min 26.0 --lon-max 30.0 \
    --major-magnitude 4.5 --window-events 64 --stride-events 8

# four pooled fault zones, leave-one-event-out
seismic-cli generate-catalog-dataset \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir dataset_catalog_pooled_loeo \
    -r 39.5,42.0,26.0,42.0 \
    -r 36.5,39.5,35.0,42.0 \
    -r 36.0,40.0,25.0,30.0 \
    -r 34.0,37.5,28.0,36.0 \
    --major-magnitude 4.5 --split-mode loeo

# quantify the leak a random split buys
seismic-cli generate-catalog-dataset \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir /tmp/dataset_catalog_leaky \
    --lat-min 40.0 --lat-max 41.5 --lon-min 26.0 --lon-max 30.0 \
    --major-magnitude 4.5 --split-mode random --seed 7
```

---

### 4.12 `generate-catalog-forecast-dataset`

The **dense** per-zone target: *will an `M ≥ threshold` event occur in this zone
within `horizon_days`?*

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--catalog-path` | str | **required** | Catalog CSV (same parsing as §4.11). |
| `--output-dir` | str | **required** | Dataset root. |
| `--window-events` | int | `64` | Events per window. |
| `--stride-events` | int | `8` | Stride. |
| `--threshold` | float | `4.5` | Magnitude defining a qualifying event. |
| `--horizon-days` | float | `30.0` | Forecast horizon. Also the split embargo width. |
| `--min-magnitude` | float | `2.0` | Drop catalog events below this. |
| `--target-n` | int | `32` | RAM image side length. |
| `--zone`, `-z` | str | all four, repeatable | Restrict to named zones. Validated against `forecast.FAULT_ZONES`. |
| `--train-ratio` | float | `0.7` | Fraction of the time span. |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | Implicit remainder. |

**No `--seed`, no `--decluster`, no `--embargo-days`.** The embargo is always
exactly `horizon_days` at each boundary, which is provably sufficient here because
every label looks forward exactly that far.

**Zones** (`forecast.FAULT_ZONES`, `(lat_min, lat_max, lon_min, lon_max)`):

| Zone | Box | Region |
|---|---|---|
| `NAFZ` | `(39.5, 42.0, 26.0, 42.0)` | North Anatolian Fault Zone |
| `EAFZ` | `(36.5, 39.5, 35.0, 42.0)` | East Anatolian Fault Zone |
| `AEGEAN` | `(36.0, 40.0, 25.0, 30.0)` | Aegean / western Anatolian extension |
| `CENTRAL` | `(34.0, 37.5, 28.0, 36.0)` | Central Anatolia / Cyprus arc |

**Two deliberate differences from `generate-catalog-dataset`**, both load-bearing:

- **No declustering.** A clustered aftershock sequence is exactly the precursor
  signal this target wants, not noise to be defined away.
- **Windows containing an `M ≥ threshold` event are kept.** The label only looks
  at events strictly *after* the window's end, so a large event inside the window
  is a feature (recent activity), not the answer leaking into its own input.

Windows whose horizon runs past the end of the catalog are dropped — the label is
unknowable, and keeping them would silently record "no record" as "no earthquake".

**Tensor contents.** `seq` and `img` are identical in construction to §4.11.
`aux` is `(11,)` — `catalog.DENSE_AUX_FEATURES`: `log_duration_days`, `log_rate`,
`log_rate_recent`, `rate_accel`, `mean_mag`, `max_mag`, `mag_std`,
`log_total_energy`, `log_energy_recent_frac`, `b_value`, `days_since_prev_major`.
The old `n_events` is dropped (constant by construction) and
`days_since_prev_major` added.

The manifest carries a binary `label` and `next_magnitude` (the magnitude of the
first qualifying event inside the horizon, `NaN` exactly when `label == 0`). The
run cross-checks that pairing and prints `OK` or
`MISMATCH -- pairing bug in build_dense_windows`.

```bash
# all four zones, M>=4.5 within 30 days
seismic-cli generate-catalog-forecast-dataset \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir dataset_catalog_forecast_dense \
    --threshold 4.5 --horizon-days 30

# two zones, tighter horizon
seismic-cli generate-catalog-forecast-dataset \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir dataset_forecast_naf_eaf_7d \
    -z NAFZ -z EAFZ --threshold 4.0 --horizon-days 7 --stride-events 4
```

---

### 4.13 `generate-groundmotion-dataset`

Response-corrected PGA/PGV labels for the Nurtas replication.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--eq-dir` | str | **required** | Directory of **60 s raw records** — *not* the anchored 3 s ones. |
| `--catalog-path` | str | **required** | Event catalog CSV (same parsing as §4.9). |
| `--output-dir` | str | **required** | Dataset root; the `train/`, `val/`, `test/` directories are flat. |
| `--cache-dir` | str | `data/station_inventory` | StationXML cache. Populated once from FDSN, then fully offline. |
| `--station-catalog` | str | none | Enables `distance_km`. |
| `--target` | str | `vel` | Stored input tensor: `vel` (cm/s, native) or `acc` (gal). **Not validated by Typer** — a bad value raises `ValueError`. |
| `--label-seconds` | float | `25.0` | Forward label duration after the input window closes. Fixed, so the target does not depend on record length. |
| `--train-ratio` | float | `0.7` | |
| `--val-ratio` | float | `0.15` | |
| `--test-ratio` | float | `0.15` | |
| `--limit-files` | int | none | Process only the first N records (smoke test). |
| `--seed` | int | `42` | Event-key shuffle for the split. |
| `--num-cores` | int | `cpu_count() - 1` | Worker processes. |

**Hard-coded, not configurable.** These must match the values `anchor.py` was run
with to write the anchored corpus, or the re-derived arrival will not correspond
to the stored input window:

```python
PICK_STA_SECONDS = 1.0   PICK_LTA_SECONDS = 10.0
TRIGGER_ON = 3.5         TRIGGER_OFF = 1.0
PRE_ARRIVAL_FRACTION = 0.2   INPUT_SECONDS = 3.0
DIGITIZER_RAIL = 2**23   CLIP_TOL = 0.01   SENS_MISMATCH_TOL = 0.05
```

The FDSN client is hard-coded to `"KOERI"` with a 60 s timeout. A station with no
retrievable response is cached as a `.missing` marker so it is not re-requested.

**What it does per (event, station).**

1. Group traces by `NET.STA`; skip stations with fewer than 3 traces.
2. Replay `anchor.py`'s STA/LTA pick on the 60 s record to recover the arrival
   sample. Stations that do not trigger are skipped — `anchor.py` wrote no input
   window for them either.
3. Input window `[a − 0.6 s, a + 2.4 s]`, reproducing
   `anchor.slice_anchored_window` exactly including its end-of-record clamp.
4. `remove_response` twice (`VEL` and `ACC`) with
   `pre_filt = (0.05, 0.1, 40.0, 45.0)`, `water_level = 60`, 5 % taper, using the
   trace's **real** `starttime` — StationXML responses are epoch-bounded, and a
   placeholder time zeroes the whole dataset.
5. Peaks are taken on the **vector magnitude** `sqrt(Z² + N² + E²)` sample by
   sample, not as the max of per-component peaks.
6. The input tensor is sliced from the *same* deconvolution, so no taper or
   water-level artifact lands on the P onset.

**Two targets, both emitted:**

- `*_fwd` — peak over `[input_end, input_end + label_seconds]`, strictly after
  everything the model saw. A genuine forecast.
- `*_full` — peak over the whole record. The paper's quantity, but it **overlaps
  the input**, so a strong result on it is partly self-prediction.

Roughly 51 % of this corpus peaks at or before the input closes, so the two differ
substantially; `peak_in_input` and `peak_rel_arrival_s` record which rows.

**Splitting is event-disjoint and not optional** — one earthquake at twenty
stations gives twenty correlated targets driven by the same source.

**Rows with no usable response produce no tensor and no manifest row.** They are
counted separately and reported *before* the quality-flag table, because
`response_ok` measured over the manifest alone is 100 % by construction.

**Output tensor** is a bare `torch` tensor of shape `(3, round(3.0 · fs))` —
`(3, 300)` at the corpus's usual 100 Hz — in cm/s or gal. Nothing resamples here,
so a station at another rate produces a differently-shaped tensor.

**Post-run report:** per-split counts, event-leakage check (must be 0), quality
flags, the clean subset (`response_ok` and `sens_mismatch <= 0.05`), target
distributions, and an attenuation fit `log10(target) ~ a·M + b·log10(distance)`.
`a` should land near +1, `b` near −1 for a direct-wave amplitude. `b` comes out
**positive** for the `_fwd` targets, which is expected and annotated
`<- see S-P moveout note` rather than hidden.

```bash
# first run: populates the StationXML cache from FDSN
seismic-cli generate-groundmotion-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_60s \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --station-catalog catalogs/stations.csv \
    --output-dir dataset_groundmotion_vel \
    --cache-dir raw/data/station_inventory \
    --target vel --label-seconds 25

# smoke test on 50 records, acceleration input
seismic-cli generate-groundmotion-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_60s \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir /tmp/gm_smoke \
    --cache-dir raw/data/station_inventory \
    --target acc --limit-files 50
```

---

### 4.14 `eval-sta-lta`

Scores the classic STA/LTA trigger on the **exact** windows a CNN trained on the
dataset would see.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--manifest-path` | str | **required** | A detection `manifest.csv` (§6.1). |
| `--split` | str | `test` | Which split to evaluate. |
| `--window-seconds` | float | `60.0` | **Must match generation.** |
| `--overlap` | float | `0.5` | **Must match generation.** |
| `--sta-seconds` | float | auto | STA length. Auto: `LTA / 10`, floored at `0.05 s`. |
| `--lta-seconds` | float | auto | LTA length. Auto: `window / 3`, capped at `10.0 s`. |

**There is no `--fs` flag.** The CLI passes `fs=100.0`, which is used **only** as
a fallback for manifests that predate the `fs` column. Current manifests carry a
per-row `fs` and it is used.

Auto-derived parameters (`eval_baseline.derive_sta_lta_params`):

| `--window-seconds` | STA | LTA |
|---|---|---|
| 60 | 1.0 | 10.0 |
| 6 | 0.2 | 2.0 |
| 3 | 0.1 | 1.0 |

At 60 s this reproduces the classic 1.0 / 10.0 exactly, so historical
long-window results are unchanged.

> **The auto-derivation is wrong for anchored windows, and the code says so.**
> `classic_sta_lta`'s characteristic function is exactly 0 for its first `nlta`
> samples. `anchor.py`'s default puts the arrival only 20 % into the window, while
> this formula's LTA is 33 % of it — so for any anchored window under ~50 s the
> arrival sits inside the forced-zero warm-up and is invisible. Measured on the
> 6 s anchored dataset: auto (STA 0.2 / LTA 2.0) gives **AUC 0.51**; a
> validation-selected (STA 0.03 / LTA 0.3) gives **AUC 0.82**.
> Pass `--sta-seconds` / `--lta-seconds` explicitly for anything from
> `anchor-windows`, with LTA comfortably under
> `pre_arrival_fraction × window_seconds`. The run prints a `[WARN]` when the
> derived LTA exceeds 15 % of the window.

**Which manifests work.** It requires the columns `split`, `class_name`,
`station_key`, `file_path`, `filename` — i.e. the detection schema written by
`core.run_balanced_preprocessing`. All seven station-split generators produce it,
and the filename regex accepts both `.png` and `.pt`. The regression, riskclass,
catalog and groundmotion manifests have no `class_name` column and will raise a
`KeyError`.

**Reconstruction is not identical to the training input.** The window is rebuilt
from raw counts and only **linearly detrended** before scoring — there is **no
bandpass**, unlike `core.clean_and_filter_1d`. The score is
`max` over the three components of `max(classic_sta_lta(...))`.

`file_path` is stored exactly as it was given at generation time, so this must run
from the same working directory. The command prints a path sanity check at
startup and warns if the sample path does not resolve.

**Reported metrics:** AUC (threshold-free — use this for comparison), then
accuracy / precision / recall at the Youden's-J threshold. That threshold is
chosen **on the evaluated split**, so the thresholded numbers are STA/LTA's upper
bound, not a fair head-to-head.

```bash
# 60 s dataset — auto parameters are correct here (unanchored windows)
seismic-cli eval-sta-lta \
    --manifest-path dataset_60s/manifest.csv \
    --split test --window-seconds 60 --overlap 0.25

# 6 s ANCHORED dataset — auto parameters would score 0.51; set them explicitly
seismic-cli eval-sta-lta \
    --manifest-path dataset_6s_max/manifest.csv \
    --split test --window-seconds 6 --overlap 0.5 \
    --sta-seconds 0.03 --lta-seconds 0.3
```

---

## 5. Module reference

Every file in `seismic_cli/`.

### `__init__.py`
Docstring and `__version__ = "0.1.0"`. Nothing else.

### `cli.py` — 14 Typer commands
The only module that defines CLI surface. Imports `anchor`, `catalog`,
`eval_baseline`, `forecast`, `ram_aux`, `ram_dual`, `regression`, `riskclass`,
`spectrogram` at module level, and `groundmotion` lazily inside its command (it
pulls obspy inventory machinery).

Its own validation, before any work: `--max` vs `--limit-pictures` mutual
exclusion; `--normalize` membership (detection commands only); `--encoding`
membership; `--region` shape; `--split-mode` membership; `--zone` membership.
Everything else is validated downstream and surfaces as a traceback (§9.4).

### `core.py` — shared transform + detection orchestrator

| Symbol | What it is |
|---|---|
| `standardize(x, mu, sigma, eps)` | Z-score against supplied stats, or the window's own. `sigma < 1e-12` is clamped. |
| `reshape_to_target_n(x, n)` | `(d, n)` matrix + depth `d = max(2, ceil(len(x)/n))`, pad/truncate as needed. |
| `ram_matrix(x, n, mu, sigma)` | The RAM transform, §3.7. |
| `to_uint8(mat)` | Clip to ±π, map to `0..255`. |
| `clean_and_filter_1d(x, fs, freqmin, freqmax)` | The signal chain, §3.1. |
| `select_components(available)` | Role-ordered `(z, n, e)` or `None`, §3.3. |
| `window_array_indexed(...)` | Indexed windowing + gap rejection, §3.2. |
| `window_array(...)` | Backwards-compatible wrapper; generation does not use it. |
| `_masked_to_filled(tr_data)` | `(filled_float64, gap_mask)`. |
| `compute_station_noise_baselines(...)` | Parallel amplitude baselines, §3.4. |
| `scan_single_mseed(args)` | Header-only per-`(file, station)` window count. |
| `RamImageEncoder` | Default encoder; RGB PNG. |
| `_window_amplitude_scores(...)` | Loudest component in station-sigma units (raw counts without a baseline). |
| `_pick_from_band(scores, quota, band)` | Evenly spread selection across a percentile slice; widens downward rather than under-filling. |
| `_pick_amplitude_matched(scores, quota, target_scores, n_bins=40, rng)` | Quantile-bin match to the event distribution; redistributes unfillable bins to the nearest that can. |
| `_score_windows_task(args)` | Amplitude-only pre-pass worker. |
| `mseed_file_to_dataset(...)` | Per-file worker: read once, window, clean, encode, return manifest rows. |
| `mseed_file_to_ram_rgb(...)` | Backwards-compatible RAM alias. |
| `_cap_station_windows(...)` | Turns a per-station cap into per-`(station, file)` quotas. |
| `_write_split_manifest(...)` | Detection manifest writer. |
| `_mine_hard_negatives_globally(...)` | Global amplitude ranking → per-file whitelists, §4.7. |
| `run_balanced_preprocessing(...)` | The five-phase detection orchestrator. |

`run_balanced_preprocessing` accepts `max_gap_fraction` (0.05) and
`hard_negative_band` as ordinary parameters; only the latter is exposed as a flag,
and only on `generate-spec-dual-dataset`.

### `anchor.py` — arrival anchoring

`select_pick_traces`, `pick_arrival_with_cft` → `(arrival_sample, max_cft)`,
`pick_arrival_sample` (backwards-compatible wrapper returning only the sample),
`slice_anchored_window`, `process_one_event_file`, `run_anchor_windows`.
Backs `anchor-windows` (§4.1) and is replayed by `groundmotion.py`.

### `ram_aux.py` — RAM + amplitude scalars

`AUX_FEATURES = ["log_snr", "log_rms"]`,
`AUX_FEATURES_PER_COMPONENT = ["log_snr_Z", "log_snr_N", "log_snr_E", "log_rms_Z", "log_rms_N", "log_rms_E"]`,
`RamAuxEncoder`, `RamAuxEncoderV2`.

The image is always built with plain per-window self-standardization; the
station baselines are baked in at construction purely for `log_snr`.

### `ram_dual.py` — `{seq, img}` with a RAM 2D channel

`_resample_to` (polyphase), `_fit_length`, `RamDualEncoder`,
`RamDualAuxEncoder`, `RamDualAuxEncoderV2`.

Implements Wang & Zhao (2025, *Applied Soft Computing* 172:112889) Fig. 7 /
Sec. 3.3.1: the two channels are independent views of the same raw window, not a
shared intermediate. `seq` is the raw standardized `(m, 3)` waveform; the LSTM
branch never sees the RAM reshape. Both branches read the **same** `(mu, sigma)`.

### `spectrogram.py` — spectrogram encoders and geometry checking

`NORMALIZE_MODES`, `stft_geometry`, `suggest_stft_params`,
`check_stft_resolution`, `_resample_to`, `_fit_length`, `SpectrogramEncoder`,
`SpectrogramDualEncoder`, `SpectrogramDualAuxEncoder`,
`SpectrogramDualAuxEncoderV2`, `compute_station_spectral_baselines`.

Geometry constants: `NFFT_WINDOW_FRACTION_WARN = 0.25`,
`NFFT_WINDOW_FRACTION_SEVERE = 0.50`, `MIN_USEFUL_FRAMES = 8`,
`MIN_SUGGESTED_NFFT = 32`.

`SpectrogramDualEncoder` **wraps** a `SpectrogramEncoder` and calls its
`normalize_spec` directly, so there is exactly one copy of the normalization.
Torch is imported lazily per worker, and `torch.set_num_threads(1)` prevents
oversubscription.

### `regression.py` — magnitude regression orchestrator

`EVENT_ID_RE`, `parse_event_id`, `_pick_column`, `load_event_catalog`,
`load_station_coords`, `haversine_km`, `_station_coord`,
`_init_regression_worker`, `_process_regression_file`,
`run_regression_preprocessing`. `parse_event_id`, `load_event_catalog`,
`load_station_coords`, `haversine_km` and `_station_coord` are reused by
`riskclass.py` and `groundmotion.py`.

### `riskclass.py` — three-class risk orchestrator

`RISK_CLASSES = ["00_noise", "01_low_risk", "02_high_risk"]`, `MANIFEST_COLUMNS`,
`_process_earthquake_risk_file`, `_process_noise_risk_file`,
`run_riskclass_preprocessing`.

### `catalog.py` — catalog windowing, both targets

Loading and geometry: `load_catalog`, `filter_region`, `haversine_km`, `_pick`.
Physics: `b_value_aki`, `max_lyapunov_rosenstein`, `energy_joules`.
Declustering: `gardner_knopoff_windows`, `decluster_gardner_knopoff`.
Tercile target: `report_major_events`, `build_windows`, `pool_regions`,
`chronological_split`, `random_split`, `assign_risk_classes`, `class_names_for`,
`_fmt_days`, `encode_and_write`, `run_catalog_dataset`.
Dense target: `DENSE_AUX_FEATURES`, `build_dense_windows`, `pool_dense_zones`,
`dense_chronological_split`, `encode_and_write_dense`,
`run_catalog_forecast_dataset`.
Feature lists: `SEQ_FEATURES` (6), `IMAGE_FEATURES` (3), `AUX_FEATURES` (9),
`RISK_CLASSES` (the *fixed* names, only correct at 1 y / 5 y boundaries).

`chronological_split` accepts a `max_horizon_days` argument that its body never
uses.

### `forecast.py` — zone definitions and the dense-target rationale

`FAULT_ZONES` (§4.12), `FEATURES` (11, mirroring `DENSE_AUX_FEATURES`),
`build_region_windows`, `build_dataset`, `chronological_split`,
`zone_major_times`, `build_blocks`, `_spearman_safe`.

**`cli.py` imports this module solely for `FAULT_ZONES`.** It fits no model —
the sklearn forecaster that once sat on top of it has been retired. `build_blocks`
is the honest evaluation unit for that target: disjoint consecutive
`horizon_days` blocks, each forecast from the last window ending strictly before
it opens, with outcomes read from the catalog rather than inherited from window
labels. Consecutive windows overlap 11–46×, so per-window AUC overstates
confidence by up to ~7× in its standard error.

### `eval_baseline.py` — STA/LTA baseline

`FILENAME_RE`, `MIN_STA_SECONDS = 0.05`, `MAX_LTA_SECONDS = 10.0`,
`derive_sta_lta_params`, `extract_window_index`, `get_window_from_mseed`,
`sta_lta_score`, `run_eval_sta_lta`.

### `groundmotion.py` — peak ground motion

`inventory_path`, `get_inventory`, `sensitivity_mismatch`,
`arrival_sample_for_station`, `_vector_magnitude`, `_log10`, `_peak_in`,
`ground_motion_for_station`, `extract_event_file`, `MANIFEST_COLUMNS` (29),
`_scan_station_count`, `_process_groundmotion_file`,
`run_groundmotion_preprocessing`, `report_groundmotion_manifest`,
`_report_attenuation`.

### `spectrograph.py` — legacy standalone script

Not imported by anything. A thin `main()` wrapper over
`run_balanced_preprocessing` + `SpectrogramEncoder`, configured by module-level
globals (`BASE_WAVEFORMS_DIR`, `NOISE_BATCH_DIR`, `TARGET_DIRECTORIES`, `N_FFT`,
`TOP_DB`, `NOMINAL_FS`, `OVERLAP`, `NORMALIZE`, `GENERATE_MAX`,
`MAX_WINDOWS_PER_STATION`). Its docstring is a list of the defects the standalone
version carried and the shared pipeline fixed. **Prefer
`generate-spectrogram-dataset`.**

### `waveform_diff.py` — recurrence-style probe library

`windowed_diffs`, `process_component`. No CLI command; consumed by the two
root-level scripts in §8.

Two metrics per consecutive **non-overlapping** window pair (`overlap=0.0`
internally). Gap-rejected windows break the chain, so only truly adjacent
indices (`idx == prev_idx + 1`) are paired:

- `raw_diff` — RMS of the elementwise difference between the two cleaned,
  bandpassed, **unnormalized** windows. Preserves amplitude by construction.
- `ram_diff` — mean absolute difference between the two RAM images. Shape-only:
  RAM's scale invariance means it structurally cannot see an amplitude-only
  change.

Returns `{"raw_diff": [...], "ram_diff": [...], "diff_idx": [...], "n_windows": N}`
where `diff_idx` holds the index of the **second** window in each pair.

---

## 6. Manifest schemas

Every generator writes `manifest.csv` at the dataset root.

### 6.1 Detection — all seven station-split commands

`core._write_split_manifest`. 6 columns:

| Column | Meaning |
|---|---|
| `split` | `train` / `val` / `test` |
| `class_name` | `01_earthquake` / `00_noise` |
| `station_key` | `NET.STA` |
| `file_path` | Source mseed path **exactly as given at generation time** (usually relative). |
| `filename` | `<source_stem>_<NET.STA>_win<NNN>.<ext>`. `NNN` is the *original* window index: `start_sample = NNN × step_samples`. |
| `fs` | The station's own sampling rate for this file. |

This is the only schema `eval-sta-lta` accepts.

### 6.2 Regression

9 columns (10 with `--per-component-aux`):

`split`, `station_key`, `event_id`, `file_path`, `filename`, `fs`, `magnitude`,
then `log_snr` **or** `log_snr_0`, `log_snr_1`, `log_snr_2`, then `distance_km`.

`log_snr` is `NaN` where no component had a usable baseline; `distance_km` is
`NaN` without `--station-catalog` or without catalog lat/lon.

### 6.3 Riskclass

`riskclass.MANIFEST_COLUMNS`, 10 columns:

`split`, `risk_class`, `station_key`, `event_id`, `file_path`, `filename`, `fs`,
`magnitude`, `log_snr`, `distance_km`.

Noise rows carry `event_id = ""`, `magnitude = NaN`, `distance_km = NaN`.

### 6.4 Catalog (tercile target)

17 columns:

`split`, `filename`, `region`, `target_time`, `start_time`, `end_time`,
`days_to_major`, `risk_class`, then the 9 `AUX_FEATURES`: `n_events`,
`log_duration_days`, `log_rate`, `mean_mag`, `max_mag`, `log_total_energy`,
`b_value`, `lyapunov`, `mag_std`.

### 6.5 Catalog (dense forecast target)

18 columns:

`split`, `filename`, `region`, `start_time`, `end_time`, `label`,
`next_magnitude`, then the 11 `DENSE_AUX_FEATURES`.

### 6.6 Ground motion

`groundmotion.MANIFEST_COLUMNS`, 29 columns:

| Group | Columns |
|---|---|
| identity | `split`, `event_id`, `station_key`, `filename`, `fs` |
| predictors | `magnitude`, `distance_km` |
| forward targets | `pga_gal_fwd`, `pgv_cms_fwd`, `log_pga_fwd`, `log_pgv_fwd` |
| full-record targets | `pga_gal_full`, `pgv_cms_full`, `log_pga_full`, `log_pgv_full` |
| input-window baseline | `log_peak_input_vel`, `log_peak_input_acc` |
| where the peak sits | `peak_rel_arrival_s`, `peak_in_input`, `arrival_s` |
| label geometry | `label_seconds`, `label_truncated` |
| quality flags | `pick_at_floor`, `max_cft`, `n_masked_frac`, `clipped`, `response_ok`, `sens_mismatch`, `peak_counts` |

Everything a scalar baseline needs is here, so a baseline never has to open a
tensor.

---

## 7. On-disk output formats

### Directory layout

```
# detection (7 commands)                # regression / riskclass / groundmotion
dataset/                                dataset/
├── train/01_earthquake/*.png|pt        ├── train/*.pt|png
├── train/00_noise/*.png|pt             ├── val/*.pt|png
├── val/...   test/...                  ├── test/*.pt|png
└── manifest.csv                        └── manifest.csv

# catalog (both targets)                # catalog, --split-mode loeo
dataset/                                dataset/
├── train/win000000.pt ...              ├── all/win000000.pt ...
├── val/...   test/...                  └── manifest.csv
└── manifest.csv
```

Detection, regression and riskclass filenames are
`<source_stem>_<NET.STA>_win<NNN><ext>`. Ground motion filenames are
`<source_stem>_<NET.STA>.pt` (no window index — one window per station-event).
Catalog filenames are `win<NNNNNN>.pt`, numbered per split.

### Tensor payloads

| Key | Shape | dtype | Notes |
|---|---|---|---|
| `seq` | `(m, 3)` | float32 | `m = round(fs · window_seconds)`; standardized Z/N/E. |
| `img` (RAM) | `(3, target_n, target_n)` | float32 | `[0, 1]`. |
| `img` (spectrogram) | `(3, n_fft//2 + 1, m // hop + 1)` | float32 | dB, after `--normalize`. |
| `aux` | `(2,)` or `(6,)` | float32 | §3.8. |
| catalog `seq` | `(window_events, 6)` | float32 | |
| catalog `img` | `(3, target_n, target_n)` | float32 | uint8-quantized, then `/255`. |
| catalog `aux` | `(9,)` or `(11,)` | float32 | tercile vs dense target. |
| groundmotion | `(3, round(3.0 · fs))` | float32 | bare tensor, cm/s or gal. |
| spectrogram-only | `(3, F, T)` | float32 | bare tensor, not a dict. |

Verified against datasets in this tree: `dataset_specdual_ponly_3p4s_hard`
→ `seq (340, 3)`, `img (3, 33, 22)` (3.4 s, `n_fft 64`, `hop 16`);
`dataset_magclass_dual_6s` → `seq (600, 3)`, `img (3, 65, 19)` (6 s, `n_fft 128`,
`hop 32`); `dataset_magreg_catalog_6s` → `seq (600, 3)`, `img (3, 129, 10)`
(6 s, defaults).

`generate-dataset`'s PNGs are RGB `target_n × target_n`, R = Z, G = N-ish,
B = E-ish.

---

## 8. Non-CLI entry points

Two root-level scripts consume `seismic_cli.waveform_diff`. Neither takes
arguments — configuration is module-level globals.

### `waveform_diff_probe.py`

```bash
python waveform_diff_probe.py
```

Compares `raw_diff` / `ram_diff` at 3 h vs 6 h before each event, paired per
`(event, station)`, with a Wilcoxon signed-rank test overall and per magnitude
tercile. Globals: `WINDOW_LENGTHS = (50.0, 145.0)`, `FREQMIN/FREQMAX = 1.0/45.0`,
`CATALOG_PATH = "catalogs/deprem_katalog_utc.csv"`,
`NOISE_3H`/`NOISE_6H` under `data/batched_noise_waveforms/`.
Writes `waveform_diff_probe_results.csv`.

The comparison is 3 h-vs-6 h rather than a dense trend because those directories
hold a single 5-minute snapshot at each fixed offset, not continuous recordings.
The script prints its own caveat: there is no non-precursor control, so a positive
result shows at most "closer-to-event windows look different from further-out
ones".

### `waveform_diff_trend.py`

```bash
python waveform_diff_trend.py
```

Dense follow-up on `data/batched_waveforms/day_before_24h` (15 pilot events,
`catalogs/pilot_continuous_sample.csv`). Computes the full diff sequence at 50 s
resolution across ~24 h before each event, Spearman-correlates it against
time-to-event per `(event, station)`, then tests the sign of those correlations
across series with a one-sample Wilcoxon. Fixed 8 worker processes. Writes
`waveform_diff_trend_results.csv`. Same control caveat.

---

## 9. Known defects and traps

Verified against the code (and, where noted, against real data in this tree)
during this pass. The first two are **wrong runtime output**, not just wrong docs.

### 9.1 `catalog.py` prints an all-zero class histogram whenever boundaries are derived

`encode_and_write` builds its per-split class-balance line from the module-level
`RISK_CLASSES = ["lt_1y", "1_5y", "gt_5y"]`, but `assign_risk_classes` labels
windows with names generated from the boundaries actually in force
(`class_names_for`). With the default tercile boundaries the two never match, so
the printed distribution reads `lt_1y=0  1_5y=0  gt_5y=0`. The `majority=` figure
and the written manifest are correct; only that one line is wrong.
*(Affects `generate-catalog-dataset` only; the dense path prints a positive rate
instead and is unaffected.)*

### 9.2 `groundmotion.py`'s `n_masked_frac` is always 0

`extract_event_file` merges with `fill_value="interpolate"`, which leaves no
masked arrays, so `core._masked_to_filled` always reports a zero mask. Measured on
this tree's `window_post_60s` corpus: 0 masked traces after
`merge(fill_value="interpolate")` versus 2 after a plain `merge()`, and every
emitted row carried `n_masked_frac = 0`. The peak amplitudes are still safe —
gaps are interpolated, not sentinel-filled — but the quality **column** carries no
information.

### 9.3 Hard-negative mining silently degrades without `--baseline`

`core._window_amplitude_scores` divides each window's amplitude by its
`(station, component)` noise sigma **only when a baseline exists**, and
`run_balanced_preprocessing` leaves `station_baselines` empty unless `--baseline`
is set. Without it the global cross-station ranking runs on raw counts, where
"loudest" mostly means "highest-gain instrument" — the exact failure the
function's own docstring warns about. Nothing warns at runtime. **Pass
`--baseline` alongside `--hard-negatives`.**

Relatedly, when `--match-negative-amplitude` is set the startup banner still
prints the `--hard-negative-band` percentages even though the band is ignored.

### 9.4 Four flags are validated too late for a clean error

`--split-by` (regression), `--normalize` (regression **and** riskclass — the
detection commands *do* validate it), and `--target` (groundmotion) raise
`ValueError` as a Python traceback rather than Typer's `Invalid value` box. All
four fail before any work is done, so nothing is written.

### 9.5 `anchor.py` does not advance `stats.starttime`

Anchored windows carry their parent record's start time, so absolute arrival time
is unrecoverable from them. Harmless for the classifiers (they never use absolute
time); it is why `groundmotion.py` re-derives the pick from the 60 s record.

### 9.6 `eval-sta-lta` scores unfiltered data

Reconstructed windows are linearly detrended but **not** bandpassed, while the
CNN's input goes through the full `clean_and_filter_1d` chain. The baseline and
the model therefore see different preprocessing on the same samples.

### 9.7 Auto-derived STA/LTA parameters are wrong for anchored windows

See the box in §4.14. AUC 0.51 versus 0.82 on the 6 s anchored dataset.

### 9.8 Repeated noise scans

`generate-dual-aux-dataset` and `generate-spec-dual-aux-dataset` scan the noise
corpus once for the aux baselines and again inside `run_balanced_preprocessing`
when `--baseline` is set. `generate-spec-dual-aux-dataset` with
`--normalize station` also scans it a third time for the spectral profiles (which
is single-threaded).

### 9.9 Split ratios are neither validated nor normalised

Nothing checks that `--train-ratio + --val-ratio + --test-ratio == 1.0`. In the
non-`--max` detection path the test split silently absorbs the remainder and can
go negative.

### 9.10 `--max` balancing works on estimates

Class balance is computed from header-scan window counts. Realised counts drift
where windows are rejected at generation time (gaps, header-vs-merged length
differences). The manifest is the ground truth for what was written.

### 9.11 Structural limits worth knowing

- **Noise-station diversity is the binding constraint.** The risk dataset can run
  on single-digit noise-station counts per split, few enough that one faulty
  instrument distorted an entire result.
- **`riskclass` cannot enforce station- and event-disjointness at once.** It
  enforces the station rule and *measures* event overlap.
- **The catalog task's real sample size is the distinct target-event count**, not
  the window count.

---

## 10. End-to-end recipes

### 10.1 Detection, 6 s anchored windows, with a fair baseline

```bash
# 1. anchor 6 s windows on the P arrival, from the 60 s corpus
seismic-cli anchor-windows \
    --source-dir raw/data/batched_waveforms/window_post_60s \
    --output-base-dir raw/data/batched_waveforms \
    -t 6

# 2. build the largest balanced dataset the data supports
seismic-cli generate-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_anchored \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_6s_max \
    --window-seconds 6 --overlap 0.5 \
    --max --max-windows-per-station 20

# 3. score STA/LTA on the identical test windows.
#    ANCHORED windows: set STA/LTA explicitly (see 4.14).
seismic-cli eval-sta-lta \
    --manifest-path dataset_6s_max/manifest.csv \
    --split test --window-seconds 6 --overlap 0.5 \
    --sta-seconds 0.03 --lta-seconds 0.3
```

### 10.2 Dual-channel spectrogram detection with amplitude-matched negatives

```bash
seismic-cli generate-spec-dual-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_3.4s_ponly \
    --noise-dir raw/data/batched_noise_waveforms \
    --output-dir dataset_specdual_ponly_3p4s_matched \
    --window-seconds 3.4 --overlap 0.5 --fs 100 \
    --n-fft 64 --hop-length 16 --normalize station \
    --max --max-windows-per-station 20 \
    --baseline --hard-negatives --match-negative-amplitude
```

`--baseline` is not optional here in practice: it is what puts the amplitude
ranking into station-sigma units (§9.3).

### 10.3 Magnitude regression, event-disjoint

```bash
seismic-cli generate-regression-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_6s_catalog \
    --noise-dir raw/data/batched_noise_waveforms \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --station-catalog catalogs/stations.csv \
    --output-dir dataset_magreg_catalog_6s \
    --encoding spectrogram --dual \
    --window-seconds 6 --n-fft 128 --hop-length 32 \
    --split-by event --seed 42
```

Check the `[leakage]` block: with `--split-by event` the events are disjoint by
construction and shared stations are expected.

### 10.4 Catalog forecasting on the validated dense target

```bash
seismic-cli generate-catalog-forecast-dataset \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir dataset_catalog_forecast_dense \
    --threshold 4.5 --horizon-days 30 \
    --window-events 64 --stride-events 8
```

Read the per-split positive rate before anything else — that is the floor a model
has to beat.

### 10.5 Peak ground motion

```bash
seismic-cli generate-groundmotion-dataset \
    --eq-dir raw/data/batched_waveforms/window_post_60s \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --station-catalog catalogs/stations.csv \
    --output-dir dataset_groundmotion_vel \
    --cache-dir raw/data/station_inventory \
    --target vel --label-seconds 25
```

Headline `*_fwd`, not `*_full`: `*_full` overlaps the model's own input. Then read
the attenuation fit — `b` near −1 on `log_pgv_full` confirms the coordinates and
the response correction; a positive `b` on the `_fwd` targets is the expected S-P
moveout artefact, not a bug.
