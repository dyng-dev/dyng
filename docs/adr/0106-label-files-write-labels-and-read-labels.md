# ADR 0106: Label files: `io::write_labels` and `io::read_labels` in LabelPropagation-CUDA's bytes (AD13 D2)

- **Status:** Accepted (2026-11-11). The format is the author's: **AD13 (sheet D2), decided by the
  author on 2026-10-28** (GOVERNANCE.md's approvals log, the row "The file formats of 0.4 (sheet
  D1-D2)": `io::write_labels` writes the original's `i %.6f` bytes). The names are AD12's
  (`io::write_labels` and `read_labels`; the row "The new public names of 0.4"). The reader's
  rules and the tests are accepted under delegation, technical scope only. M10b step 2 (the M10b
  acceptance criterion 6).
- **Date:** 2026-11-11
- **Deciders:** S M Shovan (AD13 D2, AD12); the AI assistant, on the author's behalf (the rest)

## Context

LabelPropagation-CUDA@a276a3a (O) writes its labels as `dynlp_labels.txt` with
`writeLabelsToFile` (src/utils.cu): per vertex `i`, one space, the label through its TextWriter's
`putFixed6` (`std::to_chars(double(x), chars_format::fixed, 6)`), `\n`. M10a reproduced it in an
internal writer (`detail::write_dynlp_labels`, ADR 0067) that the compat tool and the replays use;
the damped replays compare its bytes with O's goldens.

## Decision

1. `void io::write_labels(const std::string& path, array_view<const float> labels)` in
   `<dyng/io/result_io.hpp>`: the bytes of O's writer (dynG's `text_writer::put_fixed6` is O's
   `putFixed6`), vertices 0 .. n-1 in order, the parent directory created, host memory required
   (`invalid_argument_error`), `io_error` on a write failure. `to_chars(double, fixed, 6)` is
   correctly rounded with ties to even, the bytes of `printf("%.6f")` (0 `0.000000`, 1
   `1.000000`, 0.5 `0.500000`, the exact tie 0.0078125 `0.007812`); a non-finite label is written
   as to_chars writes it.
2. `std::vector<float> io::read_labels(const std::string& path, std::int64_t num_vertices)`: the
   reader shape of `read_distances` (every vertex exactly once in any order, a fast pass and a
   strict pass that reports the first problem with its line and column), the label a decimal
   token read by `std::from_chars<float>` (also `nan` / `inf`, which the writer produces for
   non-finite labels), outside the float range refused. A label reads back within 5e-7 plus half
   a float ulp of the written float; the file, not the float, is the exchange format.
3. There is one implementation: the internal `detail::write_dynlp_labels` forwards to
   `io::write_labels`, and `dyng-compat-dynlp` writes `dynlp_labels.txt` and `irlp_labels.txt`
   with it (keeping O's console message when the file cannot be opened). The compat tool's outputs
   are therefore unchanged by construction; the damped replays (byte parity of
   `dynlp_labels.txt` on P) check it against O's goldens.

## Consequences

- `Labels.WriteLabelsIsPrintfFixed6ByteForByte` checks 0, 1, 0.5, every k/128 tie and its
  neighbours and 2,000 random labels against `printf("%.6f")`; the damped replays check O's bytes.
- `fuzz_result_io` reads label files and checks the write-read round trip within that bound.
- Python's `dyng.io.write_labels` / `read_labels` and the CLI's label output (M10b step 3) use
  these functions.

## Alternatives considered

- **Shortest round-trip output** (the float's bits back exactly). Rejected by AD13 D2: the file is
  for comparison with the original's, byte for byte.
