# Third-Party Notices

This repository's top-level code is released under Apache-2.0. The `third_party/`
directories are upstream projects kept as Git submodules and remain governed by
their own licenses and notices.

## Submodules

- `third_party/FLARE/`
  - Upstream: https://github.com/ant-research/FLARE
  - See `third_party/FLARE/LICENSE.txt` and `third_party/FLARE/LEGAL.md`
- `third_party/MASt3R-SLAM/`
  - Upstream: https://github.com/rmurai0610/MASt3R-SLAM
  - See `third_party/MASt3R-SLAM/LICENSE.md`

## Important license boundary

The FLARE tree includes upstream MASt3R / DUSt3R derived files that carry their
own notices, including non-commercial restrictions in some files. For example:

- `third_party/FLARE/dust3r/dust3r/model.py`
- `third_party/FLARE/dust3r/dust3r/heads/dpt_head.py`
- `third_party/FLARE/mast3r/catmlp_dpt_head.py`
- `third_party/FLARE/mast3r/shallow_cnn.py`

Review the license headers and upstream license files before redistributing or
using those components, checkpoints, or derived artifacts.
