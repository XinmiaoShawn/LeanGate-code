# Prepared Dataset Layouts

This public repo supports only prepared layouts. It does not convert raw official downloads into these layouts.

## TUM
Each scene must live directly under the dataset root:

```text
<dataset-root>/
  rgbd_dataset_freiburg1_desk/
    rgb.txt
    groundtruth.txt
    camera_640.yaml
    rgb/
      000000.png
      ...
```

Accepted calibration filenames:
- `camera_640.yaml`
- `calib.yaml`
- `camera.yaml`
- `camera.txt`

Accepted GT filenames:
- `groundtruth.txt`
- `gt.txt`
- `traj_gt_tum.txt`

## 7Scenes
The repo discovers any scene directory that contains `rgb.txt`:

```text
<dataset-root>/
  chess/
    seq-01/
      rgb.txt
      groundtruth.txt
      calib.yaml
      rgb/
        frame-000000.color.png
        ...
```

The discovered scene id is the relative path of the directory that owns `rgb.txt`, for example `chess/seq-01`.

## EuRoC
Prepared EuRoC supports either the nested `mav0/cam0` layout or a flattened export:

```text
<dataset-root>/
  machine_hall/
    MH_01_easy/
      mav0/
        cam0/
          rgb.txt
          calib.yaml
          groundtruth.txt
          data/
            1403636579763555584.png
            ...
```

or:

```text
<dataset-root>/
  MH_01_easy/
    rgb.txt
    calib.yaml
    groundtruth.txt
    data/
      1403636579763555584.png
      ...
```

Public CLI scene ids use the short sequence name such as `MH_01_easy`.
