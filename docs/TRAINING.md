# Training

## Dataset setup

DeT follows the JamMa data pipeline. Two parts of data are needed for training: the original datasets, such as ScanNet and MegaDepth, and the offline generated dataset indices. The indices store scenes, image pairs, and other metadata used for training, validation, and testing. For MegaDepth, relative poses between training images are cached in the indexing files. For ScanNet, relative poses are not stored because the resulting index files would be very large.

### Download datasets

#### MegaDepth

We use depth maps provided in the [original MegaDepth dataset](https://www.cs.cornell.edu/projects/megadepth/) as well as undistorted images, camera intrinsics, and camera extrinsics preprocessed by [D2-Net](https://github.com/mihaidusmanu/d2-net#downloading-and-preprocessing-the-megadepth-dataset). Download them separately from the following links.

- [MegaDepth undistorted images and processed depths](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz)
  - Only the depth maps are used.
  - This path is referred to as `/path/to/megadepth` below.
- [D2-Net preprocessed images](https://drive.google.com/drive/folders/1hxpOsqOZefdrba_BqnW490XpNX_LgXPB)
  - Images are undistorted manually in D2-Net because the undistorted images from MegaDepth do not come with corresponding intrinsics.
  - This path is referred to as `/path/to/megadepth_d2net` below.

#### ScanNet

Set up ScanNet following [the official guide](https://github.com/ScanNet/ScanNet#scannet-data).

> Note: We use the [Python exported data](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python), not the C++ exported data.

### Download dataset indices

Download the required dataset indices from the [LoFTR/JamMa data link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf). After downloading, unzip the required files.

```bash
unzip downloaded-file.zip

tar xf train-data/megadepth_indices.tar
tar xf train-data/scannet_indices.tar

# Optional testing data
tar xf testdata/megadepth_test_1500.tar
tar xf testdata/scannet_test_1500.tar
```

### Build dataset symlinks

Symlink the datasets to the `data` directory under the project root.

```bash
# ScanNet train/test data
ln -s /path/to/scannet_train/* /path/to/project/data/scannet/train
ln -s /path/to/scannet_test/* /path/to/project/data/scannet/test
ln -s /path/to/scannet_indices/* /path/to/project/data/scannet/index

# MegaDepth train/test data
ln -sv /path/to/megadepth/phoenix /path/to/megadepth_d2net/Undistorted_SfM /path/to/project/data/megadepth/train
ln -sv /path/to/megadepth/phoenix /path/to/megadepth_d2net/Undistorted_SfM /path/to/project/data/megadepth/test
ln -s /path/to/megadepth_indices/* /path/to/project/data/megadepth/index
```

### Run training

```bash
bash scripts/reproduce_train/outdoor.sh
```

Checkpoints and logs are local artifacts and are ignored by Git.
