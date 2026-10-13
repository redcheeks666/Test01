# 无损二进制分片清单

大于 95,000,000 字节的原件按每片 94,371,840 字节（90 MiB）保存；每片小于 95 MB。未转码、未压缩。

在文件所在目录恢复，例如：

```sh
cat example.mp4.part[0-9][0-9] > example.mp4
shasum -a 256 -c example.mp4.sha256
```

原件路径以仓库根目录为基准。sha256 sidecar 使用原件文件名，便于在相同目录校验。

| 原路径 | 原大小（字节） | 分片数 | SHA-256 |
|---|---:|---:|---|
| assets/NA/NA-04/NA-04_02_iss_view_of_earth_nasa.mp4 | 112793370 | 2 | 2a1a22185d14ca07176e6c2d071874ab5a091d4687ebd502316161136f560362 |
| assets/NA/NA-15/NA-15_01_artemis2_launch_4k_nasa.mp4 | 276763221 | 3 | 21d9be4dd4379838e63849a6abec02a6c4b428c45e03ba73ab60278b2b92027f |
| assets/NA/NA-15/NA-15_02_orion_liftoff_camera_nasa.mp4 | 158709167 | 2 | da9bac305571fae1ee02f466a95a386789268cc252f6772694185461ba35b698 |
| assets/PD/PD-11/PD-11_01_wright_first_flight_loc.tif | 243313546 | 3 | c2df8c6ca0ab1cda3577e44573f001ec41b1a5a07acf32ef33a7cf8140e96d21 |
| assets/SX/SX-01/SX-01_03_flight14_prelaunch_payload_spacex.mp4 | 200434544 | 3 | 8d6a538b489f4a0f26a581b670d6369cab4e1fdda4fbcc92630581a7e06dcb44 |
| assets/SX/SX-01/SX-01_04_flight14_prelaunch_pad_spacex.mp4 | 632820737 | 7 | c5d775e317572c6267d2263adf22e6faa867d0552bc96fd02578590ec789909c |
| assets/SX/SX-01/SX-01_05_flight14_hot_staging_spacex.mp4 | 102234401 | 2 | 30dc0bbe451285fee6e0e78962e674ebe56002daa22a8ccd5f61cfaf44f87703 |
| assets/SX/SX-03/SX-03_03_flight14_payload_deploy_start_spacex.mp4 | 113707733 | 2 | 892b50327ba8c595c1f60d88e23316fe46868ae44b17d48e40bcd32f51dd2f72 |
| assets/SX/SX-03/SX-03_07_flight14_deorbit_complete_spacex.mp4 | 158539467 | 2 | f7eee09999adfbb49f665df33272e0decbfdf93cecdde23ea2f772cc0c98cd2a |
| assets/SX/SX-03/SX-03_09_flight14_splashdown_spacex.mp4 | 152029344 | 2 | d92346e1ab2686c6fd2b4489b9fa691d7fb31444d346484c0aa4c887040d2053 |
| assets/SX/SX-01/SX-01_12_flight14_liftoff_cut01_spacex.mp4 | 180911537 | 2 | 02dbccbbae88d6a8b15f4abd6747db0b21a8a1903a5c1c4fd26526e49e4e1949 |
| assets/SX/SX-03/SX-03_11_flight14_earth_limb_cut01_spacex.mp4 | 148443361 | 2 | cdc3035359bf139cbd7f87bcd6379acda6bc062edba53a80827b14bb3f0e5718 |
| assets/SX/SX-03/SX-03_12_flight14_deployment_cut02_spacex.mp4 | 264166797 | 3 | 92d509328d1e232f28c43a082ac0f010c71e90ee700becefe665364a88a40bfa |
