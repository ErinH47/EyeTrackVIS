# EyeTrackVIS

Automatic Control Theory@PKU 2022 Spring, Eye Tracking, Visualization, Machine Learning
by hq

**collect data:**

```shell
python collect_data.py
```

Images will be saved in the `"image/"` directory.

**train model:**

```shell
python train.py
```

Note that you should set your screen size by `height` and `width` before training.
The model will be saved as `"eye_track_model"`.

**test model:**

```shell
python prediction.py
```

It will load the model, and you can move the mouse with your eyes live.

