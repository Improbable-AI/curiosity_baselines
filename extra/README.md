# Custom Gym files

To change the default camera that gets rendered first, we need to add a few lines to `fetch_env.py`.  Copy this file to `gym/envs/robotics/fetch_env.py`. Specifically, the lines added are:

```python
# Line 5
from mujoco_py.generated import const

# Line 127
self.viewer.cam.fixedcamid = 3
self.viewer.cam.type = const.CAMERA_FIXED
```

To change the camera placement, we can adjust the cameras that are defined in `robot.xml`.  This should be placed at `gym/envs/robotics/assets/fetch/robot.xml`.  The changes are noted by comments on the camera objects.

