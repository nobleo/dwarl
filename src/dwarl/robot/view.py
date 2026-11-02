"""Grounded inspection scene, separate from the saved training robot."""

import mujoco

from dwarl.robot.model import inspect_model


def inspection_model(xml):
    spec = mujoco.MjSpec.from_string(xml)
    geometry = inspect_model(spec.compile())
    spec.worldbody.add_geom(name="inspection_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[5, 5, 0.1], rgba=[0.65, 0.7, 0.75, 1])
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
    spec.stat.extent = 2.0
    spec.stat.center = [0, 0, 0.3]
    model = spec.compile()
    data = mujoco.MjData(model)
    address = model.joint("root").qposadr[0]
    data.qpos[address + 2] = geometry.spawn_height
    mujoco.mj_forward(model, data)
    return model, data
