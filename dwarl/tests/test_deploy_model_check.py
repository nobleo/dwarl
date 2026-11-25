"""Deploy compares the compiled robot model, so a newer MuJoCo writing the same model differently passes."""

from dwarl.workflows.deploy import model_difference

MODEL = """<mujoco><option timestep="{dt}"/><worldbody><body name="base" pos="{x} 0 0">
<joint name="wheel" type="hinge"/><geom name="chassis" type="box" size="{size} 0.1 0.05"/></body></worldbody>
<actuator><general joint="wheel" {extra} biastype="affine" biasprm="0 0 -1"/></actuator></mujoco>"""


def _write(path, dt="0.002", x="0", size="0.2", extra=""):
    path.write_text(MODEL.format(dt=dt, x=x, size=size, extra=extra))
    return path


def test_same_model_written_differently_passes(tmp_path):
    trained = _write(tmp_path / "trained.xml")
    assert model_difference(trained, _write(tmp_path / "noise.xml", x="9.02056e-16")) is None


def test_velocity_input_label_passes(tmp_path):
    trained = _write(tmp_path / "trained.xml")
    assert model_difference(trained, _write(tmp_path / "labelled.xml", extra='input="vel"')) is None


def test_real_changes_are_reported(tmp_path):
    trained = _write(tmp_path / "trained.xml")
    assert model_difference(trained, _write(tmp_path / "bigger.xml", size="0.25")) is not None  # first field
    assert model_difference(trained, _write(tmp_path / "slower.xml", dt="0.004")) == "opt.timestep"
    assert model_difference(trained, tmp_path / "missing.xml") == "missing.xml"


def test_collision_mask_bits_are_compared_exactly(tmp_path):
    trained = _write(tmp_path / "trained.xml")
    target = _write(tmp_path / "target.xml")
    trained.write_text(trained.read_text().replace('name="chassis"', 'name="chassis" contype="1073741824"'))
    target.write_text(target.read_text().replace('name="chassis"', 'name="chassis" contype="1073741825"'))
    assert model_difference(trained, target) in {"body_contype", "geom_contype"}
