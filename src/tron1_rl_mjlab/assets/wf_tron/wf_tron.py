from pathlib import Path
import mujoco

from mjlab.actuator import XmlPositionActuatorCfg, XmlVelocityActuatorCfg
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

current_dir: Path = Path(__file__).parent.resolve()

WF_TRON_XML: Path = current_dir / "xml" / "robot.xml"
assert WF_TRON_XML.exists(), f"XML file not found: {WF_TRON_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(WF_TRON_XML))


WF_TRON_LEG_ACTUATORS = XmlPositionActuatorCfg(
    target_names_expr=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"),
)

WF_TRON_WHEEL_ACTUATORS = XmlVelocityActuatorCfg(
    target_names_expr=("wheel_[RL]_Joint",),
)

WF_TRON_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        WF_TRON_LEG_ACTUATORS,
        WF_TRON_WHEEL_ACTUATORS,
    ),
)

WF_TRON_CONTACT_SENSOR = ContactSensorCfg(
    name="contact_sensors",
    primary=ContactMatch(mode="body", pattern="base_Link", entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=10,
)

WF_TRON_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8 + 0.166),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    articulation=WF_TRON_ARTICULATION,
)

if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.scene import SceneCfg, Scene
    from mjlab.terrains import TerrainEntityCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": WF_TRON_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")

    viewer.launch(scene.compile())
