import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from .robot_cfg import HUAN_ROBOT_CFG

# Configuration 
ACTION_JOINTS_13 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
]

LEFT_FOOT_BODY = "left_foot_pitch_link"
RIGHT_FOOT_BODY = "right_foot_pitch_link"
UNDESIRED_CONTACT_BODIES = ["base_link"]


@configclass
class StandSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = HUAN_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        debug_vis=False,
    )

# Joint Limits and Action Scaling 
ACTIONS_MAX_13 = [2.618, 1.571, 1.571, 2.356, 0.436, 0.785, 2.618, 0.262, 1.571, 2.356, 0.436, 0.785, 2.618]
ACTIONS_MIN_13 = [-2.618, -0.262, -1.571, -0.087, -0.436, -0.785, -2.618, -1.571, -1.571, -0.087, -0.436, -0.785, -2.618]

# Nominal standing pose (degrees)
DEFAULT_JOINT_ANGLES = [
    -14.0, 0.0, 0.0, 29.5, 0.0, -13.7,  
    -14.0, 0.0, 0.0, 29.5, 0.0, -13.7,
    0.0
]

ACTION_SCALE_BY_JOINT = {}
for i, j in enumerate(ACTION_JOINTS_13):
    q0 = math.radians(DEFAULT_JOINT_ANGLES[i])
    s_pos = ACTIONS_MAX_13[i] - q0
    s_neg = q0 - ACTIONS_MIN_13[i]
    ACTION_SCALE_BY_JOINT[j] = float(max(1e-6, min(s_pos, s_neg)))



@configclass
class StandActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTION_JOINTS_13,
        scale=ACTION_SCALE_BY_JOINT,
        use_default_offset=True,
    )


@configclass
class StandObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Standard observations to the policy."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.zero_velocity_command, params={"dim": 3})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
        """Extra info to help the Critic learn faster"""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.zero_velocity_command, params={"dim": 3})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
        )
        actions = ObsTerm(func=mdp.last_action)

        base_height = ObsTerm(
            func=mdp.base_height_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        feet_contact = ObsTerm(
            func=mdp.feet_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[LEFT_FOOT_BODY, RIGHT_FOOT_BODY],
                ),
                "force_threshold": 1.0,
            },
        )
        feet_speed_xy = ObsTerm(
            func=mdp.feet_speed_xy_obs,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[LEFT_FOOT_BODY, RIGHT_FOOT_BODY],
                ),
            },
        )

        feet_height = ObsTerm(
            func=mdp.feet_height_obs, 
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_FOOT_BODY, RIGHT_FOOT_BODY])},
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class StandEventsCfg:
    """Random actions that happens during training"""
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (0, 0),
            },
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0), 
        params={
            "velocity_range": {
                "x": (-0.3, 0.3), 
                "y": (-0.3, 0.3), 
            }
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.97, 1.03), 
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13),
        },
    )


@configclass
class StandRewardsCfg:
    """Reward terms for encouraging stable standing and penalize instability"""

    # Primary Objectives (Positive)
    alive = RewTerm(func=mdp.is_alive, weight=2.0) # Reward for not falling over
    # Stay vertical
    upright = RewTerm( 
        func=mdp.upright_reward, 
        weight=10.0, 
        params={"asset_cfg": SceneEntityCfg("robot")} 
    )
    # Keep that head high
    height = RewTerm(
        func=mdp.root_height_reward,
        weight=6.0,
        params={"target_height": 0.70, "asset_cfg": SceneEntityCfg("robot")},
    )
    # Try to stick to the default standing pose
    pose = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
    )

    # Regularization (Negative) 
    # Penalize excessive motion
    # Big penalty if you mess up and fall
    terminating = RewTerm(func=mdp.is_terminated, weight=-50.0)
    # Stop drifting around the floor
    lin_vel = RewTerm(
        func=mdp.root_lin_vel_error_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # Stop spinning like a top
    ang_vel = RewTerm(
        func=mdp.root_ang_vel_error_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # No shaky
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
    )
    # Keep the control commands smooth
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Penalize if anything but the feet touches the ground
    feet_still = RewTerm(
        func=mdp.feet_speed_xy_penalty,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[LEFT_FOOT_BODY, RIGHT_FOOT_BODY],
            ),
        },
    )

    # Penalize contacts on undesired bodies to prevent the robot from using its base for support or bracing during falls
    undesired_contact = RewTerm(
        func=mdp.undesired_contact_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=UNDESIRED_CONTACT_BODIES,
            ),
            "force_threshold": 1.0,
        },
    )

    # Don't twist the waist too much
    waist_yaw_hold = RewTerm(
        func=mdp.waist_yaw_abs_l2,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"]),
        },
    )


@configclass
class StandTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Terminate if the robot base falls below 0.5 meters
    fall = DoneTerm(
        func=mdp.fall_by_height,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.5},
    )

    # Terminate if the robot tilts more than ~57 degrees (1.0 rad)
    bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={
                "asset_cfg": SceneEntityCfg("robot"), 
                "limit_angle": 1.0 # 0.65 * 3.14159 = ~2.04 radian
            },
        )

    # Hard reset if the torso hits the ground
    undesired_contact = DoneTerm(
        func=mdp.undesired_contact_termination,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=UNDESIRED_CONTACT_BODIES,
            ),
            "force_threshold": 1.0,
        },
    )


@configclass
class HuanStandEnvCfg(ManagerBasedRLEnvCfg):
    scene: StandSceneCfg = StandSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: StandObservationsCfg = StandObservationsCfg()
    actions: StandActionsCfg = StandActionsCfg()
    rewards: StandRewardsCfg = StandRewardsCfg()
    events: StandEventsCfg = StandEventsCfg()
    terminations: StandTerminationsCfg = StandTerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 0.001
        self.decimation = 10
        self.episode_length_s = 20.0 #stand for 20 seconds
        self.sim.render_interval = self.decimation

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        