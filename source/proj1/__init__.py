from gymnasium.envs.registration import register

register(
    id="Isaac-HuanPJ1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "proj1.stand_env_cfg:HuanStandEnvCfg",
        "rsl_rl_cfg_entry_point": "proj1.agents.rsl_rl_ppo_cfg:HumanoidN1PPORunnerCfg",
    },
)
