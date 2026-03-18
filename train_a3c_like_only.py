from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from pentest_env_complex import PentestEnvComplex


def main():
    print("Training A3C-like surrogate (A2C-based) ...")
    env = Monitor(PentestEnvComplex())
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./complex_tensorboard/",
        n_steps=32,
        ent_coef=0.02,
        learning_rate=7e-4,
    )
    model.learn(total_timesteps=80_000)
    model.save("models_complex/a3c_like_complex")
    print("Saved: models_complex/a3c_like_complex.zip")


if __name__ == "__main__":
    main()
