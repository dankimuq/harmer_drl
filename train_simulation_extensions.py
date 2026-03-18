"""
train_simulation_extensions.py

Train only the newly added simulation agents:
  - DPPO-like
  - GAIL-like

Usage:
    source .venv/bin/activate
    python train_simulation_extensions.py
"""

from simulation_extensions import train_dppo_like, train_gail_like


def main():
    print("[1/2] Training DPPO-like ...")
    train_dppo_like(total_timesteps=80_000, n_envs=8, save_name="dppo_complex")
    print("   Saved -> models_complex/dppo_complex.zip\n")

    print("[2/2] Training GAIL-like ...")
    train_gail_like(total_rounds=120, bc_epochs=20, disc_updates=6, policy_updates=4, save_name="gail_complex.pt")
    print("   Saved -> models_complex/gail_complex.pt\n")

    print("All simulation extension models trained.")


if __name__ == "__main__":
    main()