#!/usr/bin/env python3
"""Visualize different noise schedules for DDPM."""

import matplotlib.pyplot as plt
import torch

from ddpm import DDPM


def visualize_noise_schedules():
    """Compare different noise schedules visually."""

    num_timesteps = 1000
    schedules = ["linear", "cosine", "quadratic", "sigmoid"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, schedule in enumerate(schedules):
        # Create a temporary model to get the beta schedule
        model = DDPM(
            image_size=40,
            num_timesteps=num_timesteps,
            beta_schedule=schedule,
            beta_start=0.0001,
            beta_end=0.02,
            device="cpu",
        )

        betas = model.betas.cpu().numpy()
        alphas_cumprod = model.alphas_cumprod.cpu().numpy()

        ax = axes[idx]

        # Plot betas
        ax2 = ax.twinx()
        line1 = ax.plot(betas, "b-", label="Beta (β)", linewidth=2)
        line2 = ax2.plot(alphas_cumprod, "r-", label="Alpha Cumprod (ᾱ)", linewidth=2)

        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Beta (β)", color="b", fontsize=12)
        ax2.set_ylabel("Alpha Cumprod (ᾱ)", color="r", fontsize=12)
        ax.tick_params(axis="y", labelcolor="b")
        ax2.tick_params(axis="y", labelcolor="r")

        ax.set_title(
            f"{schedule.capitalize()} Schedule", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left", fontsize=10)

        # Add statistics text
        stats_text = (
            f"Beta range: [{betas.min():.6f}, {betas.max():.6f}]\n"
            f"Mean beta: {betas.mean():.6f}\n"
            f"Final ᾱ: {alphas_cumprod[-1]:.6f}"
        )
        ax.text(
            0.98,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle(
        "Noise Schedule Comparison for DDPM\n"
        f"Timesteps: {num_timesteps}, Beta Start: 0.0001, Beta End: 0.02",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "./out/ddpm/noise_schedules_comparison.png", dpi=150, bbox_inches="tight"
    )
    print(f"Saved visualization to ./out/ddpm/noise_schedules_comparison.png")
    plt.close()

    # Create a second figure comparing all schedules on the same plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    colors = ["blue", "red", "green", "purple"]

    for schedule, color in zip(schedules, colors):
        model = DDPM(
            image_size=40,
            num_timesteps=num_timesteps,
            beta_schedule=schedule,
            beta_start=0.0001,
            beta_end=0.02,
            device="cpu",
        )

        betas = model.betas.cpu().numpy()
        alphas_cumprod = model.alphas_cumprod.cpu().numpy()

        ax1.plot(betas, color=color, label=schedule.capitalize(), linewidth=2)
        ax2.plot(alphas_cumprod, color=color, label=schedule.capitalize(), linewidth=2)

    ax1.set_xlabel("Timestep", fontsize=12)
    ax1.set_ylabel("Beta (β)", fontsize=12)
    ax1.set_title("Beta Values Comparison", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Timestep", fontsize=12)
    ax2.set_ylabel("Alpha Cumprod (ᾱ)", fontsize=12)
    ax2.set_title("Cumulative Alpha Values Comparison", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("All Noise Schedules Overlaid", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("./out/ddpm/noise_schedules_overlay.png", dpi=150, bbox_inches="tight")
    print(f"Saved overlay visualization to ./out/ddpm/noise_schedules_overlay.png")
    plt.close()

    # Print numerical comparison
    print("\n" + "=" * 80)
    print("NOISE SCHEDULE STATISTICS")
    print("=" * 80)

    for schedule in schedules:
        model = DDPM(
            image_size=40,
            num_timesteps=num_timesteps,
            beta_schedule=schedule,
            beta_start=0.0001,
            beta_end=0.02,
            device="cpu",
        )

        betas = model.betas.cpu().numpy()
        alphas_cumprod = model.alphas_cumprod.cpu().numpy()

        print(f"\n{schedule.upper()} Schedule:")
        print(f"  Beta range:        [{betas.min():.8f}, {betas.max():.8f}]")
        print(f"  Mean beta:         {betas.mean():.8f}")
        print(f"  Std beta:          {betas.std():.8f}")
        print(f"  Initial ᾱ (t=0):   {alphas_cumprod[0]:.8f}")
        print(f"  Final ᾱ (t=T):     {alphas_cumprod[-1]:.8f}")
        print(f"  Signal ratio:      {(1 - alphas_cumprod[-1]):.8f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import os

    os.makedirs("./out/ddpm", exist_ok=True)
    visualize_noise_schedules()
    print("\nDone! Check ./out/ddpm/ for visualizations.")
