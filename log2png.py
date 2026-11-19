"""
Save all TensorBoard scalar plots from logs/rough_go2/ to logs/plots/
Usage: python save_log_plots.py
       python save_log_plots.py --logs_dir logs/rough_go2 --out_dir logs/plots
"""
import argparse
import os
import math
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_events(run_dir):
    """Return {tag: [(step, value), ...]} for all scalar events in run_dir."""
    from torch.utils.tensorboard.summary import hparams  # noqa – just checking tb is available
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("tensorboard package not found. Install with: pip install tensorboard")

    ea = EventAccumulator(run_dir)
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]
    return scalars


def save_individual_plots(scalars, out_dir, run_name):
    """One PNG per scalar tag."""
    os.makedirs(out_dir, exist_ok=True)
    for tag, data in scalars.items():
        steps, values = zip(*data)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, values, linewidth=1.2)
        ax.set_title(f"{run_name}\n{tag}", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        # sanitise tag for filename
        fname = tag.replace("/", "_").replace(" ", "_") + ".png"
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  saved {fname}")


def save_grid_plot(scalars, out_dir, run_name):
    """All scalars in one grid PNG."""
    if not scalars:
        return
    n = len(scalars)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = [axes] if n == 1 else list(axes.flat)

    for ax, (tag, data) in zip(axes, scalars.items()):
        steps, values = zip(*data)
        ax.plot(steps, values, linewidth=1.0)
        ax.set_title(tag, fontsize=8)
        ax.set_xlabel("Step", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(run_name, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "all_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved all_metrics.png ({n} scalars)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="unitree_rl_gym/logs/rough_go2")
    parser.add_argument("--out_dir",  default="unitree_rl_gym/logs/plots")
    args = parser.parse_args()

    logs_dir = args.logs_dir
    out_dir  = args.out_dir

    runs = sorted([
        d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d))
    ])

    print(f"Found {len(runs)} runs in {logs_dir}\n")

    for run in runs:
        run_dir  = os.path.join(logs_dir, run)
        plot_dir = os.path.join(out_dir, run)
        print(f"[{run}]")

        try:
            scalars = read_events(run_dir)
        except Exception as e:
            print(f"  SKIP — could not read events: {e}\n")
            continue

        if not scalars:
            print(f"  SKIP — no scalar data found\n")
            continue

        save_individual_plots(scalars, plot_dir, run)
        save_grid_plot(scalars, plot_dir, run)
        print()

    print(f"Done. Plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()