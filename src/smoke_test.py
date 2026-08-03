"""Validate imports, model instantiation, and PixelRL tensor shapes."""

from __future__ import annotations

import argparse

import torch

from neuralnet import PixelRL_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--train-step",
        action="store_true",
        help="Also run one backward/optimizer step with A3C-style losses.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_actions = 6
    model = PixelRL_model(n_actions).to(device)

    # Input: LAB image (3ch) + hidden state (64ch) = 67 channels
    batch, height, width = 1, 64, 64
    state = torch.randn(batch, 3 + 64, height, width, device=device)

    if args.train_step:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        pi, v, hidden = model.pi_and_v(state)

        # Simulate A3C loss
        target_actions = torch.randint(0, n_actions, (batch, height, width), device=device)
        probs = torch.softmax(pi, dim=1)
        log_probs = torch.log_softmax(pi, dim=1)
        selected_log_probs = log_probs.gather(1, target_actions.unsqueeze(1)).squeeze(1)
        policy_loss = -selected_log_probs.mean()
        value_loss = v.mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()
        total_loss = policy_loss + value_loss + entropy

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        print(f"train_step_loss={float(total_loss.detach()):.6f}")
    else:
        model.eval()
        with torch.inference_mode():
            pi, v, hidden = model.pi_and_v(state)

    assert pi.shape == (batch, n_actions, height, width), pi.shape
    assert v.shape == (batch, 1, height, width), v.shape
    assert hidden.shape == (batch, 64, height, width), hidden.shape

    print(f"torch={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(f"capability={torch.cuda.get_device_capability(device)}")
    print(f"pi_shape={tuple(pi.shape)}")
    print(f"value_shape={tuple(v.shape)}")
    print(f"hidden_shape={tuple(hidden.shape)}")
    if args.train_step:
        print("PixelRL backward/optimizer step passed.")
    print("PixelRL smoke test passed.")


if __name__ == "__main__":
    main()
