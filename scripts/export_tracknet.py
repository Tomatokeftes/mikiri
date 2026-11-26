"""Turn the published TrackNetV3 checkpoint into the ONNX file Mikiri loads.

Mikiri runs the learned shuttle detector through onnxruntime, which it already
needs for pose, and the authors publish a PyTorch checkpoint. This is the one
step between the two, and it is a script and not part of the app because it is
the only thing here that needs PyTorch: about 200 MB that nothing at runtime
uses. Run it once, in any environment that has ``torch`` and ``onnx``.

Getting the inputs (both MIT, by the authors' README)::

    git clone https://github.com/qaz812345/TrackNetV3
    # download TrackNetV3_ckpts.zip from the link in that README and unzip it

Then::

    python scripts/export_tracknet.py TrackNetV3 ckpts/TrackNet_best.pt

which writes ``tracknetv3.onnx`` into Mikiri's models folder, next to the pose
models. Pass ``--out`` to put it somewhere else.

The checkpoint is a pickle, and unpickling runs whatever the file says. It is
therefore loaded with ``weights_only=True``, which admits tensors and plain
containers and nothing that can execute; a checkpoint that needs more than
that is refused rather than trusted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


#: File name, input height and width, and frames per pass of the export.
EXPORT_SHAPE = ("tracknetv3.onnx", 288, 512, 8)


def main() -> int:
    """Export the checkpoint and check the exported file against the original."""
    parser = argparse.ArgumentParser(description="Export TrackNetV3 to ONNX for Mikiri.")
    parser.add_argument("repo", type=Path, help="a checkout of qaz812345/TrackNetV3")
    parser.add_argument("checkpoint", type=Path, help="TrackNet_best.pt from its checkpoints")
    parser.add_argument("--out", type=Path, default=None, help="where to write the ONNX file")
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    import torch

    from mikiri.paths import default_models_dir

    # Repeated from mikiri.tracknet rather than imported: that module pulls in
    # the whole app, and this script is meant to run in an environment that
    # has PyTorch and little else. tests/test_tracknet.py holds the two equal.
    MODEL_NAME, NET_HEIGHT, NET_WIDTH, SEQ_LEN = EXPORT_SHAPE

    sys.path.insert(0, str(args.repo))
    from model import TrackNet  # the authors' definition, not a copy of it

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    params = ckpt["param_dict"]
    if params.get("seq_len") != SEQ_LEN or params.get("bg_mode") != "concat":
        # mikiri.tracknet builds its input for exactly this configuration: the
        # background picture and then eight frames. Another one would export
        # cleanly and then be fed the wrong channels.
        print(
            f"unsupported checkpoint: seq_len={params.get('seq_len')} "
            f"bg_mode={params.get('bg_mode')!r}",
            file=sys.stderr,
        )
        return 1
    net = TrackNet(in_dim=(SEQ_LEN + 1) * 3, out_dim=SEQ_LEN)
    net.load_state_dict(ckpt["model"])
    net.eval()

    out = args.out or (default_models_dir() / MODEL_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.rand(1, (SEQ_LEN + 1) * 3, NET_HEIGHT, NET_WIDTH)
    torch.onnx.export(
        net,
        (sample,),
        str(out),
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "n"}, "y": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )

    # The export is only worth having if it is the same function.
    with torch.inference_mode():
        want = net(sample).numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
        None, {"x": sample.numpy()}
    )[0]
    worst = float(np.abs(want - got).max())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), largest difference {worst:.2e}")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
