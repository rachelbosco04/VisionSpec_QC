import sys, random, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    print("[ERROR] pip install opencv-python")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
except ImportError:
    print("[ERROR] pip install torch torchvision pillow")
    sys.exit(1)

CLASSES = ["DEFECT", "PASS"]
IMG_SIZE = 224


def eval_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


# ==========================================================
# Grad-CAM
# ==========================================================

class GradCAM:
    """
    Grad-CAM for ResNet50 / MobileNetV2 / EfficientNetB0
    """

    def __init__(self, model, arch):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        self._hook_layers(arch)

    def _hook_layers(self, arch):
        arch = arch.lower()

        if arch == "resnet50":
            target = self.model.layer4[-1]

        elif arch == "mobilenetv2":
            target = self.model.features[-1]

        else:   # efficientnetb0
            target = self.model.features[-1]

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def __call__(self, img_tensor):
        img_tensor = img_tensor.unsqueeze(0)

        self.model.zero_grad()

        out = self.model(img_tensor)
        probs = torch.softmax(out, dim=1)

        pred_idx = probs.argmax().item()
        score = probs[0, pred_idx]

        score.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam).cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, pred_idx, probs[0, pred_idx].item()


# ==========================================================
# Utility
# ==========================================================

def overlay_heatmap(original_img_np, cam, alpha=0.45):
    h, w = original_img_np.shape[:2]

    cam_up = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(
        (cam_up * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original_img_np,
        1 - alpha,
        heatmap_rgb,
        alpha,
        0
    )

    return overlay


# ==========================================================
# Main Report
# ==========================================================

def generate_gradcam_report(weights, data_dir, arch,
                            output_dir, n_samples=8, conf_thresh=0.5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------
    print(f"[INFO] Loading model: {weights}")

    from train import build_model

    model = build_model(
        arch,
        num_classes=2,
        freeze_backbone=False
    )

    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    grad_cam = GradCAM(model, arch)
    tfm = eval_transforms()

    # ------------------------------------------------------
    # Load Images
    # ------------------------------------------------------
    data_path = Path(data_dir)

    pass_imgs = list((data_path / "pass").glob("*.png")) + \
                list((data_path / "pass").glob("*.jpg"))

    defect_imgs = list((data_path / "defect").glob("*.png")) + \
                  list((data_path / "defect").glob("*.jpg"))

    if len(pass_imgs) == 0 or len(defect_imgs) == 0:
        print("[ERROR] Could not find pass/defect images.")
        sys.exit(1)

    # ------------------------------------------------------
    # Balanced Sampling (4 PASS + 4 DEFECT if n=8)
    # ------------------------------------------------------
    n_each = n_samples // 2

    selected_pass = random.sample(
        pass_imgs,
        min(n_each, len(pass_imgs))
    )

    selected_defect = random.sample(
        defect_imgs,
        min(n_each, len(defect_imgs))
    )

    samples = selected_pass + selected_defect
    random.shuffle(samples)

    print(f"[INFO] Running GradCAM on {len(samples)} images...")

    # ------------------------------------------------------
    # Plot Grid
    # ------------------------------------------------------
    rows = len(samples)
    cols = 3

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4, rows * 3.5)
    )

    fig.suptitle(
        "VisionSpec QC — Week 3 Grad-CAM Verification\n"
        "✓ DEFECT: focus near fault region\n"
        "✓ PASS: diffuse / low activation",
        fontsize=11,
        fontweight="bold",
        y=1.01
    )

    if rows == 1:
        axes = [axes]

    pass_cnt = 0
    defect_cnt = 0

    # ------------------------------------------------------
    # Per Image
    # ------------------------------------------------------
    for row, img_path in enumerate(samples):

        pil_img = Image.open(img_path).convert("RGB")
        pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))

        orig_np = np.array(pil_img)

        img_t = tfm(Image.fromarray(orig_np)).to(device)

        try:
            cam, pred_idx, conf = grad_cam(img_t)
        except Exception as e:
            print(f"[WARN] Failed on {img_path.name}: {e}")
            continue

        verdict = CLASSES[pred_idx]

        if verdict == "PASS":
            pass_cnt += 1
            color = "#27ae60"
        else:
            defect_cnt += 1
            color = "#e74c3c"

        cam_up = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        heatmap_rgb = cv2.cvtColor(
            cv2.applyColorMap(
                (cam_up * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            ),
            cv2.COLOR_BGR2RGB
        )

        overlay = overlay_heatmap(orig_np, cam)

        # Original
        ax = axes[row][0]
        ax.imshow(orig_np)
        ax.set_title(
            f"Original\n{img_path.parent.name}/{img_path.stem[:18]}",
            fontsize=7
        )
        ax.axis("off")

        # Heatmap
        ax = axes[row][1]
        ax.imshow(heatmap_rgb)
        ax.set_title(
            "Grad-CAM Heatmap",
            fontsize=7
        )
        ax.axis("off")

        # Overlay
        ax = axes[row][2]
        ax.imshow(overlay)
        ax.set_title(
            f"{verdict} ({conf*100:.1f}%)",
            fontsize=8,
            color=color,
            fontweight="bold"
        )
        ax.axis("off")

    plt.tight_layout()

    save_path = out_dir / "week3_gradcam_report.png"

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------
    print(f"\n[OK] GradCAM report saved → {save_path}")

    print("\n=======================================================")
    print("WEEK 3 SUMMARY")
    print("=======================================================")
    print(f"Samples: {len(samples)} | PASS: {pass_cnt} | DEFECT: {defect_cnt}")
    print("\nManual verification checklist:")
    print("[ ] DEFECT images focus on fault region")
    print("[ ] PASS images have diffuse activation")
    print("[ ] No background-only attention")
    print("=======================================================")


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="VisionSpec QC - Week 3 GradCAM"
    )

    parser.add_argument(
        "--weights",
        required=True
    )

    parser.add_argument(
        "--source",
        default="data/final_dataset/test"
    )

    parser.add_argument(
        "--arch",
        default="mobilenetv2",
        choices=["resnet50", "mobilenetv2", "efficientnetb0"]
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/logs/week3"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=8
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    generate_gradcam_report(
        args.weights,
        args.source,
        args.arch,
        args.output_dir,
        args.n_samples,
        args.conf
    )


if __name__ == "__main__":
    main()