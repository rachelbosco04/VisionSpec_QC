from flask import Flask, request, jsonify, render_template, render_template_string
import sys, time, io, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

try:
    import torch
    from torchvision import transforms
    from PIL import Image
except ImportError:
    print("[ERROR] pip install torch torchvision pillow")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] pip install opencv-python")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["DEFECT", "PASS"]
IMG_SIZE = 224


# ==========================================================
# MODEL WRAPPER
# ==========================================================

class VisionSpecModel:
    def __init__(self, weights_path, arch="mobilenetv2", device="auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        )

        self.arch = arch.lower()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.model = self._load(weights_path)

        print(f"[OK] Model loaded on {self.device}")

        self._warmup()

    def _load(self, path):
        from train import build_model

        model = build_model(
            self.arch,
            num_classes=2,
            freeze_backbone=False
        )

        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(self.device)

        return model

    def _warmup(self, n=5):
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(self.device)

        with torch.no_grad():
            for _ in range(n):
                self.model(dummy)

        print(f"[OK] Warm-up complete ({n} passes)")

    @torch.no_grad()
    def predict(self, image):
        t0 = time.perf_counter()

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        out = self.model(tensor)

        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

        pred = int(probs.argmax())

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "verdict": CLASSES[pred],
            "confidence": float(probs[pred]),
            "prob_defect": float(probs[0]),
            "prob_pass": float(probs[1]),
            "latency_ms": round(latency_ms, 2)
        }


# ==========================================================
# FLASK APP
# ==========================================================

def create_app(model):

    app = Flask(__name__, template_folder="../templates")

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify({
            "status": "ok",
            "model": model.arch,
            "device": str(model.device)
        })

    @app.route("/predict", methods=["POST"])
    def predict():

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        try:
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        result = model.predict(img)

        # Browser form upload
        if "text/html" in request.headers.get("Accept", ""):

            verdict = result["verdict"]
            confidence = result["confidence"]
            latency = result["latency_ms"]

            color = "#22c55e" if verdict == "PASS" else "#ef4444"

            return render_template_string(f"""
<!DOCTYPE html>
<html>
<head>
<title>Prediction Result</title>
<style>
body {{
font-family: Arial;
background:#0f172a;
color:white;
display:flex;
justify-content:center;
align-items:center;
height:100vh;
margin:0;
}}
.card {{
background:#1e293b;
padding:40px;
border-radius:18px;
width:500px;
text-align:center;
}}
.verdict {{
font-size:42px;
font-weight:bold;
color:{color};
}}
a {{
color:#60a5fa;
text-decoration:none;
}}
</style>
</head>
<body>
<div class="card">
<h1>Prediction Result</h1>
<div class="verdict">{verdict}</div>
<p>Confidence: {confidence:.2%}</p>
<p>Latency: {latency:.2f} ms</p>
<br>
<a href="/">Try Another Image</a>
</div>
</body>
</html>
""")

        return jsonify(result)

    return app


# ==========================================================
# BENCHMARK
# ==========================================================

def run_benchmark(model, n_images=200, output_dir="./outputs/logs/week4"):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[BENCHMARK] Running {n_images} inference passes...")

    dummy_imgs = [
        Image.fromarray(
            np.random.randint(
                50, 200,
                (IMG_SIZE, IMG_SIZE, 3),
                dtype=np.uint8
            )
        )
        for _ in range(n_images)
    ]

    latencies = []

    for img in dummy_imgs:
        t0 = time.perf_counter()
        model.predict(img)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)

    fps = 1000 / latencies.mean()

    print("\n==================================================")
    print("LATENCY BENCHMARK RESULTS")
    print("==================================================")
    print(f"Mean latency      : {latencies.mean():.2f} ms")
    print(f"Median latency    : {np.median(latencies):.2f} ms")
    print(f"P95 latency       : {np.percentile(latencies,95):.2f} ms")
    print(f"Throughput (FPS)  : {fps:.1f}")
    print("==================================================")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(latencies)
    ax.axhline(latencies.mean(), linestyle="--")
    ax.set_title("Inference Latency")
    ax.set_xlabel("Image Index")
    ax.set_ylabel("ms")

    save_path = out / "week4_latency_benchmark.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    report = {
        "mean_ms": float(latencies.mean()),
        "median_ms": float(np.median(latencies)),
        "fps": float(fps),
        "target_10fps_met": bool(fps >= 10)
    }

    (out / "week4_benchmark_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print(f"[OK] Saved → {save_path}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--arch",
        default="mobilenetv2",
        choices=["resnet50", "mobilenetv2", "efficientnetb0"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--output_dir",
        default="./outputs/logs/week4"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true"
    )

    args = parser.parse_args()

    model = VisionSpecModel(
        args.weights,
        args.arch,
        args.device
    )

    if args.benchmark:
        run_benchmark(model, 200, args.output_dir)
        return

    app = create_app(model)

    print(f"\n[INFO] Running on http://localhost:{args.port}")
    app.run(
        host="0.0.0.0",
        port=args.port,
        debug=False
    )


if __name__ == "__main__":
    main()