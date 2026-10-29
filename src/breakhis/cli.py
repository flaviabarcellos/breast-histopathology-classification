"""Command line interface for indexing, training, evaluation, and prediction."""
import argparse
import json
from pathlib import Path

from .data import CLASS_NAMES, index_images, split_patients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/default.json"))
    command = commands.add_parser("evaluate")
    command.add_argument("--run", type=Path, required=True)
    command = commands.add_parser("predict")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("prepare", "train"):
        config = json.loads(args.config.read_text())
        output = Path(config["output_dir"])
        if args.command == "train" and output.exists():
            raise FileExistsError("Use a new output_dir for each training run.")
        manifest = split_patients(index_images(config["data_dir"], config["magnification"]),
                                  config["train_fraction"], config["validation_fraction"],
                                  config["test_fraction"], config["seed"])
        if args.command == "prepare":
            path = Path("data/processed") / f"manifest_{config['magnification']}X.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_csv(path, index=False)
            print(manifest.groupby(["split", "label"]).size().to_string())
            print(f"Manifest saved to {path}")
        else:
            output.mkdir(parents=True)
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (output / "class_names.json").write_text(json.dumps(CLASS_NAMES) + "\n")
            manifest.to_csv(output / "manifest.csv", index=False)
            from .train import train
            train(config, manifest, output)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        print(json.dumps(evaluate(args.run), indent=2))
    else:
        import tensorflow as tf
        config = json.loads((args.run / "config.json").read_text())
        image = tf.keras.utils.load_img(args.image, target_size=config["image_size"])
        array = tf.keras.utils.img_to_array(image)[None, ...]
        model = tf.keras.models.load_model(args.run / "best.keras")
        probability = model.predict(array, verbose=0)[0]
        print(json.dumps(dict(zip(CLASS_NAMES, map(float, probability))), indent=2))


if __name__ == "__main__":
    main()
