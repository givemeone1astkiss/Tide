import argparse
from utils import *
from config import *
from pytorch_lightning import Trainer

def main(args):
    model = get_model(args.model, args.config)
    peptide_data = PeptideDataset(path=args.dataset, ratio=args.ratio, batch_size=args.batch_size)
    if args.model in MACHINE_LEARNING_MODEL:
        train_x, train_y = peptide_data.ml_train_datasets
        test_x, test_y = peptide_data.ml_val_datasets
        model.fit(train_x, train_y)
        train_loss = model.eval(train_x, train_y, args.eval_func)
        print(f"Train loss: {train_loss}")
        test_loss = model.eval(test_x, test_y, args.eval_func)
        print(f"Test loss: {test_loss}")
    elif args.model in DEEP_LEARNING_MODEL:
        trainer = Trainer(enable_progress_bar=True, max_epochs=args.epoch, default_root_dir=f"{args.save_path}/{args.model}")
        trainer.fit(model, peptide_data)
    else:
        raise ValueError("Invalid model name.")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--model", type=str, default="MLP")
    argparse.add_argument("--config", type=str, default="model.yaml")
    argparse.add_argument("--dataset", type=str, default=DATASET)
    argparse.add_argument("--ratio", type=float, default=0.8)
    argparse.add_argument("--batch_size", type=int, default=32)
    argparse.add_argument("--epochs", type=int, default=100)
    argparse.add_argument("--lr", type=int, default=0.01)
    argparse.add_argument("--eval_func", type=str, default="mse")
    argparse.add_argument("--save_path", type=str, default=SAVE_MODEL_PATH)
    argparse.add_argument("--load", type=bool, default=False)
    argparse.add_argument("--load_path", type=str, default=SAVE_MODEL_PATH)
    args = argparse.parse_args()
    main(args)