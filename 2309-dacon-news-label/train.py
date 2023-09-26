import argparse

import lightning as L
import numpy as np
import pandas as pd
import torch
from module import DeBERTaClassifier, NewsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


def load_data(news_data, submit_data, manual_data):
    """Loads data from csv files."""
    df = pd.read_csv(news_data, index_col=0)
    df_submit = pd.read_csv(submit_data, index_col=0)
    df["text"] = df["title"] + " " + df["contents"]
    df_manual = pd.read_csv(manual_data, index_col=0)
    return df, df_submit, df_manual


def train_model(
    data, manual_data, model_name, tokenizer, batch_size, learning_rate, num_epochs
):
    """Trains a model with manual labels."""
    dataset = NewsDataset(
        data.loc[manual_data.index, ["text"]].join(manual_data["category"], how="left"),
        tokenizer,
    )
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    model = DeBERTaClassifier(
        model_name,
        num_labels=6,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_batch=len(train_loader),
    )

    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        precision=16,
    )
    trainer.fit(model, train_loader)
    return model


def test_model(model, df, tokenizer):
    """Tests a model on the test set."""
    dataset_all = NewsDataset(df, tokenizer)
    test_loader = DataLoader(dataset_all, batch_size=16, shuffle=False, num_workers=4)
    model = model.to("cuda")
    model.eval()
    probs = []
    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_loader)):
            data = {key: value.to("cuda") for key, value in data.items()}
            out = model.predict_step(data, idx)
            probs.append(out.detach().cpu().numpy())
            del data, out
            torch.cuda.empty_cache()
    return probs


def select_uncertain_samples(model, unlabeled_df, tokenizer):
    """Selects top n_samples where the model is least confident."""
    dataset = NewsDataset(unlabeled_df, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    model = model.to("cuda")
    model.eval()
    probs = []
    with torch.no_grad():
        for idx, data in tqdm(enumerate(loader)):
            data = {key: value.to("cuda") for key, value in data.items()}
            out = model.predict_step(data, idx)
            probs.extend(out.detach().cpu().numpy())
            del data, out
            torch.cuda.empty_cache()

    uncertainties = -np.max(probs, axis=1)
    selected_indices = np.argsort(uncertainties)

    return unlabeled_df.iloc[selected_indices]


def main(args):
    df, df_submit, df_manual = load_data(
        args.news_data, args.submit_data, args.manual_data
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    model = train_model(
        df,
        df_manual,
        args.model_name,
        tokenizer,
        args.batch_size,
        args.learning_rate,
        args.num_epochs,
    )
    # df_uncertain = select_uncertain_samples(model, df, tokenizer).to_csv(
    #     "uncertain.csv", index=True, header=True
    # )

    logits = test_model(model, df, tokenizer)

    output = np.concatenate(logits)
    df_submit["category"] = output.argmax(axis=1)
    df_submit["category"].to_csv(args.output, index=True, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--news_data", default="../../data/dacon-hackathon-2309/news.csv"
    )
    parser.add_argument(
        "--submit_data", default="../../data/dacon-hackathon-2309/sample_submission.csv"
    )
    parser.add_argument(
        "--manual_data", default="../../data/dacon-hackathon-2309/manual.csv"
    )
    parser.add_argument(
        "--model_name", default="Yueh-Huan/news-category-classification-distilbert"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--output", default="submission.csv")

    args = parser.parse_args()
    main(args)
