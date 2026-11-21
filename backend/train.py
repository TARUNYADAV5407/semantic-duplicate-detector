from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

def main():
    print("Loading base encoder for fine-tuning...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading STS-B semantic similarity dataset...")
    train_raw = load_dataset("nyu-mll/glue", "stsb", split="train")
    val_raw = load_dataset("nyu-mll/glue", "stsb", split="validation")

    train_examples = [
        InputExample(texts=[x["sentence1"], x["sentence2"]], label=float(x["label"]) / 5.0)
        for x in train_raw
    ]

    val_s1 = [x["sentence1"] for x in val_raw]
    val_s2 = [x["sentence2"] for x in val_raw]
    val_scores = [float(x["label"]) / 5.0 for x in val_raw]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(model=model)
    evaluator = evaluation.EmbeddingSimilarityEvaluator(val_s1, val_s2, val_scores)

    print("Fine-tuning model on STS-B (1 epoch ~1-2 mins on CPU)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=1,
        warmup_steps=50,
        output_path="./my_trained_model",
        show_progress_bar=True,
    )
    print("Saved your trained model to ./my_trained_model !")

if __name__ == "__main__":
    main()