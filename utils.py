from bert_score import score
import pandas as pd


def batch_compute_bert_score(
    candidates_list, references_list, model_type="bert-base-uncased", device="cuda:0"
):
    P, R, F1 = score(
        candidates_list, references_list, model_type=model_type, device=device
    )
    return P, R, F1


if __name__ == "__main__":
    references_list = [
        "This is a test sentence for BERTScore evaluation.",
        "Another reference sentence.",
        "This is a test sentence for BERTScore.",
        "test test test",
    ]
    candidates_list = [
        "This is a test sentence for evaluating BERTScore.",
        "Another candidate sentence.",
        "This is a BERTScore test sentence.",
        "target target target target target target",
    ]
    P, R, F1 = batch_compute_bert_score(candidates_list, references_list)

    for i in range(len(P)):
        print(f"Pair {i+1}: Precision={P[i]:.3f}, Recall={R[i]:.3f}, F1={F1[i]:.3f}")
