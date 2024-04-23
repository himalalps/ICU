from bert_score import score
import pandas as pd

DIALOG_DATASETS = [
    "wizard_of_wikipedia",
    "empathetic_dialogues",
    "blended_skill_talk",
    "wizard_of_internet",
]

CLASSIFICATION_DATASETS = [
    "piqa",
    "hellaswag",
    "ai2_arc",
    "winogrande",
    "math_qa",
    "pubmed_qa",
    "copa",
    "super_glue",
]

PPL_DATASETS = ["wikitext", "pile"]

COMPLETION_DATASETS = ["lambada"]


def normalize_reply(text: str, version=2) -> str:
    """
    Standardize the capitalization and punctuation spacing of the input text.
    Version 1: Fix sentence start casing, and punctuation.
    Version 2: Add trailing period, if missing.
    """

    switch_list = [(" .", "."), (" ,", ","), (" ?", "?"), (" !", "!"), (" ' ", "'")]

    # add spaces so that words and punctuation can be seaprated
    new_text = text.lower()

    # normalize in case of human:
    for new, old in switch_list:
        new_text = new_text.replace(old, new).replace("  ", " ")

    # split on punctuation to find sentence boundaries
    # capitalize stuff
    tokens = new_text.split(" ")
    for i in range(len(tokens)):
        if i == 0:
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in ("i", "i'm", "i've", "i'll", "i'd"):
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in "?.!" and i < len(tokens) - 1:
            tokens[i + 1] = uppercase(tokens[i + 1])
    new_text = " ".join(tokens)
    new_text = " " + new_text + " "

    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])

    # get rid of surrounding whitespace
    new_text = new_text.strip()
    new_text = new_text.replace("  ", " ")

    if version > 1 and new_text and new_text[-1] not in "!.?)\"'":
        new_text += "."

    return new_text


def uppercase(string: str) -> str:
    """
    Make the first character of the string uppercase, if the string is non-empty.
    """
    if len(string) == 0:
        return string
    else:
        return string[0].upper() + string[1:]


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
