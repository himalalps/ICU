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


import sys
import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Required_parameter
    parser.add_argument(
        "--config-file",
        "--cf",
        help="pointer to the configuration file of the experiment",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )

    # Optional Params
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        "--max_pred",
        default=80,
        type=int,
        help="The maximum number of masked tokens in a sequence to be predicted.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--do_lower_case",
        default=True,
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )

    parser.add_argument(
        "--use_pretrain",
        default=False,
        action="store_true",
        help="Whether to use Bert Pretrain Weights or not",
    )

    parser.add_argument(
        "--refresh_bucket_size",
        type=int,
        default=1,
        help="This param makes sure that a certain task is repeated for this time steps to \
                            optimise on the back propogation speed with APEX's DistributedDataParallel",
    )
    parser.add_argument(
        "--finetune",
        default=False,
        action="store_true",
        help="Whether to finetune only",
    )

    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="LE",
        help="Choices LE, EE, EP (L: Linear, E: Exponetial, P: Polynomial warmup and decay)",
    )

    parser.add_argument(
        "--lr_offset", type=float, default=0.0, help="Offset added to lr."
    )

    parser.add_argument(
        "--load_training_checkpoint",
        "--load_cp",
        type=str,
        default=None,
        help="This is the path to the TAR file which contains model+opt state_dict() checkpointed.",
    )
    parser.add_argument(
        "--load_checkpoint_id",
        "--load_cp_id",
        type=str,
        default=None,
        help="Checkpoint identifier to load from checkpoint path",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="This is the path to store the output and TensorBoard results.",
    )

    parser.add_argument(
        "--rewarmup",
        default=False,
        action="store_true",
        help="Rewarmup learning rate after resuming from a checkpoint",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=sys.maxsize,
        help="Maximum number of training steps of effective batch size to complete.",
    )

    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=sys.maxsize,
        help="Maximum number of training steps of effective batch size within an epoch to complete.",
    )

    parser.add_argument(
        "--print_steps",
        type=int,
        default=100,
        help="Interval to print training details.",
    )

    parser.add_argument(
        "--data_path_prefix",
        type=str,
        default="",
        help="Path to prefix data loading, helpful for AML and other environments",
    )

    parser.add_argument(
        "--validation_data_path_prefix",
        type=str,
        default=None,
        help="Path to prefix validation data loading, helpful if pretraining dataset path is different",
    )

    parser.add_argument(
        "--deepspeed_transformer_kernel",
        default=False,
        action="store_true",
        help="Use DeepSpeed transformer kernel to accelerate.",
    )

    parser.add_argument(
        "--stochastic_mode",
        default=False,
        action="store_true",
        help="Use stochastic mode for high-performance transformer kernel.",
    )

    parser.add_argument(
        "--ckpt_to_save",
        nargs="+",
        type=int,
        help="Indicates which checkpoints to save, e.g. --ckpt_to_save 160 161, by default all checkpoints are saved.",
    )

    parser.add_argument(
        "--attention_dropout_checkpoint",
        default=False,
        action="store_true",
        help="Use DeepSpeed transformer kernel memory optimization to checkpoint dropout output.",
    )
    parser.add_argument(
        "--normalize_invertible",
        default=False,
        action="store_true",
        help="Use DeepSpeed transformer kernel memory optimization to perform invertible normalize backpropagation.",
    )
    parser.add_argument(
        "--gelu_checkpoint",
        default=False,
        action="store_true",
        help="Use DeepSpeed transformer kernel memory optimization to checkpoint GELU activation.",
    )
    parser.add_argument(
        "--deepspeed_sparse_attention",
        default=False,
        action="store_true",
        help="Use DeepSpeed sparse self attention.",
    )

    parser.add_argument(
        "--use_nvidia_dataset",
        default=False,
        action="store_true",
        help="Use Nvidia pretraining dataset.",
    )

    parser.add_argument(
        "--progressive_layer_drop",
        default=False,
        action="store_true",
        help="Whether to enable progressive layer dropping or not",
    )

    return parser


def is_time_to_exit(args, epoch_steps=0, global_steps=0):
    return (epoch_steps >= args.max_steps_per_epoch) or (global_steps >= args.max_steps)
