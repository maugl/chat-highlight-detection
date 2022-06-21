import numpy as np


def calculate_class_ratio(ds):
    # specifically used for binary cross entropy loss with logits
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    total_count = len(np.ravel(ds["train"]["hl_labels"]))
    pos_cat_count = sum(np.ravel(ds["train"]["hl_labels"]))
    neg_cat_count = total_count - pos_cat_count

    return neg_cat_count / pos_cat_count

