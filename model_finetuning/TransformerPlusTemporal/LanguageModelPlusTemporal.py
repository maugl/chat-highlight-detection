import torch
from torch import cuda
from transformers import RobertaModel, PreTrainedModel

device = 'cuda' if cuda.is_available() else 'cpu'

class LanguageModelPlusTemporal(torch.nn.Module):
    """
    TODO
    make a ModelConfig argument and Class to set the parameters for the model
    """

    def __init__(self, lm_path, additional_features_size, window_size, sequence_length, num_dist_categories,
                 num_dist_steps, main_loss_ratio=1, pos_label_ratio=7.35):
        super(LanguageModelPlusTemporal, self).__init__()

        self.additional_features_size = additional_features_size
        self.window_size = window_size
        self.loss1_ratio = main_loss_ratio
        self.loss2_ratio = 1 - main_loss_ratio
        self.pos_label_ratio = torch.Tensor([pos_label_ratio]).to(device)
        # layers
        self.l1 = RobertaModel.from_pretrained(lm_path)
        self.d1 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(128 + self.additional_features_size,
                                  4096)  # should be model_output_size (sequence length) + additional_features_size
        # maybe I should translate the additional features to a bigger
        # vectorspace so that they do not get drowned out by the sequence representation
        self.l41 = torch.nn.Linear(4096, window_size)
        # for each chunk in a sequence we have num_distcategories and each category has num_dist_steps number of discrete labels
        # self.l42 = torch.nn.Linear(1024, sequence_length)  # might have to add more here for each class and possible value
        # l42: maybe combine in-highlight output and outside-highlight output

    def forward(self, input_ids, attention_mask, additional_features, hl_labels=None, objective_simple=None, **kwargs):
        output_1 = self.l1(input_ids, attention_mask=attention_mask)  # last_hidden_state, pooler_output
        # average lm last hidden layer outputs
        # alterntively use pooled output
        output_1_average = torch.mean(output_1.last_hidden_state, axis=-1)
        # concatenate sequence representation with additional temporal features
        input_3 = torch.cat(tensors=(output_1_average, additional_features), axis=-1)
        # linear
        output_3 = self.l3(input_3)
        # activation
        output_3_act = torch.tanh(output_3)
        # dropout
        output_3_drop = self.d1(output_3_act)
        # for binary classification of main objective
        # define to return
        output = self.l41(output_3_drop)
        # for multiclass classification of additional objective
        # output_42 = self.l42(output_3)
        # outputs = [output_41, output_42]

        # print("hl_labels", hl_labels is not None)
        # print("objective_simple", objective_simple is not None)
        # for compatibility with the huggingface Trainer API
        if hl_labels is not None:
            loss = self._get_loss(output, hl_labels)
            return {"loss": loss,
                    "logits": output}

        return {"logits": output}

    def _get_loss(self, outputs, targets):
        """
        for o, t in zip(outputs[0], targets[0]):
          loss += torch.nn.BCEWithLogitsLoss()(o, t)
        """
        # does this actually do what I want?
        # compute losses for n (window_size) outputs?
        # we can add weighing for the much smaller positive class with pos_weight
        # MSELoss()
        """
        loss1 = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_label_ratio)(
            torch.reshape(outputs, (outputs.shape[0] * self.window_size, 1)),
            torch.reshape(targets, (targets.shape[0] * self.window_size, 1)))  # keep as is
        """
        outputs = torch.sigmoid(torch.reshape(outputs, (outputs.shape[0] * self.window_size, 1)))
        loss1 = torch.nn.MSELoss()(
            outputs,
            torch.reshape(targets, (targets.shape[0] * self.window_size, 1)))  # keep as is
        return loss1
