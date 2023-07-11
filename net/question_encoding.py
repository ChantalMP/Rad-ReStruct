import torch.nn as nn
from transformers import AutoConfig, RobertaModel


class QuestionEncoderBERT(nn.Module):
    def __init__(self, args):
        super(QuestionEncoderBERT, self).__init__()
        self.args = args
        # load pre-trained BERT model
        config = AutoConfig.from_pretrained(args.bert_model)
        self.BERTmodel = RobertaModel.from_pretrained(args.bert_model, config=config)

    def forward(self, input_ids, q_attn_mask):
        # feed question to BERT model
        outputs = self.BERTmodel(input_ids, attention_mask=q_attn_mask)
        # get word embeddings
        word_embeddings = outputs.last_hidden_state

        return word_embeddings
