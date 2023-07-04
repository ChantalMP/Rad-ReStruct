import _pickle as cPickle
import json
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _load_dataset(dataroot, name):
    """Load entries

    img2id: dict {img -> id} id can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """

    def assert_eq(real, expected):
        assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

    def _create_entry(img, data, answer):
        if None != answer:
            answer.pop('image_name')
            answer.pop('qid')
        entry = {
            'qid': data['qid'],
            'image_name': data['image_name'],
            'image': img,
            'question': data['question'],
            'answer': answer,
            'answer_text': data['answer'],
            'answer_type': data['answer_type'],
            'question_type': data['question_type'],
            'phrase_type': data['phrase_type']}
        return entry

    with open(os.path.join(dataroot, 'imgid2idx.json')) as f:
        img_id2val = json.load(f)

    data_path = os.path.join(dataroot, name + 'set.json')
    with open(data_path) as f:
        samples = json.load(f)
    samples = sorted(samples, key=lambda x: x['qid'])

    # loads class labels for the answers -> in the end we do not want to use class labels but the actual text answer, but for now it's ok
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    with open(answer_path, 'rb') as f:
        answers = cPickle.load(f)
    answers = sorted(answers, key=lambda x: x['qid'])
    # get all samples where the answer['labels'] is empty

    assert_eq(len(samples), len(answers))
    entries = []
    for sample, answer in zip(samples, answers):
        assert_eq(sample['qid'], answer['qid'])
        assert_eq(sample['image_name'], answer['image_name'])
        img_id = sample['image_name']
        entries.append(_create_entry(img_id2val[img_id], sample, answer))

    return entries


''' get history for training, apply history augmentation'''
def get_encoded_history(tokenizer, history, question, question_id, question_type, max_num_question_tokens, mode):
    q_level = get_question_level(question_type)

    encoded_history = []
    token_type_ids = []

    for level, level_group in history.items():
        if int(level) < q_level:
            if mode == 'train' and torch.rand(1) < 1.1:
                # random order of questions in level_group
                random.shuffle(level_group)
            for q in level_group:
                encoded_question = tokenizer.encode(q[1])[1:-1]
                encoded_answer = tokenizer.encode(str(q[2]))[1:-1]
                encoded_history.append(encoded_question + [tokenizer.sep_token_id] + encoded_answer + [tokenizer.sep_token_id])
                token_type_ids.append([2] * (len(encoded_question) + 1) + [3] * (len(encoded_answer) + 1))
        elif int(level) == q_level:
            if mode == 'train' and torch.rand(1) < 1.1:
                random.shuffle(level_group)
            for q in level_group:
                if q[0] != question_id:
                    encoded_question = tokenizer.encode(q[1])[1:-1]
                    encoded_answer = tokenizer.encode(str(q[2]))[1:-1]
                    encoded_history.append(encoded_question + [tokenizer.sep_token_id] + encoded_answer + [tokenizer.sep_token_id])
                    token_type_ids.append([2] * (len(encoded_question) + 1) + [3] * (len(encoded_answer) + 1))
                else:
                    # drop elements augmentation
                    if mode == 'train' and torch.rand(1) < 1.1:
                        if len(encoded_history) > 0:
                            keep_num = int(torch.randint(0, len(encoded_history), (1,)))  # drops at least one element
                            # randomly sample keep_num elements from history
                            keep_idxs = random.sample(range(len(encoded_history)), keep_num)
                            encoded_history = [encoded_history[i] for i in sorted(keep_idxs)]
                            token_type_ids = [token_type_ids[i] for i in sorted(keep_idxs)]

                    # add current question
                    encoded_question = tokenizer.encode(question)[1:-1]
                    encoded_history.append(encoded_question)
                    token_type_ids.append([1] * len(encoded_question))  # current question is always type 1
                    # flatten
                    encoded_history = [item for sublist in encoded_history for item in sublist]
                    token_type_ids = [item for sublist in token_type_ids for item in sublist]

                    # cut off beginning of history if it is too long
                    if len(encoded_history) > max_num_question_tokens:
                        encoded_history = encoded_history[-max_num_question_tokens:]
                        token_type_ids = token_type_ids[-max_num_question_tokens:]

                    return encoded_history, token_type_ids


''' get history without augmentation for evaluation'''
def get_encoded_history_flat(tokenizer, history, question, max_num_question_tokens):
    encoded_history = []
    token_type_ids = []
    for q in history:
        encoded_question = tokenizer.encode(q[1][0])[1:-1]
        encoded_answer = tokenizer.encode(str(q[2][0]))[1:-1]
        encoded_history.append(encoded_question + [tokenizer.sep_token_id] + encoded_answer + [tokenizer.sep_token_id])
        token_type_ids.append([2] * (len(encoded_question) + 1) + [3] * (len(encoded_answer) + 1))

    # add current question
    encoded_question = tokenizer.encode(question)[1:-1]
    encoded_history.append(encoded_question)
    token_type_ids.append([1] * len(encoded_question))  # current question is always type 1
    # flatten
    encoded_history = [item for sublist in encoded_history for item in sublist]
    token_type_ids = [item for sublist in token_type_ids for item in sublist]

    # cut off beginning of history if it is too long
    if len(encoded_history) > max_num_question_tokens:
        encoded_history = encoded_history[-max_num_question_tokens:]
        token_type_ids = token_type_ids[-max_num_question_tokens:]

    return encoded_history, token_type_ids


# img_to_q_dict already contains only train questions when in training and all questions when in validation
def encode_text_progressive(image_name, question, question_id, question_type, img_to_q_dict, tokenizer, args, mode):
    if image_name is None: #for auto-regressive evaluation / inference
        history = img_to_q_dict # here img_to_q_dict is a list of questions, manually set depending on what was already answered
        encoded_history_and_question, token_type_ids = get_encoded_history_flat(tokenizer, history, question, args.num_question_tokens)
    else:
        history = img_to_q_dict[image_name]
        encoded_history_and_question, token_type_ids = get_encoded_history(tokenizer, history, question, question_id, question_type,
                                                                           args.num_question_tokens, mode)

    tokens = [tokenizer.cls_token_id] + [tokenizer.sep_token_id] + encoded_history_and_question + [tokenizer.sep_token_id]
    token_type_ids = [1] + [1] + token_type_ids + [1]
    q_attn_mask = [1] * len(tokens)
    n_pad = args.max_position_embeddings - len(tokens)
    attn_mask = (args.num_image_tokens + len(tokens)) * [1] + (args.max_position_embeddings - len(tokens) - args.num_image_tokens) * [0]
    q_attn_mask.extend([0] * n_pad)
    tokens.extend([0] * n_pad)
    token_type_ids.extend([0] * n_pad)

    assert len(tokens) == args.max_position_embeddings
    assert len(q_attn_mask) == args.max_position_embeddings
    assert len(attn_mask) == args.max_position_embeddings
    assert len(token_type_ids) == args.max_position_embeddings

    return tokens, q_attn_mask, attn_mask, torch.tensor(token_type_ids, dtype=torch.long)


def encode_text(question, tokenizer, args):
    # get token ids and remove [CLS] and [SEP] token id
    q_tokens = tokenizer.encode(question)[1:-1]

    tokens = [tokenizer.cls_token_id] + [tokenizer.sep_token_id] + q_tokens[:args.num_question_tokens] + [tokenizer.sep_token_id]
    q_attn_mask = [1] * len(tokens)
    n_pad = args.max_position_embeddings - len(tokens)
    attn_mask = (args.num_image_tokens + len(tokens)) * [1] + (args.max_position_embeddings - len(tokens) - args.num_image_tokens) * [0]
    q_attn_mask.extend([0] * n_pad)
    tokens.extend([0] * n_pad)

    assert len(tokens) == args.max_position_embeddings
    assert len(q_attn_mask) == args.max_position_embeddings
    assert len(attn_mask) == args.max_position_embeddings

    return tokens, q_attn_mask, attn_mask


def get_question_level(question_type):
    levels = {'MODALITY': 0, 'PLANE': 1, 'ORGAN': 2,
              'PRES, ABN': 3, 'PRSE': 3, 'PRES': 3, 'COLOR, PRES': 3, 'COUNT': 3, 'ABN': 3,
              'COLOR': 4, 'PRES, COLOR': 4, 'POS': 4, 'PRES, POS': 4, 'POS, PRES': 4, 'POS, ABN': 4, 'ABN, POS': 4, 'SIZE, COLOR': 4,
              'SIZE': 4, 'SIZE, PRES': 4, 'ATTRIB, SIZE': 4, 'ATTRIB': 4, 'ATRIB': 4, 'ATTRIB, PRES': 4, 'PRES, ATTRIB': 4,
              'OTHER': 4, 'Other': 4}
    return levels[question_type]


def create_image_to_question_dict(train_df, val_df):
    # if cached file exists, load it, else create
    if os.path.exists('data/vqarad/cache/image_to_question_dict_train.json'):
        with open('data/vqarad/cache/image_to_question_dict_train.json', 'r') as f:
            image_to_question_dict_train = json.load(f)
        with open('data/vqarad/cache/image_to_question_dict_all.json', 'r') as f:
            image_to_question_dict_all = json.load(f)

    else:
        print("WARNING: img_to_question_dict not found, creating it now...")
        image_to_question_dict_train = {}
        for elem in train_df:
            image_name = elem['image_name']
            question = elem['question']
            question_id = elem['qid']
            question_type = elem['question_type']
            question_level = get_question_level(question_type)
            answer = elem['answer_text']
            if image_name not in image_to_question_dict_train:
                image_dict = {}
                for i in range(5):
                    image_dict[i] = []
                image_dict[question_level].append((question_id, question, answer))
                image_to_question_dict_train[image_name] = image_dict
            else:
                image_to_question_dict_train[image_name][question_level].append((question_id, question, answer))

        image_to_question_dict_all = {}
        df = train_df + val_df
        for elem in df:
            image_name = elem['image_name']
            question = elem['question']
            question_id = elem['qid']
            question_type = elem['question_type']
            question_level = get_question_level(question_type)
            answer = elem['answer_text']
            if image_name not in image_to_question_dict_all:
                image_dict = {}
                for i in range(5):
                    image_dict[i] = []
                image_dict[question_level].append((question_id, question, answer))
                image_to_question_dict_all[image_name] = image_dict
            else:
                image_to_question_dict_all[image_name][question_level].append((question_id, question, answer))

        # save to file
        with open('data/vqarad/cache/image_to_question_dict_train.json', 'w') as fp:
            json.dump(image_to_question_dict_train, fp)
        with open('data/vqarad/cache/image_to_question_dict_all.json', 'w') as fp:
            json.dump(image_to_question_dict_all, fp)

    return image_to_question_dict_train, image_to_question_dict_all


class VQARad(Dataset):
    """Standard per-question dataset"""
    def __init__(self, df, img_to_q_dict, tfm, args, mode='train'):
        self.df = df
        self.tfm = tfm
        self.args = args
        self.img_to_q_dict = img_to_q_dict

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)

        with open('data/vqarad/trainval_label2ans.pkl', 'rb') as f:
            self.label2ans = cPickle.load(f)

        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join('data/vqarad/images', self.df[idx]['image_name'])
        question = self.df[idx]['question']
        answer = self.df[idx]['answer']['labels'][0] if len(self.df[idx]['answer']['labels']) > 0 else -1  # answer not in train set
        answer_type = 0 if self.df[idx]['answer_type'] == 'CLOSED' else 1

        img = Image.open(path)
        if self.tfm:
            img = self.tfm(img)

        token_type_ids = torch.tensor(0)
        if self.args.progressive:
            tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(self.df[idx]['image_name'], question, self.df[idx]['qid'],
                                                                                     self.df[idx]['question_type'],
                                                                                     self.img_to_q_dict, self.tokenizer, self.args, mode=self.mode)
        else:
            tokens, q_attn_mask, attn_mask = encode_text(question, self.tokenizer, self.args)

        if self.mode == 'predict':
            image_name = self.df[idx]['image_name']
            history, _ = get_encoded_history(self.tokenizer, self.img_to_q_dict[image_name], question, self.df[idx]['qid'],
                                             self.df[idx]['question_type'], 259, mode='predict')

            return img, torch.tensor(tokens, dtype=torch.long), torch.tensor(q_attn_mask, dtype=torch.long), torch.tensor(attn_mask,
                                                                                                                          dtype=torch.long), torch.tensor(
                answer, dtype=torch.long), torch.tensor(answer_type, dtype=torch.long), \
                   token_type_ids, question, self.df[idx]['answer_text'], self.df[idx]['image_name'], self.tokenizer.decode(history), self.df[idx][
                       'qid'], torch.tensor(0)
        else:
            return img, torch.tensor(tokens, dtype=torch.long), torch.tensor(q_attn_mask, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), \
                   torch.tensor(answer, dtype=torch.long), torch.tensor(answer_type, dtype=torch.long), token_type_ids, torch.tensor(0)


class VQARadEval(Dataset):
    """ Evaluation dataset for autoregressive inference. -> returns all questions for one image at once"""
    def __init__(self, df, train_df, img_to_q_dict, tfm, args):
        self.df = df
        self.val_qids = [elem['qid'] for elem in df]
        self.train_df = train_df
        self.train_qids = [elem['qid'] for elem in train_df]
        self.image_names = sorted(list(set([elem['image_name'] for elem in df])))
        self.tfm = tfm
        self.args = args
        self.img_to_q_dict = img_to_q_dict
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        path = os.path.join('data/vqarad/images', image_name)

        img = Image.open(path)

        if self.tfm:
            img = self.tfm(img)

        history = self.img_to_q_dict[image_name]
        # flatten history
        history = [item for sublist in history.values() for item in sublist]
        items = []
        for question in history:
            if question[0] in self.train_qids:
                item = ('train', question)
                items.append(item)
            else:  # validation question
                # get index of question in validation set from the qid
                val_idx = self.val_qids.index(question[0])
                item = ('val', self.df[val_idx])
                items.append(item)

        return img, items
