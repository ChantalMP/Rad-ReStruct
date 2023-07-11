import _pickle as cPickle
import argparse
import json
import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_utils.data_vqarad import _load_dataset
from data_utils.data_vqarad import create_image_to_question_dict, VQARadEval, encode_text_progressive
from net.model import ModelWrapper

warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Finetune on VQARAD")

    parser.add_argument('--run_name', type=str, required=False, default="debug", help="run name for wandb")
    parser.add_argument('--data_dir', type=str, required=False, default="data/vqarad", help="path for data")
    parser.add_argument('--model_dir', type=str, required=False, default="",
                        help="path to load weights")
    parser.add_argument('--save_dir', type=str, required=False, default="checkpoints", help="path to save weights")
    parser.add_argument('--question_type', type=str, required=False, default=None, help="choose specific category if you want")
    parser.add_argument('--use_pretrained', action='store_true', default=False, help="use pretrained weights or not")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help="use mixed precision or not")
    parser.add_argument('--clip', action='store_true', default=False, help="clip the gradients or not")
    parser.add_argument('--bert_model', type=str, required=False, default="zzxslp/RadBERT-RoBERTa-4m", help="pretrained question encoder weights")
    parser.add_argument('--progressive', action='store_true', default=False, help="use progressive answering of questions")

    parser.add_argument('--seed', type=int, required=False, default=42, help="set seed for reproducibility")
    parser.add_argument('--num_workers', type=int, required=False, default=4, help="number of workers")
    parser.add_argument('--epochs', type=int, required=False, default=100, help="num epochs to train")

    parser.add_argument('--max_position_embeddings', type=int, required=False, default=12, help="max length of sequence")
    parser.add_argument('--batch_size', type=int, required=False, default=16, help="batch size")
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help="learning rate'")
    parser.add_argument('--hidden_dropout_prob', type=float, required=False, default=0.3, help="hidden dropout probability")
    parser.add_argument('--smoothing', type=float, required=False, default=None, help="label smoothing")

    parser.add_argument('--img_feat_size', type=int, required=False, default=14, help="dimension of last pooling layer of img encoder")
    parser.add_argument('--num_question_tokens', type=int, required=False, default=20, help="number of tokens for question")
    parser.add_argument('--hidden_size', type=int, required=False, default=768, help="hidden size")
    parser.add_argument('--vocab_size', type=int, required=False, default=30522, help="vocab size")
    parser.add_argument('--type_vocab_size', type=int, required=False, default=2, help="type vocab size")
    parser.add_argument('--heads', type=int, required=False, default=16, help="heads")
    parser.add_argument('--n_layers', type=int, required=False, default=1, help="num of layers")
    parser.add_argument('--acc_grad_batches', type=int, required=False, default=None, help="how many batches to accumulate gradients")

    ''' only relevant for radrestruct'''
    parser.add_argument('--classifier_dropout', type=float, required=False, default=0.0, help="how often should image be dropped")
    parser.add_argument('--match_instances', action='store_true', default=False, help="do optimal instance matching")

    args = parser.parse_args()

    args.num_image_tokens = args.img_feat_size ** 2
    args.max_position_embeddings = args.num_image_tokens + 3 + args.num_question_tokens  # 3 for [CLS], [SEP], [SEP]
    args.hidden_size_img_enc = args.hidden_size
    if args.progressive:
        args.max_position_embeddings = 458
        args.hidden_size_img_enc = args.hidden_size
        args.num_question_tokens = 458 - 3 - args.num_image_tokens

    # create directory for saving params
    if not os.path.exists(f'{args.save_dir}/{args.run_name}'):
        os.makedirs(f'{args.save_dir}/{args.run_name}')
    with open(os.path.join(args.save_dir, f'{args.run_name}/commandline_args.txt'), 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    pl.seed_everything(args.seed, workers=True)

    train_df = _load_dataset(args.data_dir, 'train')
    val_df = _load_dataset(args.data_dir, 'test')

    img_to_q_dict_train, img_to_q_dict_all = create_image_to_question_dict(train_df, val_df)

    args.num_classes = 458

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ModelWrapper(args, train_df, val_df)

    # use torchinfo to see model architecture and trainable parameters
    from torchinfo import summary

    summary(model)

    if args.use_pretrained:
        model.load_state_dict(torch.load(args.model_dir, map_location=torch.device('cpu'))['state_dict'])

    img_tfm = model.model.image_encoder.img_tfm
    norm_tfm = model.model.image_encoder.norm_tfm
    resize_size = model.model.image_encoder.resize_size

    test_tfm = transforms.Compose([img_tfm, norm_tfm])

    autoregr_valdataset = VQARadEval(val_df, train_df, img_to_q_dict_all, tfm=test_tfm, args=args)

    valloader = DataLoader(autoregr_valdataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    logger = pl.loggers.TensorBoardLogger('runs', name=args.run_name, version=0)

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=args.epochs,
        precision=16 if args.mixed_precision else 32,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.acc_grad_batches,
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor='Acc/val_clean', dirpath=args.save_dir, filename=args.run_name + '/{epoch}-{Acc/val_clean:.2f}', mode='max',
                            every_n_epochs=1)],
        benchmark=False,
        deterministic=True
    )

    # preds = trainer.predict(model, valloader, return_predictions=True)
    # given the valloader and the predictions pred, compute the accuracy for each batch and create a list of the wrong examples
    # load trainval_label2ans.pkl
    with open('data/vqarad/trainval_label2ans.pkl', 'rb') as f:
        label2ans = cPickle.load(f)
    correct = 0
    total = 0
    model.eval()
    for i, batch in tqdm(enumerate(valloader)):
        img, items = batch
        current_history = []
        for question in items:
            if question[0][0] == 'train':
                current_history.append(question[1])
            else:
                target = question[1]['answer']['labels'][0] if len(question[1]['answer']['labels']) > 0 else -1
                question_token, q_attention_mask, attn_mask, token_type_ids_q = \
                    encode_text_progressive(None, question[1]['question'][0], question[1]['qid'][0],
                                            question[1]['question_type'][0], current_history,
                                            autoregr_valdataset.tokenizer, args, mode='val')

                # convert all to tensor
                question_token = torch.tensor(question_token).unsqueeze(0)
                q_attention_mask = torch.tensor(q_attention_mask).unsqueeze(0)
                attn_mask = torch.tensor(attn_mask).unsqueeze(0)
                token_type_ids_q = torch.tensor(token_type_ids_q).unsqueeze(0)
                with torch.cuda.amp.autocast():
                    logits, attentions = model(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, mode='autoregressive_val')
                pred = torch.argmax(logits, dim=1)
                text_pred = label2ans[pred.item()]

                if target != -1:
                    text_gt = question[1]['answer_text'][0]
                else:
                    text_gt = question[1]['answer_text'][0]

                correct += (pred == target).sum().item()
                total += 1

                current_history.append((question[1]['qid'][0], (question[1]['question'][0],), (text_pred,)))

    print(correct)
    print(total)
    print(f'Accuracy: {correct / total * 100.:.2f}')
