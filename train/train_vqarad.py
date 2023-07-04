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

from data_utils.data_vqarad import _load_dataset, create_image_to_question_dict, VQARad
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
    parser.add_argument('--num_workers', type=int, required=False, default=12, help="number of workers")
    parser.add_argument('--epochs', type=int, required=False, default=100, help="num epochs to train")

    parser.add_argument('--max_position_embeddings', type=int, required=False, default=12, help="max length of sequence")
    parser.add_argument('--max_answer_len', type=int, required=False, default=29, help="padding length for free-text answers")
    parser.add_argument('--batch_size', type=int, required=False, default=16, help="batch size")
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help="learning rate'")
    parser.add_argument('--hidden_dropout_prob', type=float, required=False, default=0.3, help="hidden dropout probability")
    parser.add_argument('--smoothing', type=float, required=False, default=None, help="label smoothing")

    parser.add_argument('--img_feat_size', type=int, required=True, default=14, help="dimension of last pooling layer of img encoder")
    parser.add_argument('--num_question_tokens', type=int, required=False, default=259, help="number of tokens for question")
    parser.add_argument('--hidden_size', type=int, required=False, default=768, help="hidden size")
    parser.add_argument('--vocab_size', type=int, required=False, default=30522, help="vocab size")
    parser.add_argument('--type_vocab_size', type=int, required=False, default=2, help="type vocab size")
    parser.add_argument('--heads', type=int, required=False, default=16, help="heads")
    parser.add_argument('--n_layers', type=int, required=False, default=1, help="num of fusion layers")
    parser.add_argument('--acc_grad_batches', type=int, required=False, default=None, help="how many batches to accumulate gradients")

    ''' only relevant for radrestruct'''
    parser.add_argument('--classifier_dropout', type=float, required=False, default=0.0, help="how often should image be dropped")
    parser.add_argument('--match_instances', action='store_true', default=False, help="do optimal instance matching")

    args = parser.parse_args()

    args.num_image_tokens = args.img_feat_size ** 2
    args.max_position_embeddings = args.num_image_tokens + 3 + args.num_question_tokens  # 3 for [CLS], [SEP], [SEP]
    args.hidden_size_img_enc = args.hidden_size
    if args.progressive:
        args.hidden_size_img_enc = args.hidden_size
        args.num_question_tokens = args.max_position_embeddings - 3 - args.num_image_tokens

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

    summary(model, device='gpu')

    if args.use_pretrained:
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.model_dir, map_location=torch.device('cpu'))['state_dict'])
        assert len(missing_keys) == 0
        assert len(unexpected_keys) == 0

    img_tfm = model.model.image_encoder.img_tfm
    norm_tfm = model.model.image_encoder.norm_tfm
    resize_size = model.model.image_encoder.resize_size

    aug_tfm = transforms.Compose([transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                                  # Cutout(),
                                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                  transforms.RandomResizedCrop(resize_size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                                  transforms.RandomRotation(10)])

    train_tfm = transforms.Compose([img_tfm, aug_tfm, norm_tfm]) if norm_tfm is not None else transforms.Compose([img_tfm, aug_tfm])
    test_tfm = transforms.Compose([img_tfm, norm_tfm]) if norm_tfm is not None else img_tfm

    traindataset = VQARad(train_df, img_to_q_dict_train, tfm=train_tfm, args=args, mode='train')
    valdataset = VQARad(val_df, img_to_q_dict_all, tfm=test_tfm, args=args, mode='val')

    trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger = pl.loggers.TensorBoardLogger('runs', name=args.run_name, version=0)

    if args.output == 'class':
        checkpoint_callback = ModelCheckpoint(monitor='Acc/val_clean', dirpath=os.path.join(args.save_dir, args.run_name), filename='{epoch}-{Acc/val_clean:.2f}',
                                              mode='max', every_n_epochs=1, save_last=True)
    else:
        checkpoint_callback = ModelCheckpoint(monitor='Acc/val', dirpath=args.save_dir, filename=args.run_name + '/{epoch}-{Acc/val:.2f}', mode='max',
                                              every_n_epochs=1, save_last=True)
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.epochs,
        precision=16 if args.mixed_precision and torch.cuda.is_available() else 32,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.acc_grad_batches,
        logger=logger,
        callbacks=[checkpoint_callback],
        benchmark=False,
        deterministic=True
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
