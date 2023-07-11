import itertools
import json
import warnings
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils.data_radrestruct import encode_text_progressive
from data_utils.preprocessing_radrestruct import get_topic_question, get_question
from evaluation.defs import *

warnings.simplefilter("ignore", UserWarning)


def get_value(out, info, answer_options):
    option_idxs = [answer_options[option] for option in info['options']]
    soft_pred = out[0, option_idxs].sigmoid().detach().cpu().numpy()
    pred = (soft_pred > 0.5).astype(int)

    if info['answer_type'] == 'single_choice':
        # only select choice with highest score
        max_idx = np.argmax(soft_pred)
        pred = np.zeros_like(pred)
        pred[max_idx] = 1

    elif info['answer_type'] == 'multi_choice':
        if 'no selection' in info['options']:
            no_selection_idx = info['options'].index('no selection')
        elif 'unspecified' in info['options']:
            no_selection_idx = info['options'].index('unspecified')
        else:
            no_selection_idx = None
        # if more than one selection predicted, set "no_selection" to 0
        if np.sum(pred) > 1 and no_selection_idx is not None:
            pred[no_selection_idx] = 0
        # if nothing predicted, set 'no_selection' to 1
        elif np.sum(pred) == 0 and no_selection_idx is not None:
            pred[no_selection_idx] = 1

    # language answers: get all elements in info['options'] that are predicted
    language_answers = [info['options'][i] for i in range(len(pred)) if pred[i] == 1]

    return language_answers, pred


def iterate_instances_VQA(model, img, elem, question, elem_name, topic_name, area_name, history, tokenizer, args, pred_vector, report_keys,
                          max_instances, answer_options, report_vector_gt=None, match_instances=False):
    infos = elem["infos"] if topic_name != "infos" else elem
    elem_history = deepcopy(history)

    gt_instances = elem["instances"]
    instance_keys = list(infos.keys())
    if "instances" in instance_keys:
        instance_keys.remove("instances")
    # make sure all instances have same structures
    for i in range(len(gt_instances)):
        assert instance_keys == list(gt_instances[i].keys())
    no_predicted = False

    max_instance_key = f"{area_name}/{topic_name}" if topic_name == 'infos' else f"{area_name}/{elem_name}"
    max_num_occurences = max_instances[max_instance_key] if max_instance_key in max_instances else 1
    pred_instances = []
    neg_pred_instances = []
    dummy_pred_vector = deepcopy(pred_vector)

    for instance_idx in range(max_num_occurences):
        curr_pred = []
        curr_pred_neg = []
        if not no_predicted:
            if instance_idx == 0:  # prediction before was positive, otherwise iterate_instances is not called
                q_positive_pred = True
            else:
                # generate follow-up question
                question = get_question(elem_name, topic_name, area_name, first_instance=False)
                # make prediction
                tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, elem_history, tokenizer, mode='val', args=args)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                               q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                               attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                               token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')

                q_positive_pred = torch.argmax(out[0, [58, 95]]) == 1

            if not q_positive_pred:  # no was predicted for further instance
                if match_instances:
                    curr_pred_neg.extend(NO)
                    dummy_pred_vector.extend(NO)
                    lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(dummy_pred_vector):]) if
                                           key.startswith(
                                               f"{area_name}_{topic_name}_{elem_name}_" if topic_name != "infos" else f"{area_name}_{topic_name}_") and key.endswith(
                                               str(instance_idx))]

                else:
                    pred_vector.extend(NO)
                    report_key_gen = f"{area_name}_{topic_name}_{elem_name}_yes" if topic_name != 'infos' else f"{area_name}_{topic_name}_yes"
                    report_key = report_keys[len(pred_vector) - 2]
                    if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith(
                            "3") or report_key.endswith("4"):
                        report_key = report_key[:-2]
                    assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                    lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(pred_vector):]) if
                                           key.startswith(
                                               f"{area_name}_{topic_name}_{elem_name}_" if topic_name != "infos" else f"{area_name}_{topic_name}_") and key.endswith(
                                               str(instance_idx))]
                # add -2 to pred_vector for all elements in lower hierarchy ids
                if match_instances:
                    curr_pred_neg.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))
                    dummy_pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))
                else:
                    pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))
                no_predicted = True

            else:
                elem_history.append((question, ["yes"]))
                if match_instances:
                    curr_pred.extend(YES)
                    dummy_pred_vector.extend(YES)
                else:
                    pred_vector.extend(YES)
                    report_key_gen = f"{area_name}_{topic_name}_{elem_name}_yes" if topic_name != "infos" else f"{area_name}_{topic_name}_yes"
                    report_key = report_keys[len(pred_vector) - 2]
                    if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith(
                            "3") or report_key.endswith("4"):
                        report_key = report_key[:-2]
                    assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"

                for key in instance_keys:  # different "infos" instances
                    if key == "body_region":
                        question = "In which part of the body?"
                        tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, elem_history, tokenizer, mode='val',
                                                                                                 args=args)
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                                       q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                                       attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                                       token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')

                        language_answers, pred = get_value(out, infos[key], answer_options)
                        if match_instances:
                            curr_pred.extend(pred)
                            dummy_pred_vector.extend(pred)
                        else:
                            pred_vector.extend(pred)
                            report_key_gen = f"{area_name}_{topic_name}_{elem_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}" if topic_name != "infos" else f"{area_name}_{topic_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}"
                            report_key = report_keys[len(pred_vector) - 1]
                            assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                        elem_history.append((question, language_answers))
                    elif key == "localization":
                        question = "In which area?"
                        tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, elem_history, tokenizer, mode='val',
                                                                                                 args=args)
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                                       q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                                       attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                                       token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')

                        language_answers, pred = get_value(out, infos[key], answer_options)
                        if match_instances:
                            curr_pred.extend(pred)
                            dummy_pred_vector.extend(pred)
                        else:
                            pred_vector.extend(pred)
                            report_key_gen = f"{area_name}_{topic_name}_{elem_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}" if topic_name != "infos" else f"{area_name}_{topic_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}"
                            report_key = report_keys[len(pred_vector) - 1]
                            assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                        elem_history.append((question, language_answers))
                    elif key == "attributes":
                        question = "What are the attributes?"
                        tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, elem_history, tokenizer, mode='val',
                                                                                                 args=args)
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                                       q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                                       attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                                       token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')

                        language_answers, pred = get_value(out, infos[key], answer_options)
                        if match_instances:
                            curr_pred.extend(pred)
                            dummy_pred_vector.extend(pred)
                        else:
                            pred_vector.extend(pred)
                            report_key_gen = f"{area_name}_{topic_name}_{elem_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}" if topic_name != "infos" else f"{area_name}_{topic_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}"
                            report_key = report_keys[len(pred_vector) - 1]
                            assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                        elem_history.append((question, language_answers))
                    elif key == "degree":
                        question = "What is the degree?"
                        tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, elem_history, tokenizer, mode='val',
                                                                                                 args=args)
                        # with torch.cuda.amp.autocast():
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                                       q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                                       attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                                       token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')

                        language_answers, pred = get_value(out, infos[key], answer_options)
                        if match_instances:
                            curr_pred.extend(pred)
                            dummy_pred_vector.extend(pred)
                        else:
                            pred_vector.extend(pred)
                            report_key_gen = f"{area_name}_{topic_name}_{elem_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}" if topic_name != "infos" else f"{area_name}_{topic_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}"
                            report_key = report_keys[len(pred_vector) - 1]
                            assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                        elem_history.append((question, language_answers))


        else:
            # just add negative dummy answer until max_instances are filled
            if match_instances:
                curr_pred_neg.extend(NO)
                dummy_pred_vector.extend(NO)
                lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(dummy_pred_vector):]) if
                                       key.startswith(
                                           f"{area_name}_{topic_name}_{elem_name}_" if topic_name != "infos" else f"{area_name}_{topic_name}_") and key.endswith(
                                           str(instance_idx))]

            else:
                pred_vector.extend(NO)
                report_key_gen = f"{area_name}_{topic_name}_{elem_name}_yes" if topic_name != "infos" else f"{area_name}_{topic_name}_yes"
                report_key = report_keys[len(pred_vector) - 2]
                if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith(
                        "3") or report_key.endswith("4"):
                    report_key = report_key[:-2]
                assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(pred_vector):]) if
                                       key.startswith(
                                           f"{area_name}_{topic_name}_{elem_name}_" if topic_name != "infos" else f"{area_name}_{topic_name}_") and key.endswith(
                                           str(instance_idx))]
            # add -2 to pred_vector for all elements in lower hierarchy ids
            if match_instances:
                curr_pred_neg.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))
                dummy_pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))
            else:
                pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))

        if match_instances:
            if len(curr_pred) > 0:
                pred_instances.append(curr_pred)
            if len(curr_pred_neg) > 0:
                neg_pred_instances.append(curr_pred_neg)
            assert len(curr_pred) > 0 or len(curr_pred_neg) > 0, f"Neither positive nor negative prediction for {area_name}_{topic_name}_{elem_name}"

    if match_instances:
        # find best match between gt_instances and pred_instances (lowest F1 score)
        # don't match predictions to empty gt if there are other gt instances
        num_gt = len(gt_instances)
        num_pred = len(pred_instances)

        if num_gt == 0 or num_pred == 0:
            # add flattened pred_instances, followed by neg_pred_instances to pred_vector in order they were predicted
            pred_vector.extend([item for sublist in pred_instances for item in sublist])
            pred_vector.extend([item for sublist in neg_pred_instances for item in sublist])
            assert pred_vector == dummy_pred_vector

        else:
            # get vectors from target for gt_instances
            instance_len = len(pred_instances[0])
            start_idx = len(pred_vector)  # where the current finding starts
            gt_vectors = []
            for gt_instance in gt_instances:
                gt_vector = report_vector_gt[0, start_idx:start_idx + instance_len]
                start_idx += instance_len
                gt_vectors.append(gt_vector)
                assert (gt_vector[:2] == torch.tensor(YES)).all()  # should only be positive instances starti

            gt_idxs = list(range(len(gt_vectors)))  # only positive instances
            pred_idxs = list(range(len(pred_instances)))  # only positive instances

            if len(gt_idxs) > len(pred_idxs):
                gt_permutations = list(itertools.permutations(gt_idxs, len(pred_idxs)))
                matchings = [(torch.tensor(gt), torch.tensor(pred_idxs)) for gt in gt_permutations]
            else:
                pred_permutations = list(itertools.permutations(pred_idxs, len(gt_idxs)))
                matchings = [(torch.tensor(gt_idxs), torch.tensor(pred)) for pred in pred_permutations]

            # Compute F1 scores for all possible matchings
            f1_scores = {}
            for gt_idxs_match, pred_idxs_match in matchings:
                # get concatenated gt vector in order of gt_idxs
                gt_vector = torch.cat([gt_vectors[idx] for idx in gt_idxs_match], dim=0)
                # get concatenated pred vector in order of pred_idxs
                preds_vector = torch.cat([torch.tensor(pred_instances[idx]) for idx in pred_idxs_match], dim=0)
                # compute F1 score
                f1_scores[(gt_idxs_match, pred_idxs_match)] = f1_score(gt_vector, preds_vector, average='macro')

            # get best matching
            best_matching = max(f1_scores, key=f1_scores.get)
            gt_idxs_best, pred_idxs_best = best_matching

            # construct what to add to pred_vector using best_matching
            # sort gt_idxs and pred_idxs such that gt_idxs is in ascending order
            gt_idxs_best, pred_idxs_best = zip(*sorted(zip(gt_idxs_best, pred_idxs_best)))
            unmatched_pred_idxs = [idx for idx in range(len(pred_instances)) if idx not in pred_idxs_best]
            # create flattened pred vector
            # in ascending order of gt_idxs, if idx was matched add matched pred to vector, else add one of the neg_pred_instances
            for idx in range(max_num_occurences):
                if idx in gt_idxs_best:
                    pred_idx = pred_idxs_best[gt_idxs_best.index(idx)]
                    pred_vector.extend(pred_instances[pred_idx])
                else:
                    if len(unmatched_pred_idxs) > 0:
                        pred_idx = unmatched_pred_idxs.pop(0)
                        pred_vector.extend(pred_instances[pred_idx])
                    else:
                        # add one of the neg_pred_instances
                        pred_vector.extend(neg_pred_instances[0])
                        neg_pred_instances = neg_pred_instances[1:]

    return pred_vector


def iterate_area_VQA(img, area, area_name, model, tokenizer, args, max_instances, pred_vector, report_keys, answer_options, report_vector_gt,
                     match_instances):
    for topic_name, topic in area.items():
        if topic_name == 'area':
            continue

        # get prediction for topic question
        area_question = get_topic_question(topic_name, area_name)
        tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(area_question, history=[], tokenizer=tokenizer, mode='val',
                                                                                 args=args)
        # with torch.cuda.amp.autocast():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                       q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                       attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0), token_type_ids_q=token_type_ids.unsqueeze(0),
                       mode='val')

        area_positive_pred = torch.argmax(out[0, [58, 95]]) == 1  # yes was predicted

        if not area_positive_pred:  # we predicted no -> don't answer any following questions
            pred_vector.extend(NO)
            # get report_key for current question
            report_key_gen = f"{area_name}_{topic_name}_yes"
            report_key = report_keys[len(pred_vector) - 2]
            if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith("3") or report_key.endswith(
                    "4"):
                report_key = report_key[:-2]
            assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
            lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(pred_vector):]) if
                                   key.startswith(f"{area_name}_{topic_name}_")]

            # add -2 to pred_vector for all elements in lower hierarchy ids
            pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))

        else:  # predicted yes -> iterate through all elements
            history = []
            if topic_name != 'infos':
                pred_vector.extend(YES)
                report_key_gen = f"{area_name}_{topic_name}_yes"
                report_key = report_keys[len(pred_vector) - 2]
                if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith(
                        "3") or report_key.endswith("4"):
                    report_key = report_key[:-2]
                assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                history.append((area_question, ["yes"]))

                for elem_name, elem in area[topic_name].items():
                    question = get_question(elem_name, topic_name, area_name, first_instance=True)
                    tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, history, tokenizer, mode='val', args=args)
                    # with torch.cuda.amp.autocast():
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                                   q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                                   attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                                   token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')

                    elem_positive_pred = torch.argmax(out[0, [58, 95]]) == 1

                    if not elem_positive_pred:
                        pred_vector.extend(NO)
                        lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(pred_vector):]) if
                                               key.startswith(f"{area_name}_{topic_name}_{elem_name}_")]
                        report_key_gen = f"{area_name}_{topic_name}_{elem_name}_yes"
                        report_key = report_keys[len(pred_vector) - 2]
                        if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith(
                                "3") or report_key.endswith("4"):
                            report_key = report_key[:-2]
                        assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                        # add -2 to pred_vector for all elements in lower hierarchy ids
                        pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))


                    else:  # positive prediction
                        pred_vector = iterate_instances_VQA(model, img, elem, question, elem_name, topic_name, area_name, history, tokenizer, args,
                                                            pred_vector, report_keys, max_instances, answer_options, report_vector_gt,
                                                            match_instances)

            else:
                question = get_question(area_name, topic_name, area_name, first_instance=True)
                pred_vector = iterate_instances_VQA(model, img, topic, question, area_name, topic_name, area_name, history, tokenizer, args,
                                                    pred_vector, report_keys, max_instances, answer_options, report_vector_gt, match_instances)

    return pred_vector


def predict_autoregressive_VQA(model, valloader, args):
    model.eval()
    match_instances = args.match_instances
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)

    with open('data/radrestruct/max_instances.json', 'r') as f:
        max_instances = json.loads(f.read())
    with open(f'data/radrestruct/train_vectorized_answers/1.json', 'r') as f:
        report_keys = list(json.load(f).keys())  # same for all reports
    answer_options = valloader.dataset.answer_options

    preds = []

    for i, batch in tqdm(enumerate(valloader)):
        img, report, report_vector_gt = batch
        assert len(report) == 1
        report = list(report[0].values())[0]
        pred_vector = []
        # iterate through report, answer each question
        # generate history for following questions
        # stop generation if no -> collect metrics accordingly

        # iterate over all findings
        for area in report:
            if "sub_areas" in area:
                for sub_area_name, sub_area in area["sub_areas"].items():
                    pred_vector = iterate_area_VQA(img, sub_area, sub_area_name, model, tokenizer, args, max_instances, pred_vector, report_keys,
                                                   answer_options, report_vector_gt, match_instances)

            else:
                pred_vector = iterate_area_VQA(img, area, area['area'], model, tokenizer, args, max_instances, pred_vector, report_keys,
                                               answer_options, report_vector_gt, match_instances)

        assert len(pred_vector) == len(report_keys)
        preds.append(pred_vector)

    return np.array(preds, dtype=np.float32)
