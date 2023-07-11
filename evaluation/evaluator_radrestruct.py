import json

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from evaluation.defs import *


class AutoregressiveEvaluator:
    def __init__(self):
        # load report_keys
        with open('data/radrestruct/report_keys.json', 'r') as f:
            self.report_keys = json.load(f)

        # convert to list only including the actual answers (last element)
        self.report_labels = np.array([key.split('_')[-1] if key.split('_')[-1] in ['yes', 'no'] else key.split('_')[-2] for key in self.report_keys])

        with open('data/radrestruct/vectorized_choice_options.json', 'r') as f:
            self.choice_options = json.load(f)
        with open('data/radrestruct/vectorized_question_ids.json', 'r') as f:
            self.question_ids = json.load(f)

        with open('data/radrestruct/answer_options.json', 'r') as f:
            self.answer_options = json.load(f)

    def iterate_area(self, area, area_name, pred, max_instances, curr_q_id):
        for topic_name, topic in area.items():
            if topic_name == 'area':
                continue

            if topic_name == 'infos':
                if f"{area_name}/{topic_name}" in max_instances:
                    max_num_occurences = max_instances[f"{area_name}/{topic_name}"]
                else:
                    max_num_occurences = 1

                prev_pred = np.array(UNDEFINED)  # not existing yet
                for i in range(max_num_occurences):
                    # options are (yes, no) -> only continue if answer is yes
                    q_id_idxs = np.where(np.array(self.question_ids) == int(curr_q_id))
                    topic_pred = pred[q_id_idxs]
                    if (topic_pred == NOT_PREDICTED).all() or (
                            prev_pred == np.array(NO)).all():  # no answer was predicted because of previous instance no
                        assert i != 0  # should never for first iteration
                        pred[q_id_idxs] = NO
                        topic_pred = pred[q_id_idxs]
                    prev_pred = topic_pred
                    curr_q_id += 1

                    for info_name, info in topic.items():
                        if info_name != 'instances':
                            q_id_idxs = np.where(np.array(self.question_ids) == int(curr_q_id))
                            if topic_pred[1] == 1:  # pred is no -> below all binary questions should be set to no and all others have no selection
                                options = self.report_labels[q_id_idxs]
                                if 'yes' in options:
                                    pred[q_id_idxs] = NO
                                else:
                                    pred[q_id_idxs] = 0

                            else:  # pred is yes -> don't change lower predictions
                                pass
                            curr_q_id += 1


            else:
                q_id_idxs = np.where(np.array(self.question_ids) == int(curr_q_id))
                area_pred = pred[q_id_idxs]
                assert NOT_PREDICTED not in area_pred, f"topic_pred should not contain -2, but is {area_pred}"
                curr_q_id += 1

                for elem_name, elem in area[topic_name].items():
                    if f"{area_name}/{elem_name}" in max_instances:
                        max_num_occurences = max_instances[f"{area_name}/{elem_name}"]
                    else:
                        max_num_occurences = 1

                    prev_pred = np.array(UNDEFINED)  # not existing yet
                    for i in range(max_num_occurences):
                        q_id_idxs = np.where(np.array(self.question_ids) == int(curr_q_id))
                        topic_pred = pred[q_id_idxs]
                        if (topic_pred == NOT_PREDICTED).all() or (area_pred == np.array(NO)).all() or (prev_pred == np.array(
                                NO)).all():  # no answer was predicted because of higher hierarchy no (VQA) or higher pred was no (bl) or previous instance is no-> answer should be set to no
                            pred[q_id_idxs] = NO
                            topic_pred = pred[q_id_idxs]
                        prev_pred = topic_pred
                        assert NOT_PREDICTED not in topic_pred, f"topic_pred should not contain -2, but is {topic_pred}"
                        curr_q_id += 1

                        for info_name, info in elem['infos'].items():
                            q_id_idxs = np.where(np.array(self.question_ids) == int(curr_q_id))
                            if area_pred[0] == 1 and topic_pred[0] == 1:  # pred is yes -> don't adapt prediction
                                pass
                            else:  # pred is no at some point higher -> all following questions are answered with no / not answered
                                options = self.report_labels[q_id_idxs]
                                if 'yes' in options:  # binary question -> set to no
                                    pred[q_id_idxs] = NO
                                else:
                                    pred[q_id_idxs] = 0  # attribute question -> no answer is selected

                            curr_q_id += 1

        return curr_q_id, pred

    def adapt_vector_autoregressive(self, preds):
        """ensures completeness by setting all answers to no / nothing selected if a higher hierarchy answer is no already"""

        # load max_instances.json
        with open('data/radrestruct/max_instances.json', 'r') as f:
            max_instances = json.loads(f.read())

        report_path = f'data/radrestruct/template_final_clean.json'
        with open(report_path, 'r') as f:
            report_template = json.loads(f.read())

        for idx, pred in enumerate(preds):
            curr_q_id = 0
            for area in report_template:
                if "sub_areas" in area:
                    for sub_area_name, sub_area in area["sub_areas"].items():
                        curr_q_id, pred = self.iterate_area(sub_area, sub_area_name, pred, max_instances, curr_q_id)

                else:
                    curr_q_id, pred = self.iterate_area(area, area['area'], pred, max_instances, curr_q_id)
            preds[idx] = pred
            assert NOT_PREDICTED not in pred

        return preds

    def clean_preds(self, preds):
        # dont change the original preds
        preds = preds.copy()

        # pre-processing
        for idx in range(len(preds)):  # iterate through all samples
            for q_id, choice_option in self.choice_options.items():  # iterate through all questions
                if choice_option == "single_choice":
                    q_id_idxs = np.where(np.array(self.question_ids) == int(q_id))
                    # only one answer should be predicted for single_choice questions
                    assert (preds[
                                idx, q_id_idxs].sum() <= 1).all()  # 0 is ok, happens when nothing is selected because question was not even answered (higher no)

                elif choice_option == 'multi_choice':
                    q_id_idxs = np.where(np.array(self.question_ids) == int(q_id))
                    if 'unspecified' in self.report_labels[q_id_idxs]:
                        no_selection_idx = np.where(self.report_labels[q_id_idxs] == 'unspecified')[0][0]
                        no_selection_idx = q_id_idxs[0][no_selection_idx]
                    elif 'no selection' in self.report_labels[q_id_idxs]:
                        no_selection_idx = np.where(self.report_labels[q_id_idxs] == 'no selection')[0][0]
                        no_selection_idx = q_id_idxs[0][no_selection_idx]
                    else:
                        no_selection_idx = None

                    # if more than one selection predicted, set "no_selection" to 0
                    # if exactly one selection predicted, it is already either only no_selection or only one of the other options so no need to change anything
                    if sum(preds[idx][q_id_idxs]) > 1 and no_selection_idx is not None:
                        preds[idx][no_selection_idx] = 0
                    # if nothing predicted, set 'no_selection' to 1
                    elif sum(preds[idx][q_id_idxs]) == 0 and no_selection_idx is not None:
                        preds[idx][no_selection_idx] = 1

        preds = self.adapt_vector_autoregressive(preds)

        return preds

    def calculate_detailed_metrics(self, preds, targets):

        layer_1_idxs = [idx for idx, key in enumerate(self.report_keys) if key.endswith('_no') or key.endswith('_yes')]
        layer_2_idxs = [idx for idx, key in enumerate(self.report_keys) if '_no_' in key or '_yes_' in key]
        layer_2_questions_diseases_idxs = [idx for idx, key in enumerate(self.report_keys) if
                                           ('_yes_' in key or '_no_' in key) and '_diseases_' in key]
        layer_2_questions_signs_idxs = [idx for idx, key in enumerate(self.report_keys) if ('_yes_' in key or '_no_' in key) and '_signs_' in key]
        layer_2_questions_objects_idxs = [idx for idx, key in enumerate(self.report_keys) if ('_yes_' in key or '_no_' in key) and '_objects_' in key]
        layer_2_questions_body_regions_idxs = [idx for idx, key in enumerate(self.report_keys) if
                                               ('_yes_' in key or '_no_' in key) and ('_body_regions_' in key or '_infos_' in key)]
        layer_3_questions_idxs = [idx for idx, key in enumerate(self.report_keys) if
                                  '_yes' not in key and '_no_' not in key and not key.endswith("_no")]

        class_report_l1 = classification_report(targets[:, layer_1_idxs], preds[:, layer_1_idxs], output_dict=True, zero_division=1)
        class_report_l2 = classification_report(targets[:, layer_2_idxs], preds[:, layer_2_idxs], output_dict=True, zero_division=1)
        class_report_l2_diseases = classification_report(targets[:, layer_2_questions_diseases_idxs], preds[:, layer_2_questions_diseases_idxs],
                                                         output_dict=True, zero_division=1)
        class_report_l2_signs = classification_report(targets[:, layer_2_questions_signs_idxs], preds[:, layer_2_questions_signs_idxs],
                                                      output_dict=True, zero_division=1)
        class_report_l2_objects = classification_report(targets[:, layer_2_questions_objects_idxs], preds[:, layer_2_questions_objects_idxs],
                                                        output_dict=True, zero_division=1)
        class_report_l2_body_regions = classification_report(targets[:, layer_2_questions_body_regions_idxs],
                                                             preds[:, layer_2_questions_body_regions_idxs], output_dict=True, zero_division=1)
        class_report_l3 = classification_report(targets[:, layer_3_questions_idxs], preds[:, layer_3_questions_idxs], output_dict=True,
                                                zero_division=1)

        f1_l1 = np.mean([class_report_l1[class_name]['f1-score'] for class_name in class_report_l1 if class_report_l1[class_name]['support'] > 0])
        f1_l2 = np.mean([class_report_l2[class_name]['f1-score'] for class_name in class_report_l2 if class_report_l2[class_name]['support'] > 0])
        f1_l2_diseases = np.mean([class_report_l2_diseases[class_name]['f1-score'] for class_name in class_report_l2_diseases if
                                  class_report_l2_diseases[class_name]['support'] > 0])
        f1_l2_signs = np.mean([class_report_l2_signs[class_name]['f1-score'] for class_name in class_report_l2_signs if
                               class_report_l2_signs[class_name]['support'] > 0])
        f1_l2_objects = np.mean([class_report_l2_objects[class_name]['f1-score'] for class_name in class_report_l2_objects if
                                 class_report_l2_objects[class_name]['support'] > 0])
        f1_l2_body_regions = np.mean([class_report_l2_body_regions[class_name]['f1-score'] for class_name in class_report_l2_body_regions if
                                      class_report_l2_body_regions[class_name]['support'] > 0])
        f1_l3 = np.mean([class_report_l3[class_name]['f1-score'] for class_name in class_report_l3 if class_report_l3[class_name]['support'] > 0])

        prec_l1 = np.mean([class_report_l1[class_name]['precision'] for class_name in class_report_l1 if class_report_l1[class_name]['support'] > 0])
        prec_l2 = np.mean([class_report_l2[class_name]['precision'] for class_name in class_report_l2 if class_report_l2[class_name]['support'] > 0])
        prec_l2_diseases = np.mean([class_report_l2_diseases[class_name]['precision'] for class_name in class_report_l2_diseases if
                                    class_report_l2_diseases[class_name]['support'] > 0])
        prec_l2_signs = np.mean([class_report_l2_signs[class_name]['precision'] for class_name in class_report_l2_signs if
                                 class_report_l2_signs[class_name]['support'] > 0])
        prec_l2_objects = np.mean([class_report_l2_objects[class_name]['precision'] for class_name in class_report_l2_objects if
                                   class_report_l2_objects[class_name]['support'] > 0])
        prec_l2_body_regions = np.mean([class_report_l2_body_regions[class_name]['precision'] for class_name in class_report_l2_body_regions if
                                        class_report_l2_body_regions[class_name]['support'] > 0])
        prec_l3 = np.mean([class_report_l3[class_name]['precision'] for class_name in class_report_l3 if class_report_l3[class_name]['support'] > 0])

        rec_l1 = np.mean([class_report_l1[class_name]['recall'] for class_name in class_report_l1 if class_report_l1[class_name]['support'] > 0])
        rec_l2 = np.mean([class_report_l2[class_name]['recall'] for class_name in class_report_l2 if class_report_l2[class_name]['support'] > 0])
        rec_l2_diseases = np.mean([class_report_l2_diseases[class_name]['recall'] for class_name in class_report_l2_diseases if
                                   class_report_l2_diseases[class_name]['support'] > 0])
        rec_l2_signs = np.mean(
            [class_report_l2_signs[class_name]['recall'] for class_name in class_report_l2_signs if class_report_l2_signs[class_name]['support'] > 0])
        rec_l2_objects = np.mean([class_report_l2_objects[class_name]['recall'] for class_name in class_report_l2_objects if
                                  class_report_l2_objects[class_name]['support'] > 0])
        rec_l2_body_regions = np.mean([class_report_l2_body_regions[class_name]['recall'] for class_name in class_report_l2_body_regions if
                                       class_report_l2_body_regions[class_name]['support'] > 0])
        rec_l3 = np.mean([class_report_l3[class_name]['recall'] for class_name in class_report_l3 if class_report_l3[class_name]['support'] > 0])

        acc_report_l1 = accuracy_score(targets[:, layer_1_idxs], preds[:, layer_1_idxs])
        acc_report_l2 = accuracy_score(targets[:, layer_2_idxs], preds[:, layer_2_idxs])
        acc_report_l2_diseases = accuracy_score(targets[:, layer_2_questions_diseases_idxs], preds[:, layer_2_questions_diseases_idxs])
        acc_report_l2_signs = accuracy_score(targets[:, layer_2_questions_signs_idxs], preds[:, layer_2_questions_signs_idxs])
        acc_report_l2_objects = accuracy_score(targets[:, layer_2_questions_objects_idxs], preds[:, layer_2_questions_objects_idxs])
        acc_report_l2_body_regions = accuracy_score(targets[:, layer_2_questions_body_regions_idxs],
                                                    preds[:, layer_2_questions_body_regions_idxs])
        acc_report_l3 = accuracy_score(targets[:, layer_3_questions_idxs], preds[:, layer_3_questions_idxs])

        detailed_metrics = {}
        detailed_metrics['l1'] = f"F1: {f1_l1:.4f} - Acc: {acc_report_l1:.4f} - Prec: {prec_l1:.4f} - Rec: {rec_l1:.4f}"
        detailed_metrics['l2'] = f"F1: {f1_l2:.4f} - Acc: {acc_report_l2:.4f} - Prec: {prec_l2:.4f} - Rec: {rec_l2:.4f}"
        detailed_metrics[
            'l2_diseases'] = f"F1: {f1_l2_diseases:.4f} - Acc: {acc_report_l2_diseases:.4f} - Prec: {prec_l2_diseases:.4f} - Rec: {rec_l2_diseases:.4f}"
        detailed_metrics['l2_signs'] = f"F1: {f1_l2_signs:.4f} - Acc: {acc_report_l2_signs:.4f} - Prec: {prec_l2_signs:.4f} - Rec: {rec_l2_signs:.4f}"
        detailed_metrics[
            'l2_objects'] = f"F1: {f1_l2_objects:.4f} - Acc: {acc_report_l2_objects:.4f} - Prec: {prec_l2_objects:.4f} - Rec: {rec_l2_objects:.4f}"
        detailed_metrics[
            'l2_body_regions'] = f"F1: {f1_l2_body_regions:.4f} - Acc: {acc_report_l2_body_regions:.4f} - Prec: {prec_l2_body_regions:.4f} - Rec: {rec_l2_body_regions:.4f}"
        detailed_metrics['l3'] = f"F1: {f1_l3:.4f} - Acc: {acc_report_l3:.4f} - Prec: {prec_l3:.4f} - Rec: {rec_l3:.4f}"

        return detailed_metrics

    def compute_metrics(self, preds, targets):
        preds = self.clean_preds(preds)
        class_report = classification_report(targets, preds, output_dict=True, zero_division=1)
        detailed_metrics = self.calculate_detailed_metrics(preds, targets)

        # avg f1_score of all classes where support > 0
        f1 = np.mean([class_report[class_name]['f1-score'] for class_name in class_report if class_report[class_name]['support'] > 0])
        precision = np.mean([class_report[class_name]['precision'] for class_name in class_report if class_report[class_name]['support'] > 0])
        recall = np.mean([class_report[class_name]['recall'] for class_name in class_report if class_report[class_name]['support'] > 0])
        acc = accuracy_score(targets.flatten(), preds.flatten())
        acc_report = accuracy_score(targets, preds)

        return acc, acc_report, f1, precision, recall, detailed_metrics

    def evaluate(self, preds, targets):
        acc, acc_report, f1, precision, recall, detailed_metrics = self.compute_metrics(preds, targets)
        return acc, acc_report, f1, precision, recall, detailed_metrics
