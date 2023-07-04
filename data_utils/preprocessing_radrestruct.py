import copy
import json
import os
from copy import deepcopy


''' utilities for generating vqa-pairs from the Rad-ReStruct reports '''

def get_object_question(object_name, first_instance):
    if first_instance:
        if object_name == "contrast media":
            question = f"Is there {object_name}?"
        elif object_name == "tube, inserted":
            question = f"Is there a tube inserted?"
        elif object_name == "catheters, indwelling":
            question = f"Are there indwelling catheters?"
        elif object_name in ["medical device", "implanted medical device"]:
            question = f"Is there a {object_name}?"
        else:
            question = f"Are there {object_name}?"
    else:
        if object_name == "contrast media":
            question = f"Is there more {object_name}?"
        elif object_name == "tube, inserted":
            question = f"Is there another tube inserted?"
        elif object_name == "catheters, indwelling":
            question = f"Are there more indwelling catheters?"
        elif object_name in ["medical device", "implanted medical device"]:
            question = f"Is there another {object_name}?"
        else:
            question = f"Are there more {object_name}?"

    return question


def get_question(elem_name, topic_name, area_name, first_instance):
    if topic_name == "objects":
        question = get_object_question(elem_name, first_instance)
    elif topic_name == "signs":
        if first_instance:
            article = "a" if elem_name[0] in ["a", "e", "i", "o", "u"] else "an"
            question = f"Is there {article} {elem_name} in the {area_name}?"
        else:
            question = f"Is there another {elem_name} in the {area_name}?"

    elif topic_name in ["diseases"]:
        if first_instance:
            question = f"Is there {elem_name} in the {area_name}?"
        else:
            question = f"Is there another case of {elem_name} in the {area_name}?"

    elif topic_name == "body_regions":
        if elem_name == area_name:
            if first_instance:
                question = f"Is there anything abnormal in the whole {elem_name}?"
            else:
                question = f"Is there anything else abnormal in the whole {elem_name}?"
        else:
            if first_instance:
                question = f"Is there anything abnormal in the {elem_name}?"
            else:
                question = f"Is there anything else abnormal in the {elem_name}?"

    elif topic_name == "infos":
        if first_instance:
            question = f"Is there anything abnormal in the {elem_name}?"
        else:
            question = f"Is there anything else abnormal in the {elem_name}?"

    else:
        raise ValueError(f"topic_name {topic_name} not supported")

    return question


def get_topic_question(topic_name, area_name):
    if topic_name == "objects":
        question = f"Are there any foreign objects?"

    elif topic_name in ["signs", "diseases", "other diseases"]:
        question = f"Are there any {topic_name} in the {area_name}?"

    elif topic_name in ["body_regions", "infos"]:  # for larger areas they are splitted in body regions, for smaller areas they go directly to infos
        question = f"Is there anything abnormal in the {area_name}?"

    else:
        raise ValueError(f"topic_name {topic_name} not supported")

    return question


def iterate_instances(elem, question, qa_pairs, elem_name, topic_name, area_name, history, max_instances):
    global count
    bin_info = {'answer_type': 'single_choice', 'options': ['yes', 'no']}
    infos = elem["infos"] if topic_name != "infos" else elem
    elem_history = deepcopy(history)
    for idx, instance in enumerate(elem["instances"]):
        if idx == 0:
            bin_info_copy = copy.deepcopy(bin_info)
            bin_info_copy['path'] = f"{area_name}_{topic_name}" if area_name == elem_name else f"{area_name}_{topic_name}_{elem_name}"
            qa_pairs.append((question, ["yes"], deepcopy(elem_history), bin_info_copy))
            elem_history.append((question, ["yes"]))
        else: # ask for the nth occurrence
            question = get_question(elem_name, topic_name, area_name, first_instance=False)
            bin_info_copy = copy.deepcopy(bin_info)
            bin_info_copy['path'] = f"{area_name}_{topic_name}" if area_name == elem_name else f"{area_name}_{topic_name}_{elem_name}"
            qa_pairs.append((question, ["yes"], deepcopy(elem_history), bin_info_copy))
            elem_history.append((question, ["yes"]))

        for key, value in instance.items():  # different "infos" instances
            info = copy.deepcopy(infos[key])
            info['path'] = f"{area_name}_{topic_name}_{key}" if area_name == elem_name else f"{area_name}_{topic_name}_{elem_name}_{key}"
            if key == "body_region":
                question = "In which part of the body?"
                qa_pairs.append((question, value, deepcopy(elem_history), info))
                elem_history.append((question, value))
            elif key == "localization":
                question = "In which area?"
                qa_pairs.append((question, value, deepcopy(elem_history), info))
                elem_history.append((question, value))
            elif key == "attributes":
                question = "What are the attributes?"
                qa_pairs.append((question, value, deepcopy(elem_history), info))
                elem_history.append((question, value))
            elif key == "degree":
                question = "What is the degree?"
                qa_pairs.append((question, value, deepcopy(elem_history), info))
                elem_history.append((question, value))
            if len(value) == 0:
                print(f"WARNING: {key} is empty for {elem_name} in {area_name}")

    # if less instances than max_instances -> add negative answered question
    if f"{area_name}/{topic_name}" in max_instances: # for this finding multiple instances are allowed
        max_num_occurences = max_instances[f"{area_name}/{topic_name if topic_name == 'infos' else elem_name}"]
    else:
        max_num_occurences = 1

    if len(elem["instances"]) < max_num_occurences: #only add one negative instance as then execution will stop
        question = get_question(elem_name, topic_name, area_name, first_instance=False)
        bin_info_copy = copy.deepcopy(bin_info)
        bin_info_copy['path'] = f"{area_name}_{topic_name}" if area_name == elem_name else f"{area_name}_{topic_name}_{elem_name}"
        qa_pairs.append((question, ["no"], deepcopy(elem_history), bin_info_copy))

    return qa_pairs, elem_history


def iterate_area(area, area_name, qa_pairs, max_instances):
    bin_info = {'answer_type': 'single_choice', 'options': ['yes', 'no']}
    for topic_name, topic in area.items():
        if topic_name == 'area':
            continue

        # is there any element in the topic where instances > 0?
        topic_positive = False
        if topic_name == 'infos':
            topic_positive = len(topic["instances"]) > 0
        else:
            for elem_name, elem in topic.items():
                if len(elem["instances"]) > 0:
                    topic_positive = True
                    break

        if not topic_positive:  # area is normal
            question = get_topic_question(topic_name, area_name)
            bin_info_copy = copy.deepcopy(bin_info)
            bin_info_copy['path'] = f"{area_name}_{topic_name}"
            qa_pairs.append((question, ["no"], [], bin_info_copy))

        else:
            history = []
            if topic_name != 'infos':
                question = get_topic_question(topic_name, area_name)
                bin_info_copy = copy.deepcopy(bin_info)
                bin_info_copy['path'] = f"{area_name}_{topic_name}"
                qa_pairs.append((question, ["yes"], [], bin_info_copy))
                history.append((question, ["yes"]))

                for elem_name, elem in area[topic_name].items():
                    question = get_question(elem_name, topic_name, area_name, first_instance=True)

                    if len(elem["instances"]) == 0:
                        bin_info_copy = copy.deepcopy(bin_info)
                        bin_info_copy['path'] = f"{area_name}_{topic_name}" if area_name == elem_name else f"{area_name}_{topic_name}_{elem_name}"
                        qa_pairs.append((question, ["no"], history, bin_info_copy))  # question, answer, history

                    else:
                        qa_pairs, elem_history = iterate_instances(elem, question, qa_pairs, elem_name, topic_name, area_name, history, max_instances)

            else:
                question = get_question(area_name, topic_name, area_name, first_instance=True)
                qa_pairs, elem_history = iterate_instances(topic, question, qa_pairs, area_name, topic_name, area_name, history, max_instances)

    return qa_pairs


def create_training_samples(split='train'):
    with open(f'data/radrestruct/{split}_ids.json', 'r') as f:
        report_ids = json.loads(f.read())

    # load max_instances
    with open(f'data/radrestruct/max_instances.json', 'r') as f:
        max_instances = json.loads(f.read())

    # iterate over all reports
    for report_id in report_ids:
        report_path = f'data/radrestruct/new_reports/{report_id}.json'
        with open(report_path, 'r') as f:
            report = json.loads(f.read())

        qa_pairs = []

        # iterate over all findings
        for area in report:
            if "sub_areas" in area:
                for sub_area_name, sub_area in area["sub_areas"].items():
                    qa_pairs = iterate_area(sub_area, sub_area_name, qa_pairs, max_instances)

            else:
                qa_pairs = iterate_area(area, area['area'], qa_pairs, max_instances)

        # save qa_pairs
        if not os.path.exists(f'data/radrestruct/{split}_qa_pairs'):
            os.makedirs(f'data/radrestruct/{split}_qa_pairs')
        with open(f'data/radrestruct/{split}_qa_pairs/{report_id}.json', 'w') as f:
            json.dump(qa_pairs, f)


''' utilities for generating vectorized ground truth from the Rad-ReStruct reports '''

def iterate_area_vectorized(area, area_name, answer_vector, max_instances, question_ids, question_id, choice_options):
    for topic_name, topic in area.items():
        if topic_name == 'area':
            continue

        # is there any element in the topic where instances > 0?
        topic_positive = False
        if topic_name == 'infos':
            if f"{area_name}/{topic_name}" in max_instances:
                max_num_occurences = max_instances[f"{area_name}/{topic_name}"]
            else:
                max_num_occurences = 1

            for i in range(max_num_occurences):
                if i < len(topic['instances']):
                    answer_vector[f"{area_name}_{topic_name}_yes_{i}"] = True
                    answer_vector[f"{area_name}_{topic_name}_no_{i}"] = False
                    question_ids.append(question_id)
                    question_ids.append(question_id)
                    choice_options[question_id] = 'single_choice'
                    question_id += 1
                    for info_name, info in topic.items():
                        if info_name != 'instances':
                            choice_options[question_id] = info['answer_type']
                            for option in info['options']:
                                answer_vector[f"{area_name}_{topic_name}_{info_name}_{option}_{i}"] = option in topic['instances'][i][info_name]
                                question_ids.append(question_id)
                            question_id += 1
                else:
                    answer_vector[f"{area_name}_{topic_name}_yes_{i}"] = False
                    answer_vector[f"{area_name}_{topic_name}_no_{i}"] = True
                    question_ids.append(question_id)
                    question_ids.append(question_id)
                    choice_options[question_id] = 'single_choice'
                    question_id += 1
                    for info_name, info in topic.items():
                        if info_name != 'instances':
                            choice_options[question_id] = info['answer_type']  # don't want any post-processing as the question should not be asked
                            for option in info['options']:
                                answer_vector[f"{area_name}_{topic_name}_{info_name}_{option}_{i}"] = False
                                question_ids.append(question_id)
                            question_id += 1

        else:
            for elem_name, elem in topic.items():
                if len(elem["instances"]) > 0:
                    topic_positive = True
                    break
            if topic_positive:
                answer_vector[f"{area_name}_{topic_name}_yes"] = True
                answer_vector[f"{area_name}_{topic_name}_no"] = False
                question_ids.append(question_id)
                question_ids.append(question_id)
                choice_options[question_id] = 'single_choice'
                question_id += 1
            else:
                answer_vector[f"{area_name}_{topic_name}_yes"] = False
                answer_vector[f"{area_name}_{topic_name}_no"] = True
                question_ids.append(question_id)
                question_ids.append(question_id)
                choice_options[question_id] = 'single_choice'
                question_id += 1

            for elem_name, elem in area[topic_name].items():
                if f"{area_name}/{elem_name}" in max_instances:
                    max_num_occurences = max_instances[f"{area_name}/{elem_name}"]
                else:
                    max_num_occurences = 1

                for i in range(max_num_occurences):
                    if i < len(elem['instances']):
                        answer_vector[f"{area_name}_{topic_name}_{elem_name}_yes_{i}"] = True
                        answer_vector[f"{area_name}_{topic_name}_{elem_name}_no_{i}"] = False
                        question_ids.append(question_id)
                        question_ids.append(question_id)
                        choice_options[question_id] = 'single_choice'
                        question_id += 1
                        for info_name, info in elem['infos'].items():
                            choice_options[question_id] = info['answer_type']
                            for option in info['options']:
                                answer_vector[f"{area_name}_{topic_name}_{elem_name}_{info_name}_{option}_{i}"] = option in elem['instances'][i][
                                    info_name]
                                question_ids.append(question_id)
                            question_id += 1

                    else:
                        answer_vector[f"{area_name}_{topic_name}_{elem_name}_yes_{i}"] = False
                        answer_vector[f"{area_name}_{topic_name}_{elem_name}_no_{i}"] = True
                        question_ids.append(question_id)
                        question_ids.append(question_id)
                        choice_options[question_id] = 'single_choice'
                        question_id += 1
                        for info_name, info in elem['infos'].items():
                            choice_options[question_id] = info['answer_type']
                            for option in info['options']:
                                answer_vector[f"{area_name}_{topic_name}_{elem_name}_{info_name}_{option}_{i}"] = False
                                question_ids.append(question_id)
                            question_id += 1

    return answer_vector, question_ids, question_id, choice_options


def create_vectorized_samples(split):
    with open(f'data/radrestruct/{split}_ids.json', 'r') as f:
        report_ids = json.loads(f.read())

    # load max_instances.json
    with open('data/radrestruct/max_instances.json', 'r') as f:
        max_instances = json.loads(f.read())

    for report_id in report_ids:
        report_path = f'data/radrestruct/new_reports/{report_id}.json'
        with open(report_path, 'r') as f:
            report = json.loads(f.read())

            answer_vector = {}
            question_ids = []
            choice_options = {}
            question_id = 0

            for area in report:
                if "sub_areas" in area:
                    for sub_area_name, sub_area in area["sub_areas"].items():
                        answer_vector, question_ids, question_id, choice_options = iterate_area_vectorized(sub_area, sub_area_name, answer_vector,
                                                                                                           max_instances, question_ids, question_id,
                                                                                                           choice_options)

                else:
                    answer_vector, question_ids, question_id, choice_options = iterate_area_vectorized(area, area['area'], answer_vector, max_instances,
                                                                                                       question_ids, question_id, choice_options)

            # save answer_vectors
            # generate folder if not exists
            if not os.path.exists(f'data/radrestruct/{split}_vectorized_answers'):
                os.makedirs(f'data/radrestruct/{split}_vectorized_answers')
            with open(f'data/radrestruct/{split}_vectorized_answers/{report_id}.json', 'w') as f:
                json.dump(answer_vector, f)

            if split == "train":
                # save question_ids
                with open(f'data/radrestruct/vectorized_question_ids.json', 'w') as f:
                    json.dump(question_ids, f)

                # save choice_options
                with open(f'data/radrestruct/vectorized_choice_options.json', 'w') as f:
                    json.dump(choice_options, f)


if __name__ == '__main__':
    a_options_test = create_training_samples('test')
    a_options_train = create_training_samples('train')
    a_options_val = create_training_samples('val')

    create_vectorized_samples('train')
    create_vectorized_samples('val')
    create_vectorized_samples('test')