# Author: Gyan Tatiya

import csv
import os
import pickle

import cv2
import numpy as np

import matplotlib.pyplot as plt


def time_taken(start, end):
    """Human readable time between `start` and `end`
    :param start: time.time()
    :param end: time.time()
    :returns: day:hour:minute:second.millisecond
    """

    my_time = end - start
    day = my_time // (24 * 3600)
    my_time = my_time % (24 * 3600)
    hour = my_time // 3600
    my_time %= 3600
    minutes = my_time // 60
    my_time %= 60
    seconds = my_time
    milliseconds = ((end - start) - int(end - start))
    day_hour_min_sec = str('%02d' % int(day)) + ":" + str('%02d' % int(hour)) + ":" + str('%02d' % int(minutes)) + \
                       ":" + str('%02d' % int(seconds) + "." + str('%.3f' % milliseconds)[2:])

    return day_hour_min_sec


def split_train_test(n_folds, trials_per_object):

    test_size = trials_per_object // n_folds
    tt_splits = {}

    for a_fold in range(n_folds):

        train_index = []
        test_index = np.arange(test_size * a_fold, test_size * (a_fold + 1))

        if test_size * a_fold > 0:
            train_index.extend(np.arange(0, test_size * a_fold))
        if test_size * (a_fold + 1) - 1 < trials_per_object - 1:
            train_index.extend(np.arange(test_size * (a_fold + 1), trials_per_object))

        tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("train", []).extend(train_index)
        tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("test", []).extend(test_index)

    return tt_splits


def get_unique_content_weight_objects(objects, num_object_):

    # Sampling objects with unique content and weight
    sampled_objects = []
    while len(sampled_objects) < num_object_:

        sampled_object = np.random.choice(objects)
        sampled_object_content = sampled_object.split('-')[1]
        sampled_object_weight = sampled_object.split('-')[2]

        unique_content_weight = True
        for o in sampled_objects:
            content = o.split('-')[1]
            weight = o.split('-')[2]
            if sampled_object_content == content and sampled_object_weight == weight:
                unique_content_weight = False
                break

        if unique_content_weight:
            sampled_objects.append(sampled_object)

    return sampled_objects


def has_all_properties(objects, properties):

    properties = {p: False for p in properties}
    for o in objects:
        for p in properties:
            if p in o:
                properties[p] = True

        if all(value == True for value in properties.values()):
            return True

    return False


def get_random_objects(objects_labels_, n_folds, num_object_):

    objects = list(objects_labels_.keys())

    folds_objects = {}
    for a_fold in range(n_folds):
        sampled_objects = get_unique_content_weight_objects(objects, num_object_)

        sampled_objects_labels = {}
        for label_, object_name_ in enumerate(sorted(sampled_objects)):
            sampled_objects_labels[object_name_] = {'old_label': objects_labels_[object_name_], 'new_label': label_}

        folds_objects.setdefault("fold_" + str(a_fold), sampled_objects_labels)

    return folds_objects


def split_train_test_v2(n_folds, test_percentage, objects, properties, task):
    """
    Split objects into train and test.
    Make sure train and test have at least one example of each property.

    :param n_folds:
    :param test_percentage:
    :param objects:
    :param properties:
    :param task:
    :return:
    """

    num_of_test_objects = int(len(objects) * test_percentage)
    objects = list(objects.keys())
    properties = list(properties.keys())

    tt_splits = {}
    for a_fold in range(n_folds):
        while True:
            test_objects = get_unique_content_weight_objects(objects, num_of_test_objects)
            train_objects = list(set(objects) - set(test_objects))

            if has_all_properties(test_objects, properties) and has_all_properties(train_objects, properties):

                # Making sure the first training objects cover all the labels
                while True:
                    if task == 'weight':
                        w_list = [object_name_.split('-')[2] for object_name_ in train_objects[0:len(properties)]]
                    elif task == 'content':
                        w_list = [object_name_.split('-')[1] for object_name_ in train_objects[0:len(properties)]]
                    else:
                        w_list = [object_name_.split('-')[0] for object_name_ in train_objects[0:len(properties)]]

                    if len(properties) == len(set(w_list)):
                        break
                    np.random.shuffle(train_objects)

                tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("train", train_objects)
                tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("test", test_objects)
                break

    return tt_splits


def get_split_data(trials, objects_labels_, path, objects, behavior_, modality_, task):

    if isinstance(list(trials)[0], np.integer):
        trials = ["trial-" + str(trial_num) for trial_num in sorted(trials)]

    x_split = []
    y_split = []
    for object_name_ in sorted(objects):
        for trial_num in sorted(trials):
            data_path = os.sep.join([path, behavior_, object_name_, trial_num, modality_ + ".bin"])
            bin_file_ = open(data_path, "rb")
            example = pickle.load(bin_file_)
            bin_file_.close()

            if task == 'object':
                x_split.append(example.flatten())
                y_split.append(objects_labels_[object_name_])
            elif task == 'weight':
                for w in [22, 50, 100, 150]:
                    w = str(w) + 'g'
                    if w == object_name_.split('-')[2]:
                        x_split.append(example.flatten())
                        y_split.append(objects_labels_[w])
                        break
            elif task == 'content':
                for c in ['empty', 'rice', 'pasta', 'nutsandbolts', 'marbles', 'dices', 'buttons']:
                    if c == object_name_.split('-')[1]:
                        x_split.append(example.flatten())
                        y_split.append(objects_labels_[c])
                        break
            elif task == 'color':
                for c in ['white', 'red', 'blue', 'green', 'yellow']:
                    if c == object_name_.split('-')[0]:
                        # extracting histogram features from the last image
                        example = cv2.calcHist([example[-1]], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
                        # print("3D histogram shape: {}, with {} values".format(example.shape, example.flatten().shape[0]))
                        x_split.append(example.flatten())
                        y_split.append(objects_labels_[c])
                        break

    return np.array(x_split), np.array(y_split)


def get_robot_data(robots, robots_paths, behavior_list, modality_list, objects_labels_, empty=True):

    """
    Load the discretized robot data for the given behavior, modality if it already exists otherwise read all of them
    and save it to be loaded quickly later.
    It is assumed that the saved dataset_discretized.bin file contains all the behaviors and modalities.
    """

    robots_data_ = {}
    for robot_name in robots:

        print("robot_name: ", robot_name)
        db_file_name = robots_paths[robot_name] + os.sep + "dataset_metadata_discretized.bin"
        bin_file = open(db_file_name, "rb")
        metadata = pickle.load(bin_file)
        bin_file.close()

        robots_data_filepath = robots_paths[robot_name] + os.sep + "dataset_discretized.bin"
        if not os.path.exists(robots_data_filepath):

            # Read and save the examples for each behavior
            robot_ = {}
            for behavior_ in behavior_list:
                robot_.setdefault(behavior_, {})
                for modality_ in modality_list:
                    if behavior_ == 'look' and modality_ in {'audio', 'effort', 'position', 'velocity', 'torque',
                                                             'force'}:
                        continue

                    robot_[behavior_].setdefault(modality_, {})
                    for object_name_ in sorted(metadata[behavior_]['objects']):

                        label_ = None
                        for on in sorted(metadata['grasp']['objects']):
                            if empty:
                                if on == object_name_:
                                    label_ = objects_labels_[object_name_]
                                    break
                            else:
                                if 'empty' not in on and on == object_name_:
                                    label_ = objects_labels_[object_name_]
                                    break

                        if label_ is not None:
                            trials = []
                            for trial_num in sorted(metadata[behavior_]['trials']):
                                if modality_ == 'camera_rgb_image_raw':
                                    data_path = os.sep.join(
                                        [robots_paths[robot_name], behavior_, object_name_, str(trial_num), modality_ +
                                         ".bin"])
                                else:
                                    data_path = os.sep.join([robots_paths[robot_name], behavior_, object_name_,
                                                             str(trial_num), modality_ + "-discretized.bin"])
                                bin_file_ = open(data_path, "rb")
                                example = pickle.load(bin_file_)
                                bin_file_.close()

                                if modality_ == 'camera_rgb_image_raw':
                                    example = cv2.calcHist([example[-1]], [0, 1, 2], None, [4, 4, 4],
                                                           [0, 256, 0, 256, 0, 256])

                                trials.append(example.flatten())

                            robot_[behavior_][modality_].setdefault(label_, [])
                            robot_[behavior_][modality_][label_].append(trials)

                    for label_ in robot_[behavior_][modality_]:
                        robot_[behavior_][modality_][label_] = np.array(robot_[behavior_][modality_][label_])

            output_file = open(robots_data_filepath, "wb")
            pickle.dump(robot_, output_file)
            output_file.close()
        else:
            print("Loading: " + robots_data_filepath)
            bin_file = open(robots_data_filepath, "rb")
            robot_ = pickle.load(bin_file)
            bin_file.close()

        robots_data_[robot_name] = robot_

    return robots_data_


def get_property_label(object_name_, task, objects_labels_, properties_labels_, empty=True):

    label_ = None
    if task == 'object':
        return objects_labels_[object_name_]
    elif task == 'weight':
        if empty:
            for w in [22, 50, 100, 150]:
                w = str(w) + 'g'
                if w == object_name_.split('-')[2]:
                    label_ = properties_labels_[w]
                    break
        else:
            for w in [50, 100, 150]:
                w = str(w) + 'g'
                if w == object_name_.split('-')[2]:
                    label_ = properties_labels_[w]
                    break
    elif task == 'content':
        if empty:
            for c in ['empty', 'rice', 'pasta', 'nutsandbolts', 'marbles', 'dices', 'buttons']:
                if c == object_name_.split('-')[1]:
                    label_ = properties_labels_[c]
                    break
        else:
            for c in ['rice', 'pasta', 'nutsandbolts', 'marbles', 'dices', 'buttons']:
                if c == object_name_.split('-')[1]:
                    label_ = properties_labels_[c]
                    break
    elif task == 'color':
        for c in ['white', 'red', 'blue', 'green', 'yellow']:
            if c == object_name_.split('-')[0]:
                label_ = properties_labels_[c]
                break

    return label_


def get_robot_objects_data(objects, objects_labels_, properties_labels_, task, robots_data_):

    X_data = []
    y_object = []
    y_property = []
    for object_name in objects:
        label = objects_labels_[object_name]
        property_label = get_property_label(object_name, task, objects_labels_, properties_labels_)
        examples = robots_data_[label][0]
        examples = examples.reshape(-1, examples.shape[-1])
        X_data.extend(examples)
        y_object.extend(np.repeat(label, len(examples)))
        y_property.extend(np.repeat(property_label, len(examples)))
    X_data = np.array(X_data)
    y_object = np.array(y_object).reshape((-1, 1))
    y_property = np.array(y_property).reshape((-1, 1))

    return X_data, y_object, y_property


def repeat_trials(y_train_source, y_train_target, X_train_source_, X_train_target_):

    X_train_source_repeat = []
    y_train_source_repeat = []
    X_train_target_repeat = []
    y_train_target_repeat = []
    for s_idx, s_label in enumerate(y_train_source.flatten()):
        for t_idx, t_label in enumerate(y_train_target.flatten()):
            if s_label == t_label:
                X_train_source_repeat.append(X_train_source_[s_idx])
                y_train_source_repeat.append(s_label)

                X_train_target_repeat.append(X_train_target_[t_idx])
                y_train_target_repeat.append(t_label)
    X_train_source_repeat = np.array(X_train_source_repeat)
    y_train_source_repeat = np.array(y_train_source_repeat)
    X_train_target_repeat = np.array(X_train_target_repeat)
    y_train_target_repeat = np.array(y_train_target_repeat)

    return X_train_source_repeat, y_train_source_repeat, X_train_target_repeat, y_train_target_repeat


def get_new_labels(y_object, objects_labels_):

    label_count = 0
    y_object_new = []
    old_labels_new_label_ = {}
    objects_labels_new = {}
    for old_label in y_object.flatten():
        if old_label not in old_labels_new_label_:
            old_labels_new_label_[old_label] = label_count
            y_object_new.append(label_count)
            object_name_ = list(objects_labels_.keys())[list(objects_labels_.values()).index(old_label)]
            objects_labels_new[object_name_] = label_count
            objects_labels_new[old_label] = label_count
            label_count += 1
        else:
            y_object_new.append(old_labels_new_label_[old_label])
    y_object_new = np.array(y_object_new).reshape((-1, 1))

    return y_object_new, objects_labels_new, old_labels_new_label_


def augment_trials(X_data, y_object, y_property=None, num_trials_aug=5, shuffle=True):

    if y_property is None:
        y_property = []

    object_labels = set(y_object.flatten())

    X_data_aug = []
    y_object_aug = []
    y_property_aug = []
    for label in object_labels:
        indices = np.where(y_object == label)

        X_data_mean = np.mean(X_data[indices[0]], axis=0)
        X_data_std = np.std(X_data[indices[0]], axis=0)

        if len(y_property) > 0:
            property_label = y_property[indices[0]][0][0]

        for _ in range(num_trials_aug):
            data_point = np.random.normal(X_data_mean, X_data_std)
            X_data_aug.append(data_point)
            y_object_aug.append(label)
            if len(y_property) > 0:
                y_property_aug.append(property_label)

    X_data_aug = np.array(X_data_aug)
    y_object_aug = np.array(y_object_aug).reshape((-1, 1))
    y_property_aug = np.array(y_property_aug).reshape((-1, 1))

    if len(X_data_aug) > 0:
        X_data = np.concatenate((X_data, X_data_aug), axis=0)
        y_object = np.concatenate((y_object, y_object_aug), axis=0)
        if len(y_property) > 0:
            y_property = np.concatenate((y_property, y_property_aug), axis=0)

    if shuffle:
        random_idx = np.random.permutation(X_data.shape[0])
        X_data = X_data[random_idx]
        y_object = y_object[random_idx]
        if len(y_property) > 0:
            y_property = y_property[random_idx]

    if len(y_property) > 0:
        return X_data, y_object, y_property
    else:
        return X_data, y_object


def check_kema_data(kema_data):

    for x_key in kema_data:
        if "Test" not in x_key and x_key.startswith('X') and kema_data[x_key].shape[0] <= 10:
            y_key = 'Y' + x_key[1]
            print("<= 10 EXAMPLES FOR: ", x_key, y_key)

            while kema_data[x_key].shape[0] <= 10:
                idx = np.random.choice(kema_data[x_key].shape[0])
                kema_data[x_key] = np.append(kema_data[x_key], kema_data[x_key][idx].reshape(1, -1),
                                             axis=0)
                kema_data[y_key] = np.append(kema_data[y_key], kema_data[y_key][idx].reshape(1, -1),
                                             axis=0)

    return kema_data


def combine_probability(proba_acc_list_, y_test_, acc=None):

    # For each classifier, combine weighted probability based on its accuracy score
    proba_list = []
    for proba_acc in proba_acc_list_:
        y_proba = proba_acc['proba']
        if acc and proba_acc[acc] > 0:
            # Multiple the score by probability to combine each classifier's performance accordingly
            y_proba = y_proba * proba_acc[acc]  # weighted probability
        proba_list.append(y_proba)

    # Combine weighted probability of all classifiers
    y_proba_norm = np.zeros(len(proba_list[0][0]))
    for proba in proba_list:
        y_proba_norm = y_proba_norm + proba

    # Normalizing probability
    y_proba_norm_sum = np.sum(y_proba_norm, axis=1)  # sum of weighted probability
    y_proba_norm_sum = np.repeat(y_proba_norm_sum, len(proba_list[0][0]), axis=0).reshape(y_proba_norm.shape)
    y_proba_norm = y_proba_norm / y_proba_norm_sum

    y_proba_pred = np.argmax(y_proba_norm, axis=1)
    y_prob_acc = np.mean(y_test_ == y_proba_pred)

    return y_proba_norm, y_prob_acc


def update_all_modalities(modalities_proba_score, y_test_):

    # For each modality, combine weighted probability based on its accuracy score
    proba_acc_list = []
    for modality_ in modalities_proba_score:
        proba_acc = {'proba': modalities_proba_score[modality_]['proba'],
                     'train_acc': modalities_proba_score[modality_]['train_acc'],
                     'test_acc': modalities_proba_score[modality_]['test_acc']}
        proba_acc_list.append(proba_acc)

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel())
    modalities_proba_score.setdefault('all_modalities', {})
    modalities_proba_score['all_modalities']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'train_acc')
    modalities_proba_score.setdefault('all_modalities_train', {})
    modalities_proba_score['all_modalities_train']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities_train']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'test_acc')
    modalities_proba_score.setdefault('all_modalities_test', {})
    modalities_proba_score['all_modalities_test']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities_test']['test_acc'] = y_prob_acc

    return modalities_proba_score


def update_all_behaviors_modalities(behaviors_modalities_proba_score, y_test_):

    # For each behavior and modality, combine weighted probability based on its accuracy score
    proba_acc_list = []
    for behavior_ in behaviors_modalities_proba_score:
        for modality_ in behaviors_modalities_proba_score[behavior_]:
            if not modality_.startswith('all_modalities'):
                proba_acc = {'proba': behaviors_modalities_proba_score[behavior_][modality_]['proba'],
                             'train_acc': behaviors_modalities_proba_score[behavior_][modality_]['train_acc'],
                             'test_acc': behaviors_modalities_proba_score[behavior_][modality_]['test_acc']}
                proba_acc_list.append(proba_acc)

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel())
    behaviors_modalities_proba_score.setdefault('all_behaviors_modalities', {})
    behaviors_modalities_proba_score['all_behaviors_modalities']['proba'] = y_proba_norm
    behaviors_modalities_proba_score['all_behaviors_modalities']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'train_acc')
    behaviors_modalities_proba_score.setdefault('all_behaviors_modalities_train', {})
    behaviors_modalities_proba_score['all_behaviors_modalities_train']['proba'] = y_proba_norm
    behaviors_modalities_proba_score['all_behaviors_modalities_train']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'test_acc')
    behaviors_modalities_proba_score.setdefault('all_behaviors_modalities_test', {})
    behaviors_modalities_proba_score['all_behaviors_modalities_test']['proba'] = y_proba_norm
    behaviors_modalities_proba_score['all_behaviors_modalities_test']['test_acc'] = y_prob_acc

    return behaviors_modalities_proba_score


def compute_mean_accuracy(folds_behaviors_modalities_proba_score, acc='test_acc', vary_objects=True):

    behaviors_modalities_score = {}
    for fold_ in folds_behaviors_modalities_proba_score:
        if vary_objects:
            for objects_per_label_ in folds_behaviors_modalities_proba_score[fold_]:
                behaviors_modalities_score.setdefault(objects_per_label_, {})
                for behavior_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_]:
                    if behavior_.startswith('all_behaviors_modalities'):
                        behaviors_modalities_score[objects_per_label_].setdefault(behavior_, [])
                        y_prob_acc = folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_][acc]
                        behaviors_modalities_score[objects_per_label_][behavior_].append(y_prob_acc)
                    else:
                        behaviors_modalities_score[objects_per_label_].setdefault(behavior_, {})
                        for modality_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_]:
                            behaviors_modalities_score[objects_per_label_][behavior_].setdefault(modality_, [])
                            y_prob_acc = folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_][modality_][acc]
                            behaviors_modalities_score[objects_per_label_][behavior_][modality_].append(y_prob_acc)
        else:
            for behavior_ in folds_behaviors_modalities_proba_score[fold_]:
                if behavior_.startswith('all_behaviors_modalities'):
                    behaviors_modalities_score.setdefault(behavior_, [])
                    y_prob_acc = folds_behaviors_modalities_proba_score[fold_][behavior_][acc]
                    behaviors_modalities_score[behavior_].append(y_prob_acc)
                else:
                    behaviors_modalities_score.setdefault(behavior_, {})
                    for modality_ in folds_behaviors_modalities_proba_score[fold_][behavior_]:
                        behaviors_modalities_score[behavior_].setdefault(modality_, [])
                        y_prob_acc = folds_behaviors_modalities_proba_score[fold_][behavior_][modality_][acc]
                        behaviors_modalities_score[behavior_][modality_].append(y_prob_acc)

    if vary_objects:
        for objects_per_label_ in behaviors_modalities_score:
            for behavior_ in behaviors_modalities_score[objects_per_label_]:
                if behavior_.startswith('all_behaviors_modalities'):
                    behaviors_modalities_score[objects_per_label_][behavior_] = {
                        'mean': np.mean(behaviors_modalities_score[objects_per_label_][behavior_]),
                        'std': np.std(behaviors_modalities_score[objects_per_label_][behavior_])}
                else:
                    for modality_ in behaviors_modalities_score[objects_per_label_][behavior_]:
                        behaviors_modalities_score[objects_per_label_][behavior_][modality_] = {
                            'mean': np.mean(behaviors_modalities_score[objects_per_label_][behavior_][modality_]),
                            'std': np.std(behaviors_modalities_score[objects_per_label_][behavior_][modality_])}
    else:
        for behavior_ in behaviors_modalities_score:
            if behavior_.startswith('all_behaviors_modalities'):
                behaviors_modalities_score[behavior_] = {
                    'mean': np.mean(behaviors_modalities_score[behavior_]),
                    'std': np.std(behaviors_modalities_score[behavior_])}
            else:
                for modality_ in behaviors_modalities_score[behavior_]:
                    behaviors_modalities_score[behavior_][modality_] = {
                        'mean': np.mean(behaviors_modalities_score[behavior_][modality_]),
                        'std': np.std(behaviors_modalities_score[behavior_][modality_])}

    return behaviors_modalities_score


def save_cost_csv(cost, save_path, csv_name):
    """
    Save loss over iterations in a csv file
    :param cost:
    :param save_path:
    :param csv_name:
    :return:
    """

    with open(save_path + os.sep + csv_name + ".csv", 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["epoch", "Loss"])
        for i in range(1, len(cost) + 1):
            writer.writerow([i, cost[i - 1]])


def plot_loss_curve(cost, save_path, title_name, xlabel, ylabel):
    """
    Plot loss over iterations and save a plot
    :param cost:
    :param save_path:
    :param title_name:
    :param xlabel:
    :param ylabel:
    :return:
    """

    plt.plot(range(1, len(cost) + 1), cost)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_name)
    plt.savefig(save_path + os.sep + title_name + ".png", bbox_inches='tight', dpi=100)
    plt.close()


def plot_all_modalities(behaviors_modalities_score, all_modalities_type, x_points, title_name, xlabel, file_path_name,
                        task=None, ylim=True, xticks=False):

    all_scores = {}
    for examples_per_object in sorted(behaviors_modalities_score):
        for behavior in behaviors_modalities_score[examples_per_object]:
            if not behavior.startswith('all_behaviors_modalities'):
                all_scores.setdefault(behavior, {'mean': [], 'std': []})
                all_scores[behavior]['mean'].append(behaviors_modalities_score[examples_per_object][behavior][all_modalities_type]['mean'])
                all_scores[behavior]['std'].append(behaviors_modalities_score[examples_per_object][behavior][all_modalities_type]['std'])
    print("all_scores: ", all_scores)

    for behavior in sorted(all_scores):
        all_scores[behavior]['mean'] = np.array(all_scores[behavior]['mean']) * 100
        all_scores[behavior]['std'] = np.array(all_scores[behavior]['std']) * 100
        plt.plot(x_points, all_scores[behavior]['mean'], label=behavior.capitalize())
        plt.fill_between(x_points, all_scores[behavior]['mean'] - all_scores[behavior]['std'],
                         all_scores[behavior]['mean'] + all_scores[behavior]['std'], alpha=0.3)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% ' + (task.capitalize() if task else '') + ' Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(x_points)
    plt.legend(loc='upper left')
    plt.savefig(file_path_name + "_" + all_modalities_type + '.png', bbox_inches='tight', dpi=100)
    plt.close()


def plot_fold_all_behaviors_modalities(behaviors_modalities_score_bl, behaviors_modalities_score_kt, fold,
                                       all_behaviors_modalities_type, title_name, xlabel, file_path_name, task=None,
                                       ylim=True, xticks=False):

    acc_bl = []
    acc_kt = []
    num_objects = []
    for objects_per_label in behaviors_modalities_score_bl[fold]:
        num_objects.append(objects_per_label)
        acc_bl.append(behaviors_modalities_score_bl[fold][objects_per_label][all_behaviors_modalities_type]['test_acc'])
        acc_kt.append(behaviors_modalities_score_kt[fold][objects_per_label][all_behaviors_modalities_type]['test_acc'])
    acc_bl = np.array(acc_bl) * 100
    acc_kt = np.array(acc_kt) * 100

    plt.plot(num_objects, acc_bl, color='pink', label='Baseline Condition')
    plt.plot(num_objects, acc_kt, color='blue', label='Transfer Condition')

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% ' + (task.capitalize() if task else '') + ' Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(num_objects)
    plt.legend(loc='lower right')
    plt.savefig(file_path_name + "_" + all_behaviors_modalities_type + '.png', bbox_inches='tight', dpi=100)
    plt.close()


def plot_all_behaviors_modalities(behaviors_modalities_score_bl, behaviors_modalities_score_kt,
                                  all_behaviors_modalities_type, title_name, xlabel, file_path_name, task=None,
                                  ylim=True, xticks=False):

    acc_bl = []
    std_bl = []
    acc_kt = []
    std_kt = []
    num_objects = []
    for examples_per_object in sorted(behaviors_modalities_score_bl):
        num_objects.append(examples_per_object)
        acc_bl.append(behaviors_modalities_score_bl[examples_per_object][all_behaviors_modalities_type]['mean'])
        std_bl.append(behaviors_modalities_score_bl[examples_per_object][all_behaviors_modalities_type]['std'])
        acc_kt.append(behaviors_modalities_score_kt[examples_per_object][all_behaviors_modalities_type]['mean'])
        std_kt.append(behaviors_modalities_score_kt[examples_per_object][all_behaviors_modalities_type]['std'])
    acc_bl = np.array(acc_bl) * 100
    std_bl = np.array(std_bl) * 100
    acc_kt = np.array(acc_kt) * 100
    std_kt = np.array(std_kt) * 100

    plt.plot(num_objects, acc_bl, color='pink', label='Baseline Condition')
    plt.fill_between(num_objects, acc_bl - std_bl, acc_bl + std_bl, color='pink', alpha=0.3)

    plt.plot(num_objects, acc_kt, color='blue', label='Transfer Condition')
    plt.fill_between(num_objects, acc_kt - std_kt, acc_kt + std_kt, color='blue', alpha=0.3)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% ' + (task.capitalize() if task else '') + ' Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(num_objects)
    plt.legend(loc='lower right')
    plt.savefig(file_path_name + "_" + all_behaviors_modalities_type + '.png', bbox_inches='tight', dpi=100)
    plt.close()
