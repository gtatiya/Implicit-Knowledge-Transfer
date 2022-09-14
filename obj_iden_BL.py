# Author: Gyan Tatiya

import os
import pickle

import numpy as np
from constants import get_objects_labels

from sklearn.svm import SVC

from model import classifier
from utils import get_random_objects, split_train_test, update_all_modalities, \
    update_all_behaviors_modalities, compute_mean_accuracy, get_robot_data


if __name__ == "__main__":
    """
    This script trains an object identity recognition model for each behavior and modality of a target robot data.
    Fuses the predictions of all behaviors and modalities by uniform combination, train score, and test score.
    For training and testing, same amount of trials for all object class is used.
    For training, all target robot's trials are used.
    For testing, 5-fold trial-based cross validation is done.
    
    This is to find the best possible performance when the target robot has explored all trials per object.
    """

    np.random.seed(0)

    ROBOTS_PATH = {"Baxter": r"data/Baxter_Dataset/3_Binary",
                   "UR5": r"data/UR5_Dataset/3_Binary"}

    dataset_path = r"data/UR5_Dataset/3_Binary"
    db_file_name = dataset_path + os.sep + "dataset_metadata_discretized.bin"
    bin_file = open(db_file_name, "rb")
    metadata = pickle.load(bin_file)
    bin_file.close()

    ALL_ROBOTS_LIST = ["Baxter", "UR5"]

    A_TARGET_ROBOT = "Baxter"  # always 1 robot
    # A_TARGET_ROBOT = "UR5"  # always 1 robot

    SOURCE_ROBOT_LIST = []
    for robot in ALL_ROBOTS_LIST:
        if robot != A_TARGET_ROBOT:
            SOURCE_ROBOT_LIST.append(robot)

    BEHAVIOR_LIST = ["grasp", "pick", "hold", "shake", "lower", "drop", "push"]
    MODALITY_LIST = ['audio', 'effort', 'force']

    DETECTION_TASK = 'object'  # object

    objects_labels = get_objects_labels('object', objects_list=metadata['grasp']['objects'])
    print("objects_labels: ", len(objects_labels), objects_labels)

    properties_labels = get_objects_labels(DETECTION_TASK)
    print("properties_labels: ", len(properties_labels), properties_labels)

    robots_data = get_robot_data(ALL_ROBOTS_LIST, ROBOTS_PATH, BEHAVIOR_LIST, MODALITY_LIST, objects_labels)

    CLF = SVC(gamma='auto', kernel='rbf', probability=True)
    CLF_NAME = "SVM-RBF"

    NO_OF_INTERACTIONS = [1, 2, 3, 4]  # [1, 2, 3, 4]
    num_of_test_examples = 1
    folds = len(metadata['grasp']['trials']) // num_of_test_examples
    TRAIN_TEST_SPLITS = split_train_test(folds, len(metadata['grasp']['trials']))
    print("TRAIN_TEST_SPLITS: ", TRAIN_TEST_SPLITS)

    objects_folds = 10
    num_object = 12  # must be 2 to 19
    folds_objects_labels = get_random_objects(objects_labels, objects_folds, num_object)
    print("folds_objects_labels: ", folds_objects_labels)

    results_path = 'results'
    log_path = results_path + os.sep + DETECTION_TASK + '_' + A_TARGET_ROBOT + '_' + CLF_NAME + "_BEST_test-trial" + \
               os.sep + str(num_object) + "_objects"
    os.makedirs(log_path, exist_ok=True)

    folds_objects_score_best = {}
    for object_fold in folds_objects_labels:
        print("object_fold: ", object_fold)
        folds_objects_score_best.setdefault(object_fold, {})

        folds_behaviors_modalities_proba_score_best = {}
        for fold in sorted(TRAIN_TEST_SPLITS):
            print(fold, TRAIN_TEST_SPLITS[fold])
            folds_behaviors_modalities_proba_score_best.setdefault(fold, {})

            for behavior in BEHAVIOR_LIST:
                print("behavior: ", behavior)
                folds_behaviors_modalities_proba_score_best[fold].setdefault(behavior, {})

                for modality in MODALITY_LIST:
                    print("modality: ", modality)
                    folds_behaviors_modalities_proba_score_best[fold][behavior].setdefault(modality, {})

                    # Get test data
                    X_test = []
                    y_test_object = []
                    for object_name in folds_objects_labels[object_fold]:
                        label = folds_objects_labels[object_fold][object_name]['old_label']
                        examples = robots_data[A_TARGET_ROBOT][behavior][modality][label][0][TRAIN_TEST_SPLITS[fold]["test"]]  # Objects, Trials, Features
                        examples = examples.reshape(-1, examples.shape[-1])
                        X_test.extend(examples)
                        label = folds_objects_labels[object_fold][object_name]['new_label']
                        y_test_object.extend(np.repeat(label, len(examples)))
                    X_test = np.array(X_test)
                    y_test_object = np.array(y_test_object).reshape((-1, 1))
                    print("X_test: ", X_test.shape)
                    print("y_test_object: ", y_test_object.shape, y_test_object.flatten())

                    # Get train data
                    X_train = []
                    y_train_object = []
                    for object_name in folds_objects_labels[object_fold]:
                        label = folds_objects_labels[object_fold][object_name]['old_label']
                        examples = robots_data[A_TARGET_ROBOT][behavior][modality][label][0][TRAIN_TEST_SPLITS[fold]["train"]]  # Objects, Trials, Features
                        examples = examples.reshape(-1, examples.shape[-1])
                        X_train.extend(examples)
                        label = folds_objects_labels[object_fold][object_name]['new_label']
                        y_train_object.extend(np.repeat(label, len(examples)))
                    X_train = np.array(X_train)
                    y_train_object = np.array(y_train_object).reshape((-1, 1))
                    print("X_train: ", X_train.shape)
                    print("y_train_object: ", y_train_object.shape, y_train_object.flatten())

                    # Finding the best performance by training the target robots using all trials including test trials
                    if not folds_behaviors_modalities_proba_score_best[fold][behavior][modality]:
                        y_acc_test, y_pred_test, y_proba_test = classifier(CLF, X_train, X_test, y_train_object.ravel(),
                                                                           y_test_object.ravel())

                        y_proba_pred_test = np.argmax(y_proba_test, axis=1)
                        y_prob_acc_test = np.mean(y_test_object.ravel() == y_proba_pred_test)
                        print("y_proba_pred_test: ", y_proba_pred_test.shape, y_proba_pred_test)
                        print("y_prob_acc_test: ", y_prob_acc_test)

                        folds_behaviors_modalities_proba_score_best[fold][behavior][modality]['proba'] = y_proba_test
                        folds_behaviors_modalities_proba_score_best[fold][behavior][modality]['test_acc'] = y_prob_acc_test
                        folds_behaviors_modalities_proba_score_best[fold][behavior][modality]['train_acc'] = y_prob_acc_test

                folds_behaviors_modalities_proba_score_best[fold][behavior] = \
                    update_all_modalities(folds_behaviors_modalities_proba_score_best[fold][behavior], y_test_object)

            folds_behaviors_modalities_proba_score_best[fold] = \
                update_all_behaviors_modalities(folds_behaviors_modalities_proba_score_best[fold], y_test_object)

        folds_objects_score_best[object_fold] = compute_mean_accuracy(folds_behaviors_modalities_proba_score_best,
                                                                      vary_objects=False)

    behaviors_modalities_score_best = compute_mean_accuracy(folds_objects_score_best, acc='mean', vary_objects=False)

    for behavior in behaviors_modalities_score_best:
        print(behavior, behaviors_modalities_score_best[behavior])

    # Save results
    db_file_name = log_path + os.sep + DETECTION_TASK + "_" + A_TARGET_ROBOT + "_" + CLF_NAME + "_BEST.bin"
    output_file = open(db_file_name, "wb")
    pickle.dump(behaviors_modalities_score_best, output_file)
    pickle.dump(A_TARGET_ROBOT, output_file)
    pickle.dump(CLF_NAME, output_file)
    output_file.close()
