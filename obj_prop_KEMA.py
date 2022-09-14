# Author: Gyan Tatiya

import os
import pickle
from datetime import datetime

import numpy as np
import scipy.io
import matlab.engine

from constants import *

from scipy.io import loadmat

from sklearn.svm import SVC

from model import classifier
from utils import split_train_test_v2, augment_trials, update_all_modalities, update_all_behaviors_modalities,\
    compute_mean_accuracy, get_robot_data, get_property_label, plot_all_modalities, plot_all_behaviors_modalities,\
    get_robot_objects_data


if __name__ == "__main__":
    """
    This script trains an object property recognition model for each behavior and modality of a target robot data
    incrementally and source robot data.
    Fuses the predictions of all behaviors and modalities by uniform combination, train score, and test score.
    For training maximum 80% of the objects are used and for testing rest 20% of the objects are used.
    For training, the target robot data is increased incrementally.
    For testing, novel objects are used.
    Projection can be done using KEMA in 2 ways: object-identity and object-property based.
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

    # A_TARGET_ROBOT = "Baxter"  # always 1 robot
    A_TARGET_ROBOT = "UR5"  # always 1 robot

    SOURCE_ROBOT_LIST = []
    for robot in ALL_ROBOTS_LIST:
        if robot != A_TARGET_ROBOT:
            SOURCE_ROBOT_LIST.append(robot)

    # "look", "grasp", "pick", "hold", "shake", "lower", "drop", "push"
    BEHAVIOR_LIST = ["grasp", "pick", "hold", "shake", "lower", "drop", "push"]
    # 'camera_rgb_image_raw', 'audio', 'effort', 'force'
    MODALITY_LIST = ['audio', 'effort', 'force']

    DETECTION_TASK = 'weight'  # weight, content, color

    PROJECTION_TYPE = 'object'  # object, property

    NUM_TRIALS_AUGMENT = 5  # 0 for no augmentation

    objects_labels = get_objects_labels('object', objects_list=metadata['grasp']['objects'])
    print("objects_labels: ", len(objects_labels), objects_labels)

    properties_labels = get_objects_labels(DETECTION_TASK)
    print("properties_labels: ", len(properties_labels), properties_labels)

    if DETECTION_TASK == 'weight':
        NO_OF_OBJECTS = [4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 76]
        TRAIN_TEST_SPLITS = TRAIN_TEST_SPLITS_WEIGHT
    elif DETECTION_TASK == 'content':
        NO_OF_OBJECTS = [7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 60, 70, 76]
        TRAIN_TEST_SPLITS = TRAIN_TEST_SPLITS_CONTENT
    else:
        print("DETECTION_TASK NOT DETECTED.")
        exit()

    robots_data = get_robot_data(ALL_ROBOTS_LIST, ROBOTS_PATH, BEHAVIOR_LIST, MODALITY_LIST, objects_labels)

    CLF = SVC(gamma='auto', kernel='rbf', probability=True)
    CLF_NAME = "SVM-RBF"

    MATLAB_eng = matlab.engine.start_matlab()

    results_path = 'results'
    data_path_KEMA = results_path + os.sep + "KEMA_data" + os.sep + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    input_filename_KEMA = 'data_' + A_TARGET_ROBOT + '.mat'
    output_filename_KEMA = 'projections_' + A_TARGET_ROBOT + '.mat'

    os.makedirs(data_path_KEMA, exist_ok=True)

    log_path = results_path + os.sep + DETECTION_TASK + '_' + A_TARGET_ROBOT + '_' + CLF_NAME + "_" + PROJECTION_TYPE +\
               '_projection_KEMA' + ("_AUG_" + str(NUM_TRIALS_AUGMENT) + "_trials" if NUM_TRIALS_AUGMENT else '')
    os.makedirs(log_path, exist_ok=True)

    folds_behaviors_modalities_proba_score_bl = {}
    folds_behaviors_modalities_proba_score_kt = {}
    for fold in sorted(TRAIN_TEST_SPLITS):
        print(fold, TRAIN_TEST_SPLITS[fold])
        folds_behaviors_modalities_proba_score_bl.setdefault(fold, {})
        folds_behaviors_modalities_proba_score_kt.setdefault(fold, {})

        for num_of_objects in NO_OF_OBJECTS:
            print("num_of_objects: ", num_of_objects)
            folds_behaviors_modalities_proba_score_bl[fold].setdefault(num_of_objects, {})
            folds_behaviors_modalities_proba_score_kt[fold].setdefault(num_of_objects, {})

            for behavior in BEHAVIOR_LIST:
                print("behavior: ", behavior)
                folds_behaviors_modalities_proba_score_bl[fold][num_of_objects].setdefault(behavior, {})
                folds_behaviors_modalities_proba_score_kt[fold][num_of_objects].setdefault(behavior, {})

                for modality in MODALITY_LIST:
                    print("modality: ", modality)
                    if behavior == 'look' and modality in {'audio', 'effort', 'position', 'velocity', 'torque',
                                                           'force'}:
                        continue
                    elif behavior != 'look' and modality in {'camera_rgb_image_raw'}:
                        continue

                    folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior].setdefault(modality, {})
                    folds_behaviors_modalities_proba_score_kt[fold][num_of_objects][behavior].setdefault(modality, {})

                    # Get test data
                    X_test, y_test_object, y_test_property = \
                        get_robot_objects_data(TRAIN_TEST_SPLITS[fold]["test"], objects_labels,
                                               properties_labels, DETECTION_TASK,
                                               robots_data[A_TARGET_ROBOT][behavior][modality])
                    print("X_test: ", X_test.shape)
                    print("y_test_object: ", len(y_test_object), y_test_object.flatten())
                    print("y_test_property: ", len(y_test_property), y_test_property.flatten())

                    # Get train data
                    X_train, y_train_object, y_train_property = \
                        get_robot_objects_data(TRAIN_TEST_SPLITS[fold]["train"][0:num_of_objects], objects_labels,
                                               properties_labels, DETECTION_TASK,
                                               robots_data[A_TARGET_ROBOT][behavior][modality])
                    print("X_train: ", X_train.shape)
                    print("y_train_object: ", len(y_train_object))  #, y_train_object.flatten())
                    print("y_train_property: ", len(y_train_property))  #, y_train_property.flatten())

                    if NUM_TRIALS_AUGMENT:
                        X_train, y_train_object, y_train_property = \
                            augment_trials(X_train, y_train_object, y_train_property, NUM_TRIALS_AUGMENT)
                        print("After Data Augmentations:")
                        print("X_train: ", X_train.shape)
                        print("y_train_object: ", len(y_train_object))  # , y_train_object.flatten())
                        print("y_train_property: ", len(y_train_property))  # , y_train_property.flatten())

                    # Train and Test
                    y_acc, y_pred, y_proba = classifier(CLF, X_train, X_test, y_train_property.ravel(),
                                                        y_test_property.ravel())

                    y_proba_pred = np.argmax(y_proba, axis=1)
                    y_prob_acc = np.mean(y_test_property.ravel() == y_proba_pred)
                    print("y_proba_pred: ", len(y_proba_pred), y_proba_pred)
                    print("y_prob_acc: ", y_prob_acc)

                    folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior][modality]['proba'] = y_proba
                    folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior][modality]['test_acc'] = y_prob_acc

                    # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
                    # Use only training data to get a score
                    y_acc_train, y_pred_train, y_proba_train = classifier(CLF, X_train, X_train,
                                                                          y_train_property.ravel(),
                                                                          y_train_property.ravel())
                    y_proba_pred_train = np.argmax(y_proba_train, axis=1)
                    y_prob_acc_train = np.mean(y_train_property.ravel() == y_proba_pred_train)
                    print("y_proba_pred_train: ", len(y_proba_pred_train))  # , y_proba_pred_train)
                    print("y_prob_acc_train: ", y_prob_acc_train)

                    folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior][modality]['train_acc'] = y_prob_acc_train

                    KEMA_y_property = {'Y2': y_train_property}
                    KEMA_data = {'X2': X_train, 'X2_Test': X_test}
                    if PROJECTION_TYPE == 'object':
                        KEMA_data['Y2'] = y_train_object + 1  # + 1  # add 1 as in KEMA (MATLAB) labels starts from 1
                    else:
                        KEMA_data['Y2'] = y_train_property + 1  # + 1  # add 1 as in KEMA (MATLAB) labels starts from 1

                    count = 1
                    for source_robot in SOURCE_ROBOT_LIST:

                        objects_label_list = list(objects_labels.values())
                        objects_list = list(objects_labels.keys())

                        X_train, y_train_object, y_train_property = \
                            get_robot_objects_data(objects_list, objects_labels,
                                                   properties_labels, DETECTION_TASK,
                                                   robots_data[source_robot][behavior][modality])
                        print("source X_train: ", X_train.shape)
                        print("source y_train_object: ", len(y_train_object))  #, y_train_object.flatten())
                        print("source y_train_property: ", len(y_train_property))  #, y_train_property.flatten())

                        if NUM_TRIALS_AUGMENT:
                            X_train, y_train_object, y_train_property = \
                                augment_trials(X_train, y_train_object, y_train_property, NUM_TRIALS_AUGMENT)
                            print("After Data Augmentations:")
                            print("source X_train: ", X_train.shape)
                            print("source y_train_object: ", len(y_train_object))  # , y_train_object.flatten())
                            print("source y_train_property: ", len(y_train_property))  # , y_train_property.flatten())

                        KEMA_y_property['Y' + str(count)] = y_train_property
                        KEMA_data['X' + str(count)] = X_train
                        if PROJECTION_TYPE == 'object':
                            KEMA_data['Y' + str(count)] = y_train_object + 1  # + 1  # add 1 as in KEMA (MATLAB) labels starts from 1
                        else:
                            KEMA_data['Y' + str(count)] = y_train_property + 1  # + 1  # add 1 as in KEMA (MATLAB) labels starts from 1

                        count += 1

                    scipy.io.savemat(os.path.join(data_path_KEMA, input_filename_KEMA), mdict=KEMA_data)
                    MATLAB_eng.project2Domains_v2(data_path_KEMA, input_filename_KEMA, output_filename_KEMA, 1)

                    # In case Matlab messes up, we'll load and check these immediately, then delete them so we never read in an old file
                    projections = None
                    if os.path.isfile(os.path.join(data_path_KEMA, output_filename_KEMA)):
                        try:
                            projections = loadmat(os.path.join(data_path_KEMA, output_filename_KEMA))
                            Z1_train, Z2_train, Z2_Test = projections['Z1'], projections['Z2'], projections['Z2_Test']
                            os.remove(os.path.join(data_path_KEMA, output_filename_KEMA))
                            os.remove(os.path.join(data_path_KEMA, input_filename_KEMA))
                        except TypeError as e:
                            print('loadmat failed: ' + str(e))

                    X_train = np.concatenate((Z1_train, Z2_train), axis=0)
                    y_train_property = np.concatenate((KEMA_y_property['Y1'], KEMA_y_property['Y2']), axis=0)
                    print("Z X_train: ", X_train.shape)
                    print("Z y_train_property: ", len(y_train_property))  # , y_train_property.flatten())

                    if not np.isreal(X_train).all():
                        print("Complex number detected: X_train")
                        X_train = X_train.real

                    if not np.isreal(Z2_Test).all():
                        print("Complex number detected: Z2_Test")
                        Z2_Test = Z2_Test.real

                    y_acc, y_pred, y_proba = classifier(CLF, X_train, Z2_Test, y_train_property.ravel(),
                                                        y_test_property.ravel())

                    y_proba_pred = np.argmax(y_proba, axis=1)
                    y_prob_acc = np.mean(y_test_property.ravel() == y_proba_pred)
                    print("y_proba_pred: ", len(y_proba_pred), y_proba_pred)
                    print("y_prob_acc: ", y_prob_acc)

                    folds_behaviors_modalities_proba_score_kt[fold][num_of_objects][behavior][modality]['proba'] = y_proba
                    folds_behaviors_modalities_proba_score_kt[fold][num_of_objects][behavior][modality]['test_acc'] = y_prob_acc

                    # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
                    # Use only training data to get a score
                    y_acc_train, y_pred_train, y_proba_train = classifier(CLF, X_train, X_train,
                                                                          y_train_property.ravel(),
                                                                          y_train_property.ravel())
                    y_proba_pred_train = np.argmax(y_proba_train, axis=1)
                    y_prob_acc_train = np.mean(y_train_property.ravel() == y_proba_pred_train)
                    print("y_proba_pred_train: ", len(y_proba_pred_train))  # , y_proba_pred_train)
                    print("y_prob_acc_train: ", y_prob_acc_train)

                    folds_behaviors_modalities_proba_score_kt[fold][num_of_objects][behavior][modality]['train_acc'] = y_prob_acc_train

                folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior] = \
                    update_all_modalities(
                        folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior],
                        y_test_property)
                folds_behaviors_modalities_proba_score_kt[fold][num_of_objects][behavior] = \
                    update_all_modalities(
                        folds_behaviors_modalities_proba_score_kt[fold][num_of_objects][behavior],
                        y_test_property)

            folds_behaviors_modalities_proba_score_bl[fold][num_of_objects] = \
                update_all_behaviors_modalities(folds_behaviors_modalities_proba_score_bl[fold][num_of_objects],
                                                y_test_property)
            folds_behaviors_modalities_proba_score_kt[fold][num_of_objects] = \
                update_all_behaviors_modalities(folds_behaviors_modalities_proba_score_kt[fold][num_of_objects],
                                                y_test_property)

    behaviors_modalities_score_bl = compute_mean_accuracy(folds_behaviors_modalities_proba_score_bl)
    behaviors_modalities_score_kt = compute_mean_accuracy(folds_behaviors_modalities_proba_score_kt)

    for examples_per_object in behaviors_modalities_score_bl:
        for behavior in behaviors_modalities_score_bl[examples_per_object]:
            print(examples_per_object, behavior, behaviors_modalities_score_bl[examples_per_object][behavior])

    for examples_per_object in behaviors_modalities_score_kt:
        for behavior in behaviors_modalities_score_kt[examples_per_object]:
            print(examples_per_object, behavior, behaviors_modalities_score_kt[examples_per_object][behavior])

    # Save results
    db_file_name = log_path + os.sep + DETECTION_TASK + "_" + A_TARGET_ROBOT + "_" + CLF_NAME + "_" + PROJECTION_TYPE + "_projection_KEMA.bin"
    output_file = open(db_file_name, "wb")
    pickle.dump(behaviors_modalities_score_bl, output_file)
    pickle.dump(behaviors_modalities_score_kt, output_file)
    pickle.dump(NO_OF_OBJECTS, output_file)
    pickle.dump(A_TARGET_ROBOT, output_file)
    pickle.dump(CLF_NAME, output_file)
    output_file.close()

    title_name = 'Individual behavior (Baseline Condition)'
    xlabel = 'No. of training objects'
    file_path_name = log_path + os.sep + DETECTION_TASK + "_" + A_TARGET_ROBOT + "_" + CLF_NAME + "_" + PROJECTION_TYPE + "_projection_KEMA_each_behavior_bl"
    plot_all_modalities(behaviors_modalities_score_bl, 'all_modalities', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)
    plot_all_modalities(behaviors_modalities_score_bl, 'all_modalities_train', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)
    plot_all_modalities(behaviors_modalities_score_bl, 'all_modalities_test', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)

    ###################

    title_name = 'Individual behavior (Transfer Condition)'
    xlabel = 'No. of training objects'
    file_path_name = log_path + os.sep + DETECTION_TASK + "_" + A_TARGET_ROBOT + "_" + CLF_NAME + "_" + PROJECTION_TYPE + "_projection_KEMA_each_behavior_kt"
    plot_all_modalities(behaviors_modalities_score_kt, 'all_modalities', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)
    plot_all_modalities(behaviors_modalities_score_kt, 'all_modalities_train', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)
    plot_all_modalities(behaviors_modalities_score_kt, 'all_modalities_test', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)
    #####################

    title_name = 'All behaviors combined (' + A_TARGET_ROBOT + ' as Target)'
    xlabel = 'No. of training objects'
    file_path_name = log_path + os.sep + DETECTION_TASK + "_" + A_TARGET_ROBOT + "_" + CLF_NAME + "_" + PROJECTION_TYPE + "_projection_KEMA_all_behaviors_bl_kt"
    plot_all_behaviors_modalities(behaviors_modalities_score_bl, behaviors_modalities_score_kt,
                                  'all_behaviors_modalities', title_name, xlabel, file_path_name, task=DETECTION_TASK)
    plot_all_behaviors_modalities(behaviors_modalities_score_bl, behaviors_modalities_score_kt,
                                  'all_behaviors_modalities_train', title_name, xlabel, file_path_name, task=DETECTION_TASK)
    plot_all_behaviors_modalities(behaviors_modalities_score_bl, behaviors_modalities_score_kt,
                                  'all_behaviors_modalities_test', title_name, xlabel, file_path_name, task=DETECTION_TASK)
