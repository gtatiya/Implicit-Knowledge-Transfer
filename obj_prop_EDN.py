# Author: Gyan Tatiya

import csv
import os
import pickle
import time

import numpy as np

from constants import *

from sklearn.svm import SVC

import tensorflow as tf

from model import classifier, EncoderDecoderNetworkTF
from utils import split_train_test_v2, repeat_trials, augment_trials, update_all_modalities,\
    update_all_behaviors_modalities, compute_mean_accuracy, get_robot_data, get_robot_objects_data, plot_loss_curve,\
    save_cost_csv, plot_all_modalities, plot_all_behaviors_modalities, time_taken, plot_fold_all_behaviors_modalities


if __name__ == "__main__":
    """
    This script trains an object property recognition model for each behavior and modality of a target robot data
    incrementally and source robot data.
    Fuses the predictions of all the behaviors and modalities by uniform combination, train score, and test score.
    For training maximum 80% of the objects are used and for testing rest 20% of the objects are used.
    For training, the target robot data is increased incrementally.
    For testing, novel objects are used.
    Projection can be done using EDN in 2 ways: object-identity and object-property based.
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

    DETECTION_TASK = 'weight'  # content, weight, color

    PROJECTION_TYPE = 'object'  # object, property

    NUM_TRIALS_AUGMENT = 5  # 0 for no augmentation

    objects_labels = get_objects_labels('object', objects_list=metadata['grasp']['objects'])
    print("objects_labels: ", len(objects_labels), objects_labels)

    properties_labels = get_objects_labels(DETECTION_TASK)
    print("properties_labels: ", len(properties_labels), properties_labels)

    if DETECTION_TASK == 'weight':
        NO_OF_OBJECTS = [4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 76]  # [4, 20, 40, 76]
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

    results_path = 'results'
    log_path = results_path + os.sep + DETECTION_TASK + '_' + A_TARGET_ROBOT + '_' + CLF_NAME + "_" + PROJECTION_TYPE +\
               '_projection_EDN_' + ("_AUG_" + str(NUM_TRIALS_AUGMENT) + "_trials" if NUM_TRIALS_AUGMENT else '')
    os.makedirs(log_path, exist_ok=True)

    # Writing log file for execution time
    with open(log_path + os.sep + 'time_log.txt', 'w') as time_log_file:
        time_log_file.write('Time Log\n')
        main_start_time = time.time()

    # Hyper-Parameters
    TRAINING_EPOCHS = 1000
    LEARNING_RATE = 0.0001
    CODE_VECTOR = 125
    HIDDEN_LAYER_UNITS = [1000, 500, 250]
    ACTIVATION_FUNCTION = tf.nn.elu

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

                    start_time = time.time()

                    log_path_fold = log_path + os.sep + fold + os.sep + str(num_of_objects) + os.sep + \
                                    "_".join([behavior, modality]) + os.sep
                    os.makedirs(log_path_fold, exist_ok=True)

                    results = []
                    with open(log_path_fold + os.sep + "results.csv", 'w') as f:
                        writer = csv.writer(f, lineterminator="\n")
                        writer.writerow(["Baseline (Test Accuracy)", "Baseline (Train Accuracy)",
                                         "Transfer (Test Accuracy)", "Transfer (Train Accuracy)"])

                    # Get test data
                    X_test_target, y_test_object_target, y_test_property_target = \
                        get_robot_objects_data(TRAIN_TEST_SPLITS[fold]["test"], objects_labels,
                                               properties_labels, DETECTION_TASK,
                                               robots_data[A_TARGET_ROBOT][behavior][modality])

                    print("X_test_target: ", X_test_target.shape)
                    print("y_test_object_target: ", y_test_object_target.shape, y_test_object_target.flatten())
                    print("y_test_property_target: ", y_test_property_target.shape, y_test_property_target.flatten())

                    # Get train data
                    X_train_target, y_train_object_target, y_train_property_target = \
                        get_robot_objects_data(TRAIN_TEST_SPLITS[fold]["train"][0:num_of_objects], objects_labels,
                                               properties_labels, DETECTION_TASK,
                                               robots_data[A_TARGET_ROBOT][behavior][modality])

                    print("X_train_target: ", X_train_target.shape)
                    print("y_train_object_target: ", y_train_object_target.shape, y_train_object_target.flatten())
                    print("y_train_property_target: ", y_train_property_target.shape, y_train_property_target.flatten())

                    if NUM_TRIALS_AUGMENT:
                        X_train_target, y_train_object_target, y_train_property_target = \
                            augment_trials(X_train_target, y_train_object_target, y_train_property_target, NUM_TRIALS_AUGMENT)
                        print("After Data Augmentations:")
                        print("X_train_target: ", X_train_target.shape)
                        print("y_train_object_target: ", y_train_object_target.shape, y_train_object_target.flatten())
                        print("y_train_property_target: ", y_train_property_target.shape, y_train_property_target.flatten())

                    # Train and Test
                    y_acc, y_pred, y_proba = classifier(CLF, X_train_target, X_test_target,
                                                        y_train_property_target.ravel(), y_test_property_target.ravel())

                    y_proba_pred = np.argmax(y_proba, axis=1)
                    y_prob_acc = np.mean(y_test_property_target.ravel() == y_proba_pred)
                    print("y_proba_pred: ", len(y_proba_pred), y_proba_pred)
                    print("y_prob_acc: ", y_prob_acc)
                    results.append(y_prob_acc)

                    folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior][modality]['proba'] = y_proba
                    folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior][modality]['test_acc'] = y_prob_acc

                    # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
                    # Use only training data to get a score
                    y_acc_train, y_pred_train, y_proba_train = classifier(CLF, X_train_target, X_train_target,
                                                                          y_train_property_target.ravel(),
                                                                          y_train_property_target.ravel())
                    y_proba_pred_train = np.argmax(y_proba_train, axis=1)
                    y_prob_acc_train = np.mean(y_train_property_target.ravel() == y_proba_pred_train)
                    print("y_proba_pred_train: ", y_proba_pred_train.shape)  # , y_proba_pred_train)
                    print("y_prob_acc_train: ", y_prob_acc_train)
                    results.append(y_prob_acc_train)

                    folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior][modality]['train_acc'] = y_prob_acc_train

                    for source_robot in SOURCE_ROBOT_LIST:

                        X_train_source, y_train_object_source, y_train_property_source = \
                            get_robot_objects_data(TRAIN_TEST_SPLITS[fold]["train"][0:num_of_objects], objects_labels,
                                                   properties_labels, DETECTION_TASK,
                                                   robots_data[source_robot][behavior][modality])
                        print("X_train_source: ", X_train_source.shape)
                        print("y_train_object_source: ", y_train_object_source.shape, y_train_object_source.flatten())
                        print("y_train_property_source: ", y_train_property_source.shape, y_train_property_source.flatten())

                        if NUM_TRIALS_AUGMENT:
                            X_train_source, y_train_object_source, y_train_property_source = \
                                augment_trials(X_train_source, y_train_object_source, y_train_property_source,
                                               NUM_TRIALS_AUGMENT)
                            print("After Data Augmentations:")
                            print("X_train_source: ", X_train_source.shape)
                            print("y_train_object_source: ", y_train_object_source.shape, y_train_object_source.flatten())
                            print("y_train_property_source: ", y_train_property_source.shape, y_train_property_source.flatten())

                        if PROJECTION_TYPE == 'object':
                            X_train_source_repeat, y_train_object_source_repeat, X_train_target_repeat, \
                            y_train_object_target_repeat = repeat_trials(y_train_object_source,
                                                                         y_train_object_target,
                                                                         X_train_source, X_train_target)

                            print("X_train_source_repeat: ", X_train_source_repeat.shape)
                            # print("y_train_object_source_repeat: ", y_train_object_source_repeat.shape, y_train_object_source_repeat)
                            print("X_train_target_repeat: ", X_train_target_repeat.shape)
                            # print("y_train_object_target_repeat: ", y_train_object_target_repeat.shape, y_train_object_target_repeat)
                        else:
                            X_train_source_repeat, y_train_property_source_repeat, X_train_target_repeat, \
                            y_train_property_target_repeat = repeat_trials(y_train_property_source,
                                                                           y_train_property_target,
                                                                           X_train_source, X_train_target)

                            print("X_train_source_repeat: ", X_train_source_repeat.shape)
                            # print("y_train_property_source_repeat: ", y_train_property_source_repeat.shape, y_train_property_source_repeat)
                            print("X_train_target_repeat: ", X_train_target_repeat.shape)
                            # print("y_train_property_target_repeat: ", y_train_property_target_repeat.shape, y_train_property_target_repeat)

                        data = {'source': X_train_source_repeat, 'target': X_train_target_repeat}

                        # Implement the network
                        tf.reset_default_graph()
                        num_of_features_1 = data['source'].shape[-1]
                        num_of_features_2 = data['target'].shape[-1]
                        edn = EncoderDecoderNetworkTF(input_channels=num_of_features_1,
                                                      output_channels=num_of_features_2,
                                                      hidden_layer_sizes=HIDDEN_LAYER_UNITS,
                                                      n_dims_code=CODE_VECTOR,
                                                      learning_rate=LEARNING_RATE,
                                                      activation_fn=ACTIVATION_FUNCTION)

                        # Train the network
                        cost_log = edn.train_session(data['source'], data['target'], TRAINING_EPOCHS, None)

                        plot_loss_curve(cost_log, log_path_fold,
                                        title_name="-".join([behavior, modality]) + "_Loss_EDN",
                                        xlabel='Training Iterations', ylabel='Loss')
                        save_cost_csv(cost_log, log_path_fold, csv_name="-".join([behavior, modality]) + "_Loss_EDN")

                        X_test_source, y_test_object_source, y_test_property_source = \
                            get_robot_objects_data(TRAIN_TEST_SPLITS[fold]["test"], objects_labels,
                                                   properties_labels, DETECTION_TASK,
                                                   robots_data[source_robot][behavior][modality])
                        print("X_test_source: ", X_test_source.shape)
                        print("y_test_object_source: ", y_test_object_source.shape, y_test_object_source.flatten())
                        print("y_test_property_source: ", y_test_property_source.shape, y_test_property_source.flatten())

                        X_source, y_object_source, y_property_source = \
                            get_robot_objects_data(objects_labels.keys(), objects_labels,
                                                   properties_labels, DETECTION_TASK,
                                                   robots_data[source_robot][behavior][modality])
                        print("X_source: ", X_source.shape)
                        print("y_object_source: ", y_object_source.shape, y_object_source.flatten())
                        print("y_property_source: ", y_property_source.shape, y_property_source.flatten())

                        # Generate features using trained network
                        X_test_target_gen = edn.generate(X_test_source)
                        X_test_target_gen = np.array(X_test_target_gen)

                        # Test data loss
                        test_loss = edn.rmse_loss(X_test_target_gen, X_test_target)
                        print("1 GEN test_loss: ", test_loss)

                        X_train_target_gen = edn.generate(data['source'])
                        X_train_target_gen = np.array(X_train_target_gen)

                        train_loss = edn.rmse_loss(X_train_target_gen, data['target'])
                        print("2 GEN train_loss: ", train_loss)

                        X_train_target_gen = edn.generate(X_train_source)
                        X_train_target_gen = np.array(X_train_target_gen)

                        X_target_gen = edn.generate(X_source)
                        X_target_gen = np.array(X_target_gen)

                        with open(log_path_fold + os.sep + "loss_EDN.csv", 'w') as f:
                            writer = csv.writer(f, lineterminator="\n")
                            writer.writerow(["Test Loss", test_loss])
                            writer.writerow(["Train Loss", train_loss])

                    X_train = np.concatenate((X_train_target, X_test_target_gen, X_train_target_gen), axis=0)
                    y_train_property = np.concatenate((y_train_property_target, y_test_property_source, y_train_property_source), axis=0)

                    print("Z X_train: ", X_train.shape)
                    print("Z y_train_property: ", y_train_property.shape, y_train_property.flatten())

                    y_acc, y_pred, y_proba = classifier(CLF, X_train, X_test_target, y_train_property.ravel(),
                                                        y_test_property_target.ravel())

                    y_proba_pred = np.argmax(y_proba, axis=1)
                    y_prob_acc = np.mean(y_test_property_target.ravel() == y_proba_pred)
                    print("y_proba_pred: ", len(y_proba_pred), y_proba_pred)
                    print("y_prob_acc: ", y_prob_acc)
                    results.append(y_prob_acc)

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
                    results.append(y_prob_acc_train)

                    folds_behaviors_modalities_proba_score_kt[fold][num_of_objects][behavior][modality]['train_acc'] = y_prob_acc_train

                    with open(log_path_fold + os.sep + "results.csv", 'a') as f:  # append to the file created
                        writer = csv.writer(f, lineterminator="\n")
                        writer.writerow(results)

                    # Writing log file for execution time
                    file = open(log_path + os.sep + 'time_log.txt', 'a')  # append to the file created
                    end_time = time.time()
                    file.write("\n\n" + fold + os.sep + str(num_of_objects) + os.sep + "_".join([behavior, modality]))
                    file.write("\nTime: " + time_taken(start_time, end_time))
                    file.write("\nTotal Time: " + time_taken(main_start_time, end_time))
                    file.close()

                folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior] = \
                    update_all_modalities(
                        folds_behaviors_modalities_proba_score_bl[fold][num_of_objects][behavior],
                        y_test_property_target)
                folds_behaviors_modalities_proba_score_kt[fold][num_of_objects][behavior] = \
                    update_all_modalities(
                        folds_behaviors_modalities_proba_score_kt[fold][num_of_objects][behavior],
                        y_test_property_target)

            folds_behaviors_modalities_proba_score_bl[fold][num_of_objects] = \
                update_all_behaviors_modalities(folds_behaviors_modalities_proba_score_bl[fold][num_of_objects],
                                                y_test_property_target)
            folds_behaviors_modalities_proba_score_kt[fold][num_of_objects] = \
                update_all_behaviors_modalities(folds_behaviors_modalities_proba_score_kt[fold][num_of_objects],
                                                y_test_property_target)

        title_name = 'All behaviors combined (' + A_TARGET_ROBOT + ' as Target) - ' + fold
        xlabel = 'No. of training objects'
        file_path_name = log_path + os.sep + fold + os.sep + DETECTION_TASK + "_" + A_TARGET_ROBOT + "_" + CLF_NAME + \
                         "_" + PROJECTION_TYPE + "_projection_EDN_all_behaviors_bl_kt_" + fold
        plot_fold_all_behaviors_modalities(folds_behaviors_modalities_proba_score_bl,
                                           folds_behaviors_modalities_proba_score_kt,
                                           fold, 'all_behaviors_modalities', title_name, xlabel, file_path_name,
                                           task=DETECTION_TASK)
        plot_fold_all_behaviors_modalities(folds_behaviors_modalities_proba_score_bl,
                                           folds_behaviors_modalities_proba_score_kt,
                                           fold, 'all_behaviors_modalities_train', title_name, xlabel, file_path_name,
                                           task=DETECTION_TASK)
        plot_fold_all_behaviors_modalities(folds_behaviors_modalities_proba_score_bl,
                                           folds_behaviors_modalities_proba_score_kt,
                                           fold, 'all_behaviors_modalities_test', title_name, xlabel, file_path_name,
                                           task=DETECTION_TASK)

    behaviors_modalities_score_bl = compute_mean_accuracy(folds_behaviors_modalities_proba_score_bl)
    behaviors_modalities_score_kt = compute_mean_accuracy(folds_behaviors_modalities_proba_score_kt)

    for examples_per_object in behaviors_modalities_score_bl:
        for behavior in behaviors_modalities_score_bl[examples_per_object]:
            print(examples_per_object, behavior, behaviors_modalities_score_bl[examples_per_object][behavior])

    for examples_per_object in behaviors_modalities_score_kt:
        for behavior in behaviors_modalities_score_kt[examples_per_object]:
            print(examples_per_object, behavior, behaviors_modalities_score_kt[examples_per_object][behavior])

    # Save results
    db_file_name = log_path + os.sep + DETECTION_TASK + "_" + A_TARGET_ROBOT + "_" + CLF_NAME + "_" + PROJECTION_TYPE + "_projection_EDN.bin"
    output_file = open(db_file_name, "wb")
    pickle.dump(behaviors_modalities_score_bl, output_file)
    pickle.dump(behaviors_modalities_score_kt, output_file)
    pickle.dump(NO_OF_OBJECTS, output_file)
    pickle.dump(A_TARGET_ROBOT, output_file)
    pickle.dump(CLF_NAME, output_file)
    output_file.close()

    title_name = 'Individual behavior (Baseline Condition)'
    xlabel = 'No. of training objects'
    file_path_name = log_path + os.sep + DETECTION_TASK + "_" + A_TARGET_ROBOT + "_" + CLF_NAME + "_" + PROJECTION_TYPE + "_projection_EDN_each_behavior_bl"
    plot_all_modalities(behaviors_modalities_score_bl, 'all_modalities', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)
    plot_all_modalities(behaviors_modalities_score_bl, 'all_modalities_train', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)
    plot_all_modalities(behaviors_modalities_score_bl, 'all_modalities_test', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)

    ###################

    title_name = 'Individual behavior (Transfer Condition)'
    xlabel = 'No. of training objects'
    file_path_name = log_path + os.sep + DETECTION_TASK + "_" + A_TARGET_ROBOT + "_" + CLF_NAME + "_" + PROJECTION_TYPE + "_projection_EDN_each_behavior_kt"
    plot_all_modalities(behaviors_modalities_score_kt, 'all_modalities', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)
    plot_all_modalities(behaviors_modalities_score_kt, 'all_modalities_train', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)
    plot_all_modalities(behaviors_modalities_score_kt, 'all_modalities_test', NO_OF_OBJECTS, title_name, xlabel,
                        file_path_name, task=DETECTION_TASK)
    #####################

    title_name = 'All behaviors combined (' + A_TARGET_ROBOT + ' as Target)'
    xlabel = 'No. of training objects'
    file_path_name = log_path + os.sep + DETECTION_TASK + "_" + A_TARGET_ROBOT + "_" + CLF_NAME + "_" + PROJECTION_TYPE + "_projection_EDN_all_behaviors_bl_kt"
    plot_all_behaviors_modalities(behaviors_modalities_score_bl, behaviors_modalities_score_kt,
                                  'all_behaviors_modalities', title_name, xlabel, file_path_name, task=DETECTION_TASK)
    plot_all_behaviors_modalities(behaviors_modalities_score_bl, behaviors_modalities_score_kt,
                                  'all_behaviors_modalities_train', title_name, xlabel, file_path_name, task=DETECTION_TASK)
    plot_all_behaviors_modalities(behaviors_modalities_score_bl, behaviors_modalities_score_kt,
                                  'all_behaviors_modalities_test', title_name, xlabel, file_path_name, task=DETECTION_TASK)
