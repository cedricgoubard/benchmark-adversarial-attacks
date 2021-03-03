"""
Module for running white box attacks and computing their performances
"""
import numpy as np
from tqdm import tqdm

import tensorflow as tf

from tensorflow import convert_to_tensor
from foolbox.attacks import LinfFastGradientAttack, LinfDeepFoolAttack
from foolbox.models import TensorFlowModel

from adv_benchmark.metrics import degree_of_change, success_rate


def attack_performances_computer(
    model_to_attack, predicting_model, attack, image_list, labels, epsilon
):  # pylint: disable=too-many-arguments
    """
    This fonction launch an attack against a model and returns the performances of the attack
    inputs:
    -model_to_attack (tensorflow model): model that will be attacked
    -predicting_model (tensorflow model): model that will predict the label of the generated
                adv example (most of the time it is the same that model_to_attack) but sometimes
                it is usefull to have another model taking care of the prediction
    -attack (foolbox attack)
    -image_list: list of images array (32*32*3) to attack
    -labels: labels (one hot encoding) of the image
    -epsilon (float): epsilon is the amount of noise added into the image at each step

    outputs:
    -DOC_attack (float) : average degreee of change of the attack
    -SR_on_attacked_model (float): success rate of the attack on the first model of the inputs
        (model_to_attack)
    -SR_on_predicting_model (float) : success rate of the attack on the second model of the inputs
        (predicting_model)
    """

    model_to_attack = TensorFlowModel(model_to_attack, bounds=(0, 255))
    success = {"attacked_model": [], "predicting_model": []}
    adv_list = []
    labels = list(map(np.argmax, labels))

    for i, image in enumerate(tqdm(image_list, position=0)):
        image = np.asarray(image)[:, :, :3].astype("float32")
        image = convert_to_tensor(np.expand_dims(image, axis=0))
        label = tf.convert_to_tensor(np.array([labels[i]]))
        _, clipped, is_adv = attack(model_to_attack, image, label, epsilons=epsilon)

        success["attacked_model"].append(bool(is_adv[0]))
        adv_list.append(np.array(clipped[0]))
        prediction = predicting_model.predict(clipped[0])
        success["predicting_model"].append(np.argmax(prediction) != labels[i])


    return (
        degree_of_change(adv_list, image_list),
        success_rate(success["attacked_model"]),
        success_rate(success["predicting_model"])
    )


def attack_runner(
    model_to_attack,
    predicting_model,
    image_list,
    labels_list,
    epislons_list,
    attack=LinfFastGradientAttack(),
):  # pylint: disable=too-many-arguments
    """
    This fonction launch an attack against a model and returns the performances of the attack

    -model_to_attack (tensorflow model): model that will be attacked
    -predicting_model (tensorflow model): model that will predict the label of the generated
    adv example (most of the time it is the same that model_to_attack) but sometimes
    it is usefull to have another model taking care of the prediction
    -image_list: list of images array (32*32*3) to attack
    -labels: labels (one hot encoding) of the image
    -epsilon_list (list of floats): list of epsilon to try

    outputs:
    -SR_on_attacked_model (dict): result of the attack on model to attack - keys: average DOC
        values computed for one epsilon, values: average success rate (SR) values computed for one
        epsilon
    -SR_on_predicting_model (dict):result of the attack on predicting_model - keys: average DOC
        values computed for one epsilon, values: average success rate (SR) values computed for one
        epsilon
    """

    succ_rate_on_attacked_model = {}
    succ_rate_on_predicting_model = {}
    for epsilon in epislons_list:
        print("======" + "epislon: " + str(epsilon) + "======")
        deg_of_change, succ_rate_attacked, succ_rate_pred = attack_performances_computer(
            model_to_attack,
            predicting_model,
            attack,
            image_list,
            labels_list,
            [epsilon],
        )
        succ_rate_on_attacked_model[deg_of_change] = succ_rate_attacked
        succ_rate_on_predicting_model[deg_of_change] = succ_rate_pred
    return (succ_rate_on_attacked_model, succ_rate_on_predicting_model)


def three_attacks_runner(
    model_to_attack, predicting_model, image_list, labels_list, epislons_list
):
    """
    This fonction launch three attacks (FGSM, deepfool 1 step and deepfool mutliple steps)
    against a model and returns the performances of the attacks

    -model_to_attack (tensorflow model): model that will be attacked
    -predicting_model (tensorflow model): model that will predict the label of the generated
    adv example (most of the time it is the same that model_to_attack) but sometimes
    it is usefull to have another model taking care of the prediction
    -image_list: list of images array (32*32*3) to attack
    -labels: labels (one hot encoding) of the image
    -epsilon_list (list of floats): list of epsilon to try

    outputs:
    -SR_deepfool_1step_dic (dict): result of the attack on model to attack - keys: average DOC
        values computed for one epsilon,
    values: average success rate (SR) values computed for one epsilon for FGSM
    -SR_deepfool_mutliple_steps_dic (dict): result of the attack on model to attack - keys: average
        DOC values computed for one epsilon, values: average success rate (SR) values computed for
        one epsilon for deepfool with 1 step
    -SR_FGSM_dic (dict): result of the attack on model to attack - keys: average DOC values computed
    for one epsilon, values: average success rate (SR) values computed for one epsilon for deepfool

    """
    if len(labels_list[0]) <= 5:
        labels_list = list(map(np.argmax, labels_list))

    attacks = {
        "FGSM": LinfFastGradientAttack(),
        "Deepfool1s": LinfDeepFoolAttack(steps=1),
        "Deepfoolmult": LinfDeepFoolAttack()
    }

    succ_rates = {"FGSM": {}, "Deepfool1s": {}, "Deepfoolmult": {}}

    args = {
        "model_to_attack": model_to_attack,
        "predicting_model": predicting_model,
        "image_list": image_list,
        "labels": labels_list,
    }

    for epsilon in epislons_list:
        args["epsilon"] = [epsilon]
        print("======" + "epislon: " + str(epsilon) + "======")

        for attack_name, attack in attacks.items():
            deg_of_change, succ_rate, _ = attack_performances_computer(
                attack=attack,
                **args
            )
            succ_rates[attack_name][deg_of_change] = succ_rate

    return (succ_rates["Deepfool1s"], succ_rates["Deepfoolmult"], succ_rates["FGSM"])
