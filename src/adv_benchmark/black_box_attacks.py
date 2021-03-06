"""
Module for running the boundary attack
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import BoundaryAttack

from adv_benchmark.metrics import degree_of_change


def boundary_attack_run(model_to_attack, target_image, iterations=100):
    """
    This fonction runs the black box boundary attack

    inputs:
    -model_to_attack (tensorflow Model instance): model that will be attacked
    -target_image (numpy array (32*32)): image that will be attack
    -iterations (int): number of times to run the attack

    output:
    -degree_of_change (dict): keys: the number of the iteration, values: the degree of change
        between target and adversarial image

    """
    classifier = TensorFlowV2Classifier(
        model=model_to_attack,
        input_shape=(32, 32, 3),
        clip_values=(0, 255),
        nb_classes=10,
    )
    final_degree_of_change = {}
    attack = BoundaryAttack(
        estimator=classifier, targeted=False, max_iter=0, delta=0.001, epsilon=0.01
    )
    iter_step = 1
    image_list = []
    target = target_image
    x_adv = None
    for i in range(iterations):
        x_adv = attack.generate(x=np.array([target]), x_adv_init=x_adv)

        # clear_output()
        print(
            "Adversarial image at step %d." % (i * iter_step),
            "L2 error",
            np.linalg.norm(np.reshape(x_adv[0] - target, [-1])),
            "and class label %d." % np.argmax(classifier.predict(x_adv)[0]),
        )
        plt.imshow(x_adv[0][..., ::-1].astype("int32"))
        image_list.append(x_adv[0][..., ::-1].astype(np.uint))
        plt.show(block=False)
        final_degree_of_change[i * iter_step] = degree_of_change([x_adv[0]], [target])

        if hasattr(attack, "curr_delta") and hasattr(attack, "curr_epsilon"):
            attack.max_iter = iter_step
            attack.delta = attack.curr_delta
            attack.epsilon = attack.curr_epsilon
        else:
            break
    return final_degree_of_change


def gif_maker(path, image_list, gif_pictures_size=200, duration=40):
    """
    This fonction creates a gif from an image list
    inputs:
    -path: path where the gif will be saved
    -image_list (list): list of image
    -gif_pictures_size(int): the size to which the gif will be resized (because 32*32 is too small)
    -duration (int): number of milliseconds that one image will remain display

    """
    for _, image in enumerate(image_list):

        img = np.array(
            tf.image.resize(
                image.astype("uint8"),
                [gif_pictures_size, gif_pictures_size],
                method="nearest",
                preserve_aspect_ratio=True,
            )
        )
        img = Image.fromarray(img.astype("uint8"), "RGB")

    image_list[0].save(
        path,
        save_all=True,
        append_images=image_list[1:],
        optimize=False,
        duration=duration,
        loop=0,
    )
    return ()
