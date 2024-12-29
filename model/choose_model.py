import gc

import tensorflow as tf

from .net import *


def choose_model(cfg):
    model = None

    if cfg.MODEL.NAME == 'sequential_angry_fox':
        model = sequential_angry_fox(
            filters=cfg.MODEL.FILTERS,
            num_layers=cfg.MODEL.NUM_LAYERS,
            sequence_length=cfg.MODEL.SEQUENCE_LENGTH,
            lora=cfg.MODEL.LORA,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )
    elif cfg.MODEL.NAME == 'sequential_transformernet':
        model = sequential_transformernet(
            filters=cfg.MODEL.FILTERS,
            num_layers=cfg.MODEL.NUM_LAYERS,
            lora=cfg.MODEL.LORA,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )
    elif cfg.MODEL.NAME == 'simplenet':
        model = simplenet()
    elif cfg.MODEL.NAME == 'single_angry_fox':
        model = single_angry_fox(
            filters=cfg.MODEL.FILTERS,
            num_layers=cfg.MODEL.NUM_LAYERS,
            sequence_length=cfg.MODEL.SEQUENCE_LENGTH,
            lora=cfg.MODEL.LORA,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )
    else:
        print("Model not found!!")

    if cfg.MODEL.PRETRAINED_PATH != "":
        if cfg.MODEL.LORA:
            if cfg.MODEL.NAME == 'sequential_angry_fox':
                pretrained_model = sequential_angry_fox(
                    filters=cfg.MODEL.FILTERS,
                    num_layers=cfg.MODEL.NUM_LAYERS,
                    sequence_length=cfg.MODEL.SEQUENCE_LENGTH,
                    lora=False,
                    dropout_rate=cfg.MODEL.DROPOUT_RATE,
                )
            elif cfg.MODEL.NAME == 'sequential_transformernet':
                pretrained_model = sequential_transformernet(
                    filters=cfg.MODEL.FILTERS,
                    num_layers=cfg.MODEL.NUM_LAYERS,
                    lora=False,
                    dropout_rate=cfg.MODEL.DROPOUT_RATE,
                )
            pretrained_model.load_weights(cfg.MODEL.PRETRAINED_PATH)

            i = 0
            weights = []
            for weight in model.weights:
                if "LoRA" in weight.name:
                    weights.append(weight)
                else:
                    weights.append(pretrained_model.weights[i])
                    i += 1
            model.set_weights(weights)

            # for layer in model.layers:
            #     if "regressor" in layer.name:
            #         layer.trainable = True
            #     elif "LoRA" in layer.name:
            #         layer.trainable = True
            #     elif "feature_extractor" in layer.name:
            #         continue
            #     else:
            #         layer.trainable = False
            
            del pretrained_model
            tf.keras.backend.clear_session()
            gc.collect()
        else:
            model.load_weights(cfg.MODEL.PRETRAINED_PATH)
    if cfg.TEST.WEIGHT != "":
        model.load_weights(cfg.TEST.WEIGHT)

    return model
