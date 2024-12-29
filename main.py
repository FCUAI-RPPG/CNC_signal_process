import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import mlflow
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error

from config import cfg
from utils import setup_logger
from data import MachiningErrorGenerator
from model import choose_model

mpl.style.use("seaborn-v0_8")


def main(cfg, logger):
    if cfg.DATASETS.TRAIN_DATA != "":
        test_data = pd.read_csv(cfg.TEST.EVALUATE_DATA)
        train_data = pd.read_csv(cfg.DATASETS.TRAIN_DATA)
    else:
        data = pd.read_csv(cfg.DATASETS.DATA)
        # test_data = data.sample(frac=0.2).copy()
        train_data = data.iloc[0:900].copy()
        test_data = data.iloc[900:].copy()
        # test_data = data[data['setting'].isin(['DOE6-13', 'DOE6-14'])].copy()
        # train_data = data.copy()
        # train_data.drop(index=test_data.index, inplace=True)

    if cfg.TEST.EVALUATE_ONLY == "off":
        train_data.to_csv(f"{cfg.OUTPUT_DIR}/train.csv", index=False)
    test_data.to_csv(f"{cfg.OUTPUT_DIR}/test.csv", index=False)

    train_generator = MachiningErrorGenerator(
        data=train_data,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        sequence_length=cfg.MODEL.SEQUENCE_LENGTH,
        workpiece_length=cfg.DATALOADER.WORKPIECE_LENGTH,
        measure_location=cfg.TEST.LOCATION,
        training=True if cfg.TEST.EVALUATE_ONLY == "off" else False
    )
    test_generator = MachiningErrorGenerator(
        data=test_data,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        sequence_length=cfg.MODEL.SEQUENCE_LENGTH,
        workpiece_length=cfg.DATALOADER.WORKPIECE_LENGTH,
        measure_location=cfg.TEST.LOCATION,
        training=False)

    model = choose_model(cfg)

    losses = {
        'regressor_act_2': tf.keras.losses.Huber(),
    }
    lr_decayed_fn = (tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg.SOLVER.BASE_LR,
        decay_steps=len(train_generator)*cfg.SOLVER.MAX_EPOCHS
    ))
    optimizer = tfa.optimizers.AdamW(learning_rate=lr_decayed_fn, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    metrics = {
        'regressor_act_2': [
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
        ],
    }
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=f"{cfg.OUTPUT_DIR}/checkpoint.h5", verbose=1, save_weights_only=True, save_best_only=False),
        tf.keras.callbacks.CSVLogger(filename=f'{cfg.OUTPUT_DIR}/log.csv', append=True),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min', baseline=None, restore_best_weights=True, start_from_epoch=0)
    ]

    model.compile(
        loss=losses,
        optimizer=optimizer,
        metrics=metrics,
    )

    logger.info("Running with config:\n{}".format(model.summary(print_fn=logger.info)))
    logger.info(f"numbers of taining data: {len(train_generator.sampling_index)}")
    logger.info(f"numbers of testing data: {len(test_generator.sampling_index)}")

    if cfg.TEST.EVALUATE_ONLY == "on":
        for dataset_name in ['train', 'test']:
            predicts = list()
            labels = list()
            for x, y in locals()[f'{dataset_name}_generator']:
                predicts.append(model.predict_on_batch(x))
                labels.append(y)
            predicts = np.concatenate(predicts, axis=0)
            labels = np.concatenate(labels, axis=0)

            predicts = predicts / 100
            labels = labels / 100

            plt.figure(figsize=(15,10))
            ax = plt.gca()
            plt.title(f"MAE: {round(mean_absolute_error(labels, predicts),4)}", fontsize=20)
            location_nums = len(cfg.TEST.LOCATION)
            for i, label, marker in zip(np.arange(location_nums), cfg.TEST.LOCATION, ['o', 'x', '^'][:location_nums]):
                plt.scatter(
                    predicts[i::location_nums],
                    labels[i::location_nums],
                    label=label,
                    c=np.arange(len(predicts[i::location_nums])),
                    cmap=plt.get_cmap('winter'),
                    marker=marker
                )
                for x, y, t in zip(predicts[i::location_nums], labels[i::location_nums], locals()[f'{dataset_name}_data'].No):
                    plt.annotate(str(t), (x,y))
            plt.plot([cfg.TEST.LIM_MIN, cfg.TEST.LIM_MAX], [cfg.TEST.LIM_MIN, cfg.TEST.LIM_MAX], color="red")
            plt.xlabel('estimated error of machining error')
            plt.ylabel('machining error')
            plt.xlim(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX)
            plt.ylim(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX)
            plt.xticks(np.arange(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX+0.001, 0.001), rotation=90)
            plt.yticks(np.arange(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX+0.001, 0.001))
            plt.legend()
            plt.colorbar()
            ax.set_facecolor('silver')
            plt.savefig(f'{cfg.OUTPUT_DIR}/scatter_plot_{dataset_name}.png', bbox_inches='tight')
            plt.close()
            np.save(f"{cfg.OUTPUT_DIR}/predicts_{dataset_name}.npy", predicts)
            np.save(f"{cfg.OUTPUT_DIR}/labels_{dataset_name}.npy", labels)

        return None
    else:
        with mlflow.start_run() as run:
            mlflow.tensorflow.autolog()
            model.fit(
                x=train_generator,
                epochs=cfg.SOLVER.MAX_EPOCHS,
                verbose=1,
                callbacks=callbacks,
                validation_data=test_generator,
                workers=cfg.DATALOADER.NUM_WORKERS,
                max_queue_size=cfg.DATALOADER.MAX_QUEUE_SIZE,
            )

            train_generator = train_generator = MachiningErrorGenerator(
                data=train_data,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                sequence_length=cfg.MODEL.SEQUENCE_LENGTH,
                workpiece_length=cfg.DATALOADER.WORKPIECE_LENGTH,
                measure_location=cfg.TEST.LOCATION,
                training=False
            )
            for dataset_name in ['train', 'test']:
                predicts = list()
                labels = list()
                for x, y in locals()[f'{dataset_name}_generator']:
                    predicts.append(model.predict_on_batch(x))
                    labels.append(y)
                predicts = np.concatenate(predicts, axis=0)
                labels = np.concatenate(labels, axis=0)

                predicts = predicts / 100
                labels = labels / 100

                plt.figure(figsize=(15,10))
                ax = plt.gca()
                plt.title(f"MAE: {round(mean_absolute_error(labels, predicts),4)}", fontsize=20)
                location_nums = len(cfg.TEST.LOCATION)
                for i, label, marker in zip(np.arange(location_nums), cfg.TEST.LOCATION, ['o', 'x', '^'][:location_nums]):
                    plt.scatter(
                        predicts[i::location_nums],
                        labels[i::location_nums],
                        label=label,
                        c=np.arange(len(predicts[i::location_nums])),
                        cmap=plt.get_cmap('winter'),
                        marker=marker
                    )
                    for x, y, t in zip(predicts[i::location_nums], labels[i::location_nums], locals()[f'{dataset_name}_data'].No):
                        plt.annotate(str(t), (x,y))
                plt.plot([cfg.TEST.LIM_MIN, cfg.TEST.LIM_MAX], [cfg.TEST.LIM_MIN, cfg.TEST.LIM_MAX], color="red")
                plt.xlabel('estimated error of machining error')
                plt.ylabel('machining error')
                plt.xlim(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX)
                plt.ylim(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX)
                plt.xticks(np.arange(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX+0.001, 0.001), rotation=90)
                plt.yticks(np.arange(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX+0.001, 0.001))
                plt.legend()
                plt.colorbar()
                ax.set_facecolor('silver')
                plt.savefig(f'{cfg.OUTPUT_DIR}/scatter_plot_{dataset_name}.png', bbox_inches='tight')
                plt.close()
                np.save(f"{cfg.OUTPUT_DIR}/predicts_{dataset_name}.npy", predicts)
                np.save(f"{cfg.OUTPUT_DIR}/labels_{dataset_name}.npy", labels)
                mlflow.log_artifact(f"{cfg.OUTPUT_DIR}/predicts_{dataset_name}.npy")
                mlflow.log_artifact(f"{cfg.OUTPUT_DIR}/labels_{dataset_name}.npy")
                mlflow.log_artifact(f'{cfg.OUTPUT_DIR}/scatter_plot_{dataset_name}.png')

        return None


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Quality Prediction")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("Quality Prediction", output_dir)
    logger.info(args)

    if args.config_file != "":
        logger.info(f"Loaded configuration file {args.config_file}")
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info(f"Running with config:\n{cfg}")

    main(cfg, logger)
