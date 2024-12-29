import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm
from collections import defaultdict

from config import cfg
from utils import setup_logger
from data import MachiningErrorGenerator, MEGenerator2TFDataset
from model import choose_model

mpl.style.use("seaborn-v0_8")


def main(cfg, logger):
    if cfg.DATASETS.TRAIN_DATA != "":
        test_data = pd.read_csv(cfg.TEST.EVALUATE_DATA)
        train_data = pd.read_csv(cfg.DATASETS.TRAIN_DATA)
    else:
        data = pd.read_csv(cfg.DATASETS.DATA)
        train_data = data.iloc[0:700].copy()
        test_data = data.iloc[700:].copy()
        # test_data = data[data['setting'].isin(['DOE6-13', 'DOE6-14'])].copy()
        # train_data = data.copy()
        # train_data.drop(index=test_data.index, inplace=True)
        # test_data = data[(data['setting'] == "DOE6-1") & (~data['name'].isin(train_data.name))]
        # data = data.iloc[0:10].copy()
        # train_data = data.iloc[:5].copy()
        # test_data = data.query('No > 5').copy()
        # test_data = data.sample(frac=0.2).copy()
        # test_data = data.copy()
        # test_data.drop(index=train_data.index, inplace=True)
        # train_data = data.query('setting == "DOE6-1"').copy()
        # test_data = data.query('setting == "DOE6-2"').tail(33).copy()
        # test_data = data[data['setting'] == "DOE6-1"].tail(30).copy()
        # test_data_doe6_3 = data[data['setting'] == "DOE6-3"].copy()
        # train_data_doe6_3 = data[data['setting'] == "DOE6-3"].copy()
        # train_data_doe6_2 = data[data['setting'] == "DOE6-2"].tail(35).copy()
        # train_data_doe6_1 = data[data['setting'] == "DOE6-1"].head(2).copy()
        # train_data = pd.concat([train_data_doe6_1, train_data_doe6_2, train_data_doe6_3], ignore_index=True)

    if cfg.TEST.EVALUATE_ONLY == "off":
        train_data.to_csv(f"{cfg.OUTPUT_DIR}/train.csv", index=False)
    test_data.to_csv(f"{cfg.OUTPUT_DIR}/test.csv", index=False)

    train_generator = MachiningErrorGenerator(
        data=train_data,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        sequence_length=cfg.MODEL.SEQUENCE_LENGTH,
        workpiece_length=cfg.DATALOADER.WORKPIECE_LENGTH,
        measure_location=cfg.TEST.LOCATION,
        training=False if cfg.TEST.EVALUATE_ONLY == "off" else False
    )
    test_generator = MachiningErrorGenerator(
        data=test_data,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        sequence_length=cfg.MODEL.SEQUENCE_LENGTH,
        workpiece_length=cfg.DATALOADER.WORKPIECE_LENGTH,
        measure_location=cfg.TEST.LOCATION,
        training=False)
    train_dataset = MEGenerator2TFDataset(train_generator).shuffle(buffer_size=train_generator.sampling_index.shape[0], reshuffle_each_iteration=True).batch(cfg.SOLVER.BATCH_SIZE)

    model = choose_model(cfg)

    loss_fn = tf.keras.losses.Huber()
    lr_decayed_fn = (tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg.SOLVER.BASE_LR,
        decay_steps=len(train_generator)*cfg.SOLVER.MAX_EPOCHS
    ))
    optimizer = tfa.optimizers.AdamW(learning_rate=lr_decayed_fn, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

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

            # 添加 NaN 检查
            check_nan(labels, "labels", logger)
            check_nan(predicts, "predictions", logger)
            
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
            plt.plot([cfg.TEST.LIM_MIN-0.001, cfg.TEST.LIM_MAX+0.001], [cfg.TEST.LIM_MIN, cfg.TEST.LIM_MAX+0.001], color="red")
            plt.xlabel('estimated error of machining error')
            plt.ylabel('machining error')
            plt.xlim(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX)
            plt.ylim(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX)
            plt.xticks(np.arange(cfg.TEST.LIM_MIN-0.001,cfg.TEST.LIM_MAX+0.002, 0.001), rotation=90)
            plt.yticks(np.arange(cfg.TEST.LIM_MIN-0.001,cfg.TEST.LIM_MAX+0.002, 0.001))
            plt.legend()
            plt.colorbar()
            ax.set_facecolor('silver')
            plt.savefig(f'{cfg.OUTPUT_DIR}/scatter_plot_{dataset_name}.png', bbox_inches='tight')
            plt.close()
            np.save(f"{cfg.OUTPUT_DIR}/predicts_{dataset_name}.npy", predicts)
            np.save(f"{cfg.OUTPUT_DIR}/labels_{dataset_name}.npy", labels)

        return None
    else:
        @tf.function(experimental_relax_shapes=True)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = loss_fn(y, logits)
                # to_update = [i for i in model.trainable_variables if ('bias' in i.name) | ('regressor_ffn_2' in i.name) | ('LoRA' in i.name)]
                to_update = model.trainable_variables
            grads = tape.gradient(loss_value, to_update)
            optimizer.apply_gradients(zip(grads, to_update))
            return loss_value
        
        best_score = 1e10
        all_loss = defaultdict(list)
        for _ in tqdm(range(cfg.SOLVER.MAX_EPOCHS)):
            # train_loss = []
            for x_batch_train, y_batch_train in train_dataset:
                loss_value = train_step(x_batch_train, y_batch_train['regressor_act_2'])
                # train_loss.append(loss_value)
            # train_loss = np.array(train_loss)
            # all_loss['train_loss'].append(train_loss.mean())

            # predicts = list()
            # labels = list()
            # for x, y in test_generator:
            #     predicts.append(model.predict_on_batch(x))
            #     labels.append(y)
            # predicts = np.concatenate(predicts, axis=0)
            # labels = np.concatenate(labels, axis=0)
            # loss_value = loss_fn(predicts, labels)
            # all_loss['test_loss'].append(loss_value.numpy())

        #     predicts = predicts / 100
        #     labels = labels / 100

        #     score = mean_absolute_error(labels, predicts)
        #     if score <= best_score:
        #         best_score = score
        #         model.save_weights(f"{cfg.OUTPUT_DIR}/checkpoint.h5")
        # model.load_weights(f"{cfg.OUTPUT_DIR}/checkpoint.h5")
        model.save(f"{cfg.OUTPUT_DIR}/checkpoint.h5")
        # Convert the list to a DataFrame
        # Export the DataFrame to a CSV file
        # df = pd.DataFrame(all_loss)
        # df.to_csv("dataset/DOE_6/EXP_DOE4/finetune/note/1-下午權重2shot/loss.csv", index=False)
        # plt.figure(figsize=(10, 6))
        # plt.plot(df['train_loss'], label='Train Loss', marker='o')
        # plt.plot(df['test_loss'], label='Test Loss', marker='x')

        # plt.title('Train and Test Loss Over Epochs', fontsize=16)
        # plt.xlabel('Epochs', fontsize=12)
        # plt.ylabel('Loss', fontsize=12)
        # plt.legend()
        # plt.grid(True)
        # plt.savefig('dataset/DOE_6/EXP_DOE4/finetune/note/1-下午權重2shot/loss_plot.png', bbox_inches='tight')
                

        train_generator = MachiningErrorGenerator(
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
            plt.plot([cfg.TEST.LIM_MIN-0.001, cfg.TEST.LIM_MAX+0.001], [cfg.TEST.LIM_MIN-0.001, cfg.TEST.LIM_MAX+0.001], color="red")
            plt.xlabel('estimated error of machining error')
            plt.ylabel('machining error')
            plt.xlim(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX)
            plt.ylim(cfg.TEST.LIM_MIN,cfg.TEST.LIM_MAX)
            plt.xticks(np.arange(cfg.TEST.LIM_MIN-0.001,cfg.TEST.LIM_MAX+0.002, 0.001), rotation=90)
            plt.yticks(np.arange(cfg.TEST.LIM_MIN-0.001,cfg.TEST.LIM_MAX+0.002, 0.001))
            plt.legend()
            plt.colorbar()
            ax.set_facecolor('silver')
            plt.savefig(f'{cfg.OUTPUT_DIR}/scatter_plot_{dataset_name}.png', bbox_inches='tight')
            plt.close()
            np.save(f"{cfg.OUTPUT_DIR}/predicts_{dataset_name}.npy", predicts)
            np.save(f"{cfg.OUTPUT_DIR}/labels_{dataset_name}.npy", labels)

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