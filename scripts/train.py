import argparse
import uuid

import neptune as neptune
import numpy as np
import tensorflow as tf
from tqdm import trange

from cvdm.configs.utils import (
    create_data_config,
    create_eval_config,
    create_model_config,
    create_neptune_config,
    create_training_config,
    load_config_from_yaml,
)
from cvdm.diffusion_models.joint_model import instantiate_cvdm
from cvdm.utils.inference_utils import (
    log_loss,
    log_metrics,
    obtain_output_montage_and_metrics,
    save_output_montage,
    save_weights,
)
from cvdm.utils.training_utils import (
    prepare_dataset,
    prepare_model_input,
    train_on_batch_cvdm,
)

tf.keras.utils.set_random_seed(42)


def main() -> None:

    # The script accepts the following command-line arguments:
    # - `--config-path`: The path to the YAML configuration file (required).
    # - `--neptune-token`: The API token for Neptune (optional).

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", help="Path to the configuration file", required=True
    )
    parser.add_argument("--neptune-token", help="API token for Neptune")

    args = parser.parse_args()

    print("Num CPUs Available: ", len(tf.config.list_physical_devices("CPU")))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    config = load_config_from_yaml(args.config_path)
    model_config = create_model_config(config)
    data_config = create_data_config(config)
    training_config = create_training_config(config)
    eval_config = create_eval_config(config)
    neptune_config = create_neptune_config(config)

    task = config.get("task")
    assert task in [
        "biosr_sr",
        "imagenet_sr",
        "biosr_phase",
        "imagenet_phase",
        "other",
    ], "Possible tasks are biosr_sr, imagenet_sr, biosr_phase, imagenet_phase, other"

    print("Getting data...")
    batch_size = data_config.batch_size
    dataset, x_shape, y_shape = prepare_dataset(task, data_config, training=True)
    val_dataset, x_shape, y_shape = prepare_dataset(task, data_config, training=False)

    dataset = dataset.shuffle(5000, reshuffle_each_iteration=False)

    val_len = eval_config.val_len
    val_dataset = val_dataset.take(val_len)
    dataset = dataset.skip(val_len)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    epochs = training_config.epochs
    generation_timesteps = eval_config.generation_timesteps

    print("Creating model...")
    models = instantiate_cvdm(
        lr=training_config.lr,
        generation_timesteps=generation_timesteps,
        cond_shape=x_shape,
        out_shape=y_shape,
        model_config=model_config,
    )
    noise_model, joint_model, schedule_model, mu_model = models
    if model_config.load_weights is not None:
        joint_model.load_weights(model_config.load_weights)
    if model_config.load_mu_weights is not None and mu_model is not None:
        mu_model.load_weights(model_config.load_mu_weights)

    run = None
    if args.neptune_token is not None and neptune_config is not None:
        run = neptune.init_run(
            api_token=args.neptune_token,
            name=neptune_config.name,
            project=neptune_config.project,
        )
        run["config.yaml"].upload(args.config_path)

    log_freq = eval_config.log_freq
    checkpoint_freq = eval_config.checkpoint_freq
    image_freq = eval_config.image_freq
    val_freq = eval_config.val_freq
    output_path = eval_config.output_path
    diff_inp = model_config.diff_inp

    print("Starting training...")
    if model_config.zmd:
        cumulative_loss = np.zeros(6)
    else:
        cumulative_loss = np.zeros(5)
    step = 0
    run_id = str(uuid.uuid4())
    for _ in trange(epochs):
        for batch in dataset:
            batch_x, batch_y = batch

            cmap = "gray" if task in ["biosr_phase", "imagenet_phase"] else None
            cumulative_loss += train_on_batch_cvdm(
                batch_x, batch_y, joint_model, diff_inp=diff_inp
            )

            if step % log_freq == 0:
                log_loss(run=run, avg_loss=cumulative_loss / (step + 1), prefix="train")

            if step % checkpoint_freq == 0:
                save_weights(
                    run=run,
                    model=joint_model,
                    mu_model=mu_model,
                    step=step,
                    output_path=output_path,
                    run_id=run_id,
                )

            if step % image_freq == 0:
                output_montage, metrics = obtain_output_montage_and_metrics(
                    batch_x,
                    batch_y.numpy(),
                    noise_model,
                    schedule_model,
                    mu_model,
                    generation_timesteps,
                    diff_inp,
                    task,
                )
                log_metrics(run, metrics, prefix="train")
                save_output_montage(
                    run=run,
                    output_montage=output_montage,
                    step=step,
                    output_path=output_path,
                    run_id=run_id,
                    prefix="train",
                    cmap=cmap,
                )

            if step % val_freq == 0:
                if model_config.zmd:
                    val_loss = np.zeros(6)
                else:
                    val_loss = np.zeros(5)
                for batch in val_dataset:
                    batch_x, batch_y = batch
                    model_input = prepare_model_input(
                        batch_x, batch_y, diff_inp=diff_inp
                    )
                    val_loss += joint_model.evaluate(
                        model_input, np.zeros_like(batch_y), verbose=0
                    )

                log_loss(run=run, avg_loss=val_loss, prefix="val")
                # To speed up, images are only generated and metrics are calculated only for one batch.
                random_batch = val_dataset.take(1)
                for batch_x, batch_y in random_batch:
                    output_montage, metrics = obtain_output_montage_and_metrics(
                        batch_x,
                        batch_y.numpy(),
                        noise_model,
                        schedule_model,
                        mu_model,
                        generation_timesteps,
                        diff_inp,
                        task,
                    )
                    log_metrics(run, metrics, prefix="val")
                    save_output_montage(
                        run=run,
                        output_montage=output_montage,
                        step=step,
                        output_path=output_path,
                        run_id=run_id,
                        prefix="val",
                        cmap=cmap,
                    )

            step += 1

    if run is not None:
        run.stop()


if __name__ == "__main__":
    main()
