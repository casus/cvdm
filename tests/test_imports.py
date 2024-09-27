def test_imports():
    try:
        # Check if external libraries can be imported
        import argparse
        import uuid

        import neptune
        import numpy as np
        import tensorflow as tf
        from tqdm import trange

        # Check if custom modules can be imported
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

        assert True  # If no ImportError occurs, the test will pass

    except ImportError as e:
        assert False, f"Import failed: {e}"
