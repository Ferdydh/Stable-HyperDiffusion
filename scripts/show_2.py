from matplotlib import pyplot as plt
from src.core.visualize import plot_image
from src.data.inr import INR
from src.data.inr_dataset import (
    DataHandler,
)

from src.core.config import DataSelector, DatasetType, MLPExperimentConfig


if __name__ == "__main__":
    config = MLPExperimentConfig.default()
    config.data.selector = DataSelector(
        dataset_type=DatasetType.MNIST, class_label=2, sample_id=1096
    )
    mlp = INR(up_scale=16)

    data_handler = DataHandler(config=config)

    data_handler.setup()

    state_dict = data_handler.train_dataloader().dataset.get_state_dict(index=0)
    mlp.load_state_dict(state_dict)
    plot_image(mlp_model=mlp, device="cpu")

    plt.ion()
    plt.show(block=True)
