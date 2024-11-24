from matplotlib import pyplot as plt
from core.utils import load_config, get_device, plot_image
from models.inr import INR
from data.inr_dataset import DataHandler, DataSelector, DatasetType


def visualize(experiment: str = "autoencoder_sanity_check"):
    """Visualize an existing MLP from the dataset."""
    cfg = load_config(experiment)
    device = get_device()
    mlp = INR(up_scale=16)

    data_handler = DataHandler(
        hparams={**cfg["data"], "device": device},
        data_folder=cfg["data"]["data_path"],
        selectors=DataSelector(
            dataset_type=DatasetType[cfg["data"]["dataset_type"]],
            class_label=cfg["data"]["class_label"],
            sample_id=cfg["data"]["sample_id"],
        ),
    )

    state_dict = data_handler.get_state_dict(index=0)
    mlp.load_state_dict(state_dict)
    fig = plot_image(mlp_model=mlp, device=device)

    plt.ion()
    plt.show(block=True)
