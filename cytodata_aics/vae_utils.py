import logging
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
import os
import tempfile
import pandas as pd
from typing import Optional
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import mlflow
from tqdm import tqdm
import math

logger = logging.getLogger(__name__)

class BaseCallback(Callback):
    def __init__(self, test_all=False, on_train=False, on_val=False, on_test=False, *args, **kwargs):
        self.test_all = test_all

        self.on = dict()
        self.on["train"] = on_train
        self.on["val"] = on_val
        self.on["test"] = on_test


    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.__on_epoch_end("train", trainer, pl_module)


    def on_val_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.__on_epoch_end("val", trainer, pl_module)


    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.test_all:
            self.__on_epoch_end("train", trainer, pl_module)
            self.__on_epoch_end("val", trainer, pl_module)

        self.__on_epoch_end("test", trainer, pl_module)


    def __on_epoch_end(self, split, trainer, pl_module):
        if self.on[split]:
            self._on_epoch_end(split, trainer, pl_module)

    def _on_epoch_end(self, split, trainer, pl_module):
        raise NotImplementedError

    def _check_outputs(self, split, trainer, pl_module):
        if split not in pl_module._cached_outputs:
            with torch.no_grad():
                dl = getattr(trainer.datamodule, f"{split}_dataloader")()

                pl_module._cached_outputs[split] = []
                for batch_idx, batch in enumerate(dl):
                    batch[pl_module.hparams.x_label] = (
                        batch[pl_module.hparams.x_label].to(pl_module.device))

                    if pl_module.hparams.loss_mask_label is not None:
                        batch[pl_module.hparams.loss_mask_label] = (
                            batch[pl_module.hparams.loss_mask_label].to(
                                pl_module.device))

                    pl_module._cached_outputs[split].append(
                        pl_module._step(split, batch, batch_idx, False)
                    )

        return pl_module._cached_outputs[split]


class GetKLDRanks(BaseCallback):
    def _on_epoch_end(self, split, trainer, pl_module):
        with torch.no_grad():
            outputs = self._check_outputs(split, trainer, pl_module)
            log_artifacts(outputs, split, trainer.current_epoch,
                            pl_module.latent_dim)                      

def log_artifacts(outputs, prefix, current_epoch, latent_dim):

    input_key = [i for i in outputs[0].keys() if "z_parts_params" in i][0]
    input_key = input_key.split('/')[1]
    _bs = len(outputs[0][f'z_parts_params/{input_key}'])
    _len = (len(outputs) - 1) * _bs + len(outputs[-1][f"z_parts_params/{input_key}"])

    all_kld = np.zeros((_len, latent_dim))
    all_mu = np.zeros((_len, latent_dim))

    logger.info("Looping over outputs")
    for ix, output in tqdm(enumerate(outputs), total=len(outputs)):
        kld_per_element = output[f"kld/{input_key}"].numpy()
        mu_var_per_elem = output[f"z_parts_params/{input_key}"].numpy()

        if mu_var_per_elem.shape[1] != latent_dim:
            mu_per_elem = mu_var_per_elem[:, :int(mu_var_per_elem.shape[1]/2)]
        else:
            mu_per_elem = mu_var_per_elem
        recon_loss= output[f'reconstruction_loss/{input_key}'].numpy()
        kld_loss = output['kld_loss'].numpy()

        start = ix * _bs
        end = start + mu_per_elem.shape[0]
        all_kld[start:end] = kld_per_element
        all_mu[start:end] = mu_per_elem

    dataframe = {
        "dimension": [],
        "test_kld_per_latent_dim": [],
        "mu_std_per_latent_dim": [],
        "mu_mean_per_latent_dim": [],
        "explained_variance": [],
    }

    mean_kld_per_dim = np.mean(all_kld, axis=0)
    dim_std = np.std(all_mu, axis=0)
    dim_mean = np.mean(all_mu, axis=0)

    total_mean_kld = np.sum(mean_kld_per_dim)

    for k in range(len(mean_kld_per_dim)):
        dataframe["dimension"].append(k)
        dataframe["test_kld_per_latent_dim"].append(mean_kld_per_dim[k])
        dataframe["mu_std_per_latent_dim"].append(dim_std[k])
        dataframe["mu_mean_per_latent_dim"].append(dim_mean[k])
        dataframe["explained_variance"].append(
            (mean_kld_per_dim[k] / total_mean_kld) * 100
        )

    stats_per_dim = pd.DataFrame(dataframe)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_path = os.path.join(tmp_dir, "stats_per_dim_test.csv")
        stats_per_dim.to_csv(dest_path)

        mlflow.log_artifact(
            local_path=dest_path,
            artifact_path="dataframes"
        )

    # Get ranked Z dim list
    stats_per_dim_ranked = (
        stats_per_dim.loc[stats_per_dim["test_kld_per_latent_dim"] > 0]
        .sort_values(by=["test_kld_per_latent_dim"])
        .reset_index(drop=True)
    )

    ranked_z_dim_list = [i for i in stats_per_dim_ranked["dimension"][::-1]]

    # Save mu per element, dimension and conditon
    # Not averaged across batch
    dataframe2 = {
        "dimension": [],
        "mu": [],
        "element": [],
    }

    for element in range(all_mu.shape[0]):
        for dimension in range(all_mu.shape[1]):
            dataframe2["dimension"].append(dimension)
            dataframe2["element"].append(element)
            dataframe2["mu"].append(
                all_mu[element, dimension].item()
            )

    mu_per_elem_and_dim = pd.DataFrame(dataframe2)

    # Make correlation plot between top 20 latent dims
    mu_per_elem_and_dim = mu_per_elem_and_dim[["dimension", "mu", "element"]]
    table = pd.pivot_table(
        mu_per_elem_and_dim, values="mu", index=["element"], columns=["dimension"]
    )
    mu_corrs = table[ranked_z_dim_list[:20]].corr()

    plt.figure(figsize=(8, 8))
    sns.set_context("talk")
    sns.heatmap(
        mu_corrs.abs(),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        square=True,
        cbar_kws={"shrink": 0.82},
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_path = os.path.join(tmp_dir, "latent_dim_corrs.png")
        plt.savefig(dest_path, dpi=300, bbox_inches="tight")

        mlflow.log_artifact(
            local_path=dest_path,
            artifact_path="images"
        )
        
def get_ranked_dims(
    stats,
    cutoff_kld_per_dim,
    max_num_shapemodes,
):

    stats = (
        stats.loc[stats["test_kld_per_latent_dim"] > cutoff_kld_per_dim]
        .sort_values(by=["test_kld_per_latent_dim"])
        .reset_index(drop=True)
    )

    ranked_z_dim_list = stats["dimension"][::-1].tolist()
    mu_std_list = stats["mu_std_per_latent_dim"][::-1].tolist()
    mu_mean_list = stats["mu_mean_per_latent_dim"][::-1].tolist()

    if len(ranked_z_dim_list) > max_num_shapemodes:
        ranked_z_dim_list = ranked_z_dim_list[:max_num_shapemodes]
        mu_std_list = mu_std_list[:max_num_shapemodes]

    return ranked_z_dim_list, mu_std_list, mu_mean_list
        
class GetEmbeddings(Callback):
    """"""

    def __init__(
        self,
        x_label: str,
        id_label: Optional[str] = None,
    ):
        """
        Args:
            x_label: x_label from datamodule
            id_field: id_field from datamodule
        """
        super().__init__()

        self.x_label = x_label
        self.id_label = id_label
        self.cutoff_kld_per_dim = 0

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():

            embeddings = get_all_embeddings(
                trainer.datamodule.train_dataloader(),
                trainer.datamodule.val_dataloader(),
                trainer.datamodule.test_dataloader(),
                pl_module,
                self.x_label,
                self.id_label,
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                dest_path = os.path.join(tmp_dir, "embeddings.csv")
                embeddings.to_csv(dest_path)

                mlflow.log_artifact(
                    local_path=dest_path,
                    artifact_path="dataframes"
                )

def get_all_embeddings(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    pl_module: LightningModule,
    x_label: str,
    id_label: None,
):

    all_embeddings = []
    cell_ids = []
    split = []

    zip_iter = zip(
        ["train", "val", "test"], [train_dataloader, val_dataloader, test_dataloader]
    )

    with torch.no_grad():
        for split_name, dataloader in zip_iter:
            logger.info(f"Getting embeddings for split: {split_name}")

            _bs = dataloader.batch_size
            _len = len(dataloader) * dataloader.batch_size

            _embeddings = np.zeros((_len, pl_module.latent_dim))
            _split = np.empty(_len, dtype=object)
            _ids = None

            id_label = (pl_module.hparams.get("id_label", None) if id_label is None else id_label)

            for index, batch in enumerate(dataloader):
                if _ids is None:
                    if id_label is not None and id_label in batch:
                        _ids = np.empty(_len, dtype=batch[id_label].cpu().numpy().dtype)
                    else:
                        _ids = None

                for key in batch.keys():
                    if not isinstance(batch[key], list):
                        batch[key] = batch[key].to(pl_module.device)
            
                z_parts_params, z_composed = pl_module(
                    batch, decode=False, compute_loss=False)

                mu_vars = z_parts_params[x_label]
                if mu_vars.shape[1] != pl_module.latent_dim:
                    mus = mu_vars[:,:int(mu_vars.shape[1]/2)]
                else:
                    mus = mu_vars

                start = _bs * index
                end = start + len(mus)
            
                _embeddings[start:end] = mus.cpu().numpy()

                if _ids is not None:
                    _ids[start:end] = batch[id_label].detach().cpu().squeeze()
                _split[start:end] = [split_name] * len(mus)

            # diff = _bs - len(batch)
            # if diff > 0:
            #     # if last batch is smaller discard the difference
            #     _embeddings = _embeddings[:-diff]
            #     if _ids is not None:
            #         _ids = _ids[:-diff]
            #     _split = _split[:-diff]
            all_embeddings.append(_embeddings)
            cell_ids.append(_ids)
            split.append(_split)

    all_embeddings = np.vstack(all_embeddings)
    cell_ids = (np.hstack(cell_ids) if cell_ids[0] is not None else None)
    split = np.hstack(split)

    df = pd.DataFrame(
        all_embeddings,
        columns=[f"mu_{i}" for i in range(all_embeddings.shape[1])]
    )
    df["split"] = split
    if cell_ids is not None:
        df["CellId"] = cell_ids

    return df