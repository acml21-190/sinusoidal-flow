import pytorch_lightning as pl
import torch
import torch.optim as optim
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau

from project.data import ToyRegressionData
from project.models.affine import AffineFlowRegressor


class AffineFlowRegressionForToyData(pl.LightningModule):
    def __init__(self, hparams):
        super(AffineFlowRegressionForToyData, self).__init__()

        self.save_hyperparameters(hparams)
        self.affine_flow_regressor = AffineFlowRegressor(self.hparams.conditioner, self.hparams.num_layers,
                                                         use_norm=self.hparams.use_norm, in_dim=self.hparams.in_dim,
                                                         hid_dims=self.hparams.hid_dims,
                                                         use_ind_std=self.hparams.use_ind_std)

    def forward(self, x):
        return self.affine_flow_regressor.predict(x)

    def shared_step(self, batch):
        x, y = batch
        _, log_joint_dens = self.affine_flow_regressor(y, cond_var=x)
        loss = -torch.mean(log_joint_dens)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        x, y = batch
        yhat = self(x)
        rmse = torch.mean((yhat - y) ** 2).sqrt()
        self.log("test_loss", loss)
        self.log("test_rmse", rmse)

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimiser,
            "lr_scheduler": ReduceLROnPlateau(optimiser, factor=0.5, patience=5, verbose=True),
            "monitor": "val_loss"
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--dataset", required=True, type=str, help="The toy dataset to use")
        parser.add_argument("--batch_size", default=128, type=int)

        parser.add_argument("--conditioner", default="mlp", type=str, help="The conditioner to use")
        parser.add_argument("--num_layers", default=1, type=int, help="Number of affine layers")
        parser.add_argument("--use_norm", action="store_true", help="Whether to use normalisation between layers")
        parser.add_argument("--hid_dims", default=tuple(), type=tuple, help="Number of hidden layers in conditioner")
        parser.add_argument("--use_ind_std", action="store_true", help="Whether to use independent std in conditioner")

        return parser


def main(args):
    dm = ToyRegressionData(args.dataset, args.batch_size)
    dm.setup()
    args.in_dim = dm.size(0)

    model = AffineFlowRegressionForToyData(args)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer.from_argparse_args(args, num_sanity_val_steps=0, callbacks=[checkpoint_callback],
                                            weights_summary='full')
    trainer.fit(model, datamodule=dm)

    trainer.test(datamodule=dm)

    # model = AffineFlowRegressionForToyData.load_from_checkpoint(checkpoint_callback.best_model_path).eval()

    # fig, ax = plt.subplots(1)
    # X_test, y_test = dm.datasets["test"].tensors
    # y_test = y_test.reshape(-1, 1)
    # y_hat = model(X_test).reshape(-1, 1)
    # print(torch.mean((y_test - y_hat) ** 2).sqrt())
    # ax.scatter(X_test, y_test)
    # ax.scatter(X_test, y_hat)
    # plt.show()


if __name__ == "__main__":
    pl.seed_everything(42)

    parser = ArgumentParser(add_help=False)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule specific params in the module
    parser = AffineFlowRegressionForToyData.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)
