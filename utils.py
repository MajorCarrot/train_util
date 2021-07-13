# Copyright (C) 2021 Adithya Venkateswaran
#
# train_util is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# train_util is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with train_util. If not, see <http://www.gnu.org/licenses/>.

from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# NOTE: Keep these as tuples
L1_LAYERS = (torch.nn.Linear, torch.nn.Conv2d)
L2_LAYERS = (torch.nn.Linear, torch.nn.Conv2d)


@dataclass
class train_configurator:
    """Class to configure the training parameters"""

    model: torch.nn.Module
    num_epochs: int
    dl_train: DataLoader
    dl_valid: DataLoader
    dl_test: DataLoader
    optimizer: torch.optim
    lossfunction: Callable[[Tensor, Tensor]]
    lr_scheduler: torch.optim.lr_scheduler
    version: str = "FC"
    lambda_l1: float = 5e-4
    lambda_l2: float = 0
    device: str = "cuda"
    save: bool = True
    save_path: str = None
    early_stopping: int = 0


@dataclass
class train_stats:
    """Class to store train stats"""

    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    test_accuracies: List[float] = []
    train_losses: List[float] = []
    val_losses: List[float] = []


def l1_loss(model: torch.nn.Module, lambda_l1: float) -> float:
    """Function to get the L1 loss. Change `L1_LAYERS` for the target layers"""
    if lambda_l1 == 0:
        return 0

    loss = 0.0
    for child in model.children():
        if isinstance(child, L1_LAYERS):
            linear_params = torch.cat([x.view(-1) for x in child.parameters()])
            loss += torch.norm(linear_params, 1)
    return lambda_l1 * loss


def l2_loss(model: torch.nn.Module, lambda_l2: float) -> float:
    """Function to get the L2 loss. Change `L2_LAYERS` for the target layers"""
    if lambda_l2 == 0:
        return 0

    loss = 0.0
    for child in model.children():
        if isinstance(child, L2_LAYERS):
            linear_params = torch.cat([x.view(-1) for x in child.parameters()])
            # Squaring as pytorch returns the squareroot value by default
            loss += torch.norm(linear_params, 2) ** 2
    return lambda_l2 * loss


class train:
    def __init__(self, configurator: train_configurator) -> None:
        self.configurator = configurator
        self.train_stats = train_stats()
        self.best_accuracy = 0
        self.epoch_num = 0

    def _get_accuracy(self, output: Tensor, target: Tensor) -> float:
        num_correct = int(
            sum(torch.argmax(target, -1) == torch.argmax(output, -1))
        )
        return num_correct / target.shape[0]

    def _get_loss(self, output: Tensor, target: Tensor) -> float:
        loss = self.configurator.lossfunction(output, target)
        loss += l1_loss(self.configurator.model, self.configurator.lambda_l1)
        loss += l2_loss(self.configurator.model, self.configurator.lambda_l2)
        return loss

    def _get_accuracy_loss(
        self, input: Tensor, target: Tensor
    ) -> Tuple[float]:
        output = self.configurator.model(input)
        accuracy = self._get_accuracy(output, target)
        loss = self._get_loss(output, target)
        return accuracy, loss

    def _print_training_stats(self, stats: Tuple[float]) -> None:
        (
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            test_accuracy,
        ) = stats
        print(f"\nLosses:\nTraining={train_loss:.5}\nValidation={val_loss:.5}")
        print(f"\nAccuracy:\nTraining={train_accuracy:.5}")
        print(f"Validation={val_accuracy:.5}\nTest={test_accuracy:.5}")

    def _update_stats(self, stats: Tuple[float]) -> None:
        (
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            test_accuracy,
        ) = stats

        self.train_stats.train_accuracies.append(train_accuracy)
        self.train_stats.train_losses.append(train_loss)
        self.train_stats.val_accuracies.append(val_accuracy)
        self.train_stats.val_losses.append(val_loss)
        self.train_stats.test_accuracies.append(test_accuracy)

    def _train_epoch(self) -> Tuple[float]:
        train_loss = 0
        train_accuracy = 0
        self.configurator.model.train()
        with tqdm(total=len(self.configurator.dl_train)) as pbar:
            for (input, target) in tqdm(self.configurator.dl_train):
                self.configurator.optimizer.zero_grad()
                input = input.to(self.configurator.device)
                target = target.to(self.configurator.device)

                mb_acc, mb_loss = self._get_accuracy_loss(input, target)

                train_loss += mb_loss.item()
                train_accuracy += mb_acc

                mb_loss.backward()
                self.configurator.optimizer.step()

                pbar.update()

        train_accuracy /= len(self.configurator.dl_train)
        train_loss /= len(self.configurator.dl_train)

        return train_loss, train_accuracy

    def _val_epoch(self) -> Tuple[float]:
        val_loss = 0
        val_accuracy = 0
        self.configurator.model.eval()

        with torch.no_grad():
            for input, target in self.configurator.dl_valid:
                input = input.to(self.configurator.device)
                target = target.to(self.configurator.device)

                mb_acc, mb_loss = self._get_accuracy_loss(input, target)
                val_accuracy += mb_acc
                val_loss += mb_loss

        val_accuracy /= len(self.configurator.dl_valid)
        val_loss /= len(self.configurator.dl_valid)

        return val_loss, val_accuracy

    def _test_epoch(self) -> Tuple[float]:
        test_accuracy = 0
        self.configurator.model.eval()

        with torch.no_grad():

            for input, target in self.configurator.dl_test:
                input = input.to(self.configurator.device)
                target = target.to(self.configurator.device)
                output = self.configurator.model(input)
                mb_acc = self._get_accuracy(output, target)

                test_accuracy += mb_acc

        test_accuracy /= len(self.configurator.dl_test)

        return test_accuracy

    def _save(self) -> None:
        if self.configurator.lr_scheduler is not None:
            lr_scheduler_state_dict = (
                self.configurator.lr_scheduler.state_dict()
            )
        else:
            lr_scheduler_state_dict = None
        torch.save(
            {
                "epoch": self.epoch_num,
                "model_state_dict": self.configurator.model.state_dict(),
                "lr_scheduler": lr_scheduler_state_dict,
                "train_stats": self.train_stats,
            },
            self.configurator.save_path.format(self.configurator.version),
        )

    def _run_epoch(self) -> bool:
        train_loss, train_accuracy = self._train_epoch()
        val_loss, val_accuracy = self._val_epoch()
        test_accuracy = self._test_epoch()

        stats = (
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            test_accuracy,
        )

        self._update_stats(stats)
        self._print_training_stats(stats)

        if self.configurator.lr_scheduler is not None:
            self.configurator.lr_scheduler.step()

        if val_accuracy >= self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_epoch = self.epoch_num
            if self.configurator.save:
                assert self.configurator.save_path is not None
                self._save()
            print(
                f"Best validation accuracy: {self.best_accuracy} \
             at epoch: {self.best_epoch + 1}"
            )
        if self.configurator.early_stopping > 0:
            if (
                self.epoch_num - self.best_epoch
            ) > self.configurator.early_stopping:
                print("Early Stopping!\n")
                return False

        return True

    def _plot(self) -> None:
        plt.title("Loss")
        plt.plot(
            [loss for loss in self.train_stats.train_losses],
            label="Training Loss",
        )
        plt.plot(
            [loss for loss in self.train_stats.val_losses],
            label="Validation Loss",
        )
        plt.legend()
        plt.show()

        plt.title("Accuracy")
        plt.plot(
            [acc for acc in self.train_stats.train_accuracies],
            label="Training Accuracy",
        )
        plt.plot(
            [acc for acc in self.train_stats.val_accuracies],
            label="Validation Accuracy",
        )
        plt.plot(
            [acc for acc in self.train_stats.test_accuracies],
            label="Test Accuracy",
        )
        plt.legend()
        plt.show()

        print(
            f"Test accuracy at best epoch: \
            {self.stats.test_accuracies[self.best_epoch]}"
        )

    def run(self) -> None:
        try:
            for i in range(self.configurator.num_epochs):
                self.epoch_num = i
                print(f"Epoch number: {self.epoch_num + 1}")
                continue_training = self._run()
                if not continue_training:
                    break

        except KeyboardInterrupt:
            print("Training Interrupted")
            self._plot()
            raise
