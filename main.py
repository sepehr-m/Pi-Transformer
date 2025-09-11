import torch
import torch.nn as nn
import os
import numpy as np
import time
import yaml
import logging
import argparse

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from torch.backends import cudnn
from model.PiTransformer import PiTransformer
from data_loader import get_loader
from utils.utils import kl_loss, adjust_learning_rate, compute_hurst_dfa, EarlyStopping, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AnomalyDetection:
    def __init__(self, config_path, dataset):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.dataset = dataset
        self.data_dir = self.config["general"]["data_dir"]
        self.device = self.config["general"]["device"]
        self.model_save_path = self.config["general"]["model_dir"]
        self.data_path = os.path.join(self.data_dir, self.dataset)

        self.data_config = self.config["datasets"][dataset]['data']
        self.model_config = self.config['datasets'][dataset]['model']
        self.training_config = self.config['datasets'][dataset]['training']

        self.win_size = self.data_config["win_size"]
        self.batch_size = self.data_config["batch_size"]
        self.anomaly_ratio = self.data_config["anomaly_ratio"]

        self.n_heads = self.model_config['n_heads']
        self.d_model = self.model_config['d_model']
        self.enc_in = self.model_config["enc_in"]
        self.c_out = self.model_config["c_out"]
        self.e_layers = self.model_config["e_layers"]
        self.gamma = self.model_config["gamma"]
        self.sigma = self.model_config["sigma"]
        self.lambda_smooth = self.model_config["lambda_smooth"]

        self.lr = self.training_config["lr"]
        self.n_epochs = self.training_config["n_epochs"]
        self.k = self.training_config["k"]

        self.temperature = self.config['datasets'][dataset]["testing"]["temperature"]

        self.train_loader = get_loader(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="train",
            dataset=self.dataset,
        )

        self.vali_loader = get_loader(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="val",
            dataset=self.dataset,
        )

        self.test_loader = get_loader(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="test",
            dataset=self.dataset,
        )

        self.thre_loader = get_loader(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="thre",
            dataset=self.dataset,
        )

        self.criterion = nn.MSELoss()
        self.global_hurst = None
        self.device = torch.device(self.device)
        self.build_model()

    def build_model(self):
        self.model = PiTransformer(
            win_size=self.win_size,
            enc_in=self.enc_in,
            c_out=self.c_out,
            e_layers=self.e_layers,
            gamma=self.gamma,
            sigma=self.sigma,
            lambda_smooth=self.lambda_smooth,
            d_model=self.d_model,
            n_heads =self.n_heads
        )

        for m in self.model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def compute_distillation_loss(self, hurst):
        if self.global_hurst is None or torch.isnan(self.global_hurst).any():
            logger.warning("Skipping distillation loss due to invalid global hurst")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        mean_hurst = torch.mean(hurst, dim=[1, 2])  # B
        mean_global_hurst = torch.mean(self.global_hurst)  # scalar
        loss = torch.mean((mean_hurst - mean_global_hurst) ** 2)

        return loss

    def validation(self, valid_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(valid_loader):
            input = input_data.float().to(self.device)

            (
                output,
                series,
                prior,
                _,
                hurst,
                tau,
                smoothness_loss,
                beta_prior_loss,
                tau_smoothness_loss,
            ) = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                norm_prior = torch.unsqueeze(
                    torch.sum(prior[u], dim=-1), dim=-1
                ).repeat(1, 1, 1, self.win_size)

                series_kl = torch.mean(
                    kl_loss(series[u], (prior[u] / norm_prior).detach())
                )
                prior_kl = torch.mean(
                    kl_loss((prior[u] / norm_prior).detach(), series[u])
                )
                series_loss += series_kl + prior_kl
                prior_loss += torch.mean(
                    kl_loss((prior[u] / norm_prior), series[u].detach())
                ) + torch.mean(kl_loss(series[u].detach(), (prior[u] / norm_prior)))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())
        vali_loss1 = np.average(loss_1) if loss_1 else float("nan")
        vali_loss2 = np.average(loss_2) if loss_2 else float("nan")

        return vali_loss1, vali_loss2

    def train(self):
        logger.info("--------------- Train mode ---------------")
        time_now = time.time()
        path = self.model_save_path
        os.makedirs(path, exist_ok=True)
        early_stopping = EarlyStopping(
            patience=3, verbose=True, dataset_name=self.dataset
        )
        train_steps = len(self.train_loader)

        hurst_path = os.path.join(
            self.model_save_path, f"{self.dataset}_global_hurst.pt"
        )
        if os.path.exists(hurst_path):
            self.global_hurst = torch.load(hurst_path, weights_only=True).to(
                self.device
            )
            logger.info(f"Loaded global hurst from {hurst_path}.")

        if self.global_hurst is None:
            all_data = []
            for input_data, _ in self.train_loader:
                all_data.append(input_data)
            all_data = torch.cat(all_data, dim=0).float().to(self.device)
            subsample_size = max(100, len(all_data) // 10)
            indices = torch.randperm(len(all_data))[:subsample_size]
            all_data = all_data[indices]
            self.global_hurst = (
                compute_hurst_dfa(all_data, use_cpu=True).mean(dim=0).to(self.device)
            )
            torch.save(self.global_hurst, hurst_path)
            logger.info(f"Saved global Hurst to {hurst_path}.")

        # Training loop
        for epoch in range(self.n_epochs):
            iter_count = 1
            loss1_list = []
            self.model.train()
            epoch_time = time.time()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimiser.zero_grad()
                input = input_data.float().to(self.device)
                (
                    output,
                    series,
                    prior,
                    _,
                    hurst,
                    tau,
                    smoothness_loss,
                    beta_prior_loss,
                    tau_smoothness_loss,
                ) = self.model(input)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    norm_prior = torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.win_size)
                    series_kl = torch.mean(
                        kl_loss(series[u], (prior[u] / norm_prior).detach())
                    )
                    prior_kl = torch.mean(
                        kl_loss((prior[u] / norm_prior).detach(), series[u])
                    )
                    series_loss += series_kl + prior_kl
                    prior_loss += torch.mean(
                        kl_loss((prior[u] / norm_prior), series[u].detach())
                    ) + torch.mean(kl_loss(series[u].detach(), (prior[u] / norm_prior)))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                rec_loss = self.criterion(output, input)
                distillation_loss = self.compute_distillation_loss(hurst)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = (
                    rec_loss
                    - self.k * series_loss
                    + smoothness_loss
                    + beta_prior_loss
                    + tau_smoothness_loss
                    + distillation_loss
                )
                loss2 = (
                    rec_loss
                    + self.k * prior_loss
                    + smoothness_loss
                    + beta_prior_loss
                    + tau_smoothness_loss
                    + distillation_loss
                )

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.n_epochs - epoch) * train_steps - i)
                    logger.info(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 1
                    time_now = time.time()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimiser.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list) if loss1_list else float("nan")
            vali_loss1, vali_loss2 = self.validation(self.test_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1
                )
            )
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimiser, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    str(self.model_save_path), str(self.dataset) + "_checkpoint.pth"
                ),
                weights_only=True,
            )
        )
        self.model.eval()
        temperature = self.temperature
        logger.info("--------------- Test mode ---------------")
        criterion = nn.MSELoss(reduction="none")

        attens_energy = []
        attens_phase = []  
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            (
                output,
                series,
                prior,
                _,
                _,
                _,
                smoothness_loss,
                beta_prior_loss,
                tau_smoothness_loss,
            ) = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                norm_prior = torch.unsqueeze(
                    torch.sum(prior[u], dim=-1), dim=-1
                ).repeat(1, 1, 1, self.win_size)
                if u == 0:
                    series_loss = (
                        kl_loss(series[u], (prior[u] / norm_prior).detach())
                        * self.temperature
                    )
                    prior_loss = (
                        kl_loss((prior[u] / norm_prior), series[u].detach())
                        * self.temperature
                    )
                else:
                    series_loss += (
                        kl_loss(series[u], (prior[u] / norm_prior).detach())
                        * self.temperature
                    )
                    prior_loss += (
                        kl_loss((prior[u] / norm_prior), series[u].detach())
                        * self.temperature
                    )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss

            phase_stream = (series_loss + prior_loss)
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            attens_phase.append(phase_stream.detach().cpu().numpy())
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        attens_phase = np.concatenate(attens_phase, axis=0).reshape(-1)
        train_phase = np.array(attens_phase)


        med_E = np.percentile(train_energy, 50.0)
        q25_E = np.percentile(train_energy, 25.0)
        q75_E = np.percentile(train_energy, 75.0)
        iqr_E = max(q75_E - q25_E, 1e-8)

        med_P = np.percentile(train_phase, 50.0)
        q25_P = np.percentile(train_phase, 25.0)
        q75_P = np.percentile(train_phase, 75.0)
        iqr_P = max(q75_P - q25_P, 1e-8)

        zE_train = np.maximum(0.0, (train_energy - med_E) / iqr_E)
        zP_train = np.maximum(0.0, (train_phase  - med_P) / iqr_P)
        fused_train = np.maximum(zE_train, zP_train)

        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            (
                output,
                series,
                prior,
                _,
                _,
                _,
                smoothness_loss,
                beta_prior_loss,
                tau_smoothness_loss,
            ) = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                norm_prior = torch.unsqueeze(
                    torch.sum(prior[u], dim=-1), dim=-1
                ).repeat(1, 1, 1, self.win_size)
                if u == 0:
                    series_loss = (
                        kl_loss(series[u], (prior[u] / norm_prior).detach())
                        * temperature
                    )
                    prior_loss = (
                        kl_loss((prior[u] / norm_prior), series[u].detach())
                        * temperature
                    )
                else:
                    series_loss += (
                        kl_loss(series[u], (prior[u] / norm_prior).detach())
                        * temperature
                    )
                    prior_loss += (
                        kl_loss((prior[u] / norm_prior), series[u].detach())
                        * temperature
                    )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss

            phase_stream = (series_loss + prior_loss)
            E = cri.detach().cpu().numpy()
            P = phase_stream.detach().cpu().numpy()
            zE = np.maximum(0.0, (E - med_E) / iqr_E)
            zP = np.maximum(0.0, (P - med_P) / iqr_P)
            fused = np.maximum(zE, zP)
            attens_energy.append(fused)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        combined_energy = np.concatenate([fused_train.reshape(-1), test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        print("Threshold :", thresh)

        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            (
                output,
                series,
                prior,
                _,
                _,
                _,
                smoothness_loss,
                beta_prior_loss,
                tau_smoothness_loss,
            ) = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                norm_prior = torch.unsqueeze(
                    torch.sum(prior[u], dim=-1), dim=-1
                ).repeat(1, 1, 1, self.win_size)
                if u == 0:
                    series_loss = (
                        kl_loss(series[u], (prior[u] / norm_prior).detach())
                        * self.temperature
                    )
                    prior_loss = (
                        kl_loss((prior[u] / norm_prior), series[u].detach())
                        * self.temperature
                    )
                else:
                    series_loss += (
                        kl_loss(series[u], (prior[u] / norm_prior).detach())
                        * self.temperature
                    )
                    prior_loss += (
                        kl_loss((prior[u] / norm_prior), series[u].detach())
                        * self.temperature
                    )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss

            phase_stream = (series_loss + prior_loss)
            E = cri.detach().cpu().numpy()
            P = phase_stream.detach().cpu().numpy()
            zE = np.maximum(0.0, (E - med_E) / iqr_E)
            zP = np.maximum(0.0, (P - med_P) / iqr_P)
            fused = np.maximum(zE, zP)
            attens_energy.append(fused)
            test_labels.append(labels)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        pred = np.array(pred)
        gt = np.array(gt)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score = precision_recall_fscore_support(
            gt, pred, average="binary"
        )[:3]
        logger.info(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )
        return accuracy, precision, recall, f_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="SMAP")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    cudnn.benchmark = True
    set_seed()
    detector = AnomalyDetection(config_path="config.yaml", dataset=args.dataset)
    if args.mode == "train":
        detector.train()
    elif args.mode == "test":
        detector.test()
