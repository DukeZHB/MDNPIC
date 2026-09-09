"""
Trainer of the MDNPIC diffusion classification network.

Pipeline:
  1. Pre-train the MGCG network with cross-entropy to obtain the
     three-dimensional conditional guidance priors (30 epochs by default).
  2. Train the conditional denoising model eps_theta with the noise
     estimation loss + Cross-Modal Semantic Alignment Regularization (CMASR).
  3. Evaluate by ancestral sampling of the reverse diffusion process.
"""
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.manifold import TSNE
from tqdm import tqdm

from alignment import CMASR
from diffusion import (EMA, p_sample_loop, q_sample, y_0_reparam)
from mgcg import MGCG
from metrics import (adjust_learning_rate, cast_label_to_one_hot_and_prototype,
                     cohen_kappa, compute_confusion_matrix, compute_f1_score,
                     accuracy, get_dataset, get_optimizer)
from diffusion.utils import make_beta_schedule
from networks import ConditionalModel

plt.style.use('ggplot')


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.test_num_timesteps = config.diffusion.test_timesteps

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule,
                                   num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start,
                                   end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
            self.one_minus_alphas_bar_sqrt *= 0.9999
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # multi-granularity conditional guidance model (pre-trained MGCG)
        if config.diffusion.apply_aux_cls:
            self.cond_pred_model = MGCG(config).to(self.device)
            self.aux_cost_function = nn.CrossEntropyLoss()
            # cross-modal semantic alignment regularization (Eq. 20-21)
            cmasr_cfg = config.diffusion.cmasr
            self.cmasr = CMASR(tau=cmasr_cfg.tau,
                               lambda_global=cmasr_cfg.lambda_global,
                               lambda_local=cmasr_cfg.lambda_local,
                               lambda_texture=cmasr_cfg.lambda_texture).to(self.device)

        # scaling temperature for NLL and ECE computation
        self.tuned_scale_T = None

    # Compute the guidance priors as diffusion conditions
    def compute_guiding_prediction(self, x):
        """
        Compute the fused prediction y_0_hat and the three granularity
        guidance priors (y_global, y_local, y_texture).
        """
        y_pred, y_global, y_local, y_texture = self.cond_pred_model(x)
        return y_pred, y_global, y_local, y_texture

    def evaluate_guidance_model(self, dataset_loader):
        """Evaluate the guidance model by reporting the top-1 accuracy."""
        y_acc_list = []
        for step, feature_label_set in tqdm(enumerate(dataset_loader)):
            x_batch, y_labels_batch = feature_label_set
            y_labels_batch = y_labels_batch.reshape(-1, 1)
            y_pred_prob, _, _, _ = self.compute_guiding_prediction(
                x_batch.to(self.device))
            y_pred_prob = y_pred_prob.softmax(dim=1)
            y_pred_label = torch.argmax(
                y_pred_prob, 1, keepdim=True).cpu().detach().numpy()
            y_labels_batch = y_labels_batch.cpu().detach().numpy()
            y_acc = y_pred_label == y_labels_batch
            if len(y_acc_list) == 0:
                y_acc_list = y_acc
            else:
                y_acc_list = np.concatenate([y_acc_list, y_acc], axis=0)
        y_acc_all = np.mean(y_acc_list)
        return y_acc_all

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the MGCG guidance model. Cross-entropy is
        applied to the fused prediction and to each granularity branch, so
        that all three guidance priors are discriminative.
        """
        y_pred, y_global, y_local, y_texture = self.compute_guiding_prediction(x_batch)
        aux_cost = self.aux_cost_function(y_pred, y_batch) + \
                   self.aux_cost_function(y_global, y_batch) + \
                   self.aux_cost_function(y_local, y_batch) + \
                   self.aux_cost_function(y_texture, y_batch)
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()

    def train(self):
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        data_object, train_dataset, test_dataset = get_dataset(args, config)
        logging.info("loading dataset..")

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        logging.info("successfully load")
        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        model = model.to(self.device)

        optimizer = get_optimizer(self.config.optim, model.parameters())
        criterion = nn.CrossEntropyLoss()

        # apply an auxiliary optimizer for the MGCG guidance model
        if config.diffusion.apply_aux_cls:
            aux_optimizer = get_optimizer(self.config.aux_optim,
                                          self.cond_pred_model.parameters())

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):
                # load saved guidance model
                aux_states = torch.load(
                    os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                 config.diffusion.trained_aux_cls_ckpt_name),
                    map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
                self.cond_pred_model.eval()
            elif hasattr(config.diffusion, "trained_aux_cls_log_path"):
                aux_states = torch.load(
                    os.path.join(config.diffusion.trained_aux_cls_log_path, "aux_ckpt.pth"),
                    map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
                self.cond_pred_model.eval()
            else:
                # pre-train the MGCG guidance network
                assert config.diffusion.aux_cls.pre_train
                self.cond_pred_model.train()
                pretrain_start_time = time.time()
                for epoch in range(config.diffusion.aux_cls.n_pretrain_epochs):
                    for feature_label_set in train_loader:
                        x_batch, y_labels_batch = feature_label_set
                        y_one_hot_batch, y_logits_batch = \
                            cast_label_to_one_hot_and_prototype(y_labels_batch, config)
                        aux_loss = self.nonlinear_guidance_model_train_step(
                            x_batch.to(self.device),
                            y_one_hot_batch.to(self.device),
                            aux_optimizer)
                    if epoch % config.diffusion.aux_cls.logging_interval == 0:
                        logging.info(
                            f"epoch: {epoch}, guidance (MGCG) pre-training loss: {aux_loss}"
                        )
                pretrain_end_time = time.time()
                logging.info(
                    "\nPre-training of the MGCG guidance network took {:.4f} minutes.\n".format(
                        (pretrain_end_time - pretrain_start_time) / 60))
                # save the guidance model after pre-training
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
            # report accuracy on both training and test set for the pre-trained guidance model
            y_acc_aux_model = self.evaluate_guidance_model(train_loader)
            logging.info(
                "\nAfter pre-training, guidance classifier accuracy on the training set is {:.8f}.".format(
                    y_acc_aux_model))
            y_acc_aux_model = self.evaluate_guidance_model(test_loader)
            logging.info(
                "\nAfter pre-training, guidance classifier accuracy on the test set is {:.8f}.\n".format(
                    y_acc_aux_model))

        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                    map_location=self.device)
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                start_epoch = states[2]
                step = states[3]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])
                # load the guidance model
                if config.diffusion.apply_aux_cls and (
                        hasattr(config.diffusion, "trained_aux_cls_ckpt_path") is False) and (
                        hasattr(config.diffusion, "trained_aux_cls_log_path") is False):
                    aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0])
                    aux_optimizer.load_state_dict(aux_states[1])

            max_accuracy = 0.0
            if config.diffusion.noise_prior:
                logging.info("Prior distribution at timestep T has a mean of 0.")
            if args.add_ce_loss:
                logging.info("Apply cross entropy as an auxiliary loss during training.")
            for epoch in range(start_epoch, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0
                for i, feature_label_set in enumerate(train_loader):
                    x_batch, y_labels_batch = feature_label_set
                    y_one_hot_batch, y_logits_batch = \
                        cast_label_to_one_hot_and_prototype(y_labels_batch, config)
                    if config.optim.lr_schedule:
                        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, config)
                    n = x_batch.size(0)
                    # record the unflattened x as input to the guidance network
                    x_unflat_batch = x_batch.to(self.device)
                    data_time += time.time() - data_start
                    model.train()
                    self.cond_pred_model.eval()
                    step += 1

                    # antithetic sampling of the timestep
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    # noise estimation loss with conditional guidance
                    x_batch = x_batch.to(self.device)
                    # the MGCG guidance network is frozen after pre-training:
                    # the three-dimensional priors act as fixed conditions
                    with torch.no_grad():
                        y_0_hat_batch, y_0_global, y_0_local, y_0_texture = \
                            self.compute_guiding_prediction(x_unflat_batch)
                    y_0_hat_batch = y_0_hat_batch.softmax(dim=1)
                    y_0_global, y_0_local, y_0_texture = (
                        y_0_global.softmax(dim=1),
                        y_0_local.softmax(dim=1),
                        y_0_texture.softmax(dim=1))

                    y_T_mean = y_0_hat_batch
                    if config.diffusion.noise_prior:
                        y_T_mean = torch.zeros(y_0_hat_batch.shape).to(y_0_hat_batch.device)
                    y_0_batch = y_one_hot_batch.to(self.device)
                    e = torch.randn_like(y_0_batch).to(y_0_batch.device)

                    # main stream: noisy label conditioned on the fused prior
                    y_t_batch = q_sample(y_0_batch, y_T_mean,
                                         self.alphas_bar_sqrt,
                                         self.one_minus_alphas_bar_sqrt, t, noise=e)
                    # granularity streams (Eq. 9-11): noisy labels with
                    # granularity-specific guidance prior means
                    y_t_batch_global = q_sample(y_0_batch, y_0_global,
                                                self.alphas_bar_sqrt,
                                                self.one_minus_alphas_bar_sqrt, t, noise=e)
                    y_t_batch_local = q_sample(y_0_batch, y_0_local,
                                               self.alphas_bar_sqrt,
                                               self.one_minus_alphas_bar_sqrt, t, noise=e)
                    y_t_batch_texture = q_sample(y_0_batch, y_0_texture,
                                                 self.alphas_bar_sqrt,
                                                 self.one_minus_alphas_bar_sqrt, t, noise=e)

                    priors = (y_0_hat_batch, y_0_local, y_0_texture)
                    output = model(x_batch, y_t_batch, t, *priors)
                    output_global = model(x_batch, y_t_batch_global, t, *priors)
                    output_local = model(x_batch, y_t_batch_local, t, *priors)
                    output_texture = model(x_batch, y_t_batch_texture, t, *priors)

                    # noise estimation loss on the main stream
                    loss = (e - output).square().mean()
                    # cross-modal semantic alignment regularization (CMASR)
                    # aligns the three granularity streams to the shared noise target
                    loss_cmasr, cmasr_details = self.cmasr(
                        output_global, output_local, output_texture, e)
                    loss = loss + loss_cmasr

                    # cross-entropy for y_0 reparameterization
                    loss0 = torch.tensor([0])
                    if args.add_ce_loss:
                        y_0_reparam_batch = y_0_reparam(model, x_batch, y_t_batch, priors,
                                                        y_T_mean, t,
                                                        self.one_minus_alphas_bar_sqrt)
                        raw_prob_batch = -(y_0_reparam_batch - 1) ** 2
                        loss0 = criterion(raw_prob_batch, y_labels_batch.to(self.device))
                        loss += config.training.lambda_ce * loss0

                    if not tb_logger is None:
                        tb_logger.add_scalar("loss", loss, global_step=step)
                        tb_logger.add_scalar("cmasr", loss_cmasr, global_step=step)

                    if step % self.config.training.logging_freq == 0 or step == 1:
                        logging.info(
                            (
                                f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, "
                                f"Noise Estimation loss: {loss.item()}, "
                                f"CMASR: {loss_cmasr.item()}, " +
                                f"data time: {data_time / (i + 1)}"
                            )
                        )

                    # optimize the diffusion model that predicts eps_theta
                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip)
                    except Exception:
                        pass
                    optimizer.step()
                    if self.config.model.ema:
                        ema_helper.update(model)

                    # joint training of the guidance network along with the diffusion model
                    if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                        self.cond_pred_model.train()
                        aux_loss = self.nonlinear_guidance_model_train_step(
                            x_unflat_batch, y_0_batch, aux_optimizer)
                        if step % self.config.training.logging_freq == 0 or step == 1:
                            logging.info(
                                f"meanwhile, guidance (MGCG) joint-training loss: {aux_loss}")

                    # save the diffusion model
                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                states,
                                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                            )
                        # save current states
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                        # save the guidance model
                        if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                            aux_states = [
                                self.cond_pred_model.state_dict(),
                                aux_optimizer.state_dict(),
                            ]
                            if step > 1:
                                torch.save(
                                    aux_states,
                                    os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                                )
                            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

                    data_start = time.time()

                logging.info(
                    (f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, "
                     f"Noise Estimation loss: {loss.item()}, "
                     f"CMASR: {loss_cmasr.item()}, " +
                     f"data time: {data_time / (i + 1)}")
                )

                # Evaluate
                if epoch % self.config.training.validation_freq == 0 \
                        or epoch + 1 == self.config.training.n_epochs:
                    model.eval()
                    self.cond_pred_model.eval()
                    acc_avg = 0.
                    y1_true = None
                    y1_pred = None
                    for test_batch_idx, (images, target) in enumerate(test_loader):
                        images_unflat = images.to(self.device)
                        images = images.to(self.device)
                        target = target.to(self.device)
                        with torch.no_grad():
                            target_pred, y_global, y_local, y_texture = \
                                self.compute_guiding_prediction(images_unflat)
                            target_pred = target_pred.softmax(dim=1)
                            y_global, y_local, y_texture = (
                                y_global.softmax(dim=1),
                                y_local.softmax(dim=1),
                                y_texture.softmax(dim=1))
                            # prior mean at timestep T
                            y_T_mean = target_pred
                            if config.diffusion.noise_prior:
                                y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)

                            priors = (target_pred, y_local, y_texture)
                            label_t_0 = p_sample_loop(model, images, priors, y_T_mean,
                                                      self.num_timesteps, self.alphas,
                                                      self.one_minus_alphas_bar_sqrt,
                                                      only_last_sample=True)
                            y1_pred = torch.cat([y1_pred, label_t_0]) if y1_pred is not None else label_t_0
                            y1_true = torch.cat([y1_true, target]) if y1_true is not None else target
                            acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
                    kappa_avg = cohen_kappa(y1_pred.detach().cpu(), y1_true.cpu()).item()
                    f1_avg = compute_f1_score(y1_true, y1_pred).item()

                    acc_avg /= (test_batch_idx + 1)
                    if acc_avg > max_accuracy:
                        logging.info("Update best accuracy at Epoch {}.".format(epoch))
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                        aux_states = [
                            self.cond_pred_model.state_dict(),
                            aux_optimizer.state_dict(),
                        ]
                        torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt_best.pth"))
                    max_accuracy = max(max_accuracy, acc_avg)
                    if not tb_logger is None:
                        tb_logger.add_scalar('accuracy', acc_avg, global_step=step)
                    logging.info(
                        (
                            f"epoch: {epoch}, step: {step}, " +
                            f"Average accuracy: {acc_avg}, Average Kappa: {kappa_avg}, "
                            f"Average F1: {f1_avg}, Max accuracy: {max_accuracy:.2f}%"
                        )
                    )

            # save the model after training is finished
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            # save the guidance model after training is finished
            if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                # report training set accuracy if applied joint training
                y_acc_aux_model = self.evaluate_guidance_model(train_loader)
                logging.info(
                    "After joint-training, guidance classifier accuracy on the training set is {:.8f}.".format(
                        y_acc_aux_model))
                # report test set accuracy if applied joint training
                y_acc_aux_model = self.evaluate_guidance_model(test_loader)
                logging.info(
                    "After joint-training, guidance classifier accuracy on the test set is {:.8f}.".format(
                        y_acc_aux_model))

    def test(self):
        args = self.args
        config = self.config
        data_object, train_dataset, test_dataset = get_dataset(args, config)
        log_path = os.path.join(self.args.log_path)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        if getattr(self.config.testing, "ckpt_id", None) is None:
            if args.eval_best:
                ckpt_id = 'best'
                states = torch.load(os.path.join(log_path, f"ckpt_{ckpt_id}.pth"),
                                    map_location=self.device)
            else:
                ckpt_id = 'last'
                states = torch.load(os.path.join(log_path, "ckpt.pth"),
                                    map_location=self.device)
        else:
            states = torch.load(os.path.join(log_path, f"ckpt_{self.config.testing.ckpt_id}.pth"),
                                map_location=self.device)
            ckpt_id = self.config.testing.ckpt_id
        logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)

        # load the guidance model
        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):
                aux_states = torch.load(
                    os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                 config.diffusion.trained_aux_cls_ckpt_name),
                    map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
            else:
                aux_cls_path = log_path
                if hasattr(config.diffusion, "trained_aux_cls_log_path"):
                    aux_cls_path = config.diffusion.trained_aux_cls_log_path
                aux_states = torch.load(os.path.join(aux_cls_path, "aux_ckpt_best.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=False)
                logging.info(f"Loading from: {aux_cls_path}/aux_ckpt_best.pth")

            # Evaluate
            model.eval()
            self.cond_pred_model.eval()
            acc_avg = 0
            kappa_avg = 0.
            y1_true = None
            y1_pred = None
            # feature vectors collected at each reverse diffusion timestep
            feature_vectors_all = {t: [] for t in range(self.test_num_timesteps + 1)}

            for test_batch_idx, (images, target) in enumerate(test_loader):
                images_unflat = images.to(self.device)
                images = images.to(self.device)
                target = target.to(self.device)
                with torch.no_grad():
                    target_pred, y_global, y_local, y_texture = \
                        self.compute_guiding_prediction(images_unflat)
                    target_pred = target_pred.softmax(dim=1)
                    y_local, y_texture = (y_local.softmax(dim=1), y_texture.softmax(dim=1))
                    y_T_mean = target_pred
                    if config.diffusion.noise_prior:
                        y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)

                    priors = (target_pred, y_local, y_texture)
                    label_t_0 = p_sample_loop(model, images, priors, y_T_mean,
                                              self.test_num_timesteps, self.alphas,
                                              self.one_minus_alphas_bar_sqrt,
                                              only_last_sample=False)
                    # collect the feature vectors at each timestep
                    for t, label in enumerate(label_t_0):
                        feature_vectors_all[t].append(label.cpu().numpy())
                    label_t_0 = label_t_0[-1].softmax(dim=-1)
                    acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
                    kappa_avg += cohen_kappa(label_t_0.detach().cpu(), target.cpu()).item()
                    y1_pred = torch.cat([y1_pred, label_t_0]) if y1_pred is not None else label_t_0
                    y1_true = torch.cat([y1_true, target]) if y1_true is not None else target

            f1_avg = compute_f1_score(y1_true, y1_pred)
            conf_matrix = compute_confusion_matrix(y1_true, y1_pred)
            acc_avg /= (test_batch_idx + 1)
            kappa_avg /= (test_batch_idx + 1)
            logging.info(
                (
                    f"[Test:] Average accuracy: {acc_avg}, Average Kappa: {kappa_avg}, "
                    f"F1: {f1_avg}, Confusion_Matrix: {conf_matrix}"
                )
            )

            # generate the dynamic t-SNE visualization of the reverse process
            self.generate_tsne_plots(feature_vectors_all, y1_true)

    def generate_tsne_plots(self, feature_vectors_all, y1_true):
        """Visualize the denoising (classification) procedure with t-SNE."""
        num_timesteps = len(feature_vectors_all) - 1
        tsne_dir = os.path.join(self.args.log_path, "t-sne")
        os.makedirs(tsne_dir, exist_ok=True)
        for t, feature_vectors in feature_vectors_all.items():
            feature_vectors = np.vstack(feature_vectors)
            # remove rows containing NaN
            if np.isnan(feature_vectors).any():
                mask = ~np.isnan(feature_vectors).any(axis=1)
                feature_vectors = feature_vectors[mask]
                y1_true = y1_true[mask]
            # reversed timestep index
            t_r = num_timesteps - t
            tsne = TSNE(n_components=2, random_state=0, perplexity=50)
            feature_vectors_tsne = tsne.fit_transform(feature_vectors)
            plt.figure(figsize=(10, 7))
            scatter = plt.scatter(feature_vectors_tsne[:, 0], feature_vectors_tsne[:, 1],
                                  c=y1_true.cpu().numpy(), cmap='viridis')
            plt.axis('off')
            plt.grid(False)
            plt.savefig(os.path.join(tsne_dir, f'tsne_t_{t_r}.png'),
                        bbox_inches='tight', pad_inches=0)
            plt.close()
            logging.info(
                f"t-SNE plot at timestep {t_r} saved to {tsne_dir}/tsne_t_{t_r}.png")
