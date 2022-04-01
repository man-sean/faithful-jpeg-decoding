import math
import torch
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel
from basicsr.losses import r1_penalty
from basicsr.utils import get_root_logger
from basicsr.utils.batch_util import expand_batch, restore_expanded_batch


@MODEL_REGISTRY.register()
class ESRGANModel(SRGANModel):
    """ESRGAN model for single image super-resolution."""

    def feed_data(self, data):
        # add support to first-moment penalty
        # if self.cri_stability:
        #     # expand batch
        #     penalty_batch_size = self.cri_stability.batch_size
        #     expansion = self.cri_stability.expansion
        #     self.original_batch_size = data["lq"].shape[0]
        #     inputs = ["gt", "lq", "qf"]
        #     xdata = {k: expand_batch(v[:penalty_batch_size], expansion) for k, v in data.items() if k in inputs}
        #     data = {k: torch.cat((data[k], v), dim=0) for k, v in xdata.items()}
        super(ESRGANModel, self).feed_data(data=data)
        # add support for jpeg qf
        if 'qf' in data:
            self.qf = data['qf'].to(self.device)

    def expand_penalty_batch(self):
        penalty_batch_size = self.cri_stability.batch_size
        expansion = self.cri_stability.expansion
        penalty_lq = expand_batch(self.lq[:penalty_batch_size], expansion)
        penalty_qf = expand_batch(self.qf[:penalty_batch_size], expansion)
        self.lq = torch.cat((self.lq, penalty_lq), dim=0)
        self.qf = torch.cat((self.qf, penalty_qf), dim=0)

    def restore_penalty_batch(self, output, calc_std=False):
        expansion = self.cri_stability.expansion
        batch_size = self.gt.shape[0]
        self.lq = self.lq[:batch_size]
        self.qf = self.qf[:batch_size]
        penalty_mean = restore_expanded_batch(output[batch_size:], expansion).mean(dim=0)
        penalty_std = restore_expanded_batch(output[batch_size:], expansion).std(dim=0) if calc_std else None
        output = output[:batch_size]
        return output, penalty_mean, penalty_std

    def current_weight(self, loss_type, current_iter):
        """
        return modified loss weight if annealing options are set, else return base weight
        """
        annealing_opt = self.opt['train'][loss_type].get('annealing_opt', None)
        if annealing_opt:
            # follow CosineAnnealingLR formula:
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
            eta_min = annealing_opt['min']
            eta_max = annealing_opt['max']
            total_iter = self.opt['train']['total_iter']
            direction = annealing_opt['direction']
            if direction == 'up':
                return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * (1 + current_iter / total_iter)))
            elif direction == 'down':
                return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * (current_iter / total_iter)))
            else:
                raise ValueError(f'Unknown direction {direction}, should be one of [up, down]')
        else:
            return self.opt['train'][loss_type]['loss_weight']

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()

        if self.cri_stability and current_iter % self.cri_stability.iters == 0:
            self.expand_penalty_batch()
            output = self.net_g(self.lq, self.qf)
            if isinstance(output, tuple):  # in case we get two versions of the output image
                output_unconstrained, output = output
            else:
                output_unconstrained = output
            self.output, penalty_mean_output, penalty_std_output = self.restore_penalty_batch(output, calc_std=self.cri_variability)
            self.output_unconstrained, _, _ = self.restore_penalty_batch(output_unconstrained, calc_std=self.cri_variability)
        else:
            self.output = self.net_g(self.lq, self.qf)
            if isinstance(self.output, tuple):  # in case we get two versions of the output image
                self.output_unconstrained, self.output = self.output
            else:
                self.output_unconstrained = self.output

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                weight_pix = self.current_weight('pixel_opt', current_iter)
                l_g_pix = self.cri_pix(self.output_unconstrained, self.gt, qf=self.qf, loss_weight=weight_pix)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
                loss_dict['w_g_pix'] = torch.tensor(weight_pix, dtype=torch.float32, device=l_g_pix.device)
            # stability loss
            if self.cri_stability and current_iter % self.cri_stability.iters == 0:
                l_g_stab = self.cri_stability(penalty_mean_output, self.gt)
                l_g_total += l_g_stab
                loss_dict['l_g_stab'] = l_g_stab
            # variability loss
            if self.cri_variability and current_iter % self.cri_stability.iters == 0:
                l_g_var = self.cri_variability(penalty_std_output)
                l_g_total += l_g_var
                loss_dict['l_g_var'] = l_g_var
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            weight_gan = self.current_weight('gan_opt', current_iter)
            fake_g_pred = self.net_d(x=self.output, y=self.lq)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False, loss_weight=weight_gan)

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            loss_dict['w_g_gan'] = torch.tensor(weight_gan, dtype=torch.float32, device=l_g_gan.device)

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        # real
        if self.r1_penalty_coef > 0.0:
            self.gt.requires_grad = True
        real_d_pred = self.net_d(x=self.gt, y=self.lq)
        if self.r1_penalty_coef > 0.0:
            l_d_r1 = r1_penalty(real_d_pred, self.gt)
            l_d_r1 = ((self.r1_penalty_coef / 2) * l_d_r1 + 0 * real_d_pred[0])
            # TODO: why do we need to add 0 * real_pred, otherwise, a runtime
            # error will arise: RuntimeError: Expected to have finished
            # reduction in the prior iteration before starting a new one.
            # This error indicates that your module has parameters that were
            # not used in producing loss.
            loss_dict['l_d_r1'] = l_d_r1.detach().mean()
        else:
            l_d_r1 = 0
        # logger = get_root_logger()
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        # logger.warning(f'Loss shape #1: {l_d_real.shape=}, {l_d_r1.shape=}')
        l_d_real = l_d_real + l_d_r1
        # logger.warning(f'Loss shape #1: {l_d_real.shape=}, {l_d_real[0].shape=}')
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(x=self.output.detach(), y=self.lq)
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        l_d_fake.backward()
        self.optimizer_d.step()

        # if l_d_real.shape != l_d_fake.shape:
        #     logger = get_root_logger()
        #     logger.warning(f'Loss shape mismatch: {l_d_real.shape=}, {l_d_fake.shape=}, {l_d_r1.shape=}')

        loss_dict['l_d_real'] = l_d_real.reshape([])
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        # print(f'[>>>>]')
        # for k, v in loss_dict.items():
        #     print(f'[>] [{v.device}] {k} = {v}')
        # print(f'[<<<<]')

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
