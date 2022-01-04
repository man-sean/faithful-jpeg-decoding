import torch
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel
from basicsr.losses import r1_penalty


@MODEL_REGISTRY.register()
class ESRGANModel(SRGANModel):
    """ESRGAN model for single image super-resolution."""

    def feed_data(self, data):
        # add support for jpeg qf
        super(ESRGANModel, self).feed_data(data=data)
        if 'qf' in data:
            self.qf = data['qf'].to(self.device)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt, qf=self.qf)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
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
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

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
        real_d_pred = self.net_d(self.gt)
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
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True) + l_d_r1
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
