import torch
from collections import OrderedDict
from os import path as osp

from tqdm import tqdm, trange
from basicsr.utils.matlab_functions import imresize
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.diffjpeg import DiffJPEG
from .base_model import BaseModel
import numpy as np
import tempfile
from torch.nn.functional import interpolate

from ..utils.collage_util import log_collage, get_collage


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_lq_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_lq_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, save_lq_img):
        dataset_name = dataloader.dataset.opt['name']
        qf = dataloader.dataset.opt['qf']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        if self.opt['is_train']:
            log_collage(dataloader=dataloader, model=self, global_step=current_iter)

        with tempfile.TemporaryDirectory() as tmpdirname:
            jpeger = DiffJPEG(differentiable=True)
            for idx, val_data in enumerate(dataloader):
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
                self.feed_data(val_data)
                self.test()

                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['result']])
                sr_img_tensor = visuals['result']
                metric_data['img'] = sr_img
                metric_data['gt_lq'] = tensor2img([visuals['lq']])
                metric_data['scale'] = self.opt['scale']
                imwrite(sr_img, osp.join(tmpdirname, f'sr_{idx}.png'))

                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']])
                    metric_data['img2'] = gt_img
                    del self.gt

                # tentative for out of GPU memory
                del self.lq
                del self.output
                torch.cuda.empty_cache()

                if save_img is not False:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}')
                    imwrite(sr_img, save_img_path + '.png')
                    if save_lq_img:
                        imwrite(metric_data['gt_lq'], save_img_path + '_gt_lq.png')
                        # sr_lq = interpolate(sr_img_tensor, scale_factor=1.0 / self.opt['scale'], mode='nearest')
                        fake_lq = jpeger(sr_img_tensor, quality=qf)
                        fake_lq = tensor2img(fake_lq.squeeze(0))
                        # sr_lq = imresize(sr_img, 1.0 / self.opt['scale'], antialiasing=True)
                        imwrite(fake_lq, save_img_path + '_fake_lq.png')
                        lq_diff = fake_lq - metric_data['gt_lq']
                        min = np.min(lq_diff)
                        max = np.max(lq_diff)
                        lq_diff -= min
                        lq_diff = lq_diff / (max - min)
                        imwrite(lq_diff * 255, save_img_path + '_lq_diff.png')

                if with_metrics:
                    # calculate metrics
                    for name, opt_ in self.opt['val']['metrics'].items():
                        if 'fid' not in name:
                            self.metric_results[name] += calculate_metric(metric_data, opt_)
                if use_pbar:
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
            if use_pbar:
                pbar.close()

            if with_metrics:
                if 'fid' in self.opt['val']['metrics']:
                    fid_metric_data = {}
                    fid_metric_data['fake_path'] = tmpdirname
                    fid_metric_data['device'] = self.device
                    self.metric_results['fid'] = calculate_metric(fid_metric_data, self.opt['val']['metrics']['fid']) * (idx + 1)
                for metric in self.metric_results.keys():
                    self.metric_results[metric] /= (idx + 1)
                    # update the best metric result
                    self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def test_fid(self, dataloader, repeat):
        dataset_name = dataloader.dataset.opt['name']
        fids = []

        for _ in trange(repeat):
            with tempfile.TemporaryDirectory() as tmpdirname:
                for idx, val_data in enumerate(dataloader):
                    self.feed_data(val_data)
                    self.test()

                    visuals = self.get_current_visuals()
                    sr_img = tensor2img([visuals['result']])
                    imwrite(sr_img, osp.join(tmpdirname, f'sr_{idx}.png'))

                    if 'gt' in visuals:
                        del self.gt

                    # tentative for out of GPU memory
                    del self.lq
                    del self.output
                    torch.cuda.empty_cache()

                fid_metric_data = {}
                fid_metric_data['fake_path'] = tmpdirname
                fid_metric_data['device'] = self.device
                fid_result = calculate_metric(fid_metric_data, self.opt['fid']['opt'])
                fids.append(fid_result)

        fids = np.array(fids)
        fid_mean = fids.mean()
        fid_std = fids.std()

        logger = get_root_logger()
        logger.info(f"FID for {dataset_name}, repeated {repeat} times: {fid_mean} ±{fid_std}")

    def test_collage(self, dataloader):
        def _rgb_to_bgr(img):
            img = torch.from_numpy(img[0].transpose(2, 0, 1))
            img = tensor2img(img)
            return img

        dataset_name = dataloader.dataset.opt['name']
        idx_to_save = self.opt['collage'].get('idx_to_save', [])
        realizations = self.opt['collage'].get('realizations', 5)

        pbar = tqdm(total=len(idx_to_save), unit='image')

        for idx, val_data in enumerate(dataloader):
            if idx not in idx_to_save:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            path = osp.join(self.opt['path']['results_root'], 'collage', dataset_name, img_name)

            real, compressed, fake, recompressed, diff, std, caption = get_collage(batch=val_data,
                                                                                   model=self,
                                                                                   expansion=32,
                                                                                   keep=realizations,
                                                                                   )
            print(f'{caption=}')

            imwrite(_rgb_to_bgr(real), path + '_real.png')
            imwrite(_rgb_to_bgr(compressed), path + '_compressed.png')
            imwrite(_rgb_to_bgr(std), path + '_std.png')
            for r in range(realizations):
                imwrite(_rgb_to_bgr(fake[r]), path + f'_fake_{r}.png')
                imwrite(_rgb_to_bgr(recompressed[r]), path + f'_fake_compressed_{r}.png')
                imwrite(_rgb_to_bgr(diff[r]), path + f'_diff_{r}.png')

            pbar.update()

        pbar.close()


