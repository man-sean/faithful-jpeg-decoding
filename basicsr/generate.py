import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, _ = parse_options(root_path, is_train=False)

    # if opt['test_type'] not in ['val', 'fid', 'collage']:
    #     raise ValueError(f'unknown test type: {opt["test_type"]}, expected one of: [val, fid, collage]')

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"gen_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')

        for idx, val_data in enumerate(test_loader):
            # print(f"[{idx: <4}]: {val_data['lq'].shape}")
            img_h, img_w = val_data['lq'].shape[-2], val_data['lq'].shape[-1]
            split = 1 + ((img_h * img_w) // (2000 * 1000))
            if split > 1: print(f"[!] Using {split=} due to image resolution ({img_h}, {img_w})")

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            model.feed_data(val_data)
            model.test(split=split)

            visuals = model.get_current_visuals()

            sr_img = tensor2img([visuals['result']])
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            path = osp.join(opt['path']['results_root'], 'fakes', test_set_name, img_name)

            imwrite(sr_img, path + f'_fake.png')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
