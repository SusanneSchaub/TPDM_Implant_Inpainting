import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import argparse
import datetime
import logging
import nibabel as nib
import torch

from configs.ve import inpainting_256 as configs
from utils import clear
import models.ncsnpp
from sampling import get_predictor, get_corrector
from physics import inpainting
import controllable_generation
import datasets
import tpdm_utils as tutils


def main():
    # parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--label_path', type=Path, default=Path(f""), help="Path to the label data.")
    argparser.add_argument('--primary-ckpt', type=Path, default=Path(f""),
                            help="(optional) Path to primary model checkpoint.")
    argparser.add_argument('--auxiliary-ckpt', type=Path, default=Path(f""),
                            help="(optional) Path to auxiliary model checkpoint.")
    argparser.add_argument('--problem-name', type=str, default="Inpainting",
                            help="(optional) Name of the problem which will be used to name the result directory.")
    argparser.add_argument('--batch-size', type=int, default=3,
                            help="(optional) Batch size for sampling.")
    argparser.add_argument('--dps-weight', type=float, default=None,
                            help="(optional) Weight of DPS step.")
    argparser.add_argument('--K', type=tutils.int_or_float, default=2,
                            help="(optional) Sampling contribution ratio of primary and auxiliary models. " + 
                            "Int inputs use a deterministic scheduler, while float inputs use a stochastic scheduler." + 
                            "Int K means primary model and auxiliary model will be updated K-1 times and 1 time, respectively." + 
                            "Float K means 1-(1/K) probability of updating the primary model and 1/K probability of updating the auxiliary model.")
    args = argparser.parse_args()

    if args.dps_weight is None:
        args.dps_weight = 1
        logging.warning("Using auto DPS weight for the given ZSR factor. For the best performance, please specify the DPS weight manually.")


    save_root = Path(f"./invp_results/{args.problem_name}/K{args.K}/lamb{args.dps_weight}/{''}/")
    config = configs.get_config()
    config.eval.batch_size = args.batch_size


    # setup sampling environment
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    scaler = datasets.get_data_scaler(config)

    sde, sigmas = tutils.get_tpdm_sde(config)

    score_model_pri, score_model_aux = tutils.get_tpdm_models(config, args.primary_ckpt, args.auxiliary_ckpt)

    predictor = get_predictor(config.sampling.predictor)
    corrector = get_corrector(config.sampling.corrector)

    #forward_op = inpainting.Inpainting_mask()

    tpdm_inpainting = controllable_generation.get_tpdm_inpainting(sde,
                                                                    predictor, corrector,
                                                                    inverse_scaler,
                                                                    config,
                                                                    save_progress=True,
                                                                    save_root=save_root,
                                                                    dps_weight=args.dps_weight,
                                                                    K=args.K,
                                                                    denoise=True)


    # load data
    label, fname_list = tutils.load_tpdm_label_data(args.label_path)
    label = label.to(config.device)
    label = scaler(label)


    def forward_fct_(x: torch.Tensor):

        def visualize(img):
            _min = img.min()
            _max = img.max()
            normalized_img = (img - _min) / (_max - _min)
            return normalized_img

        mask = nib.load("")
        mask = np.asarray(mask.dataobj)
        zzz = torch.from_numpy(mask).to(config.device)
        zzz=zzz[:,None,...]
        result=x*zzz
        N, C, Y, Z = x.shape

        assert C == 1
        return result

    def get_msk():
        mask = nib.load("./masks") #path to the mask
        mask = np.asarray(mask.dataobj)
        zzz = torch.from_numpy(mask).to(config.device)
        #zzz = zzz[:, None, ...]

        return zzz



    # prepair measurements
    measure_dagger = forward_fct_(label)
    forward_mask=get_msk()

    # prepair result directory
    save_root.mkdir(parents=True, exist_ok=True)
    irl_types = ['measure', 'recon', 'label', 'measure_dagger']
    for t in irl_types:
        if t == 'recon':
            save_root_f = save_root / t / 'progress'
            save_root_f.mkdir(exist_ok=True, parents=True)
        else:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)


    # save input data
    for i, (fname, _) in enumerate(fname_list):
        just_name = fname.split('.')[0]
        plt.imsave(save_root / 'measure' / f'{just_name}.png', clear(inverse_scaler(measure_dagger[i])), cmap='gray')
        plt.imsave(save_root / 'measure_dagger' / f'{just_name}.png', clear(inverse_scaler(measure_dagger[i])), cmap='gray')
        plt.imsave(save_root / 'label' / f'{just_name}.png', clear(inverse_scaler(label[i])), cmap='gray')


    # start reconstruction
    recon = tpdm_inpainting(score_model_pri, score_model_aux, measure_dagger, forward_mask)
    label = inverse_scaler(label)
    print(recon.shape)

    # save result
    for recon_img, label_one, (fname, _) in zip(recon, label, fname_list):
        just_name = fname.split('.')[0]
        plt.imsave(save_root / 'recon' / f'{just_name}.png', clear(recon_img), cmap='gray')
        np.save(str(save_root / 'label' / f'{just_name}.npy'), label_one.detach().cpu().numpy())
        np.save(str(save_root / 'recon' / f'{just_name}.npy'), recon_img.detach().cpu().numpy())


    # evaluate result
    tutils.print_and_save_eval_result(recon, label, save_root)


if __name__ == '__main__':
    main()
