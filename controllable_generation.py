import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import functools

from sampling import shared_corrector_update_fn, shared_predictor_update_fn
from utils import clear, fft2
from tpdm_utils import check_K, is_primary_tern


def get_tpdm_inpainting(sde, predictor, corrector, inverse_scaler, config, dps_weight, K, save_root, save_progress, denoise=True, eps=1e-5, z_mask_idxs=None):
    check_K(K)

    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=config.sampling.probability_flow,
                                            continuous=config.training.continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=config.training.continuous,
                                            snr=config.sampling.snr,
                                            n_steps=config.sampling.n_steps_each)


    def get_tpdm_inpainting_update_fn(update_fn):
        def tpdm_inpainting_update_fn(model, measure_dagger, mask, x, t):

            vec_t = torch.ones(x.shape[0], device=x.device) * t

            # input to the score function
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            # x0 hat prediction
            _, bt = sde.marginal_prob(x, vec_t)
            hatx0 = x + (bt[..., None, None, None] ** 2) * score

            # DPS step for the data consistency
            norm = torch.norm(hatx0*mask - measure_dagger)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

            x_next = x_next - norm_grad * dps_weight
            x_next_mean = x_next_mean - norm_grad * dps_weight

            x_next = x_next.detach()
            x_next_mean = x_next_mean.detach()

            return x_next, x_next_mean

        return tpdm_inpainting_update_fn

    def get_tpdm_uncond_update_fn(update_fn):
        def tpdm_uncond_update_fn(model, x, t):
            vec_t = torch.ones(x.shape[0], device=x.device) * t

            with torch.no_grad():
                x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            return x_next, x_next_mean

        return tpdm_uncond_update_fn

    predictor_tpdm_inpainting_update_fn = get_tpdm_inpainting_update_fn(predictor_update_fn)
    corrector_tpdm_inpainting_update_fn = get_tpdm_inpainting_update_fn(corrector_update_fn)
    predictor_tpdm_uncond_update_fn = get_tpdm_uncond_update_fn(predictor_update_fn)
    corrector_tpdm_uncond_update_fn = get_tpdm_uncond_update_fn(corrector_update_fn)

    def tpdm_inpainting(model_pri, model_aux, measure_dagger, mask):
        batch_len = len(measure_dagger) // config.eval.batch_size if len(
            measure_dagger) % config.eval.batch_size == 0 else len(measure_dagger) // config.eval.batch_size + 1
        B, C, H, W = measure_dagger.shape
        shape = (B, C, H, 256)

        # Initial sample
        x = sde.prior_sampling(shape).to(measure_dagger.device)
        timesteps = torch.linspace(sde.T, eps, sde.N)
        dataloader_md = torch.tensor_split(measure_dagger, batch_len)

        for i in tqdm(range(sde.N), colour="blue", unit="step", smoothing=0):
            primary_tern = is_primary_tern(i, K)

            if not primary_tern:
                # [X, 1, Y, Z] -> [Z, 1, X, Y]
                x = x.permute(3, 1, 0, 2)

            dataloader = torch.tensor_split(x, batch_len)
            dataloader_mask = torch.tensor_split(mask, batch_len)

            x_batch_s = []
            x_mean_batch_s = []

            for x_batch, gray_scale_img_batch, mask_batch in tqdm(zip(dataloader, dataloader_md, dataloader_mask), total=batch_len, colour="blue",
                                                      unit="mb", leave=False):
                if primary_tern:  # reverse diffusion with primary model + DPS
                    t = timesteps[i]
                    x_batch, x_mean_batch = corrector_tpdm_inpainting_update_fn(model_pri, gray_scale_img_batch, mask_batch, x_batch, t)
                    x_batch, x_mean_batch = predictor_tpdm_inpainting_update_fn(model_pri, gray_scale_img_batch, mask_batch, x_batch, t)

                else:  # reverse diffusion with auxiliary model
                    t = timesteps[i]
                    x_batch, x_mean_batch = corrector_tpdm_uncond_update_fn(model_aux, x_batch, t)
                    x_batch, x_mean_batch = predictor_tpdm_uncond_update_fn(model_aux, x_batch, t)

                x_batch_s.append(x_mean_batch)
                x_mean_batch_s.append(x_mean_batch)

            x = torch.cat(x_batch_s, dim=0)
            x_mean = torch.cat(x_mean_batch_s, dim=0)

            if not primary_tern:
                # [Z, 1, X, Y] -> [X, 1, Y, Z]
                x = x.permute(2, 1, 3, 0)   #check this
                x_mean = x_mean.permute(2, 1, 3, 0)

            if save_progress:
                if (i % 50) == 0:
                    plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png',
                               clear(x_mean[len(x_mean) // 2:len(x_mean) // 2 + 1]), cmap='gray')

        return inverse_scaler(x_mean if denoise else x)

    return tpdm_inpainting