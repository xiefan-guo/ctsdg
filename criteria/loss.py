import torch
import torch.nn as nn

from utils.misc import gram_matrix


def generator_loss_func(
    mask, output, ground_truth, edge, output_pred, 
    vgg_comp, vgg_output, vgg_ground_truth, 
    projected_image, projected_edge,
    output_edge):

    l1 = nn.L1Loss()
    criterion = nn.BCELoss()

    # ---------
    # hole loss
    # ---------
    loss_hole = l1((1 - mask) * output, (1 - mask) * ground_truth)

    # ----------
    # valid loss
    # ----------
    loss_valid = l1(mask * output, mask * ground_truth)

    # ---------------
    # perceptual loss
    # ---------------
    loss_perceptual = 0.0
    for i in range(3):
        loss_perceptual += l1(vgg_output[i], vgg_ground_truth[i])
        loss_perceptual += l1(vgg_comp[i], vgg_ground_truth[i])

    # ----------
    # style loss
    # ----------
    loss_style = 0.0
    for i in range(3):
        loss_style += l1(gram_matrix(vgg_output[i]), gram_matrix(vgg_ground_truth[i]))
        loss_style += l1(gram_matrix(vgg_comp[i]), gram_matrix(vgg_ground_truth[i]))

    # ----------------
    # adversarial loss
    # ----------------
    real_target = torch.tensor(1.0).expand_as(output_pred)
    if torch.cuda.is_available():
        real_target = real_target.cuda()
    loss_adversarial = criterion(output_pred, real_target) + criterion(output_edge, edge)
    
    # -----------------
    # intermediate loss
    # -----------------
    loss_intermediate = criterion(projected_edge, edge) + l1(projected_image, ground_truth)

    return {
        'loss_hole': loss_hole.mean(), 
        'loss_valid': loss_valid.mean(), 
        'loss_perceptual': loss_perceptual.mean(), 
        'loss_style': loss_style.mean(), 
        'loss_adversarial': loss_adversarial.mean(), 
        'loss_intermediate': loss_intermediate.mean()
    }


def discriminator_loss_func(real_pred, fake_pred, real_pred_edge, fake_pred_edge, edge):

    criterion = nn.BCELoss()
    
    real_target = torch.tensor(1.0).expand_as(real_pred)
    fake_target = torch.tensor(0.0).expand_as(fake_pred)
    if torch.cuda.is_available():
        real_target = real_target.cuda()
        fake_target = fake_target.cuda()

    loss_adversarial = criterion(real_pred, real_target) + criterion(fake_pred, fake_target) + \
                    criterion(real_pred_edge, edge) + criterion(fake_pred_edge, edge)

    return {
        'loss_adversarial': loss_adversarial.mean()
    }

