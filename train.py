from model import Generator, Discriminator, TrainSet
from torch.utils.data import DataLoader
from config import *

import torch
import pickle


def UNIT_Train():
    # initialize models
    gen_a = Generator()
    gen_b = Generator()
    dis_a = Discriminator()
    dis_b = Discriminator()

    # set up data loader
    with open('./tmp_pkl_data/a_b_max_min_context.pkl', 'rb') as r:
        data_a, data_b, _, _, _ = pickle.load(r)
    data_a = data_a[:int(train_size * data_a.shape[0])]
    data_b = data_b[:int(train_size * data_b.shape[0])]
    train_dataset = TrainSet(data_a, data_b)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    print('data_a shape: {}'.format(data_a.shape))
    print('data_b shape: {}'.format(data_b.shape))

    # set up parameters
    gen_params = list(gen_a.parameters()) + list(gen_b.parameters())
    dis_params = list(dis_a.parameters()) + list(dis_b.parameters())
    gen_opt = torch.optim.Adam(gen_params, lr=gen_learning_rate)
    dis_opt = torch.optim.Adam(dis_params, lr=dis_learning_rate)

    # start training
    for epoch in range(num_epochs):
        gen_loss_sum = 0
        dis_loss_sum = 0
        for i, (x_a, x_b) in enumerate(train_loader):
            x_a = x_a.to(torch.float32)
            x_b = x_b.to(torch.float32)

            # train discriminator
            dis_opt.zero_grad()
            # encode
            h_a, n_a = gen_a.encode(x_a)
            h_b, n_b = gen_b.encode(x_b)
            # decode (cross domain)
            x_ba = gen_a.decode(h_b + n_b)
            x_ab = gen_b.decode(h_a + n_a)
            # discriminator loss
            loss_dis_a = dis_a.cal_dis_loss(x_ba, x_a)
            loss_dis_b = dis_b.cal_dis_loss(x_ab, x_b)
            loss_dis_all = loss_dis_a + loss_dis_b
            dis_loss_sum += loss_dis_all
            loss_dis_all.backward()
            dis_opt.step()

            # train generator
            gen_opt.zero_grad()
            # encode
            h_a, n_a = gen_a.encode(x_a)
            h_b, n_b = gen_b.encode(x_b)
            # decode (within domain)
            x_a_recon = gen_a.decode(h_a + n_a)
            x_b_recon = gen_b.decode(h_b + n_b)
            # decode (cross domain)
            x_ba = gen_a.decode(h_b + n_b)
            x_ab = gen_b.decode(h_a + n_a)
            # encode again
            h_b_recon, n_b_recon = gen_a.encode(x_ba)
            h_a_recon, n_a_recon = gen_b.encode(x_ab)
            # decode again
            x_bab = gen_b.decode(h_b_recon + n_b_recon)
            x_aba = gen_a.decode(h_a_recon + n_a_recon)
            # generator loss
            # reconstruction loss
            loss_gen_recon_x_a = torch.mean(torch.abs(x_a_recon - x_a))
            loss_gen_recon_x_b = torch.mean(torch.abs(x_b_recon - x_b))
            # GAN loss
            loss_gen_adv_a = dis_a.cal_gen_loss(x_ba)
            loss_gen_adv_b = dis_b.cal_gen_loss(x_ab)
            # cycle-consistency loss
            loss_gen_cycle_cons_a = torch.mean(torch.abs(x_aba - x_a))
            loss_gen_cycle_cons_b = torch.mean(torch.abs(x_bab - x_b))
            # total loss
            loss_gen_all = loss_gen_recon_x_a + loss_gen_recon_x_b + \
                           loss_gen_adv_a + loss_gen_adv_b + \
                           loss_gen_cycle_cons_a + loss_gen_cycle_cons_b
            gen_loss_sum += loss_gen_all
            loss_gen_all.backward()
            gen_opt.step()

        print('Epoch {}: Generator loss: {} \t Discriminator loss: {}'.format(epoch + 1, gen_loss_sum, dis_loss_sum))

    torch.save(gen_a.state_dict(), gen_a_save_path)
    torch.save(gen_b.state_dict(), gen_b_save_path)


if __name__ == '__main__':
    UNIT_Train()


