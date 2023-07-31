def define_generator(opt):
    net_G_opt = opt["model"]["generator"]

    from .egvsr_nets import FRNet

    return FRNet(
        in_nc=net_G_opt["in_nc"],
        out_nc=net_G_opt["out_nc"],
        # nf=net_G_opt["nf"],
        nb=net_G_opt["nb"],
        scale=opt["scale"],
    )


def define_discriminator(opt):
    net_D_opt = opt["model"]["discriminator"]
    spatial_size = 128  # opt["dataset"]["train"]["crop_size"]  TODO

    from .tecogan_nets import SpatioTemporalDiscriminator

    return SpatioTemporalDiscriminator(
        in_nc=net_D_opt["in_nc"],
        spatial_size=spatial_size,
        tempo_range=net_D_opt["tempo_range"],
        scale=opt["scale"],
    )
