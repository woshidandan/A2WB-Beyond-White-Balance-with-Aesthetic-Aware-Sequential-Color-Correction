import copy

from action.DeepWB.DeepWB_arch import deep_wb_single_task


def splitNetworks(net):
    net_awb = deep_wb_single_task.deepWBnet()
    net_t = deep_wb_single_task.deepWBnet()
    net_s = deep_wb_single_task.deepWBnet()
    # AWB weights
    net_awb.encoder_inc = copy.deepcopy(net.encoder_inc)
    net_awb.encoder_down1 = copy.deepcopy(net.encoder_down1)
    net_awb.encoder_down2 = copy.deepcopy(net.encoder_down2)
    net_awb.encoder_down3 = copy.deepcopy(net.encoder_down3)
    net_awb.encoder_bridge_down = copy.deepcopy(net.encoder_bridge_down)
    net_awb.decoder_bridge_up = copy.deepcopy(net.awb_decoder_bridge_up)
    net_awb.decoder_up1 = copy.deepcopy(net.awb_decoder_up1)
    net_awb.decoder_up2 = copy.deepcopy(net.awb_decoder_up2)
    net_awb.decoder_up3 = copy.deepcopy(net.awb_decoder_up3)
    net_awb.decoder_out = copy.deepcopy(net.awb_decoder_out)
    # Tungsten WB weights
    net_t.encoder_inc = copy.deepcopy(net.encoder_inc)
    net_t.encoder_down1 = copy.deepcopy(net.encoder_down1)
    net_t.encoder_down2 = copy.deepcopy(net.encoder_down2)
    net_t.encoder_down3 = copy.deepcopy(net.encoder_down3)
    net_t.encoder_bridge_down = copy.deepcopy(net.encoder_bridge_down)
    net_t.decoder_bridge_up = copy.deepcopy(net.tungsten_decoder_bridge_up)
    net_t.decoder_up1 = copy.deepcopy(net.tungsten_decoder_up1)
    net_t.decoder_up2 = copy.deepcopy(net.tungsten_decoder_up2)
    net_t.decoder_up3 = copy.deepcopy(net.tungsten_decoder_up3)
    net_t.decoder_out = copy.deepcopy(net.tungsten_decoder_out)
    # Shade WB weights
    net_s.encoder_inc = copy.deepcopy(net.encoder_inc)
    net_s.encoder_down1 = copy.deepcopy(net.encoder_down1)
    net_s.encoder_down2 = copy.deepcopy(net.encoder_down2)
    net_s.encoder_down3 = copy.deepcopy(net.encoder_down3)
    net_s.encoder_bridge_down = copy.deepcopy(net.encoder_bridge_down)
    net_s.decoder_bridge_up = copy.deepcopy(net.shade_decoder_bridge_up)
    net_s.decoder_up1 = copy.deepcopy(net.shade_decoder_up1)
    net_s.decoder_up2 = copy.deepcopy(net.shade_decoder_up2)
    net_s.decoder_up3 = copy.deepcopy(net.shade_decoder_up3)
    net_s.decoder_out = copy.deepcopy(net.shade_decoder_out)

    return net_awb, net_t, net_s
