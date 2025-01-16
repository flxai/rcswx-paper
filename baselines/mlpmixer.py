channel_mixer = "sequential(sequential(routing[permute21, computation<linear_x4>, identity], computation<leakyrelu>), routing[identity, computation<linear_4th>, permute21])"
token_mixer = "sequential(computation<linear256>, computation<leakyrelu>, computation<linear512>)"
mlpmixer_layer = "branching(2)(clone(2), sequential(computation<norm>, sequential(sequential(routing[permute21, computation<linear_x4>, identity], computation<leakyrelu>), computation<linear256>), computation<identity>, add_tensors))"
mlpmixer = f"sequential(sequential(routing[im2col4k4s0p, computation<linear512>, identity], computation<pos_enc>), sequential(sequential({mlpmixer_layer}, {mlpmixer_layer}), sequential({mlpmixer_layer}, {mlpmixer_layer})))"
