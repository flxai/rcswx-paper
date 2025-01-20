sdpa = f"""
    branching(2)[
        clone(2),
        sequential[
            branching(2)[
                clone(2),
                routing[identity, computation[linear64], permute21],
                computation[linear64],
                dot_product(scaled=True)
            ],
            computation[softmax]
        ],
        computation[linear64],
        dot_product(scaled=False)
    ]"""
mhsa_h4 = f"""
    sequential[
        branching(4)[
            clone(4),
            {sdpa},
            cat(4,2)
        ],
        computation[linear512]
    ]"""
mhsa_h8 = f"""
    sequential[
        branching(8)[
            clone(8),
            {sdpa},
            cat(8,2)
        ],
        computation[linear512]
    ]"""
ffn = """
    sequential[
        sequential[
            computation[linear2048],
            computation[relu]
        ],
        computation[linear512]
    ]"""
transformer_layer = f"""
    sequential[
        branching(2)[
            clone(2),
            {mhsa_h4},
            computation[identity],
            add(2)
        ],
        norm
    ]"""
prenorm_transformer_layer = f"""
    sequential[
        branching(2)[
            clone(2),
            sequential[
                computation[norm],
                {mhsa_h4}
            ],
            computation[identity],
            add(2)
        ],
        branching(2)[
            clone(2),
            sequential[
                computation[norm],
                {ffn}
            ],
            computation[identity],
            add(2)
        ]
    ]"""
vit_d2 = f"""
    sequential[
        sequential[
            routing[im2col4k4s0p, computation[linear512], identity],
            computation[pos_enc]
        ],
        sequential[
            {transformer_layer},
            {transformer_layer}
        ]
    ]"""
vit_d4 = f"""
    sequential[
        sequential[
            routing[im2col4k4s0p, computation[linear512], identity],
            computation[pos_enc]
        ],
        sequential[
            sequential[
                {transformer_layer},
                {transformer_layer}
            ],
            sequential[
                {transformer_layer},
                {transformer_layer}
            ]
        ]
    ]"""
vit_d8 = f"""
    sequential[
        sequential[
            routing[im2col4k4s0p, computation[linear512], identity],
            computation[pos_enc]
        ],
        sequential[
            sequential[
                sequential[
                    {transformer_layer},
                    {transformer_layer}
                ],
                sequential[
                    {transformer_layer},
                    {transformer_layer}
                ]
            ],
            sequential[
                sequential[
                    {transformer_layer},
                    {transformer_layer}
                ],
                sequential[
                    {transformer_layer},
                    {transformer_layer}
                ]
            ]
        ]
    ]"""
