# convert the old format into the new format which contains the function names and brackets in a super simple form
# this is an example of old vs new format

# old format
# {
#     "fn": sequential_module,
#     "children": {
#         "first_fn": {
#             "fn": sequential_module,
#             "children": {
#                 "first_fn": {
#                     "fn": routing_module,
#                     "children": {
#                         "prerouting_fn": {
#                             "fn": im2col3k1s1p
#                         },
#                         "inner_fn": {
#                             "fn": computation_module,
#                             "children": {
#                                 "computation_fn": linear64
#                             }
#                         },
#                         "postrouting_fn": {
#                             "fn": col2im
#                         }
#                     }
#                 },
#                 "second_fn": {
#                     "fn": norm
#                 }
#             }
#         },
#         "second_fn": {
#             "fn": relu
#         }
#     }
# }

# new format
# sequential(
#     sequential(
#         routing[
#             im2col3k1s1p,
#             computation<linear64>,
#             col2im
#         ],
#         norm
#     ),
#     relu
# )

# now for the conversion
# einspace_resnet18_no_maxpool_architecture_dict = OrderedDict(
#     {
#         "fn": sequential_module,
#         "children": OrderedDict(
#             {
#                 "first_fn": OrderedDict(
#                     {
#                         "fn": sequential_module,
#                         "children": OrderedDict(
#                             {
#                                 "first_fn": einspace_resnet_stem_no_maxpool_architecture_dict,
#                                 "second_fn": OrderedDict(
#                                     {
#                                         "fn": sequential_module,
#                                         "children": OrderedDict(
#                                             {
#                                                 "first_fn": OrderedDict(
#                                                     {
#                                                         "fn": sequential_module,
#                                                         "children": OrderedDict(
#                                                             {
#                                                                 "first_fn": einspace_resnet_block_architecture_dict(
#                                                                     linear64,
#                                                                     linear64,
#                                                                 ),
#                                                                 "second_fn": einspace_resnet_block_architecture_dict(
#                                                                     linear64,
#                                                                     linear64,
#                                                                 ),
#                                                             }
#                                                         ),
#                                                     }
#                                                 ),
#                                                 "second_fn": OrderedDict(
#                                                     {
#                                                         "fn": sequential_module,
#                                                         "children": OrderedDict(
#                                                             {
#                                                                 "first_fn": einspace_resnet_strided_block_architecture_dict(
#                                                                     linear128,
#                                                                     linear128,
#                                                                 ),
#                                                                 "second_fn": einspace_resnet_block_architecture_dict(
#                                                                     linear128,
#                                                                     linear128,
#                                                                 ),
#                                                             }
#                                                         ),
#                                                     }
#                                                 ),
#                                             }
#                                         ),
#                                     }
#                                 ),
#                             }
#                         ),
#                     }
#                 ),
#                 "second_fn": OrderedDict(
#                     {
#                         "fn": sequential_module,
#                         "children": OrderedDict(
#                             {
#                                 "first_fn": OrderedDict(
#                                     {
#                                         "fn": sequential_module,
#                                         "children": OrderedDict(
#                                             {
#                                                 "first_fn": einspace_resnet_strided_block_architecture_dict(
#                                                     linear256,
#                                                     linear256,
#                                                 ),
#                                                 "second_fn": einspace_resnet_block_architecture_dict(
#                                                     linear256,
#                                                     linear256,
#                                                 ),
#                                             }
#                                         ),
#                                     }
#                                 ),
#                                 "second_fn": OrderedDict(
#                                     {
#                                         "fn": sequential_module,
#                                         "children": OrderedDict(
#                                             {
#                                                 "first_fn": einspace_resnet_strided_block_architecture_dict(
#                                                     linear512,
#                                                     linear512,
#                                                 ),
#                                                 "second_fn": einspace_resnet_block_architecture_dict(
#                                                     linear512,
#                                                     linear512,
#                                                 ),
#                                             }
#                                         ),
#                                     }
#                                 ),
#                             }
#                         ),
#                     }
#                 ),
#             }
#         ),
#     }
# )

# new format
# conv1x1 = lambda linear: f"routing[im2col1k1s0p, computation<{linear}>, col2im]"
# conv3x3 = lambda linear: f"routing[im2col3k1s1p, computation<{linear}>, col2im]"
# strided_conv3x3 = lambda linear: f"routing[im2col3k2s1p, computation<{linear}>, col2im]"
resnet_stem_no_maxpool = """
    sequential[
        sequential[
            sequential[
                routing[im2col3k1s1p, computation[linear64], col2im],
                computation[norm]
            ],
            computation[relu]
        ],
        routing[im2col3k2s1p, computation[linear64], col2im]
    ]"""
resnet_conv7x7_stem_no_maxpool = """
    sequential[
        sequential[
            sequential[
                routing[im2col7k2s3p, computation[linear64], col2im],
                computation[norm]
            ],
            computation[relu]
        ],
        routing[im2col3k2s1p, computation[linear64], col2im]
    ]"""
resnet_block = lambda a, b: f"""
    sequential[
        branching(2)[
            clone(2),
            sequential[
                sequential[
                    sequential[
                        routing[im2col3k1s1p, computation[{a}], col2im],
                        computation[norm]
                    ],
                    computation[relu]
                ],
                sequential[
                    routing[im2col3k1s1p, computation[{b}], col2im],
                    computation[norm]
                ]
            ],
            identity,
            add(2)
        ],
        computation[relu]
    ]"""
resnet_strided_block = lambda a, b: f"""
    sequential[
        branching(2)[
            clone(2),
            sequential[
                sequential[
                    sequential[
                        routing[im2col3k2s1p, computation[{a}], col2im],
                        computation[norm]
                    ],
                    computation[relu]
                ],
                sequential[
                    routing[im2col3k1s1p, computation[{b}], col2im],
                    computation[norm]
                ]
            ],
            sequential[
                routing[im2col1k2s0p, computation[{b}], col2im],
                computation[norm]
            ]
            add(2)
        ],
        computation[relu]
    ]"""
resnet18_no_maxpool = f"""
    sequential[
        sequential[
            {resnet_stem_no_maxpool},
            sequential[
                sequential[
                    {resnet_block('linear64', 'linear64')},
                    {resnet_block('linear64', 'linear64')}
                ],
                sequential[
                    {resnet_strided_block('linear128', 'linear128')},
                    {resnet_block('linear128', 'linear128')}
                ]
            ]
        ],
        sequential[
            sequential[
                {resnet_strided_block('linear256', 'linear256')},
                {resnet_block('linear256', 'linear256')}
            ],
            sequential[
                {resnet_strided_block('linear512', 'linear512')},
                {resnet_block('linear512', 'linear512')}
            ]
        ]
    ]"""
resnet18_conv7x7_no_maxpool = f"""
    sequential[
        sequential[
            {resnet_conv7x7_stem_no_maxpool},
            sequential[
                sequential[
                    {resnet_block('linear64', 'linear64')},
                    {resnet_block('linear64', 'linear64')}
                ],
                sequential[
                    {resnet_strided_block('linear128', 'linear128')},
                    {resnet_block('linear128', 'linear128')}
                ]
            ]
        ],
        sequential[
            sequential[
                {resnet_strided_block('linear256', 'linear256')},
                {resnet_block('linear256', 'linear256')}
            ],
            sequential[
                {resnet_strided_block('linear512', 'linear512')},
                {resnet_block('linear512', 'linear512')}
            ]
        ]
    ]"""

# print("Imported baselines")
# print("resnet18_no_maxpool", resnet18_no_maxpool)
# print("resnet18_conv7x7_no_maxpool", resnet18_conv7x7_no_maxpool)
