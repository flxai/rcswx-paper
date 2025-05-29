from os.path import join
from os import makedirs

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from networkx.drawing.nx_agraph import graphviz_layout

from search_state import DerivationTreeNode
from grammars.einspace import clone, add


colours = {
    # einspace colours
    "root":             '#8d3b2fff',    # hex for dark brown #3d2b1fff
    "network":          '#999999ff',    # grey
    "module":           '#999999ff',    # grey
    "branching_fn_2":   '#ffd966ff',    # light yellow 1
    "branching_fn_4":   '#ffd966ff',    # light yellow 1
    "branching_fn_8":   '#ffd966ff',    # light yellow 1
    "aggregation_fn_2": '#a64d79ff',    # dark magenta 1
    "aggregation_fn_4": '#a64d79ff',    # dark magenta 1
    "aggregation_fn_8": '#a64d79ff',    # dark magenta 1
    "prerouting_fn":    '#93c47dff',    # light green 1
    "postrouting_fn":   '#93c47dff',    # light green 1
    "computation_fn":   '#6fa8dcff',    # light blue 1
    "input":            '#333333ff',    # black
    "output":           '#cc4125ff',    # light red berry 1
    "mutation":         '#ff007fff',    # bright pink
    # architecture operations
    "linear16": '#6fa8dcff',  # light blue 1
    "linear32": '#6fa8dcff',  # light blue 1
    "linear64": '#6fa8dcff',  # light blue 1
    "linear128": '#6fa8dcff',  # light blue 1
    "linear256": '#6fa8dcff',  # light blue 1
    "linear512": '#6fa8dcff',  # light blue 1
    "linear1024": '#6fa8dcff',  # light blue 1
    "linear2048": '#6fa8dcff',  # light blue 1
    "identity": '#6fa8dcff',  # light blue 1
    "relu": '#6fa8dcff',  # light blue 1
    "softmax": '#6fa8dcff',  # light blue 1
    "pos_enc": '#6fa8dcff',  # light blue 1
    "clone(2)": '#ffd966ff',    # light yellow 1
    "clone(4)": '#ffd966ff',    # light yellow 1
    "clone(8)": '#ffd966ff',    # light yellow 1
    "add(2)": '#a64d79ff',    # dark magenta 1
    "add(4)": '#a64d79ff',    # dark magenta 1
    "add(8)": '#a64d79ff',    # dark magenta 1
    "permute(0,2,1)": '#93c47dff',    # light green 1
    "permute(1,0,2)": '#93c47dff',    # light green 1
    "permute(1,2,0)": '#93c47dff',    # light green 1

    # hnasbench201 colours
    "network":          '#999999ff',    # dark brown
    "D1":               '#ffd966ff',    # light yellow 1
    "D0":               '#ffd966ff',    # light yellow 1
    "D":                '#ffd966ff',    # light yellow 1
    "C":                '#93c47dff',    # light green 1
    "CL":               '#93c47dff',    # light green 1
    "OP":               '#6fa8dcff',    # light blue 1
    "ACT":              '#6fa8dcff',    # light blue 1
    "CONV":             '#6fa8dcff',    # light blue 1
    "NORM":             '#6fa8dcff',    # light blue 1
    "DOWN":             '#a64d79ff',    # dark magenta 1
    # architecture operations
    "down":             '#a64d79ff',    # dark magenta 1
    "zero":             '#999999ff',    # black
    "identity":         '#6fa8dcff',    # black,
    "avg_pool":         '#93c47dff',    # black,
    "relu":             '#6fa8dcff',    # light blue 1
    "hardswish":        '#6fa8dcff',    # light blue 1
    "mish":             '#6fa8dcff',    # light blue 1
    "conv1x1":          '#93c47dff',    # black,
    "conv3x3":          '#93c47dff',    # black,
    "dconv3x3":         '#93c47dff',    # black,
    "batchnorm":        '#6fa8dcff',    # light blue 1
    "instancenorm":     '#6fa8dcff',    # light blue 1
    "layernorm":        '#6fa8dcff',    # light blue 1
}




# a function which takes as input an architecture (as represented by nested dictionaries) and returns a networkx graph
# def architecture_to_nx(architecture, mutation_id=None):
#     net = nx.Graph()
#     net.add_node(-1, label="input", color=colours["input"], title=shape_to_string(architecture["input_shape"]), x=0, y=1000)
#     parent = recurse_add_node(architecture, net, parents=[-1], mutation_id=mutation_id)
#     print(net[parent])
#     net.add_node(parent + 1, label="output", color=colours["output"], title=net[parent], x=0, y=-1000)
#     net.add_edge(parent, parent + 1)
#     return net

def shape_to_string(shape):
    return f"[{', '.join(map(str, shape))}]"

# def recurse_add_node(node, net, parents=[], parent_fn="", mutation_id=None, override_colour=None):
#     if "node_type" in node and node["node_type"] == "terminal":
#         if "node_id" in node and node["node_id"] == mutation_id:
#             c = colours["mutation"]
#         elif override_colour:
#             c = override_colour
#         else:
#             c = colours[parent_fn]
#     elif "node_type" in node and node["node_type"] == "nonterminal":
#         if "node_id" in node and node["node_id"] == mutation_id:
#             override_colour = colours["mutation"]
#             c = override_colour

#     if "node_type" in node and node["node_type"] == "terminal":
#         net.add_node(node["node_id"], label=node["fn"].__name__, title=f"{node['node_id']}: {shape_to_string(node['output_shape'])}", color=c, level=node["depth"])
#         if parents:
#             for parent in parents:
#                 net.add_edge(parent, node["node_id"])
#         return node["node_id"]
#     elif "fn" in node and node["fn"].__name__ == "sequential_module":
#         parent = recurse_add_node(node["children"]["first_fn"], net, parents=parents, parent_fn="first_fn", mutation_id=mutation_id, override_colour=override_colour)
#         parent = recurse_add_node(node["children"]["second_fn"], net, parents=[parent], parent_fn="second_fn", mutation_id=mutation_id, override_colour=override_colour)
#     elif "fn" in node and node["fn"].__name__ == "branching_module":
#         parent = recurse_add_node(node["children"]["branching_fn"], net, parents=parents, parent_fn="branching_fn", mutation_id=mutation_id, override_colour=override_colour)
#         inner_parents = []
#         for child in node["children"]["inner_fn"]:
#             p = recurse_add_node(child, net, parents=[parent], parent_fn="inner_fn", mutation_id=mutation_id, override_colour=override_colour)
#             inner_parents.append(p)
#         parent = recurse_add_node(node["children"]["aggregation_fn"], net, parents=inner_parents, parent_fn="aggregation_fn", mutation_id=mutation_id, override_colour=override_colour)
#     elif "fn" in node and node["fn"].__name__ == "routing_module":
#         parent = recurse_add_node(node["children"]["prerouting_fn"], net, parents=parents, parent_fn="prerouting_fn", mutation_id=mutation_id, override_colour=override_colour)
#         parent = recurse_add_node(node["children"]["inner_fn"], net, parents=[parent], parent_fn="inner_fn", mutation_id=mutation_id, override_colour=override_colour)
#         parent = recurse_add_node(node["children"]["postrouting_fn"], net, parents=[parent], parent_fn="postrouting_fn", mutation_id=mutation_id, override_colour=override_colour)
#     elif "fn" in node and node["fn"].__name__ == "computation_module":
#         parent = recurse_add_node(node["children"]["computation_fn"], net, parents, parent_fn="computation_fn", mutation_id=mutation_id, override_colour=override_colour)
#     return parent

### convert the above code which uses an old version of the einspace representation (ased on dictionaries)
### to the new version which uses a tree representation (based on classes)


def recurse_add_node(node, net, parents=[]):
    if node.operation.type == "terminal":
        print(f"Adding node {node.id} with operation {node.operation.name}, colour {colours[node.operation.name]}, inputs {node.input_params}, outputs {node.output_params}")
        net.add_node(
            node.id, label=node.operation.name, title=f"{node.id}: {shape_to_string(node.output_params['shape'])}",
            color=colours[node.operation.name], level=node.level
        )
        if parents:
            print(f"Adding parents {parents}")
            for parent in parents:
                net.add_edge(parent, node.id)
        return node.id
    elif "sequential" in node.operation.name:
        for i, child in enumerate(node.children):
            if i == 0:
                parent = recurse_add_node(child, net, parents=parents)
            else:
                parent = recurse_add_node(child, net, parents=[parent])
    elif node.operation.name == "branching_module":
        parent = recurse_add_node(node.children[0], net, parents=parents)
        inner_parents = []
        for child in node.children[1:-1]:
            p = recurse_add_node(child, net, parents=[parent])
            inner_parents.append(p)
        parent = recurse_add_node(node.children[-1], net, parents=inner_parents)
    elif node.operation.name == "routing_module":
        parent = recurse_add_node(node.children[0], net, parents=parents)
        parent = recurse_add_node(node.children[1], net, parents=[parent])
        parent = recurse_add_node(node.children[2], net, parents=[parent])
    elif node.operation.name == "computation_module":
        parent = recurse_add_node(node.children[0], net, parents)
    elif "residual" in node.operation.name:
        max_id = node.get_root().serialise()[-1].id
        parent = recurse_add_node(
            DerivationTreeNode(
                id=max_id + 1, operation=clone(2), parent=node,
                input_params=node.input_params, output_params=node.input_params
            ),
            net, parents=parents
        )
        for i, child in enumerate(node.children):
            if i == len(node.children) - 2:
                residual = recurse_add_node(child, net, parents=[parent])
            elif i == len(node.children) - 1:
                main = recurse_add_node(child, net, parents=[parent])
            else:
                parent = recurse_add_node(child, net, parents=[parent])
        parent = recurse_add_node(
            DerivationTreeNode(
                id=max_id + 2, operation=add(2), parent=node,
                input_params=node.input_params, output_params=node.input_params
            ),
            net, parents=[residual, main]
        )
    elif "diamond" in node.operation.name:
        max_id = node.get_root().serialise()[-1].id
        parent = recurse_add_node(
            DerivationTreeNode(
                id=max_id + 1, operation=clone(2), parent=node,
                input_params=node.input_params, output_params=node.input_params
            ),
            net, parents=parents
        )
        for i, child in enumerate(node.children):
            if i == 0:
                left = recurse_add_node(child, net, parents=[parent])
            elif i == 1:
                right = recurse_add_node(child, net, parents=[parent])
            elif i % 2 == 0:
                left = recurse_add_node(child, net, parents=[left])
            else:
                right = recurse_add_node(child, net, parents=[right])
        parent = recurse_add_node(
            DerivationTreeNode(
                id=max_id + 2, operation=add(2), parent=node,
                input_params=node.input_params, output_params=node.input_params
            ),
            net, parents=[left, right]
        )
    elif "cell" in node.operation.name:
        max_id = node.get_root().serialise()[-1].id
        clone1 = recurse_add_node(
            DerivationTreeNode(
                id=max_id + 1, operation=clone(2), parent=node,
                input_params=node.input_params, output_params=node.input_params
            ),
            net, parents=parents
        )
        clone2 = recurse_add_node(
            DerivationTreeNode(
                id=max_id + 2, operation=clone(2), parent=node,
                input_params=node.input_params, output_params=node.input_params
            ),
            net, parents=[clone1]
        )
        # a_out = self.a(x)
        a = recurse_add_node(node.children[0], net, parents=[clone1])
        # b_out = self.b(x)
        b = recurse_add_node(node.children[1], net, parents=[clone2])
        # c_out = self.c(a_out)
        c = recurse_add_node(node.children[2], net, parents=[a])
        # d_out = self.d(x)
        d = recurse_add_node(node.children[3], net, parents=[clone2])
        # e_out = self.e(a_out)
        e = recurse_add_node(node.children[4], net, parents=[a])
        # f_out = self.f(b_out + c_out)
        bc = recurse_add_node(
            DerivationTreeNode(
                id=max_id + 3, operation=add(2), parent=node,
                input_params=node.input_params, output_params=node.input_params
            ),
            net, parents=[b, c]
        )
        f = recurse_add_node(node.children[5], net, parents=[bc])
        # out = d_out + e_out + f_out
        de = recurse_add_node(
            DerivationTreeNode(
                id=max_id + 4, operation=add(2), parent=node,
                input_params=node.input_params, output_params=node.input_params
            ),
            net, parents=[d, e]
        )
        parent = recurse_add_node(
            DerivationTreeNode(
                id=max_id + 5, operation=add(2), parent=node,
                input_params=node.input_params, output_params=node.input_params
            ),
            net, parents=[de, f]
        )
    return parent


def architecture_to_nx(node):
    net = nx.Graph()
    net.add_node(-1, label="input", color=colours["input"], title=shape_to_string(node.input_params["shape"]), x=0, y=1000)
    parent = recurse_add_node(node, net, parents=[-1])
    net.add_node(parent + 1, label="output", color=colours["output"], title=net[parent], x=0, y=-1000)
    net.add_edge(parent, parent + 1)
    return net


def visualise_architecture(node, title="", node_size=300, iteration=None, save_path=None, score=None, show=False):
    # create networkx network object
    net = architecture_to_nx(node)
    pos = graphviz_layout(net, prog="neato")
    # negate positions in dict
    pos = {k: (-v[0], v[1]) for k, v in pos.items()}
    # function to get node properties
    def get_property(net, prop, return_dict=False):
        prop_dict = nx.get_node_attributes(net, prop)
        if return_dict:
            return prop_dict
        return [prop_dict.get(node, None) for node in net.nodes()]
    # plot the tree
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    if score is not None:
        ax.set_title(f"Architecture at iteration {iteration} with score {score:.2f}")
    nx.draw(
        net, pos, with_labels=False, labels=get_property(net, "label", return_dict=True),
        node_color=get_property(net, "color"),
        node_size=node_size, font_size=4, font_weight='bold',
        ax=ax,
    )
    if save_path is not None:
        makedirs(save_path, exist_ok=True)
        plt.savefig(join(save_path, f"architecture_{iteration}.png"))
        plt.savefig(join(save_path, f"architecture.pdf"))
    if show:
        plt.show()
    plt.close()


def visualise_derivation_tree(root, stack=None, current_node_id=None, scale=1, iteration=None, save_path=None, score=None, show=False):
    def add_edges(graph, root, stack=None, current_node_id=None):
        if root is not None:
            op_name = root.operation.name if root.operation else ""
            input_shape = list(root.input_params["shape"]) if "shape" in root.input_params else ""
            output_shape = list(root.output_params["shape"]) if "shape" in root.output_params else ""
            other_shape = list(root.input_params["other_shape"]) if "other_shape" in root.input_params and root.input_params["other_shape"] is not None else ""
            on_stack = True if stack is not None and root.id in stack else False
            current = True if current_node_id == root.id else False
            if current:
                edgecolor =  "#00ff00"
            elif on_stack:
                edgecolor = "magenta"
            else:
                edgecolor = "none"
            # print(f"Adding node {root.id} with operation {op_name}")
            graph.add_node(
                root.id,
                input_shape=input_shape,
                output_shape=output_shape,
                other_shape=other_shape,
                op_name=op_name,
                color=colours[root.level] if root.id > 1 else colours["root"],
                edgecolor=edgecolor,
            )
            for child in root.children:
                add_edges(graph, child, stack, current_node_id)
                # print(f"Adding edge from {root.id} to {child.id}")
                graph.add_edge(root.id, child.id)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    stack_ids = [a.id for a, _ in stack] if stack is not None else None
    add_edges(G, root, stack_ids, current_node_id)

    # get the depth of the tree
    depth = max(nx.shortest_path_length(G, source=1).values()) + 1
    # get max degree of tree
    max_degree = max([G.out_degree(node) for node in G.nodes])
    # and the width of the tree (i.e. number of leaves)
    width = max_degree ** (depth - 1)

    # Draw the graph
    # tree layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    labels = {}
    for i, node in enumerate(G.nodes):
        labels[node] = f"{node}\n{nx.get_node_attributes(G, 'op_name')[node]}\n" \
            f"{nx.get_node_attributes(G, 'input_shape')[node]}\n" \
            f"{nx.get_node_attributes(G, 'output_shape')[node]}\n" \
            f"{nx.get_node_attributes(G, 'other_shape')[node]}"
        # colors = ["skyblue" if node.out_degree() > 0 else "salmon" for node in G.nodes]
    colors = nx.get_node_attributes(G, 'color').values()
    # draw border around nodes on stack
    border_colors = nx.get_node_attributes(G, 'edgecolor').values()

    fig = plt.figure(figsize=(max(8, scale * (width / 20)), max(6, scale * (depth / 8))))
    ax = fig.add_subplot(111)
    # ax.set_title(f"Derivation Tree at iteration {iteration}")
    if score is not None:
        ax.set_title(f"Derivation Tree at iteration {iteration} with score {score:.2f}")
    nx.draw(
        G, pos, labels=labels, with_labels=True,
        node_size=(60 * scale) ** 2, node_color=colors, edgecolors=border_colors, linewidths=2,
        font_size=6 * scale, font_color="white", font_weight="bold",
        arrows=True, arrowsize=20 * scale, arrowstyle="->",
        edge_color="gray", width=2 * scale, ax=ax
    )
    # edit the text of the nodes
    # labels = nx.get_edge_attributes(G, "output_val")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # change the size of the figure
    # plt.gcf().set_size_inches(8, 6)
    # extend the margins
    plt.margins(x=0, y=0.05 + 0.05 * scale)
    if save_path is not None:
        makedirs(save_path, exist_ok=True)
        plt.savefig(join(save_path, f"derivation_tree_{iteration}.png"))
        plt.savefig(join(save_path, f"derivation_tree.pdf"))
    if show:
        plt.show()
    plt.close()


def visualise_search_tree(root, children, Q, N, path=None, scale=1, layout="twopi", iteration=None):
    def add_edges(graph, root, children, Q, N):
        if root is not None:
            graph.add_node(
                root.id,
                op_name=root.operation.name if root.operation else "",
                score=Q[root],
                visits=N[root],
                # color=colours[root.node.level] if root.id > 1 else colours["root"],
            )
            if root in children:
                for child in children[root]:
                    add_edges(graph, child, children, Q, N)
                    edge_color = "magenta" if (root.id, child.id) in path else "grey"
                    thickness = (Q[child] / (N[child] + 0.01)) * 5 + 0.2
                    graph.add_edge(root.id, child.id, color=edge_color, thickness=thickness)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    add_edges(G, root, children, Q, N)

    # get the depth of the tree
    depth = max(nx.shortest_path_length(G, source=1).values()) + 1
    # get max degree of tree
    max_degree = max([G.out_degree(node) for node in G.nodes])
    # and the width of the tree (i.e. number of leaves)
    width = max_degree ** (depth - 1)

    # Draw the graph
    # tree layout
    pos = nx.nx_agraph.graphviz_layout(G, prog=layout)
    labels = {}
    for i, node in enumerate(G.nodes):
        labels[node] = f"{node}\n{nx.get_node_attributes(G, 'op_name')[node]}\n" \
            f"{nx.get_node_attributes(G, 'score')[node]:.2f}/" \
            f"{nx.get_node_attributes(G, 'visits')[node]}"
    colors = nx.get_node_attributes(G, 'color').values()
    edge_color = [G[u][v]["color"] for u, v in G.edges]
    edge_thickness = [G[u][v]["thickness"] * scale for u, v in G.edges]

    fig = plt.figure(figsize=(12 * scale, 12 * scale))
    ax = fig.add_subplot(111)
    ax.set_title(f"Search Tree at iteration {iteration}")
    nx.draw(
        G, pos, labels=labels, with_labels=True,
        node_size=(42 * scale) ** 2, node_color=colors, linewidths=2,
        font_size=6 * scale, font_color="white", font_weight="bold",
        arrows=True, arrowsize=20 * scale, arrowstyle="->",
        edge_color=edge_color, width=edge_thickness, ax=ax
    )
    # edit the text of the nodes
    # labels = nx.get_edge_attributes(G, "output_val")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # change the size of the figure
    # plt.gcf().set_size_inches(8, 6)
    # extend the margins
    plt.margins(0.05 + 0.05 * scale)
    plt.show()


def visualise_search_tree_2(root, children, Q, N, path=None, score_fn=None, scale=1, layout="twopi", iteration=None, save_path=None, show=False):
    def add_edges(graph, node, parent, children, Q, N):
        if node is not None:
            graph.add_node(
                node.id,
                op_name=node.operation.name if node.operation else "",
                score=score_fn(node, parent) if parent is not None and score_fn is not None and (node in children) and N[node] != 0 else 0,
                visits=N[node],
                # color=colours[node.node.level] if node.id > 1 else colours["root"],
            )
            if node in children:
                for child in children[node]:
                    add_edges(graph, child, node, children, Q, N)
                    edge_color = "#ab3396" if (node.id, child.id) in path else "grey"
                    thickness = 2 if (node.id, child.id) in path else 1
                    graph.add_edge(node.id, child.id, color=edge_color, thickness=thickness)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    add_edges(G, root, None, children, Q, N)

    # get the depth of the tree
    depth = max(nx.shortest_path_length(G, source=1).values()) + 1
    # get max degree of tree
    max_degree = max([G.out_degree(node) for node in G.nodes])
    # and the width of the tree (i.e. number of leaves)
    width = max_degree ** (depth - 1)

    # Draw the graph
    # tree layout
    pos = nx.nx_agraph.graphviz_layout(G, prog=layout)
    labels = {}
    for i, node in enumerate(G.nodes):
        labels[node] = f"{node}\n{nx.get_node_attributes(G, 'op_name')[node]}\n" \
            f"{nx.get_node_attributes(G, 'score')[node]:.2f}/" \
            f"{nx.get_node_attributes(G, 'visits')[node]}"
    n_colors = 100
    palette = sns.color_palette("ch:start=.2,rot=-.3", n_colors=n_colors)
    # assign the colours according to their score/visits on a scale of 0 to 9
    scores = nx.get_node_attributes(G, 'score').values()
    # if any score is above 1
    if max(scores) > 1:
        scores = [score / max(scores) for score in scores]
    visits = nx.get_node_attributes(G, 'visits').values()
    colors = [palette[int((n_colors - 1) * score)] if node != 1 else '#000000' for node, score in zip(G.nodes, scores)]
    sizes = [(10 * scale) ** 2 if visit > 0 else (10 * scale) ** 2 for visit in visits]
    edge_color = [G[u][v]["color"] for u, v in G.edges]
    edge_thickness = [G[u][v]["thickness"] * scale for u, v in G.edges]

    fig = plt.figure(figsize=(12 * scale, 12 * scale))
    ax = fig.add_subplot(111)
    ax.set_title(f"Search Tree at iteration {iteration}")
    nx.draw(
        G, pos, labels=None, with_labels=False,
        node_size=sizes, node_color=colors, linewidths=2,
        font_size=6 * scale, font_color="white", font_weight="bold",
        arrows=True, arrowsize=20 * scale, arrowstyle="-",
        edge_color=edge_color, width=edge_thickness, ax=ax
    )
    # edit the text of the nodes
    # labels = nx.get_edge_attributes(G, "output_val")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # change the size of the figure
    # plt.gcf().set_size_inches(8, 6)
    # extend the margins
    # plt.margins(0.05 + 0.05 * scale)
    if save_path is not None:
        makedirs(save_path, exist_ok=True)
        plt.savefig(join(save_path, f"search_tree_{iteration}.png"))
        plt.savefig(join(save_path, f"search_tree.pdf"))
    if show:
        plt.show()
    plt.close()
