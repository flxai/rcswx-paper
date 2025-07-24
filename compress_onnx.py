# %% [markdown]
# ### Converting ONNX to look more like einspace through pattern matching and parallel branch reduction

# %% [markdown]
# Reducing branches in graph:
# 1. Start from input, initialise empty stack, start traversing graph through children
# 2. When we find a Split, put it on the stack, keep putting Splits on the stack if we find more
# 3. When we find a Concat/Add/etc we pop the most recent Split off the stack and process the branching component to reduce it
# - 3.1. Find the number of branches X (input Const to Split node)
# - 3.2. Check that all branches are identical
# - 3.3. Remove all but one branch
# - 3.4. Output a string representation of branch that says it is repeated X times
# 4. Keep going until we reach output node

# %%
import onnx
import pandas as pd
import networkx as nx
from functools import partial
import matplotlib.pyplot as plt


def load_onn_file(i):
    path = f"/localdisk/data2/eintool/onnx/einspace/cifar10/seed=0/{i}.onnx"
    # load and parse the ONNX model
    nodes, onnx_graph = parse_onnx_model(path)
    df = pd.read_csv("/localdisk/data2/eintool/encodings/einspace/cifar10_seed=0.csv")
    tree_string = df["tree_encoding"][i]
    return nodes, onnx_graph, tree_string


class Node():
    def __init__(self, name, op_type, inputs, outputs, attributes=None):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes if attributes is not None else {}

    def __repr__(self):
        return f"Node(name={self.name}, op_type={self.op_type}, inputs={self.inputs}, outputs={self.outputs}, attributes={self.attributes})"

    def __eq__(self, other):
        return isinstance(other, Node) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

def parse_onnx_model(onnx_file_path: str):
    nodes = []
    model_proto = onnx.load(onnx_file_path)
    graph_proto = model_proto.graph

    # print(f"Model Name: {model_proto.graph.name}")
    # print("\nGraph Inputs:")
    for i, input_info in enumerate(graph_proto.input):
        # print(f"- {input_info.name} (Type: {input_info.type.tensor_type.elem_type}, Shape: {[d.dim_value for d in input_info.type.tensor_type.shape.dim]})")
        n = Node(
            name=input_info.name,
            op_type="Input" if i == 0 else "Parameter",
            inputs=[],
            outputs=[input_info.name],
            attributes={"shape": [d.dim_value for d in input_info.type.tensor_type.shape.dim]}
        )
        nodes.append(n)

    # print("\nGraph Nodes (Operations):")
    for node in graph_proto.node:
        # print(f"- Op Type: {node.op_type}")
        # print(f"  Inputs: {', '.join(node.input)}")
        # print(f"  Outputs: {', '.join(node.output)}")
        # if node.attribute:
            # print("  Attributes:")
            # for attr in node.attribute:
            #     print(f"    - {attr.name}: {attr}") # This will print the full attribute proto
        n = Node(
            name=node.name,
            op_type=node.op_type,
            inputs=node.input,
            outputs=node.output,
            attributes={str(attr.name): str(attr) for attr in node.attribute}
        )
        nodes.append(n)
    return nodes, graph_proto


from collections import defaultdict, deque

def topological_sort(nodes):
    # Map from output name to the node that produces it
    output_to_node = {}
    for node in nodes:
        for output in node.outputs:
            output_to_node[output] = node

    # Build graph: node -> list of dependent nodes (those that consume its outputs)
    graph = defaultdict(list)
    in_degree = defaultdict(int)  # Node -> number of unmet dependencies

    for node in nodes:
        for input_name in node.inputs:
            if input_name in output_to_node:
                producer = output_to_node[input_name]
                graph[producer].append(node)
                in_degree[node] += 1

    # Queue of nodes with no dependencies
    zero_in_degree = deque([node for node in nodes if in_degree[node] == 0])
    
    sorted_nodes = []

    while zero_in_degree:
        current = zero_in_degree.popleft()
        sorted_nodes.append(current)
        for dependent in graph[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                zero_in_degree.append(dependent)

    if len(sorted_nodes) != len(nodes):
        raise ValueError("Graph has a cycle or disconnected component!")

    return sorted_nodes


def get_node_helpers(nodes):
    # now, nodes have a list of inputs and outputs
    # we need to match these together to build the DAG
    # e.g. if node 1 has outputs [A] and node 2 has inputs [A], then node 2 is a consumer of node 1
    # and node 1 is a producer of node 2
    name_to_node = {node.name: node for node in nodes}
    children = defaultdict(list)
    parents = defaultdict(list)
    for node in nodes:
        for output in node.outputs:
            for other_node in nodes:
                if output in other_node.inputs:
                    parents[other_node.name].append(node)
                    children[node.name].append(other_node)
    return name_to_node, children, parents


def is_connected(nodes, parents, children):
    # check if the graph is connected
    for node in nodes:
        if node.name not in parents and node.name not in children:
            return False
    return True


def plot_graph(G, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    # use dot layout for better visualization
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    labels = {k: v for k, v in nx.get_node_attributes(G, 'op_type').items()}
    nx.draw(G, pos, with_labels=False, node_size=2000, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', arrows=True)
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='k', font_size=8)
    # plt.title("ONNX Model DAG")
    plt.show()

# %%
def convert_to_networkx(nodes, children):
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node.name, op_type=node.op_type, inputs=node.inputs, outputs=node.outputs, attributes=node.attributes)
        for child in children[node.name]:
            G.add_edge(node.name, child.name)
    return G


def match_fn(n1, n2):
    op_type_match = n1['op_type'] == n2['op_type']
    shape_match = True
    if 'attributes' in n1 and 'attributes' in n2:
        if "shape" in n1["attributes"] and "shape" in n2["attributes"]:
            # check if the shapes match
            shape_match = len(n1['attributes']['shape']) == len(n2['attributes']['shape'])
    return op_type_match and shape_match


def match_pattern(G, pattern, match_fn=match_fn, update_fn=None):
    """
    Matches a pattern in the graph G using the provided match function.
    If unique is True, it ensures that the match is unique by checking if the subgraph
    is already in matches.
    """
    print(pattern)
    match_id = 0
    while True:
        match = next(nx.algorithms.isomorphism.DiGraphMatcher(
            G, pattern, node_match=match_fn
        ).subgraph_isomorphisms_iter(), None)
        if match is None:
            break
        match = {v: k for k, v in match.items()}
        G = update_fn(G=G, match=match, match_id=match_id)
        match_id += 1
    return G


def update_pattern(
        G, match, match_id, pattern_name,
        new_node_name, new_node_op_type,
        pattern_input_names, pattern_output_name,
        attributes_dict={}, inherit_attributes_from=None
    ):
    new_node_name = f"{new_node_name}_{pattern_name}_{match_id}"
    # find the inputs and outputs to the subgraph
    inputs_to_subgraph = []
    for pattern_input_name in pattern_input_names:
        inputs_to_subgraph += [n for n in G.predecessors(match[pattern_input_name]) if G.nodes[n]["op_type"] not in ["Parameter", "Constant"]]
    if attributes_dict:
        new_node_attributes = {f"{node}_{attr}": G.nodes[match[node]]["attributes"].get(attr, []) for node, attr in attributes_dict.items()}
    else:
        new_node_attributes = G.nodes[match[inherit_attributes_from]]["attributes"]
    outputs_to_subgraph = list(G.successors(match[pattern_output_name]))

    # remove the matched nodes from the graph
    for _, node_name in match.items():
        G.remove_node(node_name)

    # add the new node to the graph
    G.add_node(new_node_name, op_type=new_node_op_type, attributes=new_node_attributes)

    # add edges from the matched nodes to the new node
    for input_node_name in inputs_to_subgraph:
        G.add_edge(input_node_name, new_node_name)

    # add edges from the new node to its outputs
    for output_node_name in outputs_to_subgraph:
        G.add_edge(new_node_name, output_node_name)

    # print(f"Updated {new_node_name}")

    return G


def replace_component(
    G, pattern_name, pattern,
    new_node_name, new_node_op_type,
    pattern_input_names, pattern_output_name,
    attributes_dict, inherit_attributes_from
):
    G = match_pattern(
        G, pattern, update_fn=partial(update_pattern, pattern_name=pattern_name,
            new_node_name=new_node_name, new_node_op_type=new_node_op_type,
            pattern_input_names=pattern_input_names,pattern_output_name=pattern_output_name,
            attributes_dict=attributes_dict, inherit_attributes_from=inherit_attributes_from
        )
    )
    # plot_graph(G)
    # print(f"Nodes in G: {G.size()} after {pattern_name} update")


# %%
# unfold mess -> unfold
unfold_pattern = nx.DiGraph()
unfold_pattern.add_node("shape_0", op_type="Shape")
unfold_pattern.add_node("constant_0_0", op_type="Constant")
unfold_pattern.add_node("gather_0", op_type="Gather")
unfold_pattern.add_node("constant_0_1", op_type="Constant")
unfold_pattern.add_node("mul_0", op_type="Mul")
unfold_pattern.add_node("constant_0_2", op_type="Constant")
unfold_pattern.add_node("unsqueeze_0", op_type="Unsqueeze")
unfold_pattern.add_edge("shape_0", "gather_0")
unfold_pattern.add_edge("constant_0_0", "gather_0")
unfold_pattern.add_edge("gather_0", "mul_0")
unfold_pattern.add_edge("constant_0_1", "mul_0")
unfold_pattern.add_edge("mul_0", "unsqueeze_0")
unfold_pattern.add_edge("constant_0_2", "unsqueeze_0")

unfold_pattern.add_node("shape_1", op_type="Shape")
unfold_pattern.add_node("constant_1_0", op_type="Constant")
unfold_pattern.add_node("gather_1", op_type="Gather")
unfold_pattern.add_node("constant_1_1", op_type="Constant")
unfold_pattern.add_node("unsqueeze_1", op_type="Unsqueeze")
unfold_pattern.add_edge("shape_1", "gather_1")
unfold_pattern.add_edge("constant_1_0", "gather_1")
unfold_pattern.add_edge("gather_1", "unsqueeze_1")
unfold_pattern.add_edge("constant_1_1", "unsqueeze_1")

unfold_pattern.add_node("constant_concat", op_type="Constant")
unfold_pattern.add_node("concat_0_and_1", op_type="Concat")
unfold_pattern.add_edge("unsqueeze_0", "concat_0_and_1")
unfold_pattern.add_edge("unsqueeze_1", "concat_0_and_1")
unfold_pattern.add_edge("constant_concat", "concat_0_and_1")

unfold_pattern.add_node("shape_2", op_type="Shape")
unfold_pattern.add_node("constant_2_0", op_type="Constant")
unfold_pattern.add_node("gather_2", op_type="Gather")
unfold_pattern.add_node("constant_2_1", op_type="Constant")
unfold_pattern.add_node("add_2_0", op_type="Add")
unfold_pattern.add_node("constant_2_2", op_type="Constant")
unfold_pattern.add_node("sub_2", op_type="Sub")
unfold_pattern.add_node("constant_2_3_0", op_type="Constant")
unfold_pattern.add_node("constant_2_3_1", op_type="Constant")
unfold_pattern.add_node("range_2", op_type="Range")
unfold_pattern.add_node("constant_2_4", op_type="Constant")
unfold_pattern.add_node("unsqueeze_2", op_type="Unsqueeze")
unfold_pattern.add_node("constant_2_5", op_type="Constant")
unfold_pattern.add_node("add_2_1", op_type="Add")
unfold_pattern.add_edge("shape_2", "gather_2")
unfold_pattern.add_edge("constant_2_0", "gather_2")
unfold_pattern.add_edge("gather_2", "add_2_0")
unfold_pattern.add_edge("constant_2_1", "add_2_0")
unfold_pattern.add_edge("add_2_0", "sub_2")
unfold_pattern.add_edge("constant_2_2", "sub_2")
unfold_pattern.add_edge("sub_2", "range_2")
unfold_pattern.add_edge("constant_2_3_0", "range_2")
unfold_pattern.add_edge("constant_2_3_1", "range_2")
unfold_pattern.add_edge("range_2", "unsqueeze_2")
unfold_pattern.add_edge("constant_2_4", "unsqueeze_2")
unfold_pattern.add_edge("unsqueeze_2", "add_2_1")
unfold_pattern.add_edge("constant_2_5", "add_2_1")

unfold_pattern.add_node("pad", op_type="Pad")
unfold_pattern.add_node("constant_pad", op_type="Constant")
unfold_pattern.add_edge("constant_pad", "pad")

unfold_pattern.add_node("shape_3", op_type="Shape")
unfold_pattern.add_node("constant_3_0", op_type="Constant")
unfold_pattern.add_node("gather_3", op_type="Gather")
unfold_pattern.add_node("constant_3_1", op_type="Constant")
unfold_pattern.add_node("add_3_0", op_type="Add")
unfold_pattern.add_node("constant_3_2", op_type="Constant")
unfold_pattern.add_node("sub_3", op_type="Sub")
unfold_pattern.add_node("constant_3_3_0", op_type="Constant")
unfold_pattern.add_node("constant_3_3_1", op_type="Constant")
unfold_pattern.add_node("range_3", op_type="Range")
unfold_pattern.add_node("constant_3_4", op_type="Constant")
unfold_pattern.add_node("unsqueeze_3", op_type="Unsqueeze")
unfold_pattern.add_node("constant_3_5", op_type="Constant")
unfold_pattern.add_node("add_3_1", op_type="Add")
unfold_pattern.add_edge("shape_3", "gather_3")
unfold_pattern.add_edge("constant_3_0", "gather_3")
unfold_pattern.add_edge("gather_3", "add_3_0")
unfold_pattern.add_edge("constant_3_1", "add_3_0")
unfold_pattern.add_edge("add_3_0", "sub_3")
unfold_pattern.add_edge("constant_3_2", "sub_3")
unfold_pattern.add_edge("sub_3", "range_3")
unfold_pattern.add_edge("constant_3_3_0", "range_3")
unfold_pattern.add_edge("constant_3_3_1", "range_3")
unfold_pattern.add_edge("range_3", "unsqueeze_3")
unfold_pattern.add_edge("constant_3_4", "unsqueeze_3")
unfold_pattern.add_edge("unsqueeze_3", "add_3_1")
unfold_pattern.add_edge("constant_3_5", "add_3_1")

unfold_pattern.add_node("gather_pad_and_3", op_type="Gather")
unfold_pattern.add_node("gather_2_and_3", op_type="Gather")
unfold_pattern.add_node("transpose", op_type="Transpose")
unfold_pattern.add_node("reshape", op_type="Reshape")
unfold_pattern.add_edge("pad", "gather_pad_and_3")
unfold_pattern.add_edge("add_3_1", "gather_pad_and_3")
unfold_pattern.add_edge("gather_pad_and_3", "gather_2_and_3")
unfold_pattern.add_edge("add_2_1", "gather_2_and_3")
unfold_pattern.add_edge("gather_2_and_3", "transpose")
unfold_pattern.add_edge("transpose", "reshape")
unfold_pattern.add_edge("concat_0_and_1", "reshape")

# matmul+add -> linear
matmul_add_pattern = nx.DiGraph()
matmul_add_pattern.add_node("weight", op_type="Parameter")
matmul_add_pattern.add_node("MatMul", op_type="MatMul")
matmul_add_pattern.add_node("bias", op_type="Parameter")
matmul_add_pattern.add_node("Add", op_type="Add")
matmul_add_pattern.add_edge("weight", "MatMul")
matmul_add_pattern.add_edge("bias", "Add")
matmul_add_pattern.add_edge("MatMul", "Add")

# gemm -> linear
gemm_pattern = nx.DiGraph()
gemm_pattern.add_node("weight", op_type="Parameter", attributes={"shape": [None, None]})
gemm_pattern.add_node("bias", op_type="Parameter", attributes={"shape": [None]})
gemm_pattern.add_node("Gemm", op_type="Gemm")
gemm_pattern.add_edge("weight", "Gemm")
gemm_pattern.add_edge("bias", "Gemm")

# batchnorm+parameters -> batchnorm
batchnorm_pattern = nx.DiGraph()
batchnorm_pattern.add_node("weight", op_type="Parameter", attributes={"shape": [None]})
batchnorm_pattern.add_node("bias", op_type="Parameter", attributes={"shape": [None]})
batchnorm_pattern.add_node("running_mean", op_type="Parameter", attributes={"shape": [None]})
batchnorm_pattern.add_node("running_var", op_type="Parameter", attributes={"shape": [None]})
batchnorm_pattern.add_node("BatchNormalization", op_type="BatchNormalization")
batchnorm_pattern.add_edge("weight", "BatchNormalization")
batchnorm_pattern.add_edge("bias", "BatchNormalization")
batchnorm_pattern.add_edge("running_mean", "BatchNormalization")
batchnorm_pattern.add_edge("running_var", "BatchNormalization")

# add+parameter -> add (positional encoding)
pos_enc_pattern = nx.DiGraph()
pos_enc_pattern.add_node("weight", op_type="Parameter")
pos_enc_pattern.add_node("Add", op_type="Add")
pos_enc_pattern.add_edge("weight", "Add")

# 8xSplit+Constant -> Split
split_x8_pattern = nx.DiGraph()
split_x8_pattern.add_node("Split_0", op_type="Split")
split_x8_pattern.add_node("Split_1", op_type="Split")
split_x8_pattern.add_node("Split_2", op_type="Split")
split_x8_pattern.add_node("Split_3", op_type="Split")
split_x8_pattern.add_node("Split_4", op_type="Split")
split_x8_pattern.add_node("Split_5", op_type="Split")
split_x8_pattern.add_node("Split_6", op_type="Split")
split_x8_pattern.add_node("Split_7", op_type="Split")
split_x8_pattern.add_node("constant", op_type="Constant")
split_x8_pattern.add_edge("constant", "Split_0")
split_x8_pattern.add_edge("constant", "Split_1")
split_x8_pattern.add_edge("constant", "Split_2")
split_x8_pattern.add_edge("constant", "Split_3")
split_x8_pattern.add_edge("constant", "Split_4")
split_x8_pattern.add_edge("constant", "Split_5")
split_x8_pattern.add_edge("constant", "Split_6")
split_x8_pattern.add_edge("constant", "Split_7")

# 4xSplit+Constant -> Split
split_x4_pattern = nx.DiGraph()
split_x4_pattern.add_node("Split_0", op_type="Split")
split_x4_pattern.add_node("Split_1", op_type="Split")
split_x4_pattern.add_node("Split_2", op_type="Split")
split_x4_pattern.add_node("Split_3", op_type="Split")
split_x4_pattern.add_node("constant", op_type="Constant")
split_x4_pattern.add_edge("constant", "Split_0")
split_x4_pattern.add_edge("constant", "Split_1")
split_x4_pattern.add_edge("constant", "Split_2")
split_x4_pattern.add_edge("constant", "Split_3")

# Split+Constant -> Split
split_pattern = nx.DiGraph()
split_pattern.add_node("Split", op_type="Split")
split_pattern.add_node("constant", op_type="Constant")
split_pattern.add_edge("constant", "Split")

# constant+unsqueeze+concat+reducesum -> aggregation_add
aggregation_add_8_pattern = nx.DiGraph()
aggregation_add_8_pattern.add_node("unsqueeze_0", op_type="Unsqueeze")
aggregation_add_8_pattern.add_node("constant_0", op_type="Constant")
aggregation_add_8_pattern.add_node("unsqueeze_1", op_type="Unsqueeze")
aggregation_add_8_pattern.add_node("constant_1", op_type="Constant")
aggregation_add_8_pattern.add_node("unsqueeze_2", op_type="Unsqueeze")
aggregation_add_8_pattern.add_node("constant_2", op_type="Constant")
aggregation_add_8_pattern.add_node("unsqueeze_3", op_type="Unsqueeze")
aggregation_add_8_pattern.add_node("constant_3", op_type="Constant")
aggregation_add_8_pattern.add_node("unsqueeze_4", op_type="Unsqueeze")
aggregation_add_8_pattern.add_node("constant_4", op_type="Constant")
aggregation_add_8_pattern.add_node("unsqueeze_5", op_type="Unsqueeze")
aggregation_add_8_pattern.add_node("constant_5", op_type="Constant")
aggregation_add_8_pattern.add_node("unsqueeze_6", op_type="Unsqueeze")
aggregation_add_8_pattern.add_node("constant_6", op_type="Constant")
aggregation_add_8_pattern.add_node("unsqueeze_7", op_type="Unsqueeze")
aggregation_add_8_pattern.add_node("constant_7", op_type="Constant")
aggregation_add_8_pattern.add_node("concat", op_type="Concat")
aggregation_add_8_pattern.add_node("reduce_sum", op_type="ReduceSum")
aggregation_add_8_pattern.add_edge("constant_0", "unsqueeze_0")
aggregation_add_8_pattern.add_edge("constant_1", "unsqueeze_1")
aggregation_add_8_pattern.add_edge("constant_2", "unsqueeze_2")
aggregation_add_8_pattern.add_edge("constant_3", "unsqueeze_3")
aggregation_add_8_pattern.add_edge("constant_4", "unsqueeze_4")
aggregation_add_8_pattern.add_edge("constant_5", "unsqueeze_5")
aggregation_add_8_pattern.add_edge("constant_6", "unsqueeze_6")
aggregation_add_8_pattern.add_edge("constant_7", "unsqueeze_7")
aggregation_add_8_pattern.add_edge("unsqueeze_0", "concat")
aggregation_add_8_pattern.add_edge("unsqueeze_1", "concat")
aggregation_add_8_pattern.add_edge("unsqueeze_2", "concat")
aggregation_add_8_pattern.add_edge("unsqueeze_3", "concat")
aggregation_add_8_pattern.add_edge("unsqueeze_4", "concat")
aggregation_add_8_pattern.add_edge("unsqueeze_5", "concat")
aggregation_add_8_pattern.add_edge("unsqueeze_6", "concat")
aggregation_add_8_pattern.add_edge("unsqueeze_7", "concat")
aggregation_add_8_pattern.add_edge("concat", "reduce_sum")
aggregation_add_8_pattern.add_edge("constant_8", "reduce_sum")

aggregation_add_4_pattern = nx.DiGraph()
aggregation_add_4_pattern.add_node("unsqueeze_0", op_type="Unsqueeze")
aggregation_add_4_pattern.add_node("constant_0", op_type="Constant")
aggregation_add_4_pattern.add_node("unsqueeze_1", op_type="Unsqueeze")
aggregation_add_4_pattern.add_node("constant_1", op_type="Constant")
aggregation_add_4_pattern.add_node("unsqueeze_2", op_type="Unsqueeze")
aggregation_add_4_pattern.add_node("constant_2", op_type="Constant")
aggregation_add_4_pattern.add_node("unsqueeze_3", op_type="Unsqueeze")
aggregation_add_4_pattern.add_node("constant_3", op_type="Constant")
aggregation_add_4_pattern.add_node("concat", op_type="Concat")
aggregation_add_4_pattern.add_node("constant_4", op_type="Constant")
aggregation_add_4_pattern.add_node("reduce_sum", op_type="ReduceSum")
aggregation_add_4_pattern.add_edge("constant_0", "unsqueeze_0")
aggregation_add_4_pattern.add_edge("constant_1", "unsqueeze_1")
aggregation_add_4_pattern.add_edge("constant_2", "unsqueeze_2")
aggregation_add_4_pattern.add_edge("constant_3", "unsqueeze_3")
aggregation_add_4_pattern.add_edge("unsqueeze_0", "concat")
aggregation_add_4_pattern.add_edge("unsqueeze_1", "concat")
aggregation_add_4_pattern.add_edge("unsqueeze_2", "concat")
aggregation_add_4_pattern.add_edge("unsqueeze_3", "concat")
aggregation_add_4_pattern.add_edge("concat", "reduce_sum")
aggregation_add_4_pattern.add_edge("constant_4", "reduce_sum")

# %%
pattern_dicts = [
    {
        "pattern_name": "Unfold", "pattern": unfold_pattern, "new_node_name": "Unfold",
        "new_node_op_type": "Unfold", "pattern_input_names": ["shape_0"], "pattern_output_name": "reshape",
        "attributes_dict": {}, "inherit_attributes_from": "reshape"
    },
    {
        "pattern_name": "MatMul+Add", "pattern": matmul_add_pattern, "new_node_name": "Linear",
        "new_node_op_type": "Linear", "pattern_input_names": ["MatMul"], "pattern_output_name": "Add",
        "attributes_dict": {"weight": "shape", "bias": "shape"}, "inherit_attributes_from": None
    },
    {
        "pattern_name": "Gemm", "pattern": gemm_pattern, "new_node_name": "Linear",
        "new_node_op_type": "Linear", "pattern_input_names": ["Gemm"], "pattern_output_name": "Gemm",
        "attributes_dict": {"weight": "shape", "bias": "shape"}, "inherit_attributes_from": None
    },
    {
        "pattern_name": "BatchNormalization", "pattern": batchnorm_pattern, "new_node_name": "BatchNormalization",
        "new_node_op_type": "BatchNormalization", "pattern_input_names": ["BatchNormalization"], "pattern_output_name": "BatchNormalization",
        "attributes_dict": {"weight": "shape", "bias": "shape", "running_mean": "shape", "running_var": "shape"}, "inherit_attributes_from": None
    },
    {
        "pattern_name": "PositionalEncoding", "pattern": pos_enc_pattern, "new_node_name": "PositionalEncoding",
        "new_node_op_type": "PositionalEncoding", "pattern_input_names": ["Add"], "pattern_output_name": "Add",
        "attributes_dict": {"weight": "shape"}, "inherit_attributes_from": None
    },
    {
        "pattern_name": "Split+Constant", "pattern": split_pattern, "new_node_name": "Split",
        "new_node_op_type": "Split", "pattern_input_names": ["Split"], "pattern_output_name": "Split",
        "attributes_dict": {}, "inherit_attributes_from": "Split"
    },
    {
        "pattern_name": "Aggregation Add(8)", "pattern": aggregation_add_8_pattern, "new_node_name": "Aggregation Add(8)",
        "new_node_op_type": "Aggregation Add(8)", "pattern_input_names": [
            "unsqueeze_0", "unsqueeze_1", "unsqueeze_2", "unsqueeze_3",
            "unsqueeze_4", "unsqueeze_5", "unsqueeze_6", "unsqueeze_7"
        ], "pattern_output_name": "reduce_sum", "attributes_dict": {"constant_8": "value"}, "inherit_attributes_from": None
    },
    {
        "pattern_name": "Aggregation Add(4)", "pattern": aggregation_add_4_pattern, "new_node_name": "Aggregation Add(4)",
        "new_node_op_type": "Aggregation Add(4)", "pattern_input_names": [
            "unsqueeze_0", "unsqueeze_1", "unsqueeze_2", "unsqueeze_3"
        ], "pattern_output_name": "reduce_sum", "attributes_dict": {"constant_4": "value"}, "inherit_attributes_from": None
    },
]

# %%
# reduce any parallel branches in the graph
def rollout_branch(G, node_name, end_node):
    """
    This function takes a branch (a list of nodes) and an end node.
    It returns the sequence of nodes from the start of the branch to the end node.
    """
    if node_name == end_node:
        return []
    for child in G.successors(node_name):
        return [node_name] + rollout_branch(G, child, end_node)


def reduce_subgraph(G, start_of_graph, end_of_graph):
    """
    This function takes as input two nodes, the start and end nodes of a graph.
    This graph is expected to be a set of K parallel linear sequences of operations, known as branches.
    The function confirms that the different branches are identical and returns a single branch,
    representing the reduced graph.
    """
    # starting from the start_of_graph node, we add each child to a list (one list per branch)
    # and then we check that all nodes along those branches are the same
    branches = [rollout_branch(G, child, end_of_graph) for child in G.successors(start_of_graph)]
    print(f"Found {len(branches)} branches")
    for i, branch in enumerate(branches):
        print(i, [(G.nodes[n]["op_type"], G.nodes[n]["attributes"]) for n in branch])
    # Now we have a list of branches, we can check that they are all the same
    first_branch = branches[0]
    for branch in branches[1:]:
        if len(branch) != len(first_branch):
            # raise ValueError("Branches are not the same length.")
            return G
        for i, (node1, node2) in enumerate(zip(first_branch, branch)):
            if G.nodes[node1]["op_type"] != G.nodes[node2]["op_type"] or G.nodes[node1]["attributes"] != G.nodes[node2]["attributes"]:
                # raise ValueError(f"Branches are not the same at index {i}: {node1} != {node2}")
                return G
    # remove all but one branch in G
    for branch in branches[1:]:
        for node in branch:
            G.remove_node(node)
    return G


def reduce_graph(G, queue, visited=[], stack=[]):
    """
    This function takes a branch and an end node, and rolls out the branch
    until it reaches the end node.
    """
    # find next node to process
    print("Queue:")
    for branching_depth in range(len(queue)):
        print(f"\tDepth {branching_depth}: {queue[branching_depth]}")
    branching_depth = len(stack)
    print("Graph nodes", G.nodes)
    print("Visited", visited)
    node_name = queue[branching_depth].pop(-1)
    while node_name not in G.nodes or node_name in visited:
        node_name = queue[branching_depth].pop(-1)
    visited.append(node_name)
    print(f"Stack: {[(n, k) for n, k in stack]}")
    print(f"Processing node: {node_name} ({G.nodes[node_name]['op_type']})")

    if len(list(G.successors(node_name))) == 0:
        # If we reach the end node, we can stop
        return G
    elif G.nodes[node_name]["op_type"] == "Split":
        # If the start node is a Split, we need to add it to the stack
        num_children = len(list(G.successors(node_name)))
        stack.append((node_name, num_children))
        print(f"Added {G.nodes[node_name]['op_type']} node to stack")
        print(f"Stack now: {[(n, k) for n, k in stack]}")
        queue.append([])
        branching_depth += 1
    elif G.nodes[node_name]["op_type"] in ["Concat", "Aggregation Add(4)", "Aggregation Add(8)"]:
        print(f"Found Concat node: {node_name}, num_parents={len(list(G.predecessors(node_name)))}, num_children={len(list(G.successors(node_name)))}")
        if len(stack) > 0 and len(list(G.predecessors(node_name))) == stack[-1][1]:
            split_node_name, num_children = stack.pop()
            queue.pop(-1)
            print(f"Popped {split_node_name, num_children} from the stack")
            print(f"Reducing branching module between {split_node_name} ({G.nodes[split_node_name]['op_type']}) and {node_name} ({G.nodes[node_name]['op_type']})")
            G = reduce_subgraph(G, split_node_name, node_name)
            # plot_graph(G, figsize=(20, 20))
            branching_depth -= 1

    # add more nodes to the processing queue
    for child in G.successors(node_name):
        queue[branching_depth].insert(0, child)

    # call function recursively
    G = reduce_graph(G, queue, visited, stack)
    return G

# %%
import networkx as nx
from collections import defaultdict, deque

def find_split_nodes(G):
    return [n for n in G.nodes if G.nodes[n].get("op_type") == "Split"]

def is_concat_node(G, node):
    return G.nodes[node].get("op_type") == "Concat"

def build_concat_tree(G, split_outputs):
    visited = set()
    queue = deque(split_outputs)
    concat_parents = defaultdict(set)  # concat -> list of child tensors or concat nodes

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        print("queue", queue)
        print("visited", visited)
        print("current node", node)

        for succ in G.successors(node):
            if is_concat_node(G, succ):
                concat_parents[succ].update(set(G.predecessors(succ)))
            if len(queue) == 0 and len(list(G.successors(node))) == 1:
                print(f"Found final concat", node)
                break
            if succ not in queue:
                queue.append(succ)

    return concat_parents

def match_split_branches(concat_tree, split_node):
    """
    This function takes a concat tree and a split node, and rolls back the branch
    from each input to the concat tree to its corresponding child of the split node.
    """
    def get_branch_start(node):
        parents = list(G.predecessors(node))
        if parents and parents[0] == split_node:
            return node
        return get_branch_start(parents[0])
    matches = {}
    for node in concat_tree:
        for parent in G.predecessors(node):
            if parent not in concat_tree:
                branch_start = get_branch_start(parent)
                matches[parent] = branch_start
    return matches

def build_split_tree(node, concat_tree):
    children = concat_tree[node]
    if len(children) == 1:
        return build_split_tree(children[0])
    return {
        "name": node,
        "op_type": "Split" if G.nodes[node]["op_type"] == "Concat" else G.nodes[node]["op_type"],
        "children": [build_split_tree(child, concat_tree) for child in children]
    }

def insert_split_tree(G, original_split_node, split_tree, prefix="split_rewrite"):
    def create_split_nodes(tree, G=nx.DiGraph(), parent_input=None, level=0, index=0):
        node_name = f"{prefix}_L{level}_N{index}" if "Concat" in tree["name"] else tree["name"]
        G.add_node(node_name, op_type=tree["op_type"], attributes={})
        if parent_input:
            G.add_edge(parent_input, node_name)

        for child in tree["children"]:
            if child:
                create_split_nodes(child, G, node_name, level + 1, len(G.nodes))
        return G

    # Save input to the original Split
    preds = list(G.predecessors(original_split_node))
    if len(preds) != 1:
        raise ValueError(f"Split node {original_split_node} should have exactly one input.")
    split_input = preds[0]

    split_graph = create_split_nodes(split_tree)

    # Remove the original Split and its outputs
    # old_outputs = list(G.successors(original_split_node))
    # for o in old_outputs:
    #     if G.has_node(o):
    #         G.remove_node(o)
    # if G.has_node(original_split_node):
    #     G.remove_node(original_split_node)

    # Insert split_graph into G
    G.remove_node(original_split_node)
    G = nx.compose(G, split_graph)
    G.add_edge(split_input, f"{prefix}_L0_N0")
    return G

def fix_split_nodes(G, queue, visited=[], stack=[]):
    """
    This function takes a branch and an end node, and rolls out the branch
    until it reaches the end node.
    """
    # find next node to process
    print("Queue:")
    for branching_depth in range(len(queue)):
        print(f"\tDepth {branching_depth}: {queue[branching_depth]}")
    branching_depth = len(stack)
    print("Graph nodes", G.nodes)
    print("Visited", visited)
    node_name = queue[branching_depth].pop(-1)
    while node_name not in G.nodes or node_name in visited:
        node_name = queue[branching_depth].pop(-1)
    visited.append(node_name)
    print(f"Stack: {[(n, k) for n, k in stack]}")
    print(f"Processing node: {node_name} ({G.nodes[node_name]['op_type']})")

    if len(list(G.successors(node_name))) == 0:
        # If we reach the end node, we can stop
        return G
    elif G.nodes[node_name]["op_type"] == "Split":
        # If the start node is a Split, we need to add it to the stack
        num_children = len(list(G.successors(node_name)))
        stack.append((node_name, num_children))
        print(f"Added {G.nodes[node_name]['op_type']} node to stack")
        print(f"Stack now: {[(n, k) for n, k in stack]}")
        queue.append([])
        branching_depth += 1
    elif G.nodes[node_name]["op_type"] in ["Concat", "Aggregation Add(4)", "Aggregation Add(8)"]:
        print(f"Found Concat node: {node_name}, num_parents={len(list(G.predecessors(node_name)))}, num_children={len(list(G.successors(node_name)))}")
        if len(stack) > 0 and len(list(G.predecessors(node_name))) == stack[-1][1]:
            split_node_name, num_children = stack.pop()
            queue.pop(-1)
            branching_depth -= 1
        elif len(list(G.predecessors(node_name))) != stack[-1][1]:
            split_node_name, num_children = stack.pop()
            concat_tree = build_concat_tree(G, list(G.successors(split_node_name)))
            print("Concat Tree")
            for key, value in concat_tree.items():
                print(f"{key}: {value}")
            concat_root = [n for n in concat_tree if not any([n in vals for vals in concat_tree.values()])][0]
            print("Concat Root", concat_root)
            split_tree = build_split_tree(concat_root, concat_tree)
            print("Split Tree")
            for key, value in split_tree.items():
                print(f"{key}: {value}")
            G = insert_split_tree(G, split_node_name, split_tree)
            # call function recursively
            return fix_split_nodes(G, queue=[[list(G.nodes)[0]]], visited=[], stack=[])

    # add more nodes to the processing queue
    for child in G.successors(node_name):
        queue[branching_depth].insert(0, child)

    # call function recursively
    return fix_split_nodes(G, queue, visited, stack)

# %% [markdown]
# ### Time to test it on some einspace architectures!

# %%
# i = 45
for i in range(3, 4):
    nodes, onnx_graph, tree_string = load_onn_file(i)
    nodes = topological_sort(nodes)
    name_to_node, children, parents = get_node_helpers(nodes)
    G = convert_to_networkx(nodes, children)

    print(f"Tree encoding: {tree_string}")
    print(f"Graph connected: {nx.is_connected(G.to_undirected())}")
    plot_graph(G, figsize=(20, 20))
    print(f"Nodes in G: {G.size()}")

    # compress patterns
    for pattern_dict in pattern_dicts:
        replace_component(G, **pattern_dict)

    print("Below is the graph after pattern matching")
    plot_graph(G, figsize=(20, 20))
    print(f"Nodes in G: {G.size()}")

    # disagregate split nodes
    G = fix_split_nodes(G, queue=[[list(G.nodes)[0]]], visited=[], stack=[])

    print("Below is the graph after split disaggregation")
    plot_graph(G, figsize=(20, 20))
    print(f"Nodes in G: {G.size()}")

    # compress parallel branches
    G = reduce_graph(G, [[list(G.nodes)[0]]], visited=[], stack=[])

    print("Below is the graph after parallel branch reduction")
    plot_graph(G, figsize=(20, 20))
    print(f"Nodes in G: {G.size()}")

# %%



