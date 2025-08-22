import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import random
import time
import copy

from scipy.stats import skewnorm

from search_state import Operation, DerivationTreeNode
from grammars import einspace


class MatrixCell():
    # This class represents each position within a distance matrix and holds information regarding its distance to the starting position and path to get to it 
    def __init__(self):
        self.top = [] # list of costs of continuing each path of the above matrix position downwards
        self.left = [] # list of costs of continuing each path of the matrix to the left position rightwards
        self.corner = [] # list of costs of continuing each path of the matrix on the left-top corner position down-rightwards
        self.paths = [] # list of valid, most optimal paths to reach this position in the matrix
        self.value = np.nan # distance to the starting position

    def __str__(self):
        return "Value of "+str(self.value)+" with posible paths:\n"+"".join([str(path)+"\n" for path in self.paths])

    def __repr__(self):
        return str(self)

class MatrixOperation(object):
    # This class represents a single mutation to be performed to a model
    def __init__(self, op_id = None, op_type = None, node1_id = None, node2_id = None, i = None, j = None, ii = None, jj = None, value = 0, disabler_ops = [], enabler_ops = []):
        self.id = op_id # operation identifier
        self.op_type = op_type # operation type (can be "add", "rem" or "mut", followed by "_wrap" if dealing with a routing or branching module, and followed by "_sep" or "end" if dealing with the branch/routing separator tokens)
        self.i = i # corresponding matrix row (position within the first model) where the operation starts
        self.j = j # corresponding matrix column (position within the second model) where the operation starts
        self.ii = ii # corresponding matrix row (position within the first model) where the operation ends if it's a branching or routing wrapper, or list of rows for the branch separators in a branching(2)
        self.jj = jj # corresponding matrix column (position within the first model) where the operation ends if it's a branching or routing wrapper, or list of columns for the branch separators in a branching(2)
        self.node1_id = node1_id # id of the node corresponding to position i of the first model
        self.node2_id = node2_id # id of the node corresponding to position j of the second model
        self.value = value # cost of performing this operation
        self.i_swapped = False # wether or not we had to swap the order of the branches in the first model if it corresponds to a branching(2) node
        self.j_swapped = False # wether or not we had to swap the order of the branches in the second model if it corresponds to a branching(2) node
        self.disabler_ops = disabler_ops # list of operations that, if all are performed, do not allow this operation to be performed (for instance, removing all nodes that would be wrapped by a routing dissable the adition of said routing, as it would have nothing to wrap around)
        self.enabler_ops = enabler_ops # list of operations that, if any is performed, allow this operation to be performed even if all disabler operations are performed (in the above example, adding anoder node to be wrapped by the routing before adding the routing)
        
    def __str__(self):
        string = self.op_type+" (id: "+str(self.id)+") with a cost of "+str(self.value)+". "
        if len(self.disabler_ops):
            string = string[:-2]+" (disabled by "
            for branch in self.disabler_ops:
                if type(branch) != list: branch = [branch]
                for op in branch: string += str(op.id)+", "
            if string[-3:] == "by ": string += "none, "
            string = string[:-2]+")."
        if len(self.enabler_ops):
            string = string[:-2]+" ("*(len(self.disabler_ops)==0)+"; "*(len(self.disabler_ops)>0)+"enabled by "
            for branch in self.enabler_ops:
                if type(branch) != list: branch = [branch]
                for op in branch: string += str(op.id)+", "
            if string[-3:] == "by ": string += "none, "
            string = string[:-2]+")."
        return string
        
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.id == other.id
    
class DecoyOperation(object):
    # This is an empty class to hold an operation name as if it was a node's operation
    def __init__(self, name):
        self.name = name
    
class DecoyNode(object):
    # This is an empty class to hold an operation name and id as if it was a node, as well as its parent and branch number if it is an branch separator/end decoy node
    def __init__(self, parent, branch, name):
        self.parent = parent # node parent (branching/routing to close if it is a separator/end node)
        self.operation = DecoyOperation(name) # decoy operation
        self.children = [] # decoy nodes (start node and separator/end nodes) do not have children
        if parent is None: self.id = -1 # if it is a start node, its id will be -1
        else: self.id = self.parent.id # otherwise, it will inherit the id of the node it is separating/closing

    def is_root(self):
        return self.parent is not None
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __str__(self):
        return str(self.operation.name)+" (id "+str(self.id)+")"

    def __repr__(self):
        return str(self)

class AlignmentMatrix():
    
    def __init__(self, model1, model2, priorities = ("mut", "add", "rem"), collapse_corners = False, verbose = False):
        self.verbose = verbose
        self.collapse_corners = collapse_corners
        
        self.model1 = model1
        if self.verbose: print("First model: ", self.model1)
        self.model_ops1 = [DecoyNode(None, None, "start_node")] + self.breakdown(self.model1)
        
        self.model2 = model2
        self.new_node_id = max([node.id for node in self.model1.serialise()])+1
        for node in self.model2.serialise(): self.update_id(node) # We reset the models' node ids to avoid anything breaking when we combine the models because of id repetitions
        if self.verbose: print("Second model:", self.model2)
        self.model_ops2 = [DecoyNode(None, None, "start_node")] + self.breakdown(self.model2)
        
        self.operations = []
        self.nontrivial_ops = []
        
        timestart = time.time()
        self.matrix = self.calculate_matrix(self.model_ops1, self.model_ops2, corner_value = 0, corner_op = "start", block_idx=(0,0), priorities = priorities)
        self.compute_time = time.time()-timestart
        #self.trace_back()
        self.distance = self.matrix[-1][-1].value
        if self.verbose:
            print("\nDistance of", round(self.distance,2), "through", len(self.nontrivial_ops), "operations, calculated in" ,round((self.compute_time)*1000,2),"ms."), self.visualize_alignment_matrix()
    
    def breakdown(self, model):
        model_ops = []
        
        condition = model.operation
        if model.parent: condition = condition and ("computation" not in model.parent.operation.name) 
            
        if condition:
            if not ("sequential" in model.operation.name):
                model_ops = [model]
                
            if len(model.children) > 2: # If we have several children (branchings, routings...)
                for child in range(1, len(model.children)-1): # And then add all the children's operations to the list
                    model_ops += self.breakdown(model.children[child])
                    model_ops += [DecoyNode(model, child, "wrap_"+"end"*(child==len(model.children)-2)+"sep"*(child!=len(model.children)-2))]
                
            else:
                for child in model.children:
                    model_ops += self.breakdown(child)
    
        return model_ops
                
    def visualize_alignment_matrix(self, matrix = None, model_ops1 = None, model_ops2 = None, i = -1, j = -1, saveimg = False):
        if matrix == None: matrix = self.matrix
        if model_ops1 == None: model_ops1 = self.model_ops1
        if model_ops2 == None: model_ops2 = self.model_ops2
        
        size = (len(matrix), len(matrix[0]))
        m = np.zeros(size)
        for ii in range(size[0]):
            for jj in range(size[1]):
                m[ii,jj] = matrix[ii][jj].value
        plt.figure(figsize=(10,10))
        plt.imshow(m)
        ax = plt.gca()
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_yticks([x for x in range(len(model_ops1))])
        ax.set_yticklabels([self.get_op_name(op).replace("wrap", self.get_op_name(op.parent)) if "wrap" in self.get_op_name(op) else self.get_op_name(op) for op in model_ops1], rotation=0)
        ax.set_xticks([y for y in range(len(model_ops2))])
        ax.set_xticklabels([self.get_op_name(op).replace("wrap", self.get_op_name(op.parent)) if "wrap" in self.get_op_name(op) else self.get_op_name(op) for op in model_ops2], rotation=90)
        
        length = 1.7
        interval_space = 0.7
        linewidth = 2
        cmap = plt.get_cmap('ocean', len(matrix[i][j].paths))
        for p, path in enumerate(matrix[i][j].paths):
            position = [0, 0]
            for op in path[1:]:
                color = cmap(p)
                #plt.text(op.j-0.5, op.i+0.2, str(matrix[op.i][op.j].value), color="cyan", fontsize=12, rotation=45, rotation_mode='default')
                pos_i = np.linspace(position[0], op.i, 30)
                pos_j = np.linspace(position[1], op.j, 30)
                linestyle = "-"
                if abs(op.i-position[0]) > 1:
                    pos_j += np.linspace(-0.6, 0.6, 30)**2-0.6**2
                    linestyle = "--"
                if abs(op.j-position[1]) > 1:
                    pos_i += np.linspace(-0.6, 0.6, 30)**2-0.6**2
                    linestyle = "--"
                    
                plt.plot(pos_j, pos_i, linestyle = linestyle, color = color, linewidth=(0.75+op.value)*linewidth)
                position  = [op.i, op.j]
                
        if saveimg: plt.savefig("example_crossover.svg", format='svg')
        plt.show()

    def get_op_name(self, op):
        #if ("routing" in op.operation.name or "branching" in op.operation.name) and op.operation.name != "branching(2)": return op.operation.name.split("(")[0]
        if "computation" in op.operation.name: return "comp<"+op.children[0].operation.name+">"
        else: return op.operation.name

    def print_m(self, matrix):
        m = np.array([[row.value for row in column] for column in matrix])
        plt.imshow(m)
        plt.colorbar()
        ax = plt.gca()
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_yticks([x for x in range(len(self.model_ops1))])
        ax.set_yticklabels([self.get_op_name(op).replace("wrap", self.get_op_name(op.parent)) if "wrap" in self.get_op_name(op) else self.get_op_name(op) for op in self.model_ops1], rotation=0)
        ax.set_xticks([y for y in range(len(self.model_ops2))])
        ax.set_xticklabels([self.get_op_name(op).replace("wrap", self.get_op_name(op.parent)) if "wrap" in self.get_op_name(op) else self.get_op_name(op) for op in self.model_ops2], rotation=90)
        plt.show()
    
    def calculate_matrix(self, model_ops1, model_ops2, corner_value = 0, corner_op = "start", block_idx=(0,0), priorities = ("mut", "add", "rem")):
        priorities = sorted(range(len(priorities)), key=priorities.__getitem__)
        priorities = sorted(range(len(priorities)), key=priorities.__getitem__)
        
        matrix = []
        size = (len(model_ops1), len(model_ops2))
        for i in range(size[0]): # We initialize the whole matrix with empty cells
            row = []
            for j in range(size[1]):
                row.append(MatrixCell())
            matrix.append(row)

        for i in range(size[0]):
            matrix[i][0].left = [np.inf ]
            matrix[i][0].corner = [np.inf]
        for j in range(size[1]):
            matrix[0][j].top = [np.inf]
            matrix[0][j].corner = [np.inf] 
        matrix[0][0].value = 0
        matrix[0][0].paths = [[MatrixOperation(op_id = 0, op_type = "start", node1_id = self.model_ops1[0].id, node2_id = self.model_ops2[0].id, i = 0, j = 0)]]

        for i in range(size[0]):
            for j in range(size[1]):
                if matrix[i][j].top == []:
                    for path in matrix[i-1][j].paths:
                        if model_ops1[i].operation.name in ["wrap_end", "wrap_sep"]: # If we are trying to close a new branch,
                            op = 0
                            level = 0 # we keep track of how many branches/routings we go into or exit through the current path with this "level" tracker
                            closed_branches = 0 # and the branches that we close
                            while (model_ops1[i].id != path[op].node1_id) or (path[op].op_type not in ["add_wrap", "add_wrap_jump"]):
                                op -= 1
                                # We update the depth level and the number of branches we closed
                                if (model_ops1[i].id == path[op].node1_id) and ("add" in path[op].op_type): closed_branches += 1
                                if path[op].op_type in ["rem_wrap", "mut_wrap"]: level -= 1
                                elif path[op].op_type in ["rem_wrap_end", "mut_wrap_end"]: level += 1
                                if (level == -1) or (path[op].op_type == "start"): break # If we exit the outer branch/rout, we break this loop
                            # We check that we have closed all the branches we were meant to
                            branches_to_close = len(model_ops1[i].parent.children) - 2 - (model_ops1[i].operation.name=="wrap_sep")
                            if ("_jump" in path[op].op_type): branches_to_close = 3 - branches_to_close
                            if (level==0) and (branches_to_close == closed_branches): matrix[i][j].top += [matrix[i-1][j].value] # If the branch was added within this depth, we allow the ending of the branch
                            else: matrix[i][j].top += [np.inf]
                        else: matrix[i][j].top += [matrix[i-1][j].value + 1] # If we are not trying to close a branch, we simply sum the cost of adding whatever we're adding

                if matrix[i][j].left == []:
                    for path in matrix[i][j-1].paths:
                        if model_ops2[j].operation.name in ["wrap_end", "wrap_sep"]: # If we are trying to close a branch we removed,
                            op = 0
                            level = 0 # we keep track of how many branches/routings we go into or exit through the current path with this "level" tracker
                            closed_branches = 0 # and the branches that we close
                            while (model_ops2[j].id != path[op].node2_id) or (path[op].op_type not in ["rem_wrap", "rem_wrap_jump"]):
                                op -= 1
                                # We update the depth level and the number of branches we closed
                                if (model_ops2[j].id == path[op].node2_id) and ("rem" in path[op].op_type): closed_branches += 1
                                if path[op].op_type in ["add_wrap", "mut_wrap"]: level -= 1
                                elif path[op].op_type in ["add_wrap_end", "mut_wrap_end"]: level += 1
                                if (level == -1) or (path[op].op_type == "start"): break # If we exit the outer branch/rout, we break this loop
                            # We check that we have closed all the branches we were meant to
                            branches_to_close = len(model_ops2[j].parent.children) - 2 - (model_ops2[j].operation.name=="wrap_sep")
                            if ("_jump" in path[op].op_type): branches_to_close = 3 - branches_to_close
                            if (level==0) and (branches_to_close == closed_branches): matrix[i][j].left += [matrix[i][j-1].value] # If the branch was removed within this depth, we allow the ending of the branch
                            else: matrix[i][j].left += [np.inf]
                        else: matrix[i][j].left += [matrix[i][j-1].value + 1] # If we are not trying to close a branch, we simply sum the cost of removing whatever we're removing

                if matrix[i][j].corner == []:
                    mutcost = self.cost_mut(model_ops1[i], model_ops2[j])
                    for path in matrix[i-1][j-1].paths:
                        if (model_ops1[i].operation.name in ["wrap_end", "wrap_sep"]) and (model_ops2[j].operation.name == model_ops1[i].operation.name): # If we are trying to close a branch we mutated into another,
                            op = 0
                            closed_branches1 = 0 # We keep track of how many branches we close on either model
                            closed_branches2 = 0
                            while not ((model_ops1[i].id == path[op].node1_id) and (model_ops2[j].id == path[op].node2_id) and (path[op].op_type in ["mut_wrap", "mut_wrap_jump"])): # we simply have to look at where we added/removed/mutated both branches
                                op -= 1
                                if (model_ops1[i].id == path[op].node1_id) and ("mut" in path[op].op_type): closed_branches1 += 1
                                if (model_ops2[j].id == path[op].node2_id) and ("mut" in path[op].op_type): closed_branches2 += 1
                                if (path[op].op_type == "start"): break
                            # We check that we have closed all the branches we were meant to
                            branches_to_close1 = len(model_ops1[i].parent.children) - 2 - (model_ops1[i].operation.name=="wrap_sep")
                            if ("_jump" in path[op].op_type) and ("1" in path[op].op_type): branches_to_close1 = 3 - branches_to_close1
                            branches_to_close2 = len(model_ops2[j].parent.children) - 2 - (model_ops2[j].operation.name=="wrap_sep")
                            if ("_jump" in path[op].op_type) and ("2" in path[op].op_type): branches_to_close2 = 3 - branches_to_close2
                            if (branches_to_close1 == closed_branches1) and (branches_to_close2 == closed_branches2): matrix[i][j].corner += [matrix[i-1][j-1].value] # and if the branches were mutated, we allow the ending of the branch through mutation
                            else: matrix[i][j].corner += [np.inf]
                        else: matrix[i][j].corner += [matrix[i-1][j-1].value + mutcost] # If we are not trying to close a branch, we simply sum the cost of mutating whatever we're mutating
                    
                if np.isnan(matrix[i][j].value):
                    matrix[i][j].value = np.nanmin(matrix[i][j].top + matrix[i][j].corner + matrix[i][j].left) # We check which would be the cheapest path
                    
                    for idx, top in enumerate(matrix[i][j].top):
                        if top == matrix[i][j].value:
                            path = matrix[i-1][j].paths[idx]
                            if np.isinf(matrix[i][j].value): op_value = matrix[i][j].value
                            else: op_value = matrix[i][j].value-matrix[i-1][j].value
                            if self.model_ops1[i].operation.name == "wrap_end": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "add_wrap_end", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j)]]
                            elif self.model_ops1[i].operation.name == "wrap_sep": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "add_wrap_sep", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j)]]    
                            elif (len(self.model_ops1[i].children) >= 3): matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "add_wrap", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j, value = op_value)]]
                            else: matrix[i][j].paths += [path + [MatrixOperation(op_id = len(self.operations), op_type = "add_module", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j, value = op_value)]]
                
                    for idx, left in enumerate(matrix[i][j].left):
                        if left == matrix[i][j].value:
                            path = matrix[i][j-1].paths[idx]
                            if np.isinf(matrix[i][j].value): op_value = matrix[i][j].value
                            else: op_value = matrix[i][j].value-matrix[i][j-1].value
                            if self.model_ops2[j].operation.name == "wrap_end": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "rem_wrap_end", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j)]]
                            elif self.model_ops2[j].operation.name == "wrap_sep": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "rem_wrap_sep", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j)]]
                            elif (len(self.model_ops2[j].children) >= 3): matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "rem_wrap", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j, value = op_value)]]
                            else: matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "rem", node2_id = self.model_ops2[j].id, i = i, j = j, value = op_value)]]
                    
                    for idx, corner in enumerate(matrix[i][j].corner):
                        if corner == matrix[i][j].value:
                            path = matrix[i-1][j-1].paths[idx]
                            if np.isinf(matrix[i][j].value): op_value = matrix[i][j].value
                            else: op_value = matrix[i][j].value-matrix[i-1][j-1].value
                            if self.model_ops1[i].operation.name == "wrap_end": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "mut_wrap_end", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j)]]
                            elif self.model_ops1[i].operation.name == "wrap_sep": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "mut_wrap_sep", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j)]]
                            elif (len(self.model_ops2[j].children) >= 3): matrix[i][j].paths += [path + [MatrixOperation(op_id = len(self.operations), op_type = "mut_wrap", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j, value = op_value)]]
                            else: matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "mut", node1_id = self.model_ops1[i].id, node2_id = self.model_ops2[j].id, i = i, j = j, value = op_value)]]

                if (i>0) and (j>0): matrix[i-1][j-1].paths = []
                if self.collapse_corners and (((j-i) >= len(self.model_ops2)*0.25) or ((i-j) >= len(self.model_ops1)*0.25)): matrix[i][j].paths = [matrix[i][j].paths[0]]

        return matrix

    def cost_mut(self, op1, op2, max_cost = np.inf):
        if "wrap_" in op1.operation.name or "wrap_" in op2.operation.name:
            return max_cost
        elif op1.operation.name.split("(")[0] == op2.operation.name.split("(")[0]:
            if max_cost == 1: # This only happens when comparing submodules
                if op1.operation == op2.operation: return 0.
                else: return 0.5
            elif sum([op.operation.name == "branching(2)" for op in (op1,op2)]) == 1: return max_cost # We can't change a branching(2) into a branching(8) for instance
            elif len(op1.children)>2: return ((self.cost_mut(op1.children[0], op2.children[0], 1))+(self.cost_mut(op1.children[-1], op2.children[-1], 1)))/4 # If we substitute a branching/branching(2)/rounting module by another, we compare the subtype of module
            else: return (self.cost_mut(op1.children[0], op2.children[0], 1))/2 # If we substitute a computation module by another, we compare the subtype of module
        else:
            return max_cost
            
    def update_id(self, node):
        node.id = self.new_node_id
        self.new_node_id += 1

    def split_sequentials(self, original_node, split_id):
        if split_id not in [n.id for n in original_node.serialise()]: raise Exception("Provided id is not within provided sequential node")
        parent_node = original_node.parent
        if not original_node.is_root(): child_idx = parent_node.children.index(original_node)
        nodes_list = original_node.children

        sequential_in_list = True
        while sequential_in_list:
            sequential_in_list = False
            for n, node in enumerate(nodes_list):
                if node.operation.name == "sequential":
                    sequential_in_list = True
                    nodes_list = nodes_list[:n] + node.children + nodes_list[n+1:]
                    break

        for n, node in enumerate(nodes_list):
            if node.id == split_id: break
        
        list1 = nodes_list[:n]
        list2 = nodes_list[n:]
        
        for _ in range(len(list1)-1):
            child2 = list1.pop()
            child1 = list1.pop()
    
            list1 += [DerivationTreeNode(0,
                                         level=node.level,
                                         parent=node.parent,
                                         input_params=node.input_params,
                                         depth=node.depth,
                                         limiter=node.limiter,
                                         operation = Operation(name="sequential",
                                                               build=einspace.build_sequential_module,
                                                               infer=einspace.infer_sequential_module,
                                                               valid=einspace.valid_sequential_module,
                                                               inherit = [einspace.inherit_first_child,einspace.inherit_other_child],
                                                               give_back = [einspace.give_back_default,einspace.give_back_default],
                                                               type="nonterminal",
                                                               child_levels=["module","module"])
                                        )]
            
            self.update_id(list1[-1])
            list1[-1].children=[child1, child2]
            child1.parent = list1[-1]
            child2.parent = list1[-1]
        
        for _ in range(len(list2)-1):
            child2 = list2.pop()
            child1 = list2.pop()
    
            list2 += [DerivationTreeNode(0,
                                         level=node.level,
                                         parent=node.parent,
                                         input_params=node.input_params,
                                         depth=node.depth,
                                         limiter=node.limiter,
                                         operation = Operation(name="sequential",
                                                               build=einspace.build_sequential_module,
                                                               infer=einspace.infer_sequential_module,
                                                               valid=einspace.valid_sequential_module,
                                                               inherit = [einspace.inherit_first_child,einspace.inherit_other_child],
                                                               give_back = [einspace.give_back_default,einspace.give_back_default],
                                                               type="nonterminal",
                                                               child_levels=["module","module"])
                                        )]
            
            self.update_id(list2[-1])
            list2[-1].children=[child1, child2]
            child1.parent = list2[-1]
            child2.parent = list2[-1]

        if len(list2):
            resequentialized_node = DerivationTreeNode(0,
                                                       level=node.level,
                                                       parent=node.parent,
                                                       input_params=node.input_params,
                                                       depth=node.depth,
                                                       limiter=node.limiter,
                                                       operation = Operation(name="sequential",
                                                                             build=einspace.build_sequential_module,
                                                                             infer=einspace.infer_sequential_module,
                                                                             valid=einspace.valid_sequential_module,
                                                                             inherit = [einspace.inherit_first_child,einspace.inherit_other_child],
                                                                             give_back = [einspace.give_back_default,einspace.give_back_default],
                                                                             type="nonterminal",
                                                                             child_levels=["module","module"])
                                                      )
            self.update_id(resequentialized_node)
            resequentialized_node.children=[list1[0], list2[0]]
            list1[0].parent = resequentialized_node
            list2[0].parent = resequentialized_node
        else:
            raise Exception("Unable to resequentialize as requested")

        if not original_node.is_root(): parent_node.children[child_idx] = resequentialized_node
        resequentialized_node.parent = parent_node
        return resequentialized_node

    def trace_back(self, from_pos = "end", prioritize = []):
        if from_pos == "end": from_pos = (len(self.model_ops1)-1, len(self.model_ops2)-1)
        if self.verbose: print("\nOperations to change from model 1 to model 2:")
        self.operations = []
        self.distance = self.matrix[-1][-1].value
        i=from_pos[0]
        j=from_pos[1]
        while not (i == 0 and j == 0):
            op = None
            for priority_op in prioritize: # We check if the prioritized operations are available (in order of priority)
                if priority_op in self.matrix[i][j].operation:
                    op = priority_op
                    break
            if not op: # If there was no priority, we simply select a random one
                op = random.choice(self.matrix[i][j].operation)
            
            if op == "add":
                value = self.matrix[i][j].value-self.matrix[i-1][j].value
                if self.model_ops1[i].operation.name == "wrap_end": self.operations += [MatrixOperation(op_id = len(self.operations),
                                                                                          op_type = "add_wrap_end",
                                                                                          node1_id = self.model_ops1[i].id,
                                                                                          node2_id = self.model_ops2[j].id,
                                                                                          i = i,
                                                                                          j = j)]
                elif self.model_ops1[i].operation.name == "wrap_sep": self.operations += [MatrixOperation(op_id = len(self.operations),
                                                                                                  op_type = "add_wrap_sep",
                                                                                                  node1_id = self.model_ops1[i].id,
                                                                                                  node2_id = self.model_ops2[j].id,
                                                                                                  i = i,
                                                                                                  j = j)]
                    
                elif (len(self.model_ops1[i].children) == 4): # If we have are adding a branching(2) (that is, parallelizing some modules),
                    for sep_idx, sep_operation in enumerate(self.operations): # we look for the corresponding branch separator
                        if (sep_operation.op_type == "add_wrap_sep") and (sep_operation.node1_id == self.model_ops1[i].id): break
                    for end_idx, end_operation in enumerate(self.operations): # and the corresponding branch end
                        if (end_operation.op_type == "add_wrap_end") and (end_operation.node1_id == self.model_ops1[i].id): break

                    disabler_ops = []
                    enabler_ops = []
                    adds = [[],[]]
                    muts = [[],[]]
                    rems = [[],[]]
                    for inside_op in self.operations[sep_idx+1:]:
                        if inside_op.op_type == "add_module": adds[0] = adds[0] + [inside_op]
                        if inside_op.op_type == "mut": muts[0] = muts[0] + [inside_op]
                        if inside_op.op_type == "rem": rems[0] = rems[0] + [inside_op]
                    if not len(muts[0]):
                        for rem_op in rems[0]:
                            rem_op.enabler_ops = rem_op.enabler_ops + adds[0]
                            rem_op.disabler_ops = rem_op.disabler_ops + [rem_op2 for rem_op2 in rems[0] if rem_op2 != rem_op]
                        disabler_ops = disabler_ops + [rems[0]]
                        enabler_ops = enabler_ops + [adds[0]*len(rems[0])]
                    for inside_op in self.operations[end_idx+1:sep_idx]:
                        if inside_op.op_type == "add_module": adds[1] = adds[1] + [inside_op]
                        if inside_op.op_type == "mut": muts[1] = muts[1] + [inside_op]
                        if inside_op.op_type == "rem": rems[1] = rems[1] + [inside_op]
                    if not len(muts[1]):
                        for rem_op in rems[1]:
                            rem_op.enabler_ops = rem_op.enabler_ops + adds[1]
                            rem_op.disabler_ops = rem_op.disabler_ops + [rem_op2 for rem_op2 in rems[1] if rem_op2 != rem_op]
                        disabler_ops = disabler_ops + [rems[1]]
                        enabler_ops = enabler_ops + [adds[1]*len(rems[1])]
                    
                    self.operations += [MatrixOperation(op_id = len(self.operations),
                                                  op_type = "add_branch2",
                                                  node1_id = self.model_ops1[i].id,
                                                  node2_id = self.model_ops2[j].id,
                                                  i = i,
                                                  j = j,
                                                  ii = (sep_operation.i,end_operation.i),
                                                  jj = (sep_operation.j,end_operation.j),
                                                  value = value,
                                                  disabler_ops = disabler_ops,
                                                  enabler_ops = enabler_ops)]
                    
                    if (len(adds[0]) and (not len(rems[0])) and (not len(muts[0]))) or (len(adds[1]) and (not len(rems[1])) and (not len(muts[1]))): # If we don't remove anything from any branch, and we have to add everything that's inside,
                        self.operations[-1].disabler_ops = self.operations[-1].disabler_ops + [[self.operations[-1]],[self.operations[-1]]] # the operation becomes its own disabler for both branches,
                        self.operations[-1].enabler_ops = self.operations[-1].enabler_ops + adds # only enabled by adding anything inside each branch first
                    if value and self.verbose: print(f"\t(+{value}) Parallelize using {self.get_op_name(self.model_ops1[i])} (id: {self.model_ops1[i+1].id}) from indexes {(i,j)} to {(end_operation.i,end_operation.j)}")
                
                elif (len(self.model_ops1[i].children) == 3): # If we have some of those "group-M-cat" or "rout-M-rout" situations,
                    for end_idx, end_operation in enumerate(self.operations): # we look for the corresponding branch end
                        if (end_operation.op_type == "add_wrap_end") and (end_operation.node1_id == self.model_ops1[i].id): break
                    
                    disabler_ops = []
                    enabler_ops = []
                    adds = []
                    muts = []
                    rems = []
                    for inside_op in self.operations[end_idx+1:]:
                        if inside_op.op_type == "add_module": adds += [inside_op]
                        if inside_op.op_type == "mut": muts += [inside_op]
                        if inside_op.op_type == "rem": rems += [inside_op]
                    if not len(muts):
                        for rem_op in rems:
                            rem_op.enabler_ops = rem_op.enabler_ops + adds
                            rem_op.disabler_ops = rem_op.disabler_ops + [rem_op2 for rem_op2 in rems if rem_op2 != rem_op]
                        disabler_ops = disabler_ops + rems
                        enabler_ops = enabler_ops + adds*len(rems)
                    
                    self.operations += [MatrixOperation(op_id = len(self.operations),
                                                  op_type = "add_wrap",
                                                  node1_id = self.model_ops1[i].id,
                                                  node2_id = self.model_ops2[j].id,
                                                  i = i,
                                                  j = j,
                                                  ii = end_operation.i,
                                                  jj = end_operation.j,
                                                  value = value,
                                                  disabler_ops = disabler_ops,
                                                  enabler_ops = enabler_ops)]

                    if len(adds) and (not len(rems)) and (not len(muts)): # If we don't remove anything, and we have to add everything that's inside,
                        self.operations[-1].disabler_ops = self.operations[-1].disabler_ops + [self.operations[-1]] # the operation becomes its own disabler,
                        self.operations[-1].enabler_ops = self.operations[-1].enabler_ops + adds # only enabled by adding anything inside the wrapper first
                    if value and self.verbose: print(f"\t(+{value}) Add wrapper {self.get_op_name(self.model_ops1[i])} (id: {self.model_ops1[i+1].id}) from indexes {(i,j)} to {(end_operation.i,end_operation.j)}")
                else:
                    self.operations += [MatrixOperation(op_id = len(self.operations),
                                                      op_type = "add_module",
                                                      node1_id = self.model_ops1[i].id,
                                                      node2_id = self.model_ops2[j].id,
                                                      i = i,
                                                      j = j,
                                                      value = value)]
                    if value and self.verbose: print(f"\t(+{value}) Add {self.get_op_name(self.model_ops1[i])} (id: {self.model_ops1[i].id}) at indexes {(i,j)}")
                i -= 1

            elif op == "rem":
                if self.model_ops2[j].operation.name == "wrap_end": self.operations += [MatrixOperation(op_id = len(self.operations),
                                                                                          op_type = "rem_wrap_end",
                                                                                          node1_id = self.model_ops1[i].id,
                                                                                          node2_id = self.model_ops2[j].id,
                                                                                          i = i,
                                                                                          j = j)]
                elif self.model_ops2[j].operation.name == "wrap_sep": self.operations += [MatrixOperation(op_id = len(self.operations),
                                                                                                  op_type = "rem_wrap_sep",
                                                                                                  node1_id = self.model_ops1[i].id,
                                                                                                  node2_id = self.model_ops2[j].id,
                                                                                                  i = i,
                                                                                                  j = j)]
                else:
                    value = self.matrix[i][j].value-self.matrix[i][j-1].value
                    self.operations += [MatrixOperation(op_id = len(self.operations),
                                                  op_type = "rem",
                                                  node2_id = self.model_ops2[j].id,
                                                  i = i,
                                                  j = j,
                                                  value = value)]
                    
                    if (len(self.model_ops2[j].children) == 3): # If we have some of those "group-M-cat" or "rout-M-rout" situations,
                        for end_idx, end_operation in enumerate(self.operations):
                            if (end_operation.op_type == "rem_wrap_end") and (end_operation.node2_id == self.model_ops2[j].id): break # we look for the corresponding branch end,
                        adds = []
                        muts = []
                        rems = []
                        for inside_op in self.operations[end_idx+1:-1]: # check the operations we perform and,
                            if inside_op.op_type == "add_module": adds += [inside_op]
                            if inside_op.op_type == "mut": muts += [inside_op]
                            if inside_op.op_type == "rem": rems += [inside_op]
                        if not len(muts): # if we are not forced to have modules inside regardless of the operations we perform,
                            for rem_op in rems:
                                rem_op.disabler_ops = rem_op.disabler_ops + rems # we forbid removing all layers inside the wrap
                                rem_op.enabler_ops = rem_op.enabler_ops + adds + [self.operations[-1]] # unless we add any layer, or remove the wrap itself
                    
                    elif (len(self.model_ops2[j].children) == 4): # If we have are removing a branching(2) (that is, serializing some modules),
                        for sep_idx, sep_operation in enumerate(self.operations): # we look for the corresponding branch separator
                            if (sep_operation.op_type == "rem_wrap_sep") and (sep_operation.node2_id == self.model_ops2[j].id): break
                        for end_idx, end_operation in enumerate(self.operations): # and the corresponding branch end
                            if (end_operation.op_type == "rem_wrap_end") and (end_operation.node2_id == self.model_ops2[j].id): break
                        adds = [[],[]]
                        muts = [[],[]]
                        rems = [[],[]] # and, for each branch
                        for inside_op in self.operations[sep_idx+1:-1]:
                            if inside_op.op_type == "add_module": adds[0] += [inside_op]
                            if inside_op.op_type == "mut": muts[0] += [inside_op]
                            if inside_op.op_type == "rem": rems[0] += [inside_op]
                        if not len(muts[0]): # if we are not forced to have modules inside the branch regardless of the operations we perform,
                            for rem_op in rems[0]:
                                rem_op.disabler_ops = rem_op.disabler_ops + rems[0] # we forbid removing all layers inside the branch
                                rem_op.enabler_ops = rem_op.enabler_ops + adds[0] + [self.operations[-1]] # unless we add any layer, or remove the wrap itself
                        for inside_op in self.operations[end_idx+1:sep_idx]:
                            if inside_op.op_type == "add_module": adds[1] += [inside_op]
                            if inside_op.op_type == "mut": muts[1] += [inside_op]
                            if inside_op.op_type == "rem": rems[1] += [inside_op]
                        if not len(muts[1]):
                            for rem_op in rems[1]:
                                rem_op.disabler_ops = rem_op.disabler_ops + rems[1]
                                rem_op.enabler_ops = rem_op.enabler_ops + adds[1] + [self.operations[-1]]
                        
                    if value and self.verbose: print(f"\t(+{value}) Remove {self.get_op_name(self.model_ops2[j])} (id: {self.model_ops2[j].id}) at indexes {(i,j)}")
                j -=1

            elif op == "mut":
                if self.model_ops1[i].operation.name == "wrap_end": self.operations += [MatrixOperation(op_id = len(self.operations),
                                                                                          op_type = "mut_wrap_end",
                                                                                          node1_id = self.model_ops1[i].id,
                                                                                          node2_id = self.model_ops2[j].id,
                                                                                          i = i,
                                                                                          j = j)]
                elif self.model_ops1[i].operation.name == "wrap_sep": self.operations += [MatrixOperation(op_id = len(self.operations),
                                                                                          op_type = "mut_wrap_sep",
                                                                                          node1_id = self.model_ops1[i].id,
                                                                                          node2_id = self.model_ops2[j].id,
                                                                                          i = i,
                                                                                          j = j)]
                else:
                    value = self.matrix[i][j].value-self.matrix[i-1][j-1].value
                    self.operations += [MatrixOperation(op_id = len(self.operations),
                                                  op_type = "mut",
                                                  node1_id = self.model_ops1[i].id,
                                                  node2_id = self.model_ops2[j].id,
                                                  i = i,
                                                  j = j,
                                                  value = value)]
                    if value and self.verbose: print(f"\t(+{value}) Substitute {self.get_op_name(self.model_ops2[j])} (id: {self.model_ops2[j].id}) by {self.get_op_name(self.model_ops1[i])} (id: {self.model_ops1[i].id}) at indexes {(i,j)}")
                i -= 1
                j -= 1
            
        self.nontrivial_ops = [operation for operation in self.operations if operation.value]
    
    def generate_offspring(self, selected_ops = None):
        if selected_ops == None: selected_ops = self.nontrivial_ops
        if self.verbose: print(">>>Parent model 1\n",colored(self.model2, "red"), "\n>>>Parent model 2\n",colored(self.model1, "green"),"\n")
        
        self.performed_ops = []
        offspring = self.apply_all_operations(selected_ops, copy.deepcopy(self.model2))
        
        if self.verbose: print(">>>Final model\n", colored(offspring, "yellow"))
        return offspring

    def apply_all_operations(self, selected_ops, offspring):
        for op in selected_ops:
            if len(op.enabler_ops): # If we have operations to perform beforehand, we perform them
                for branch in op.enabler_ops:
                    if type(branch) != list: branch = [branch]
                    for op in branch: offspring = self.apply_all_operations([r_op for r_op in branch if r_op in selected_ops], offspring)
            offspring = self.apply_op(op, offspring)
        return offspring

    def apply_op(self, op, offspring):
        if op.id not in self.performed_ops:
            self.performed_ops += [op.id]
            if op.op_type == "mut":
                for node in self.model_ops1:
                    if node.id == op.node1_id:
                        node1 = copy.deepcopy(node)
                        break

                for node in offspring.serialise():
                    if (node.id == op.node2_id) or (node.id == op.node1_id):
                        node2 = copy.deepcopy(node)
                        break
                if (not node1.id == op.node1_id) or ((not node.id == op.node2_id) and (not node.id == op.node1_id)): raise Exception("Nodes to mutate not found")
                
                node1str = str(node1)
                if ("branching" in node1.operation.name): node1str = node1str.split(")")[0]+")"+node1str.split(")")[1]+")...}"
                if ("routing" in node1.operation.name): node1str = node1str.split(")")[0]+")...]"
                node2str = str(node2)
                if ("branching" in node2.operation.name): node2str = node2str.split(")")[0]+")"+node2str.split(")")[1]+")...}"
                if ("routing" in node2.operation.name): node2str = node2str.split(")")[0]+")...]"
                if self.verbose: print(">>>Mutating",colored(node2str, "red"),"into", colored(node1str, "green"))
                
                node1.parent = node2.parent
                if not node2.is_root(): node2.parent.children[node2.parent.children.index(node2)] = node1
                
                if ("branching" in node1.operation.name) or ("routing" in node1.operation.name):
                    node1.children[1] = node2.children[1]
                    node1.children[1].parent = node1
                    if ("branching(2)" in node1.operation.name):
                        node1.children[2] = node2.children[2]
                        node1.children[2].parent = node1
                
                offspring = node1.get_root()
                                    
            elif op.op_type == "rem":
                for node in offspring.serialise():
                    if node.id == op.node2_id:
                        break
                if (not node.id == op.node2_id): print(">>>Tried to remove module with id", colored(op.node2_id, "red"), "but it was not found.")
                else:
                    if "branching(2)" in node.operation.name:
                        b1 = node.children[1]
                        b2 = node.children[2]
                        if self.verbose: print(">>>Serializing branches",colored(str(b1), "red"),"and",colored(str(b2), "red"))
                        
                        sequential_node = DerivationTreeNode(0,
                                                             level=node.level,
                                                             parent=node.parent,
                                                             input_params=node.input_params,
                                                             depth=node.depth,
                                                             limiter=node.limiter,
                                                             operation = Operation(name="sequential",
                                                                                   build=einspace.build_sequential_module,
                                                                                   infer=einspace.infer_sequential_module,
                                                                                   valid=einspace.valid_sequential_module,
                                                                                   inherit = [einspace.inherit_first_child,einspace.inherit_other_child],
                                                                                   give_back = [einspace.give_back_default,einspace.give_back_default],
                                                                                   type="nonterminal",
                                                                                   child_levels=["module","module"])
                                                        )
                        self.update_id(sequential_node)
                        sequential_node.children=[b1, b2]
                        b1.parent = sequential_node
                        b2.parent = sequential_node
                        if node.parent: node.parent.children[node.parent.children.index(node)] = sequential_node
                        offspring = sequential_node.get_root()
                        
                    else:
                        nodestr = str(node)
                        if ("branching" in node.operation.name): nodestr = nodestr.split(")")[0]+")"+nodestr.split(")")[1]+")...}"
                        if ("routing" in node.operation.name): nodestr = nodestr.split(")")[0]+")...]"
                        if self.verbose: print(">>>Removing", colored(nodestr, "red"))
                            
                        parent_node = node.parent
                        
                        if ("branching" in node.operation.name) or ("routing" in node.operation.name):
                            if parent_node: parent_node.children[parent_node.children.index(node)] = node.children[1]
                            node.children[1].parent = node.parent
                            offspring = node.children[1].get_root()
                        else:
                            sibling_node = parent_node.children[parent_node.children.index(node)-1]
                            if not parent_node.is_root():
                                sibling_node.parent = parent_node.parent
                                parent_node.parent.children[parent_node.parent.children.index(parent_node)] = sibling_node
                            else:
                                sibling_node.parent = None
                                
                            offspring = sibling_node.get_root()
                            
            elif op.op_type == "add_module":
                for node in self.model_ops1:
                    if node.id == op.node1_id:
                        node1 = copy.deepcopy(node)
                        break
                if (not node1.id == op.node1_id): raise Exception("Node",op.node1_id,"not found from model 1 when attempting module addition")
                if op.node2_id == -1:
                    node2 = offspring
                    add_at_the_end = False
                else:
                    last_node_pos = [0,0] # We look for the id of the node we want to add the module before
                    jjj = 1 # which will come from the second model if we parallelized the first branch
                    if op.j+jjj < len(self.model_ops2):
                        while self.model_ops2[op.j+jjj].id not in [n.id for n in offspring.serialise()]:
                            jjj += 1
                            if op.j+jjj == len(self.model_ops2): break
                    if op.j+jjj < len(self.model_ops2):
                        op_name2 = self.model_ops2[op.j+jjj].operation.name
                        while offspring.serialise()[last_node_pos[0]].id != self.model_ops2[op.j+jjj].id:
                            last_node_pos[0] = last_node_pos[0] + 1
                            if last_node_pos[0] == len(offspring.serialise()): break
                    else:
                        op_name2 = ""
                        last_node_pos[0] = len(offspring.serialise())
                    iii = 0 # or from the first model if we added the first branch
                    if op.i+iii < len(self.model_ops1):
                        while self.model_ops1[op.i+iii].id not in [n.id for n in offspring.serialise()]:
                            iii += 1
                            if op.i+iii == len(self.model_ops1): break
                    if op.i+iii < len(self.model_ops1):
                        op_name1 = self.model_ops1[op.i+iii].operation.name
                        while offspring.serialise()[last_node_pos[1]].id != self.model_ops1[op.i+iii].id:
                            last_node_pos[1] = last_node_pos[1] + 1
                            if last_node_pos[1] == len(offspring.serialise()): break
                    else:
                        op_name1 = ""
                        last_node_pos[1] = len(offspring.serialise())

                    if min(last_node_pos) == len(offspring.serialise()):
                        node2 = node1
                        node1 = offspring
                        add_at_the_end = True
                    
                    else:
                        pos_scores = last_node_pos.copy()
                        if (pos_scores[0] < len(offspring.serialise())) and ("wrap_" in op_name2): pos_scores[0] = pos_scores[0] + num_of_children(offspring.serialise()[last_node_pos[0]])
                        if (pos_scores[1] < len(offspring.serialise())) and ("wrap_" in op_name1): pos_scores[1] = pos_scores[1] + num_of_children(offspring.serialise()[last_node_pos[1]])
                        chosen_pos = pos_scores.index(min(pos_scores))
                        if ["wrap_" in op_name2, "wrap_" in op_name1][chosen_pos]:
                            node2 = node1
                            node1 = offspring.serialise()[last_node_pos[chosen_pos]].children[-2]
                            add_at_the_end = True
                        else:
                            node2 = offspring.serialise()[last_node_pos[chosen_pos]]
                            add_at_the_end = False
                
                sequential_node = DerivationTreeNode(0,
                                                     level=node.level,
                                                     parent=None,
                                                     input_params=node.input_params,
                                                     depth=node2.depth,
                                                     limiter=node.limiter,
                                                     operation = Operation(name="sequential",
                                                                           build=einspace.build_sequential_module,
                                                                           infer=einspace.infer_sequential_module,
                                                                           valid=einspace.valid_sequential_module,
                                                                           inherit = [einspace.inherit_first_child,einspace.inherit_other_child],
                                                                           give_back = [einspace.give_back_default,einspace.give_back_default],
                                                                           type="nonterminal",
                                                                           child_levels=["module","module"])
                                                )

                self.update_id(sequential_node)
                
                if add_at_the_end:
                    sequential_node.parent = node1.parent
                    if not node1.is_root():
                        node1.parent.children[node1.parent.children.index(node1)] = sequential_node
                else:
                    sequential_node.parent = node2.parent
                    if not node2.is_root():
                        node2.parent.children[node2.parent.children.index(node2)] = sequential_node
                    
                sequential_node.children=[node1, node2]
                node1.parent = sequential_node
                node2.parent = sequential_node
                
                offspring = sequential_node.get_root()
                
                if self.verbose:
                    if add_at_the_end: print(">>>Adding", colored(str(node2), "green"), "after", colored(str(node1), "red"))
                    else: print(">>>Adding", colored(str(node1), "green"), "before", colored(str(node2), "red"))
                
            elif op.op_type == "add_wrap":
                for node in self.model_ops1:
                    if node.id == op.node1_id:
                        node1 = copy.deepcopy(node)
                        break
                
                split_pos = [0,0] # We look for the id of the node we want to start the wrap at
                jjj = 1 # which will come from the second model if we parallelized the first branch
                if op.j+jjj < len(self.model_ops2):
                    while ("wrap_" in self.model_ops2[op.j+jjj].operation.name) or (self.model_ops2[op.j+jjj].id not in [n.id for n in offspring.serialise()]):
                        jjj += 1
                        if op.j+jjj == len(self.model_ops2): break
                if op.j+jjj < len(self.model_ops2):
                    while offspring.serialise()[split_pos[0]].id != self.model_ops2[op.j+jjj].id:
                        split_pos[0] = split_pos[0] + 1
                        if split_pos[0] == len(offspring.serialise()): break
                else: split_pos[0] = len(offspring.serialise())
                iii = 0 # or from the first model if we added the first branch
                if op.i+iii < len(self.model_ops1):
                    while ("wrap_" in self.model_ops1[op.i+iii].operation.name) or (self.model_ops1[op.i+iii].id not in [n.id for n in offspring.serialise()]):
                        iii += 1
                        if op.i+iii == len(self.model_ops1): break
                if op.i+iii < len(self.model_ops1):
                    while offspring.serialise()[split_pos[1]].id != self.model_ops1[op.i+iii].id:
                        split_pos[1] = split_pos[1] + 1
                        if split_pos[1] == len(offspring.serialise()): break
                else: split_pos[1] = len(offspring.serialise())
                
                node = offspring.serialise()[min(split_pos)]
                # If the previous layer we found is the start of a branch/rout coming from the first model, we are interested on its children
                if (len(node.children) == 3) and (min(split_pos) == split_pos[0]): node = node.children[1]
                starting_node = node
                split_pos = [0,0] # We look for the id of the next node after we end the wrap
                jjj = 0 # which will come from the second model if we didn't have to add the inside modules
                if op.jj+jjj < len(self.model_ops2):
                    op_name2 = self.model_ops2[op.jj+jjj].operation.name
                    while self.model_ops2[op.jj+jjj].id not in [n.id for n in offspring.serialise()]:
                        jjj += 1
                        if op.jj+jjj == len(self.model_ops2): break
                if op.jj+jjj < len(self.model_ops2):
                    while offspring.serialise()[split_pos[0]].id != self.model_ops2[op.jj+jjj].id:
                        split_pos[0] = split_pos[0] + 1
                        if split_pos[0] == len(offspring.serialise()): break
                else:
                    op_name2 = ""
                    split_pos[0] = len(offspring.serialise())
                iii = 0 # or from the first model if we did
                if op.ii+iii < len(self.model_ops1):
                    while self.model_ops1[op.ii+iii].id not in [n.id for n in offspring.serialise()]:
                        iii += 1
                        if op.ii+iii == len(self.model_ops1): break
                if op.ii+iii < len(self.model_ops1):
                    op_name1 = self.model_ops1[op.ii+iii].operation.name
                    while offspring.serialise()[split_pos[1]].id != self.model_ops1[op.ii+iii].id:
                        split_pos[1] = split_pos[1] + 1
                        if split_pos[1] == len(offspring.serialise()): break
                else:
                    op_name1 = ""
                    split_pos[1] = len(offspring.serialise())
                
                if min(split_pos) == len(offspring.serialise()): # If we need to wrap up until the end of the model
                    split_pos = len(offspring.serialise())
                    found_parent_sequential = False
                    while not found_parent_sequential: # We need to find the sequential node that holds the modules up to the end of the model
                        if not node.is_root():
                            if (node.parent.operation.name == "sequential") and (offspring.serialise()[-1] not in node.serialise()): node = node.parent 
                            else: found_parent_sequential = True
                        else: found_parent_sequential = True
                else: # If we need to wrap up until a certain module
                    if split_pos[0] == len(offspring.serialise()): split_pos[0] = -1
                    if split_pos[1] == len(offspring.serialise()): split_pos[1] = -1
                    chosen_pos = split_pos.index(max(split_pos))
                    split_pos = split_pos[chosen_pos]
                    found_parent_sequential = False
                    while not found_parent_sequential: # We need to find the sequential node that holds up to the module we woould like to wrap
                        if not node.is_root():
                            if (node.parent.operation.name == "sequential") and (offspring.serialise()[split_pos] not in node.serialise()): node = node.parent 
                            else: found_parent_sequential = True
                        else: found_parent_sequential = True
                
                if (split_pos == len(offspring.serialise())) or ["wrap_" in op_name2, "wrap_" in op_name1][chosen_pos]: end_at_id = -1 # If we are wrapping until the end of the model nor a wrap_end,
                else: end_at_id = offspring.serialise()[split_pos].id # we save the id of the last node to split the sequentials at further on
                
                if node.operation.name == "sequential":
                    try:
                        node = self.split_sequentials(node, starting_node.id).children[1] # We resequentialize the modules to be able to fit the wrap where we want it to be
                    except: # If we couldn't split it, it means that the original sequential already started with the first module we are insterested in
                        node = node
                else: node = node
                if node.operation.name == "sequential": # If  we are wrapping a sequential module
                    if end_at_id != -1: # and we have a valid end id,
                        try: node2 = self.split_sequentials(node, end_at_id).children[0] # we resequentialize the modules to be able to fit the wrap where we want it to be
                        except: node2 = node
                    else: node2 = node
                else: node2 = node

                if self.verbose: print(">>>Adding wrapper", colored(node1.operation.name, "green"), "around", colored(str(node2), "red"))
                node1.parent = node2.parent
                if not node2.is_root(): node2.parent.children[node2.parent.children.index(node2)] = node1
                node1.children[1] = node2
                node2.parent = node1
                offspring = node1.get_root()

            elif op.op_type == "add_branch2":
                for node in self.model_ops1:
                    if node.id == op.node1_id:
                        node1 = copy.deepcopy(node)
                        break

                split_pos = [0, 0] # We look for the id of the node we want to start the parallelization at
                jjj = 1 # which will come from the second model if we already had the first node from the first branch
                if op.j+jjj < len(self.model_ops2):
                    while ("wrap_" in self.model_ops2[op.j+jjj].operation.name) or (self.model_ops2[op.j+jjj].id not in [n.id for n in offspring.serialise()]):
                        jjj += 1
                        if op.j+jjj == len(self.model_ops2): break
                if op.j+jjj < len(self.model_ops2):
                    while offspring.serialise()[split_pos[0]].id != self.model_ops2[op.j+jjj].id:
                        split_pos[0] = split_pos[0] + 1
                        if split_pos[0] == len(offspring.serialise()): break
                else: split_pos[0] = len(offspring.serialise())
                
                iii = 0 # or from the first model if we had to add the first node from the first branch
                if op.i+iii < len(self.model_ops1):
                    while ("wrap_" in self.model_ops1[op.i+iii].operation.name) or (self.model_ops1[op.i+iii].id not in [n.id for n in offspring.serialise()]):
                        iii += 1
                        if op.i+iii == len(self.model_ops1): break
                if op.i+iii < len(self.model_ops1):
                    while offspring.serialise()[split_pos[1]].id != self.model_ops1[op.i+iii].id:
                        split_pos[1] += 1
                        if split_pos[1] == len(offspring.serialise()): break
                else: split_pos[1] = len(offspring.serialise())

                node = offspring.serialise()[min(split_pos)]
                starting_node = node

                split_pos = [0,0] # We look for the id of the next node after we end the second branch
                jjj = 1 # which will come from the second model if we already had it in the offspring
                if op.jj[1]+jjj < len(self.model_ops2):
                    op_name2 = self.model_ops2[op.jj[1]+jjj].operation.name
                    while self.model_ops2[op.jj[1]+jjj].id not in [n.id for n in offspring.serialise()]:
                        jjj += 1
                        if op.jj[1]+jjj == len(self.model_ops2): break
                if op.jj[1]+jjj < len(self.model_ops2):
                    while offspring.serialise()[split_pos[0]].id != self.model_ops2[op.jj[1]+jjj].id:
                        split_pos[0] = split_pos[0] + 1
                        if split_pos[0] == len(offspring.serialise()): break
                else:
                    op_name2 = ""
                    split_pos[0] = len(offspring.serialise())
                    
                iii = 0 # or from the first model if we didn't
                if op.ii[1]+iii < len(self.model_ops1):
                    while self.model_ops1[op.ii[1]+iii].id not in [n.id for n in offspring.serialise()]:
                        iii += 1
                        if op.ii[1]+iii == len(self.model_ops1): break
                if op.ii[1]+iii < len(self.model_ops1):
                    op_name1 = self.model_ops1[op.ii[1]+iii].operation.name
                    while offspring.serialise()[split_pos[1]].id != self.model_ops1[op.ii[1]+iii].id:
                        split_pos[1] = split_pos[1] + 1
                        if split_pos[1] == len(offspring.serialise()): break
                else:
                    op_name1 = ""
                    split_pos[1] = len(offspring.serialise())

                if min(split_pos) == len(offspring.serialise()): # If we need to wrap up until the end of the model
                    split_pos = len(offspring.serialise())
                    found_parent_sequential = False
                    while not found_parent_sequential: # We need to find the sequential node that holds the modules up to the end of the model
                        if not node.is_root():
                            if (node.parent.operation.name == "sequential") and (offspring.serialise()[-1] not in node.serialise()): node = node.parent 
                            else: found_parent_sequential = True
                        else: found_parent_sequential = True
                else: # If we need to wrap up until a certain module
                    if split_pos[0] == len(offspring.serialise()): split_pos[0] = -1
                    if split_pos[1] == len(offspring.serialise()): split_pos[1] = -1
                    chosen_pos = split_pos.index(max(split_pos))
                    split_pos = split_pos[chosen_pos]
                    found_parent_sequential = False
                    while not found_parent_sequential: # We need to find the sequential node that holds up to the module we woould like to wrap
                        if not node.is_root():
                            if (node.parent.operation.name == "sequential") and (offspring.serialise()[split_pos] not in node.serialise()): node = node.parent 
                            else: found_parent_sequential = True
                        else: found_parent_sequential = True

                if (split_pos == len(offspring.serialise())) or ["wrap_" in op_name2, "wrap_" in op_name1][chosen_pos]: end_at_id = -1 # If we are wrapping until the end of the model nor a wrap_end,
                else: end_at_id = offspring.serialise()[split_pos].id # we save the id of the last node to split the sequentials at further on
                    
                if node.operation.name == "sequential":
                    try:
                        node = self.split_sequentials(node, starting_node.id).children[1] # We resequentialize the modules to be able to fit the branch(2) where we want it to be
                    except: # If we couldn't split it, it means that the original sequential already started with the first module we are insterested in
                        node = node
                else: node = node

                if node.operation.name == "sequential": # If  we are wrapping a sequential module
                    if end_at_id != -1: # and we have a valid end id,
                        try: node2 = self.split_sequentials(node, end_at_id).children[0] # we resequentialize the modules to be able to fit the wrap where we want it to be
                        except: node2 = node
                    else: node2 = node
                else: node2 = node

                split_pos = [0, 0] # We look for the id of the node we want to start the second branch at
                jjj = 1 # which will come from the second model if we already had the second branch
                if op.jj[0]+jjj < len(self.model_ops2):
                    while ("wrap_" in self.model_ops2[op.jj[0]+jjj].operation.name) or (self.model_ops2[op.jj[0]+jjj].id not in [n.id for n in node2.serialise()]):
                        jjj += 1
                        if op.jj[0]+jjj == len(self.model_ops2): break
                if op.jj[0]+jjj < len(self.model_ops2):
                    while node2.serialise()[split_pos[0]].id != self.model_ops2[op.jj[0]+jjj].id:
                        split_pos[0] = split_pos[0] + 1
                        if split_pos[0] == len(node2.serialise()): break
                else: split_pos[0] = len(node2.serialise())
                
                iii = 0 # or from the first model if we added the second branch
                if op.ii[0]+iii < len(self.model_ops1):
                    while ("wrap_" in self.model_ops1[op.ii[0]+iii].operation.name) or (self.model_ops1[op.ii[0]+iii].id not in [n.id for n in node2.serialise()]):
                        iii += 1
                        if op.ii[0]+iii == len(self.model_ops1): break
                if op.ii[0]+iii < len(self.model_ops1):
                    while node2.serialise()[split_pos[1]].id != self.model_ops1[op.ii[0]+iii].id:
                        split_pos[1] += 1
                        if split_pos[1] == len(node2.serialise()): break
                else: split_pos[1] = len(node2.serialise())

                if (node2.operation.name == "sequential") and (min(split_pos) < len(node2.serialise())):
                    node2 = self.split_sequentials(node2, node2.serialise()[min(split_pos)].id) # We resequentialize the modules to be able to split the branches right where we want to
                else: node2 = node2

                if self.verbose: print(">>>Parallelizing modules", colored(str(node2.children[0]), "red"), "and", colored(str(node2.children[1]), "red"), "using", colored(node1.operation.name, "green"))

                parent_node = node2.parent
                if not node2.is_root(): parent_node.children[parent_node.children.index(node2)] = node1
                node1.parent = parent_node
                node1.children[1] = node2.children[0]
                node1.children[2] = node2.children[1]
                node2.children[0].parent = node1
                node2.children[1].parent = node1
                offspring = node1.get_root()
            
            if self.verbose and ("wrap_end" not in op.op_type) and ("wrap_sep" not in op.op_type): print("",colored(offspring, "light_grey"), "\n")
        return offspring


def num_of_children(node, n = 0):
    for child in node.children:
        n = n + 1 + num_of_children(child)
    return n


def select_operations(operations, skewness = 0):
    # positive skewness means sampling architectures closer to parent2
    # negative skewness means sampling architectures closer to parent1
    combinations = {}
    for i in range(2**len(operations)):
        combo_str = bin(i)[2:].zfill(len(operations))
        ops = [op for idx, op in enumerate(operations) if combo_str[idx] == "1"]
        value = sum([op.value for op in ops])
        for op in ops:
            # We take all enabler operations and separate them by branch (we add non-branch operations as if they were a branch)
            enablers = [op_en for op_en in op.enabler_ops if type(op_en) == list] + [[op_en for op_en in op.enabler_ops if type(op_en) != list]]
            # We do the same for the disabler operations
            disablers = [op_dis for op_dis in op.disabler_ops if type(op_dis) == list] + [[op_dis for op_dis in op.disabler_ops if type(op_dis) != list]]
            for b in range(len(enablers)):
                if len(disablers[b]) and all([disabler in ops for disabler in disablers[b]]) and (not any([enabler in ops for enabler in enablers[b]])): value = np.nan
        if not np.isnan(value): combinations[combo_str] = value

    sknorm = skewnorm(skewness)
    sample_resolution = 4
    sample_at = np.linspace(sknorm.ppf(0.01), sknorm.ppf(0.99), int(combinations[max(combinations)]*sample_resolution))
    samples = sknorm.pdf(sample_at)
    probs = [samples[int(combinations[c]*sample_resolution)-1] for c in combinations]
    probs /= np.sum(probs)
    
    selected = np.random.choice([c for c in combinations], p = probs)
    
    return [operations[i] for i, v in enumerate(selected) if v == "1"]


def constrained_smith_waterman_crossover(parent1, parent2, skewness=0):
    # build alignment matrix
    matrix = AlignmentMatrix(parent1, parent2, priorities=("mut", "add", "rem"), verbose=False)
    operations = matrix.nontrivial_ops
    if len(operations) == 0:
        return parent1, [], [], 0, 0, 0
    else:
        # sample random operations along the shortest path
        selected_ops = select_operations(operations, skewness=skewness)
        # perform the operations to generate the offspring
        child = matrix.generate_offspring(selected_ops)
        distance_between_parents = matrix.distance
        distance_to_parent2 = sum([op.value for op in selected_ops])
        distance_to_parent1 = distance_between_parents - distance_to_parent2
        # print("Distances:")
        # print(f"\tp1 -> p2 = {distance_between_parents}")
        # print(f"\tp1 -> c = {distance_to_parent1}")
        # print(f"\tp2 -> c = {distance_to_parent2}")
        # The triangle inequality doesn't seem to hold for these
        # since the following assertions fail
        # m = AlignmentMatrix(parent1, child, priorities=("mut", "add", "rem"), verbose=False)
        # assert distance_to_parent1 == sum([op.value for op in m.nontrivial_ops]), f"AssertionError: {distance_to_parent1} != {sum([op.value for op in m.nontrivial_ops])}"
        # m = AlignmentMatrix(parent2, child, priorities=("mut", "add", "rem"), verbose=False)
        # assert distance_to_parent2 == sum([op.value for op in m.nontrivial_ops]), f"AssertionError: {distance_to_parent2} != {sum([op.value for op in m.nontrivial_ops])}"

        # FIXME Check for dangling pointers
        child = copy.deepcopy(child)
        selected_ops = copy.deepcopy(selected_ops)
        operations = copy.deepcopy(operations)
        del matrix

        return child, selected_ops, operations, distance_to_parent1, distance_to_parent2, distance_between_parents


if __name__ == "__main__":
    import os, sys
    sys.path.append("../")

    from functools import partial
    from pprint import pprint
    from termcolor import colored

    import torch

    from tqdm import tqdm

    from search_strategies import create_search_strategy
    from pcfg import PCFG
    from grammars import grammars
    from evaluation import evaluation_fn
    from arguments import parse_arguments
    from data import get_data_loaders
    from utils import load_config, Limiter

    import sys
    sys.path.append(".")


    class ARGS():
        def __init__(self, config, device):
            self.config = config
            self.device = device
            self.acquisition_fn = "uct"
            self.exploration_weight=1.0
            self.incubent_type="parent"
            self.reward_mode="sum"
            self.regularised=True
            self.vis_interval = 10
            self.load_from = None
            self.generational = False
            self.add_full_paths = False
            
    args = ARGS("configs/einspace/evolution_one_point_crossover_sweep/language/evolution.yaml", "cpu")
    args = load_config(args)
    args.device = "cuda:5"
    # args.individual_mem_limit = 4096
    #args.time_limit = 10
    pprint(vars(args))

    # set the seed
    torch.manual_seed(1)

    # create the limiter
    # this makes sure that the search does not exceed
    # time, memory (GPU and RAM), depth, or node limits during the search
    limiter = Limiter(
        limits={
            "time": args.time_limit,
            "max_id": args.max_id_limit,
            "depth": args.depth_limit,
            "memory": args.mem_limit,
            "individual_memory": args.individual_mem_limit,
        }
    )

    # create the grammar
    grammar = PCFG(
        grammar=grammars[args.search_space],
        limiter=limiter,
    )
    #print(grammar)

    train_loader, val_loader, _, _ = get_data_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        root="../einspace/data",
        load_in_gpu=args.load_in_gpu,
        device=args.device,
        log=args.verbose_eval,
    )

    eval_fn = partial(
        evaluation_fn,
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # create the input parameters
    input_params = {
        "shape": torch.Size([1, args.channels, *args.image_size]),
        "other_shape": None,
        "mode": "im",
        "other_mode": None,
        "branching_factor": 1,
        "last_im_shape": None,
    }

    # create the search strategy
    search = create_search_strategy(args, grammar, eval_fn, limiter, input_params)

    # sampling first parent
    print("Sampling parent 1...")
    done = False
    while not done:
        try:
            search.limiter.timer.start()
            parent1 = search.evolver.sample(search.input_params)
            print(parent1)
            done = True
        except:
            continue
    # sampling second parent
    print("Sampling parent 2...")
    done = False
    while not done:
        try:
            search.limiter.timer.start()
            parent2 = search.evolver.sample(search.input_params)
            print(parent2)
            done = True
        except:
            continue
    # crossover
    print("Crossover...")
    start_time = time.time()
    offspring, operations = constrained_smith_waterman_crossover(parent1, parent2)
