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
    def __init__(self):
        self.top = []
        self.left = []
        self.corner = []
        self.paths = []
        self.value = np.nan

    def __str__(self):
        return "Value of "+str(self.value)+" with posible paths:\n"+"".join([str(path)+"\n" for path in self.paths])

    def __repr__(self):
        return str(self)

class MatrixOperation(object):
    def __init__(self, op_id = None, op_type = None, node1_id = None, node2_id = None, i = None, j = None, ii = None, jj = None, value = 0, disabler_ops = [], enabler_ops = []):
        self.id = op_id
        self.op_type = op_type
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.i = i
        self.j = j
        self.ii = ii
        self.jj = jj
        self.value = value
        self.i_swapped = False
        self.j_swapped = False
        self.disabler_ops = disabler_ops
        self.enabler_ops = enabler_ops
        
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
    def __init__(self, name):
        self.name = name
    
class Decoy(object):
    # This is an empty object to hold an operation name and id as if it was a node, as well as its parent and branch number if it is an end of branch decoy node
    def __init__(self, parent, branch, name):
        self.parent = parent
        self.branch = branch
        self.operation = DecoyOperation(name)
        self.children = []
        if parent is not None: self.id = self.parent.id
        else: self.id = -1

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

class AlignmentMatrixRecursive():
    def __init__(self, model1, model2, verbose = False, img_name = "example_matrix"):
        self.verbose = verbose
        self.img_name = img_name
        
        self.model1 = model1
        if self.verbose: print("First model: ", self.model1)
        self.model_ops1 = [Decoy(None, None, "start_node")] + self.breakdown(self.model1)
        
        self.model2 = model2
        self.new_node_id = max([node.id for node in self.model1.serialise()])+1
        for node in self.model2.serialise(): self.update_id(node) # We reset the models' node ids to avoid anything breaking when we combine the models because of id repetitions
        if self.verbose: print("Second model:", self.model2)
        self.model_ops2 = [Decoy(None, None, "start_node")] + self.breakdown(self.model2)
        
        self.size = (len(self.model_ops1), len(self.model_ops2))
        self.operations = []
        self.nontrivial_ops = []
        
        timestart = time.time()
        self.matrix = self.initialize_matrix()
        self.matrix = self.calculate_matrix()
        self.distance = self.matrix[-1][-1].value
        self.compute_time = time.time()-timestart
        self.calculate_restrictions()
        if self.verbose:
            print("\nDistance of", round(self.distance,2), "through", len(self.nontrivial_ops), "operations, calculated in" ,round((self.compute_time)*1000,2),"ms."), self.print_operations(), self.print_alignment_matrix()
    
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
                    model_ops += [Decoy(model, child, "wrap_"+"end"*(child==len(model.children)-2)+"sep"*(child!=len(model.children)-2))]
                
            else:
                for child in model.children:
                    model_ops += self.breakdown(child)
    
        return model_ops

    def initialize_matrix(self, model_ops1 = None, model_ops2 = None):
        matrix = []
        if (model_ops1 == None) and (model_ops2 == None): complete_matrix = True
        else: complete_matrix = False
        if model_ops1 == None: model_ops1 = self.model_ops1
        if model_ops2 == None: model_ops2 = self.model_ops2
        size = (len(model_ops1), len(model_ops2))
        for i in range(size[0]): # We initialize the whole matrix with empty cells
            row = []
            for j in range(size[1]):
                row.append(MatrixCell())
            matrix.append(row)

        if complete_matrix:
            for i in range(size[0]):
                matrix[i][0].left = [np.inf ]
                matrix[i][0].corner = [np.inf]
            for j in range(size[1]):
                matrix[0][j].top = [np.inf]
                matrix[0][j].corner = [np.inf] 
            matrix[0][0].value = 0
            matrix[0][0].paths = [[MatrixOperation(op_id = 0, op_type = "start", node1_id = model_ops1[0].id, node2_id = model_ops2[0].id, i = 0, j = 0)]]
        return matrix
        
    def calculate_matrix(self, matrix = None, model_ops1 = None, model_ops2 = None, start_i = 0, start_j = 0):
        if matrix == None: matrix = self.matrix
        matrix_iswap, matrix_jswap = False, False
        if model_ops1 == None: model_ops1 = self.model_ops1
        if model_ops2 == None: model_ops2 = self.model_ops2
        prev_i,prev_j = 0, 0
        
        while np.isnan(matrix[-1][-1].value):
            # We define the block that we will calculate next
            compute_submatrix = [False, False]
            if (model_ops1[prev_i].operation.name == "branching(2)"):
                compute_submatrix[0] = True
                for mid_i in range(prev_i+1, len(model_ops1)):
                    if (model_ops1[prev_i].id == model_ops1[mid_i].id) and ("sep" in model_ops1[mid_i].operation.name): break
                for max_i in range(mid_i+1, len(model_ops1)):
                    if (model_ops1[prev_i].id == model_ops1[max_i].id) and ("end" in model_ops1[max_i].operation.name): break
            else:
                for max_i in range(prev_i+1, len(model_ops1)):
                    if (model_ops1[max_i].operation.name == "branching(2)"): break
                        
            if (model_ops2[prev_j].operation.name == "branching(2)"):
                compute_submatrix[1] = True
                for mid_j in range(prev_j+1, len(model_ops2)):
                    if (model_ops2[prev_j].id == model_ops2[mid_j].id) and ("sep" in model_ops2[mid_j].operation.name): break
                for max_j in range(mid_j+1, len(model_ops2)):
                    if (model_ops2[prev_j].id == model_ops2[max_j].id) and ("end" in model_ops2[max_j].operation.name): break
            else:
                for max_j in range(prev_j+1, len(model_ops2)):
                    if (model_ops2[max_j].operation.name == "branching(2)"): break
            
            max_i += 1
            max_j += 1
            if (compute_submatrix[0] or compute_submatrix[1]) and ((prev_i, prev_j) != (0, 0)):
                if matrix_iswap == False: matrix_iswap = copy.deepcopy(matrix)
                if matrix_jswap == False: matrix_jswap = copy.deepcopy(matrix)
                    
                aux_model_ops1 = model_ops1[prev_i:max_i]
                aux_model_ops2 = model_ops2[prev_j:max_j]
                if compute_submatrix[0]: aux_model_ops1_swap = [model_ops1[prev_i]]+model_ops1[mid_i+1:max_i-1]+[model_ops1[mid_i]]+model_ops1[prev_i+1:mid_i]+[model_ops1[max_i-1]]
                if compute_submatrix[1]: aux_model_ops2_swap = [model_ops2[prev_j]]+model_ops2[mid_j+1:max_j-1]+[model_ops2[mid_j]]+model_ops2[prev_j+1:mid_j]+[model_ops2[max_j-1]]
                
                aux_matrix = self.calculate_matrix(matrix = [[row[j] for j in range(prev_j,max_j)] for row in matrix[prev_i:max_i]], model_ops1 = aux_model_ops1, model_ops2 = aux_model_ops2, start_i = prev_i, start_j = prev_j)
                
                if compute_submatrix[0]:
                    aux_matrix_iswap = self.calculate_matrix(matrix = [[row[j] for j in range(prev_j,max_j)] for row in matrix_iswap[prev_i:max_i]], model_ops1 = aux_model_ops1_swap, model_ops2 = aux_model_ops2, start_i = prev_i, start_j = prev_j)
                    # We fix the indices to account for the swap at the bottom end of the submatrix
                    #for j in range(len(aux_matrix_iswap[0])):
                    #    for path in aux_matrix_iswap[-1][j].paths:
                    #        for op in path:
                    #            if (op.i >= prev_i) and (op.i <= max_i-1): op.i_swapped = True
                
                if compute_submatrix[1]:
                    aux_matrix_jswap = self.calculate_matrix(matrix = [[row[j] for j in range(prev_j,max_j)] for row in matrix_jswap[prev_i:max_i]], model_ops1 = aux_model_ops1, model_ops2 = aux_model_ops2_swap, start_i = prev_i, start_j = prev_j)
                    # We fix the indices to account for the swap at the right end of the submatrix
                    #for i in range(len(aux_matrix_jswap)):
                    #    for path in aux_matrix_jswap[i][-1].paths:
                    #        for op in path:
                    #            if (op.j >= prev_j) and (op.j <= max_j-1): op.j_swapped = True
                    #print(prev_i,prev_j)
                    #self.print_alignment_matrix(matrix = aux_matrix, model_ops1 = aux_model_ops1, model_ops2 = aux_model_ops2, start_i = prev_i, start_j = prev_j)
                    #self.print_alignment_matrix(matrix = aux_matrix_jswap, model_ops1 = aux_model_ops1, model_ops2 = aux_model_ops2_swap, start_i = prev_i, start_j = prev_j)
                
                if compute_submatrix[0] and compute_submatrix[1]:
                    aux_matrix_ijswap = self.initialize_matrix(model_ops1 = aux_model_ops1_swap, model_ops2 = aux_model_ops2_swap)
                    aux_matrix_ijswap[0] = matrix_jswap[prev_i][prev_j:max_j]
                    for i, pos in enumerate([matrix_iswap[prev_i+i][prev_j] for i in range(len(aux_matrix_ijswap))]): aux_matrix_ijswap[i][0] = pos
                    aux_matrix_ijswap = self.calculate_matrix(matrix = aux_matrix_ijswap, model_ops1 = aux_model_ops1_swap, model_ops2 = aux_model_ops2_swap, start_i = prev_i, start_j = prev_j)
                    # We fix the indices to account for the swaps at the both ends of the submatrix, and replace the single swaps with double swaps when necessary
                    for i in range(len(aux_matrix_ijswap)):
                        if aux_matrix_ijswap[i][-1].value < aux_matrix_jswap[i][-1].value:
                            aux_matrix_jswap[i][-1] = aux_matrix_ijswap[i][-1]
                            for path in aux_matrix_ijswap[i][-1].paths:
                                for op in path:
                                    if (op.i >= prev_i) and (op.i <= max_i-1): op.i_swapped = True
                        elif i == 0:
                            for path in aux_matrix_jswap[0][0].paths: path[-1].i = prev_i
                    
                    for j in range(len(aux_matrix_ijswap[0])):
                        if aux_matrix_ijswap[-1][j].value < aux_matrix_iswap[-1][j].value:
                            aux_matrix_iswap[-1][j] = aux_matrix_ijswap[-1][j]
                            for path in aux_matrix_ijswap[-1][j].paths:
                                for op in path:
                                    if (op.j >= prev_j) and (op.j <= max_j-1): op.j_swapped = True
                        elif j == 0:
                            for path in aux_matrix_iswap[0][0].paths: path[-1].j = prev_j

                    # cccccccccccccccccccccccccc
                    #if aux_matrix[-1][-1].value == 4.125:
                    #if (prev_i == 1) and (prev_j==1):
                    #    print("Matrix", aux_matrix[-1][-1].value, "i", aux_matrix_iswap[-1][-1].value, "j", aux_matrix_jswap[-1][-1].value, "ij", aux_matrix_ijswap[-1][-1].value)
                for i in range(prev_i, max_i):
                    for j in range(prev_j, max_j):
                        matrix[i][j] = aux_matrix[i-prev_i][j-prev_j]
                        if compute_submatrix[0]: matrix_iswap[i][j] = aux_matrix_iswap[i-prev_i][j-prev_j]
                        else: matrix_iswap[i][j] = aux_matrix[i-prev_i][j-prev_j]
                        if compute_submatrix[1]: matrix_jswap[i][j] = aux_matrix_jswap[i-prev_i][j-prev_j]
                        else: matrix_jswap[i][j] = aux_matrix[i-prev_i][j-prev_j]
                if compute_submatrix[0] and (not compute_submatrix[1]):
                    for j in range(prev_j, max_j):
                        if matrix[max_i-1][j].value < matrix_iswap[max_i-1][j].value:
                            matrix_iswap[max_i-1][j] = matrix[max_i-1][j]
                            for path in matrix[max_i-1][j].paths: path[-1].i_swapped = False
                        elif matrix[max_i-1][j].value > matrix_iswap[max_i-1][j].value:
                            matrix[max_i-1][j] = matrix_iswap[max_i-1][j]
                            for path in matrix_iswap[max_i-1][j].paths: path[-1].i_swapped = True
                        #else:
                        #    matrix[max_i-1][j].paths += matrix_iswap[max_i-1][j].paths
                        #    matrix_iswap[max_i-1][j].paths = matrix[max_i-1][j].paths
                        matrix_jswap[max_i-1][j] = matrix[max_i-1][j]
                elif compute_submatrix[1] and (not compute_submatrix[0]):
                    for i in range(prev_i, max_i):
                        if matrix[i][max_j-1].value < matrix_jswap[i][max_j-1].value:
                            matrix[i][max_j-1] = matrix[i][max_j-1]
                            for path in matrix_jswap[i][max_j-1].paths: path[-1].j_swapped = False
                        elif matrix[i][max_j-1].value > matrix_jswap[i][max_j-1].value:
                            matrix[i][max_j-1] = matrix_jswap[i][max_j-1]
                            for path in matrix_jswap[i][max_j-1].paths: path[-1].j_swapped = True
                        #else:
                        #    matrix[i][max_j-1].paths += matrix_jswap[i][max_j-1].paths
                        #    matrix_jswap[i][max_j-1].paths = matrix[i][max_j-1].paths
                        matrix_iswap[i][max_j-1] = matrix[i][max_j-1]
                else:
                    if matrix[max_i-1][max_j-1].value < matrix_iswap[max_i-1][max_j-1].value:
                        matrix_iswap[max_i-1][max_j-1] = matrix[max_i-1][max_j-1]
                        for path in matrix[max_i-1][max_j-1].paths: path[-1].i_swapped = False
                    elif matrix[max_i-1][max_j-1].value > matrix_iswap[max_i-1][max_j-1].value:
                        matrix[max_i-1][max_j-1] = matrix_iswap[max_i-1][max_j-1]
                        for path in matrix_iswap[max_i-1][max_j-1].paths: path[-1].i_swapped = True
                    #else:
                    #    matrix[max_i-1][max_j-1].paths += matrix_iswap[max_i-1][max_j-1].paths
                    #    matrix_iswap[max_i-1][max_j-1].paths = matrix[max_i-1][max_j-1].paths
                    
                    if matrix[max_i-1][max_j-1].value < matrix_jswap[max_i-1][max_j-1].value:
                        matrix_jswap[max_i-1][max_j-1] = matrix[max_i-1][max_j-1]
                        for path in matrix[max_i-1][max_j-1].paths: path[-1].j_swapped = False
                    elif matrix[max_i-1][max_j-1].value > matrix_jswap[max_i-1][max_j-1].value:
                        matrix[max_i-1][max_j-1] = matrix_jswap[max_i-1][max_j-1]
                        for path in matrix_jswap[max_i-1][max_j-1].paths: path[-1].j_swapped = True
                    #else:
                    #    matrix[max_i-1][max_j-1].paths += matrix_jswap[max_i-1][max_j-1].paths
                    #    matrix_jswap[max_i-1][max_j-1].paths = matrix[max_i-1][max_j-1].paths
                    #if (prev_i == 1) and (prev_j==1):
                    #    print("Matrix", aux_matrix[-1][-1].value, "i", aux_matrix_iswap[-1][-1].value, "j", aux_matrix_jswap[-1][-1].value, "ij", aux_matrix_ijswap[-1][-1].value)
                    #    print("Matrix", aux_matrix[-1][-1].paths[0][-1].i_swapped, aux_matrix[-1][-1].paths[0][-1].j_swapped)
                    #    print("i", aux_matrix_iswap[-1][-1].paths[0][-1].i_swapped, aux_matrix_iswap[-1][-1].paths[0][-1].j_swapped)
                    #    print("j", aux_matrix_jswap[-1][-1].paths[0][-1].i_swapped, aux_matrix_jswap[-1][-1].paths[0][-1].j_swapped)
            else:
                #for i in range(prev_i, max_i):
                #    for j in range(prev_j, max_j):
                sequence = self.expanding_corner_loop(max_i-prev_i, max_j-prev_j)
                for position in sequence:
                    i = position[0] + prev_i
                    j = position[1] + prev_j
                    if True:
                        if np.isnan(matrix[i][j].value):
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
                                    else:
                                        matrix[i][j].left += [matrix[i][j-1].value + 1] # If we are not trying to close a branch, we simply sum the cost of removing whatever we're removing
            
                            if matrix[i][j].corner == []:
                                for path in matrix[i-1][j-1].paths:
                                    if (model_ops1[i].operation.name in ["wrap_end", "wrap_sep"]) and (model_ops2[j].operation.name == model_ops1[i].operation.name): # If we are trying to close a branch we mutated into another,
                                        op = 0
                                        closed_branches1 = 0 # We keep track of how many branches we close on either model
                                        closed_branches2 = 0
                                        
                                        while not (((model_ops1[i].id == path[op].node1_id) and (model_ops2[j].id == path[op].node2_id)) and (path[op].op_type == "mut_wrap")): # we simply have to look at where we added/removed/mutated both branches
                                            op -= 1
                                            if (model_ops1[i].id == path[op].node1_id) and ("mut" in path[op].op_type): closed_branches1 += 1
                                            if (model_ops2[j].id == path[op].node2_id) and ("mut" in path[op].op_type): closed_branches2 += 1
                                            if (path[op].op_type == "start"): break
                                        # We check that we have closed all the branches we were meant to
                                        branches_to_close1 = len(model_ops1[i].parent.children) - 2 - (model_ops1[i].operation.name=="wrap_sep")
                                        branches_to_close2 = len(model_ops2[j].parent.children) - 2 - (model_ops2[j].operation.name=="wrap_sep")
                                        if (branches_to_close1 == closed_branches1) and (branches_to_close2 == closed_branches2): matrix[i][j].corner += [matrix[i-1][j-1].value] # and if the branches were mutated, we allow the ending of the branch through mutation
                                        else: matrix[i][j].corner += [np.inf]
                                    else: matrix[i][j].corner += [matrix[i-1][j-1].value + self.cost_mut(model_ops1[i], model_ops2[j])] # If we are not trying to close a branch, we simply sum the cost of mutating whatever we're mutating

                            matrix[i][j].value = np.nanmin(matrix[i][j].top + matrix[i][j].corner + matrix[i][j].left) # We check which would be the cheapest path
                            for idx, top in enumerate(matrix[i][j].top):
                                if top == matrix[i][j].value:
                                    path = matrix[i-1][j].paths[idx]
                                    if np.isinf(matrix[i][j].value): op_value = matrix[i][j].value
                                    else: op_value = matrix[i][j].value-matrix[i-1][j].value
                                    if model_ops1[i].operation.name == "wrap_end": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "add_wrap_end", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j)]]
                                    elif model_ops1[i].operation.name == "wrap_sep": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "add_wrap_sep", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j)]]    
                                    elif (len(model_ops1[i].children) >= 3): matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "add_wrap", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j, value = op_value)]]
                                    else: matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "add_module", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j, value = op_value)]]
                        
                            for idx, left in enumerate(matrix[i][j].left):
                                if left == matrix[i][j].value:
                                    path = matrix[i][j-1].paths[idx]
                                    if np.isinf(matrix[i][j].value): op_value = matrix[i][j].value
                                    else: op_value = matrix[i][j].value-matrix[i][j-1].value
                                    if model_ops2[j].operation.name == "wrap_end": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "rem_wrap_end", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j)]]
                                    elif model_ops2[j].operation.name == "wrap_sep": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "rem_wrap_sep", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j)]]
                                    elif (len(model_ops2[j].children) >= 3): matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "rem_wrap", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j, value = op_value)]]
                                    else: matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "rem", node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j, value = op_value)]]
                                    
                            for idx, corner in enumerate(matrix[i][j].corner):
                                if corner == matrix[i][j].value:
                                    path = matrix[i-1][j-1].paths[idx]
                                    if np.isinf(matrix[i][j].value): op_value = matrix[i][j].value
                                    else: op_value = matrix[i][j].value-matrix[i-1][j-1].value
                                    if model_ops1[i].operation.name == "wrap_end": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "mut_wrap_end", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j)]]
                                    elif model_ops1[i].operation.name == "wrap_sep": matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "mut_wrap_sep", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j)]]
                                    elif (len(model_ops2[j].children) >= 3): matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "mut_wrap", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j, value = op_value)]]
                                    else: matrix[i][j].paths += [path + [MatrixOperation(op_id = len(path), op_type = "mut", node1_id = model_ops1[i].id, node2_id = model_ops2[j].id, i = i+start_i, j = j+start_j, value = op_value)]]
    
                            if matrix_iswap != False: matrix_iswap[i][j] = matrix[i][j]
                            if matrix_jswap != False: matrix_jswap[i][j] = matrix[i][j]
                        #if (i==len(matrix)-1): matrix[i][j-1].paths = []
                        #if (j==len(matrix[0])-1): matrix[i-1][j].paths = []
                        #if start_i == 4: self.print_alignment_matrix(matrix = matrix, model_ops1 = model_ops1, model_ops2 = model_ops2, i = i, j = j, start_i = start_i, start_j = start_j, saveimg = False)

            # llllllllllllllllllll
            # Get red of this because we don't need them anymore to compute anything and we liberate some memory
            if (model_ops1 == self.model_ops1) and (model_ops2 == self.model_ops2): 
                for i in range(prev_i, max_i-(max_i<len(model_ops1))):
                    for j in range(prev_j, max_j-(max_j<len(model_ops2))):
                        if (i<len(self.model_ops1)-1) and (j<len(self.model_ops2)-1):
                            matrix[i][j].paths = []
                            if matrix_iswap != False: matrix_iswap[i][j].paths = []
                            if matrix_jswap != False: matrix_jswap[i][j].paths = []
            #for i in range(prev_i+(model_ops1 != self.model_ops1), max_i-(max_i<len(model_ops1))):
            #    for j in range(prev_j+(model_ops2 == self.model_ops2), max_j-(max_j<len(model_ops2))):
            #        matrix[i][j].paths = []
            #        if matrix_iswap != False: matrix_iswap[i][j].paths = []
            #        if matrix_jswap != False: matrix_jswap[i][j].paths = []
                            
            #if (model_ops1 == self.model_ops1) and (model_ops2 == self.model_ops2): print(len(matrix[max_i-1][max_j-1].paths), "paths"), self.print_alignment_matrix()
            if max_j >= len(model_ops2):
                prev_j = 0
                prev_i = max_i-1
            else: prev_j = max_j-1
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

    def expanding_corner_loop(self, rows, cols):
        sequence = []
        # We'll process anti-diagonals where i + j = d
        for d in range(rows + cols - 1):
            if d % 2 == 0:
                # Even diagonals: process top to bottom
                i_start = max(0, d - (cols - 1))
                i_end = min(d, rows - 1)
                for i in range(i_start, i_end + 1):
                    j = d - i
                    sequence.append((i, j))
            else:
                # Odd diagonals: process bottom to top
                i_end = max(0, d - (cols - 1))
                i_start = min(d, rows - 1)
                for i in range(i_start, i_end - 1, -1):
                    j = d - i
                    sequence.append((i, j))
        return sequence
    
    def print_alignment_matrix(self, matrix = None, model_ops1 = None, model_ops2 = None, i = -1, j = -1, start_i = 0, start_j = 0, saveimg = True):
        if matrix == None: matrix = self.matrix
        if model_ops1 == None: model_ops1 = self.model_ops1
        if model_ops2 == None: model_ops2 = self.model_ops2
        
        size = (len(matrix), len(matrix[0]))
        m = np.zeros(size)
        for ii in range(size[0]):
            for jj in range(size[1]):
                m[ii,jj] = matrix[ii][jj].value
        plt.figure(figsize=(16,16))
        plt.imshow(m)
        plt.xlim((-0.5, size[1]-0.5))
        plt.ylim((size[0]-0.5, -0.5))
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
            position = [start_i, start_j]
            for op in path[1:]:
                color = cmap(p)
                #if ((op.i-start_i)>=0) and ((op.j-start_j)>=0): plt.text(op.j-start_j-0.5, op.i-start_i+0.2, str(matrix[op.i-start_i][op.j-start_j].value), color="cyan", fontsize=12, rotation=45, rotation_mode='default')
                pos_i = np.linspace(position[0], op.i-start_i, 30)
                pos_j = np.linspace(position[1], op.j-start_j, 30)
                linestyle = "-"
                if (abs(op.i-start_i-position[0]) > 1) or (abs(op.j-start_j-position[1]) > 1):
                    linestyle = "--"
                    r = np.sqrt((op.i-start_i-position[0])**2 + (op.j-start_j-position[1])**2)/2
                    try: a0 = np.arctan((op.i-start_i-position[0])/(op.j-start_j-position[1]))
                    except: a0 = np.pi/2
                    #a0 += np.pi*(a0<0)
                    #pos_i = (op.i+position[0])/2 + abs(op.i-position[0])/2 * np.sin(a0 + np.linspace(np.pi, 2*np.pi, 30))
                    #pos_j = (op.j+position[1])/2 + abs(op.j-position[1])/2 * np.cos(a0 + np.linspace(np.pi, 2*np.pi, 30))
                    pos_i = (op.i-start_i+position[0])/2 + r * np.sin(a0 + np.linspace(np.pi, 2*np.pi, 30))
                    pos_j = (op.j-start_j+position[1])/2 + r * np.cos(a0 + np.linspace(np.pi, 2*np.pi, 30))
                    
                plt.plot(pos_j, pos_i, linestyle = linestyle, color = color, linewidth=(0.75+op.value)*linewidth)
                position  = [op.i-start_i, op.j-start_j]
                
        if saveimg: plt.savefig(self.img_name+".svg", format='svg')
        plt.show()

    def get_op_name(self, op):
        #if ("routing" in op.operation.name or "branching" in op.operation.name) and op.operation.name != "branching(2)": return op.operation.name.split("(")[0]
        if "computation" in op.operation.name: return "comp<"+op.children[0].operation.name+">"
        if "wrap_" in op.operation.name: return op.parent.operation.name+op.operation.name[4:]
        else: return op.operation.name
            
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
        
        
    def calculate_restrictions(self, position = (-1, -1), path_n = 0, swap_all = True):
        #self.matrix[position[0]][position[1]].paths = [self.matrix[position[0]][position[1]].paths[0]]
        if swap_all: path_idxs = [path_idx for path_idx in range(len(self.matrix[position[0]][position[1]].paths))]
        else: path_idxs = [path_n]
        for path_idx in path_idxs:
            operations = copy.deepcopy(self.matrix[position[0]][position[1]].paths[path_idx])

            depths1 = self.calculate_depth(self.model1.serialise(), operations)
            depths2 = self.calculate_depth(self.model2.serialise(), operations)
            for depth in range(max(depths1), 0, -1):
                for idx, op in enumerate(operations):
                    if ("wrap_end" in op.op_type) and (depths1[idx]==depth):
                        if (("add" in op.op_type) or ("mut" in op.op_type)) and (len(self.model_ops1[op.i].parent.children) == 4):
                            #print("depth",depths1[idx], op.i,op.j,"op",op)
                            i = [in_i for in_i, in_node in enumerate(self.model_ops1) if in_node.id == op.node1_id]
                            if op.i_swapped:
                                for inside_op in operations:
                                    if (inside_op.i >= i[0]) and (inside_op.i < (i[0]+i[2]-i[1])):
                                        inside_op.i += (i[1]-i[0])
                                    elif (inside_op.i >= (i[0]+i[2]-i[1])) and (inside_op.i < i[2]):
                                        inside_op.i -= (i[2]-i[1])
                            for i_op in [in_op for in_op in operations if (in_op.op_type[:3] == op.op_type[:3]) and (in_op.node1_id == op.node1_id)]:
                                i_op.i_swapped = op.i_swapped
                         
            for depth in range(max(depths2), 0, -1):
                for idx, op in enumerate(operations): 
                    if ("wrap_end" in op.op_type) and (depths2[idx]==depth):  
                        if (("rem" in op.op_type) or ("mut" in op.op_type)) and (len(self.model_ops2[op.j].parent.children) == 4):
                            j = [in_j for in_j, in_node in enumerate(self.model_ops2) if in_node.id == op.node2_id]
                            if op.j_swapped:
                                for inside_op in operations:
                                    if (inside_op.j >= j[0]) and (inside_op.j < (j[0]+j[2]-j[1])):
                                        inside_op.j += (j[1]-j[0])
                                    elif (inside_op.j >= (j[0]+j[2]-j[1])) and (inside_op.j < j[2]):
                                        inside_op.j -= (j[2]-j[1])
                            for j_op in [in_op for in_op in operations if (in_op.op_type[:3] == op.op_type[:3]) and (in_op.node2_id == op.node2_id)]:
                                j_op.j_swapped = op.j_swapped
            
            self.matrix[position[0]][position[1]].paths[path_n] = operations
            if path_idx == path_n: self.operations = operations[1:]
        self.operations.reverse()
        self.nontrivial_ops = [operation for operation in self.operations if operation.value]

    def print_operations(self):
        print("\nOperations to change from model 1 to model 2:")
        self.nontrivial_ops.reverse()
        for op in self.nontrivial_ops:
            node1 = self.model_ops1[op.i]
            if not node1.is_root:
                if node1.id == node1.parent.id: node1 = node1.parent
            node2 = self.model_ops2[op.j]
            if not node2.is_root:
                if node2.id == node2.parent.id: node2 = node2.parent
            
            if ("end" in op.op_type) or ("sep" in op.op_type):
                if "add" in op.op_type:
                    print(f"\t(+{op.value}) Close branch {self.get_op_name(node1)} (id: {node1.id}) at indices {(op.i,op.j)}")
                elif "rem" in op.op_type:
                    print(f"\t(+{op.value}) Close branch {self.get_op_name(node2)} (id: {node2.id}) at indices {(op.i,op.j)}")
                else:
                    print(f"\t(+{op.value}) Close branches {self.get_op_name(node1)} (id: {node1.id}) and {self.get_op_name(node2)} (id: {node2.id}) at indices {(op.i,op.j)}")
                    
            else:
                if "add" in op.op_type:
                    if (len(node1.children) == 4):
                        print(f"\t(+{op.value}) Parallelize using {self.get_op_name(node1)} (id: {self.model_ops1[op.i].id}) from indices {(op.i,op.j)} to {(op.ii,op.jj)}")
                    elif (len(node1.children) == 3):
                        print(f"\t(+{op.value}) Add wrapper {self.get_op_name(node1)} (id: {node1.id}) from indices {(op.i,op.j)} to {(op.ii,op.jj)}")
                    else:
                        print(f"\t(+{op.value}) Add {self.get_op_name(node1)} (id: {node1.id}) at indices {(op.i,op.j)}")
                elif "rem" in op.op_type:
                    print(f"\t(+{op.value}) Remove {self.get_op_name(node2)} (id: {node2.id}) at indices {(op.i,op.j)}")
                else:
                    print(f"\t(+{op.value}) Substitute {self.get_op_name(node2)} (id: {node2.id}) by {self.get_op_name(node1)} (id: {node1.id}) at indices {(op.i,op.j)}")
        self.nontrivial_ops.reverse()

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
                if (type(op.enabler_ops) == list) and (op.i_swapped or op.j_swapped):
                    op.enabler_ops.reverse()
                for branch in op.enabler_ops:
                    if type(branch) != list: branch = [branch]
                    offspring = self.apply_all_operations([r_op for r_op in branch if r_op in selected_ops], offspring)
            offspring = self.apply_op(op, offspring)
        return offspring

    def apply_op(self, op, offspring):
        if op.id not in self.performed_ops:
            self.performed_ops += [op.id]
            if "mut" in op.op_type:
                for node in self.model_ops1:
                    if node.id == self.model_ops1[op.i].id:
                        node1 = copy.deepcopy(node)
                        break

                for node in offspring.serialise():
                    if (node.id == self.model_ops2[op.j].id) or (node.id == self.model_ops1[op.i].id):
                        node2 = copy.deepcopy(node)
                        break
                if (not node1.id == self.model_ops1[op.i].id) or ((not node.id == self.model_ops2[op.j].id) and (not node.id == self.model_ops1[op.i].id)): raise Exception("Nodes to mutate not found")
                
                node1str = str(node1)
                if ("branching" in node1.operation.name): node1str = node1str.split(")")[0]+")"+node1str.split(")")[1]+")...}"
                if ("routing" in node1.operation.name): node1str = node1str.split(")")[0]+")...]"
                node2str = str(node2)
                if ("branching" in node2.operation.name): node2str = node2str.split(")")[0]+")"+node2str.split(")")[1]+")...}"
                if ("routing" in node2.operation.name): node2str = node2str.split(")")[0]+")...]"
                if self.verbose: print(">>>Mutating",colored(node2str, "red"),"into", colored(node1str, "green"))
                
                node1.parent = node2.parent
                if not node2.is_root(): node2.parent.children[node2.parent.children.index(node2)] = node1

                if ("branching(2)" in node1.operation.name):
                    if sum([op.i_swapped, op.j_swapped])==1:
                        node1.children[2] = node2.children[1]
                        node1.children[1] = node2.children[2]
                    else:
                        node1.children[1] = node2.children[1]
                        node1.children[2] = node2.children[2]
                    node1.children[1].parent = node1
                    node1.children[2].parent = node1
                elif ("branching" in node1.operation.name) or ("routing" in node1.operation.name):
                    node1.children[1] = node2.children[1]
                    node1.children[1].parent = node1
                
                offspring = node1.get_root()
                                    
            elif "rem" in op.op_type:
                for node in offspring.serialise():
                    if node.id == self.model_ops2[op.j].id:
                        break
                if (not node.id == self.model_ops2[op.j].id): print(">>>Tried to remove module with id", colored(self.model_ops2[op.j].id, "red"), "but it was not found.")
                else:
                    #eeeeeeeeeeeeeeeeeeee
                    if "branching(2)" in node.operation.name:
                        b1 = node.children[1+op.j_swapped]
                        b2 = node.children[2-op.j_swapped]
                        #print(op.i_swapped, op.j_swapped)
                        if self.verbose: print(">>>Serializing branches",colored(str(b1), "red"),"and",colored(str(b2), "red")+" (swapping branches)"*op.j_swapped)
                        
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
                        
                    else: #fffffffffffffffffffffff
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
                            if parent_node.operation.name == "sequential": sibling_node = parent_node.children[parent_node.children.index(node)-1]
                            else: sibling_node = parent_node.children[1+(parent_node.children.index(node)==1)]
                            if not parent_node.is_root():
                                sibling_node.parent = parent_node.parent
                                parent_node.parent.children[parent_node.parent.children.index(parent_node)] = sibling_node
                            else:
                                sibling_node.parent = None
                            offspring = sibling_node.get_root()
                            
            elif op.op_type == "add_wrap":
                for node in self.model_ops1:
                    if node.id == self.model_ops1[op.i].id:
                        node1 = copy.deepcopy(node)
                        break
                offspring_serialised = offspring.serialise()
                if len(node.children) == 3:
                    split_pos = [0,0] # We look for the id of the node we want to start the wrap at
                    jjj = 1 # which will come from the second model if we parallelized the first branch
                    if op.j+jjj < len(self.model_ops2):
                        while ("wrap_" in self.model_ops2[op.j+jjj].operation.name) or (self.model_ops2[op.j+jjj].id not in [n.id for n in offspring_serialised]):
                            jjj += 1
                            if op.j+jjj == len(self.model_ops2): break
                    if op.j+jjj < len(self.model_ops2):
                        while offspring_serialised[split_pos[0]].id != self.model_ops2[op.j+jjj].id:
                            split_pos[0] = split_pos[0] + 1
                            if split_pos[0] == len(offspring_serialised): break
                    else: split_pos[0] = len(offspring_serialised)
                    iii = 0 # or from the first model if we added the first branch
                    if op.i+iii < len(self.model_ops1):
                        while ("wrap_" in self.model_ops1[op.i+iii].operation.name) or (self.model_ops1[op.i+iii].id not in [n.id for n in offspring_serialised]):
                            iii += 1
                            if op.i+iii == len(self.model_ops1): break
                    if op.i+iii < len(self.model_ops1):
                        while offspring_serialised[split_pos[1]].id != self.model_ops1[op.i+iii].id:
                            split_pos[1] = split_pos[1] + 1
                            if split_pos[1] == len(offspring_serialised): break
                    else: split_pos[1] = len(offspring_serialised)
                    
                    node = offspring_serialised[min(split_pos)]
                    # If the previous layer we found is the start of a branch/rout coming from the first model, we are interested on its children
                    if (len(node.children) == 3) and (min(split_pos) == split_pos[0]): node = node.children[1]
                    starting_node = node
                    
                    #iiiiiiiiiiiiiiiiiiiiiii
                    after_layer = [0,0]
                    closing_op = [in_idx for in_idx, in_op in enumerate(self.operations_unordered) if (in_op.node1_id==op.node1_id) and ("_end" in in_op.op_type)][0]
                    try:
                        op_distances0 = []
                        depths = []
                        depth = 0
                        #print("j")
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            depths += [depth]
                            if (("rem_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised):
                                if ("end" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth -= 1
                                elif (len(self.model_ops2[aux_op.j].children)==4): depth -= 1
                            elif (("add_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised):
                                if ("end" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth -= 1
                                elif (len(self.model_ops1[aux_op.i].children)==4): depth -= 1
                            #print(aux_idx, aux_op.j, aux_op.op_type, depth)
                            #print(aux_op.j, self.model_ops2.index(offspring_serialised[split_pos[0]]), aux_op)
                            if (("rem" in aux_op.op_type) or ("mut" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised) and (("_sep" not in self.model_ops2[aux_op.j].operation.name) and ("_end" not in self.model_ops2[aux_op.j].operation.name)): op_distances0 += [aux_idx-closing_op]
                            else: op_distances0 += [np.inf]
                            #print(depths[-1], depth, self.model_ops2[aux_op.j].operation.name)
                            #print(op_distances0[-1], aux_op.j, self.model_ops2[aux_op.j], aux_op)
                        #print("op_distances0",op_distances0)
                        #print(depths[self.operations_unordered.index(op)],"depths",depths)
                        op_distances0 = [op_dist if (depths[op_idx]==depths[self.operations_unordered.index(op)]) and (op_dist!=0) else np.inf for op_idx, op_dist in enumerate(op_distances0)]
                        #print("op_distances0 filtered",op_distances0)
                        selected_idx0 = np.argmin([abs(op_dist0) for op_dist0 in op_distances0])
                        after_layer[0] = op_distances0[selected_idx0] < 0
                        op_distances0 = abs(op_distances0[selected_idx0])
                        #print("op_distances0 final",op_distances0, self.model_ops2[self.operations_unordered[selected_idx0].j])
                    except:
                        selected_idx0 = -1
                        op_distances0 = np.inf
                        after_layer[0] = False
                    try:
                        op_distances1 = []
                        depths = []
                        depth = 0
                        #print("i")
                        #print("serialized offspring", [node.operation.name for node in offspring_serialised])
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            depths += [depth]
                            if (("rem_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised):
                                #print(aux_op.j, self.model_ops2[aux_op.j].operation.name, (self.model_ops2[aux_op.j] in offspring_serialised)) 
                                if ("end" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth -= 1
                                elif (len(self.model_ops2[aux_op.j].children)==4): depth -= 1
                            elif (("add_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised):
                                if ("end" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth -= 1
                                elif (len(self.model_ops1[aux_op.i].children)==4): depth -= 1
                            if (("add" in aux_op.op_type) or ("mut" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised) and (("_sep" not in self.model_ops1[aux_op.i].operation.name) and ("_end" not in self.model_ops1[aux_op.i].operation.name)): op_distances1 += [aux_idx-closing_op]
                            else: op_distances1 += [np.inf]
                            #print(op_distances1[-1], self.model_ops1[aux_op.i].operation.name)
                            #print(op_distances1[-1], aux_op.i, self.model_ops1[aux_op.i], aux_op)
                        #print("\nop_distances1",op_distances1)
                        #print(depths[self.operations_unordered.index(op)],"depths",depths)
                        op_distances1 = [op_dist if (depths[op_idx]==depths[self.operations_unordered.index(op)]) and (op_dist!=0) else np.inf for op_idx, op_dist in enumerate(op_distances1)]
                        #print("op_distances1",op_distances1)
                        selected_idx1 = np.argmin([abs(op_dist1) for op_dist1 in op_distances1])
                        #print("op_distances1 filtered",op_distances1)
                        after_layer[1] = op_distances1[selected_idx1] < 0
                        op_distances1 = abs(op_distances1[selected_idx1])
                        #print("selected_idx1",selected_idx1,"op_distances1 final",op_distances1)
                    except:
                        selected_idx1 = -1
                        op_distances1 = np.inf
                        after_layer[1] = False

                    #qqqqqqqqqqqqqq
                    chosen_pos = np.argmin([op_distances0, op_distances1])
                    split_pos = [selected_idx0, selected_idx1][chosen_pos]
                    #print("split_pos", [selected_idx0,selected_idx1],"(j",self.operations_unordered[selected_idx0].j,self.model_ops2[self.operations_unordered[selected_idx0].j], "i",self.operations_unordered[selected_idx1].i,self.model_ops1[self.operations_unordered[selected_idx1].i], ") after_layer", after_layer, "chosen_pos",chosen_pos)
                    after_layer = after_layer[chosen_pos]
                    final_layer_id = [self.model_ops2[self.operations_unordered[selected_idx0].j].id, self.model_ops1[self.operations_unordered[selected_idx1].i].id][chosen_pos]
                    #print(selected_idx0,selected_idx1)
                    try:
                        if chosen_pos == 0: split_pos = offspring_serialised.index(self.model_ops2[self.operations_unordered[selected_idx0].j])
                        else: split_pos = offspring_serialised.index(self.model_ops1[self.operations_unordered[selected_idx1].i])
                    except: split_pos = len(offspring_serialised)
                    
                    found_parent_sequential = False
                    if split_pos == len(offspring_serialised): # If we need to wrap up until the end of the model
                        while not found_parent_sequential: # We need to find the sequential node that holds the modules up to the end of the model
                            if not node.is_root():
                                if (node.parent.operation.name == "sequential") and (offspring_serialised[-1] not in node.serialise()): node = node.parent 
                                else: found_parent_sequential = True
                            else: found_parent_sequential = True
                    else: # If we need to wrap up until a certain modulefound_parent_sequential = False
                        while not found_parent_sequential: # We need to find the sequential node that holds up to the module we woould like to wrap
                            if not node.is_root():
                                if (node.parent.operation.name == "sequential") and (offspring_serialised[split_pos] not in node.serialise()): node = node.parent 
                                else: found_parent_sequential = True
                            else: found_parent_sequential = True

                    if (split_pos == len(offspring_serialised)): end_at_id = -1
                    else: # If we are wrapping until the end of the model nor a wrap_end, we save the id of the last node to split the sequentials at further on
                        if after_layer and (split_pos < len(offspring_serialised)):
                            #split_pos += 1
                            while (offspring_serialised[split_pos] not in self.model_ops1+self.model_ops2) or (final_layer_id == offspring_serialised[split_pos].id):
                                split_pos += 1
                                if split_pos == len(offspring_serialised): break
                            if split_pos < len(offspring_serialised): end_at_id = offspring_serialised[split_pos].id
                            else: end_at_id = -1
                        else: end_at_id = offspring_serialised[split_pos].id
                    #print("end_at_id",end_at_id)
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

                elif len(node.children) == 4:
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

                    after_layer = [0,0]
                    closing_op = [in_idx for in_idx, in_op in enumerate(self.operations_unordered) if (in_op.node1_id==op.node1_id) and ("_end" in in_op.op_type)][0]
                    try:
                        op_distances0 = []
                        depths = []
                        depth = 0
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            depths += [depth]
                            if (("rem_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised):
                                if ("end" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth -= 1
                                elif (len(self.model_ops2[aux_op.j].children)==4): depth -= 1
                            elif (("add_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised):
                                if ("end" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth -= 1
                                elif (len(self.model_ops1[aux_op.i].children)==4): depth -= 1
                            if (("rem" in aux_op.op_type) or ("mut" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised) and (("_sep" not in self.model_ops2[aux_op.j].operation.name) and ("_end" not in self.model_ops2[aux_op.j].operation.name)): op_distances0 += [aux_idx-closing_op]
                            else: op_distances0 += [np.inf]
                        op_distances0 = [op_dist if (depths[op_idx]==depths[self.operations_unordered.index(op)]) and (op_dist!=0) else np.inf for op_idx, op_dist in enumerate(op_distances0)]
                        selected_idx0 = np.argmin([abs(op_dist0) for op_dist0 in op_distances0])
                        after_layer[0] = op_distances0[selected_idx0] < 0
                        op_distances0 = abs(op_distances0[selected_idx0])
                    except:
                        selected_idx0 = -1
                        op_distances0 = np.inf
                        after_layer[0] = False
                    try:
                        op_distances1 = []
                        depths = []
                        depth = 0
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            depths += [depth]
                            if (("rem_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised):
                                if ("end" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth -= 1
                                elif (len(self.model_ops2[aux_op.j].children)==4): depth -= 1
                            elif (("add_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised):
                                if ("end" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth -= 1
                                elif (len(self.model_ops1[aux_op.i].children)==4): depth -= 1
                            if (("add" in aux_op.op_type) or ("mut" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised) and (("_sep" not in self.model_ops1[aux_op.i].operation.name) and ("_end" not in self.model_ops1[aux_op.i].operation.name)): op_distances1 += [aux_idx-closing_op]
                            else: op_distances1 += [np.inf]
                        op_distances1 = [op_dist if (depths[op_idx]==depths[self.operations_unordered.index(op)]) and (op_dist!=0) else np.inf for op_idx, op_dist in enumerate(op_distances1)]
                        selected_idx1 = np.argmin([abs(op_dist1) for op_dist1 in op_distances1])
                        after_layer[1] = op_distances1[selected_idx1] < 0
                        op_distances1 = abs(op_distances1[selected_idx1])
                    except:
                        selected_idx1 = -1
                        op_distances1 = np.inf
                        after_layer[1] = False

                    chosen_pos = np.argmin([op_distances0, op_distances1])
                    split_pos = [selected_idx0, selected_idx1][chosen_pos]
                    #print("split_pos", [selected_idx0,selected_idx1],"(j",self.operations_unordered[selected_idx0].j,self.model_ops2[self.operations_unordered[selected_idx0].j], "i",self.operations_unordered[selected_idx1].i,self.model_ops1[self.operations_unordered[selected_idx1].i], ") after_layer", after_layer, "chosen_pos",chosen_pos)
                    after_layer = after_layer[chosen_pos]
                    final_layer_id = [self.model_ops2[self.operations_unordered[selected_idx0].j].id, self.model_ops1[self.operations_unordered[selected_idx1].i].id][chosen_pos]
                    #print(selected_idx0,selected_idx1)
                    try:
                        if chosen_pos == 0: split_pos = offspring_serialised.index(self.model_ops2[self.operations_unordered[selected_idx0].j])
                        else: split_pos = offspring_serialised.index(self.model_ops1[self.operations_unordered[selected_idx1].i])
                    except: split_pos = len(offspring_serialised)

                    found_parent_sequential = False
                    if split_pos == len(offspring.serialise()): # If we need to wrap up until the end of the model
                        while not found_parent_sequential: # We need to find the sequential node that holds the modules up to the end of the model
                            if not node.is_root():
                                if (node.parent.operation.name == "sequential") and (offspring.serialise()[-1] not in node.serialise()): node = node.parent 
                                else: found_parent_sequential = True
                            else: found_parent_sequential = True
                    else: # If we need to wrap up until a certain module
                        while not found_parent_sequential: # We need to find the sequential node that holds up to the module we woould like to wrap
                            if not node.is_root():
                                if (node.parent.operation.name == "sequential") and (offspring.serialise()[split_pos] not in node.serialise()): node = node.parent 
                                else: found_parent_sequential = True
                            else: found_parent_sequential = True

                    if (split_pos == len(offspring_serialised)): end_at_id = -1
                    else: # If we are wrapping until the end of the model nor a wrap_end, we save the id of the last node to split the sequentials at further on
                        if after_layer and split_pos < len(offspring_serialised):
                            split_pos += 1
                            while offspring_serialised[split_pos] not in self.model_ops1+self.model_ops2:
                                split_pos += 1
                                if split_pos == len(offspring_serialised): break
                            if split_pos < len(offspring_serialised): end_at_id = offspring_serialised[split_pos].id
                            else: end_at_id = -1
                        else: end_at_id = offspring_serialised[split_pos].id

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

                    after_layer = [0,0]
                    closing_op = [in_idx for in_idx, in_op in enumerate(self.operations_unordered) if (in_op.node1_id==op.node1_id) and ("_sep" in in_op.op_type)][0]
                    try:
                        op_distances0 = []
                        depths = []
                        depth = 0
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            depths += [depth]
                            if (("rem_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised):
                                if ("end" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth -= 1
                                elif (len(self.model_ops2[aux_op.j].children)==4): depth -= 1
                            elif (("add_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised):
                                if ("end" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth -= 1
                                elif (len(self.model_ops1[aux_op.i].children)==4): depth -= 1
                            if (("rem" in aux_op.op_type) or ("mut" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised) and (("_sep" not in self.model_ops2[aux_op.j].operation.name) and ("_end" not in self.model_ops2[aux_op.j].operation.name)): op_distances0 += [aux_idx-closing_op]
                            else: op_distances0 += [np.inf]
                        op_distances0 = [op_dist if (depths[op_idx]==depths[self.operations_unordered.index(op)]) and (op_dist!=0) else np.inf for op_idx, op_dist in enumerate(op_distances0)]
                        selected_idx0 = np.argmin([abs(op_dist0) for op_dist0 in op_distances0])
                        after_layer[0] = op_distances0[selected_idx0] < 0
                        op_distances0 = abs(op_distances0[selected_idx0])
                    except:
                        selected_idx0 = -1
                        op_distances0 = np.inf
                        after_layer[0] = False
                    try:
                        op_distances1 = []
                        depths = []
                        depth = 0
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            depths += [depth]
                            if (("rem_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised):
                                if ("end" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4): depth -= 1
                                elif (len(self.model_ops2[aux_op.j].children)==4): depth -= 1
                            elif (("add_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised):
                                if ("end" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth += 2
                                elif ("sep" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4): depth -= 1
                                elif (len(self.model_ops1[aux_op.i].children)==4): depth -= 1
                            if (("add" in aux_op.op_type) or ("mut" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised) and (("_sep" not in self.model_ops1[aux_op.i].operation.name) and ("_end" not in self.model_ops1[aux_op.i].operation.name)): op_distances1 += [aux_idx-closing_op]
                            else: op_distances1 += [np.inf]
                        op_distances1 = [op_dist if (depths[op_idx]==depths[self.operations_unordered.index(op)]) and (op_dist!=0) else np.inf for op_idx, op_dist in enumerate(op_distances1)]
                        selected_idx1 = np.argmin([abs(op_dist1) for op_dist1 in op_distances1])
                        after_layer[1] = op_distances1[selected_idx1] < 0
                        op_distances1 = abs(op_distances1[selected_idx1])
                    except:
                        selected_idx1 = -1
                        op_distances1 = np.inf
                        after_layer[1] = False

                    #kkkkkkkkkkkkkkkkk
                    chosen_pos = np.argmin([op_distances0, op_distances1])
                    split_pos = [selected_idx0, selected_idx1][chosen_pos]
                    #print("split_pos", [selected_idx0,selected_idx1],"(j",self.operations_unordered[selected_idx0].j,self.model_ops2[self.operations_unordered[selected_idx0].j], "i",self.operations_unordered[selected_idx1].i,self.model_ops1[self.operations_unordered[selected_idx1].i], ") after_layer", after_layer, "chosen_pos",chosen_pos)
                    after_layer = after_layer[chosen_pos]
                    final_layer_id = [self.model_ops2[self.operations_unordered[selected_idx0].j].id, self.model_ops1[self.operations_unordered[selected_idx1].i].id][chosen_pos]
                    #print(selected_idx0,selected_idx1)
                    try:
                        if chosen_pos == 0: split_pos = offspring_serialised.index(self.model_ops2[self.operations_unordered[selected_idx0].j])
                        else: split_pos = offspring_serialised.index(self.model_ops1[self.operations_unordered[selected_idx1].i])
                    except: split_pos = len(offspring_serialised)

                    if (split_pos == len(offspring_serialised)): end_at_id = -1
                    else: # If we are wrapping until the end of the model nor a wrap_end, we save the id of the last node to split the sequentials at further on
                        if after_layer and split_pos < len(offspring_serialised):
                            split_pos += 1
                            while offspring_serialised[split_pos] not in self.model_ops1+self.model_ops2:
                                split_pos += 1
                                if split_pos == len(offspring_serialised): break
                            if split_pos < len(offspring_serialised): end_at_id = offspring_serialised[split_pos].id
                            else: end_at_id = -1
                        else: end_at_id = offspring_serialised[split_pos].id

                    #print(node2)
                    if (node2.operation.name == "sequential") and (split_pos < len(node2.serialise())):
                        node2 = self.split_sequentials(node2, offspring_serialised[split_pos].id) # We resequentialize the modules to be able to split the branches right where we want to
                    else: node2 = node2

                    #print(node2)
                    if self.verbose: print(">>>Parallelizing modules", colored(str(node2.children[0]), "red"), "and", colored(str(node2.children[1]), "red"), "using", colored(node1.operation.name, "green"))
    
                    parent_node = node2.parent
                    if not node2.is_root(): parent_node.children[parent_node.children.index(node2)] = node1
                    node1.parent = parent_node
                    node1.children[1] = node2.children[0]
                    node1.children[2] = node2.children[1]
                    node2.children[0].parent = node1
                    node2.children[1].parent = node1
                    offspring = node1.get_root()

            elif "add" in op.op_type:
                for node in self.model_ops1:
                    if node.id == self.model_ops1[op.i].id:
                        node1 = copy.deepcopy(node)
                        break
                if (not node1.id == self.model_ops1[op.i].id): raise Exception("Node",self.model_ops1[op.i].id,"not found from model 1 when attempting module addition")
                if self.model_ops2[op.j].id == -1:
                    node2 = offspring
                    after_layer = False
                else:
                    offspring_serialised = offspring.serialise()



                    depths = self.calculate_depth(offspring_serialised)
                    #print([(aaaa.i, aaaa.j) for aaaa in self.operations_unordered])
                    after_layer = [0,0]
                    closing_op = self.operations_unordered.index(op)
                    #print(depths[self.operations_unordered.index(op)],"depths",depths)
                    try:
                        if (self.model_ops2[op.j] in offspring_serialised) and ("_end" in self.model_ops2[op.j].operation.name):
                            selected_idx0, op_distances0 = [(aux_idx, aux_idx-closing_op) for aux_idx, aux_op in enumerate(self.operations_unordered) if aux_op.j == op.j][0]
                            after_layer[0] = True
                        else:
                            op_distances0 = []
                            for aux_idx, aux_op in enumerate(self.operations_unordered):
                                if (("rem" in aux_op.op_type) or ("mut" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised) and (("_sep" not in self.model_ops2[aux_op.j].operation.name) and ("_end" not in self.model_ops2[aux_op.j].operation.name)) and ((aux_idx-closing_op > 0) or (len(self.model_ops2[aux_op.j].children)<2)): op_distances0 += [aux_idx-closing_op]
                                else: op_distances0 += [np.inf]
                            #print("j unfiltered", op_distances0)
                            op_distances0 = [op_dist if (depths[op_idx]==depths[self.operations_unordered.index(op)]) and (op_dist!=0) else np.inf for op_idx, op_dist in enumerate(op_distances0)]
                            #print("j filtered", op_distances0)
                            selected_idx0 = np.argmin([abs(op_dist0) for op_dist0 in op_distances0])
                            after_layer[0] = op_distances0[selected_idx0] < 0
                            op_distances0 = abs(op_distances0[selected_idx0])
                    except:
                        selected_idx0 = -1
                        op_distances0 = np.inf
                        after_layer[0] = False
                    try:
                        op_distances1 = []
                        for aux_idx, aux_op in enumerate(self.operations_unordered):
                            # and ((aux_idx-closing_op > 0) or ("wrap" not in aux_op.op_type))
                            if (("add" in aux_op.op_type) or ("mut" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised) and (("_sep" not in self.model_ops1[aux_op.i].operation.name) and ("_end" not in self.model_ops1[aux_op.i].operation.name)) and ((aux_idx-closing_op > 0) or (len(self.model_ops1[aux_op.i].children)<2)): op_distances1 += [aux_idx-closing_op]
                            else: op_distances1 += [np.inf]
                        #print("i unfiltered", op_distances1)
                        op_distances1 = [op_dist if (depths[op_idx]==depths[self.operations_unordered.index(op)]) and (op_dist!=0) else np.inf for op_idx, op_dist in enumerate(op_distances1)]
                        #print("i filtered", op_distances1)
                        selected_idx1 = np.argmin([abs(op_dist1) for op_dist1 in op_distances1])
                        after_layer[1] = op_distances1[selected_idx1] < 0
                        op_distances1 = abs(op_distances1[selected_idx1])
                    except:
                        selected_idx1 = -1
                        op_distances1 = np.inf
                        after_layer[1] = False

                    if np.isinf(np.min([op_distances0, op_distances1])):# If we need to wrap up until the end of the model
                        node2 = node1
                        node1 = offspring
                        after_layer = True
                        
                    #jjjjjjjjjjjjjjj
                    else:
                        chosen_pos = np.argmin([op_distances0, op_distances1])
                        after_layer = after_layer[chosen_pos]
                        if chosen_pos == 0: split_pos = offspring_serialised.index(self.model_ops2[self.operations_unordered[selected_idx0].j])
                        else: split_pos = offspring_serialised.index(self.model_ops1[self.operations_unordered[selected_idx1].i])

                        if after_layer:
                            node2 = node1
                            node1 = offspring.serialise()[split_pos]
                        else:
                            node2 = offspring.serialise()[split_pos]
                
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
                
                if after_layer:
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
                    if after_layer: print(">>>Adding", colored(str(node2), "green"), "after", colored(str(node1), "red"))
                    else: print(">>>Adding", colored(str(node1), "green"), "before", colored(str(node2), "red"))
            
            if self.verbose and ("wrap_end" not in op.op_type) and ("wrap_sep" not in op.op_type): print("",colored(offspring, "light_grey"), "\n")
        return offspring
    
    def calculate_depth(self, offspring_serialised, operations = None):
        if operations == None: operations = self.operations_unordered
        next_depth = 1
        stack = [0]
        ids_stack = []
        for aux_idx, aux_op in enumerate(operations):
            ids_stack.append(stack[-1])
            if (("rem_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops2[aux_op.j] in offspring_serialised):
                if ("end" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4):
                    stack.pop()
                    stack.pop()
                elif ("sep" in self.model_ops2[aux_op.j].operation.name) and (len(self.model_ops2[aux_op.j].parent.children)==4):
                    stack.append(next_depth)
                    next_depth += 1
                elif (len(self.model_ops2[aux_op.j].children)==4):
                    stack.append(next_depth)
                    next_depth += 1
            elif (("add_wrap" in aux_op.op_type) or ("mut_wrap" in aux_op.op_type)) and (self.model_ops1[aux_op.i] in offspring_serialised):
                if ("end" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4):
                    stack.pop()
                    stack.pop()
                elif ("sep" in self.model_ops1[aux_op.i].operation.name) and (len(self.model_ops1[aux_op.i].parent.children)==4):
                    stack.append(next_depth)
                    next_depth += 1
                elif (len(self.model_ops1[aux_op.i].children)==4):
                    stack.append(next_depth)
                    next_depth += 1
        return(ids_stack)

def num_of_children(node, n = 0):
    for child in node.children:
        n = n + 1 + num_of_children(child)
    return n


def select_operations(operations, skewness = 0):
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

def recursive_constrained_smith_waterman_crossover(parent1, parent2, skewness=0):
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
        distance_between_parents = matrix.distance # sum([op.value for op in matrix.nontrivial_ops])
        distance_to_parent2 = sum([op.value for op in selected_ops])
        distance_to_parent1 = distance_between_parents - distance_to_parent2

        # FIXME Check for dangling pointers
        child = copy.deepcopy(child)
        selected_ops = copy.deepcopy(selected_ops)
        operations = copy.deepcopy(operations)
        del matrix

        return child, selected_ops, operations, distance_to_parent1, distance_to_parent2, distance_between_parents

def compile_fn(node, args):
    backbone = node.build(node, set_memory_checkpoint=True)
    return Network(
        backbone,
        node.output_params["shape"],
        args.num_classes,
        vars(args)
    ).to(args.device)
