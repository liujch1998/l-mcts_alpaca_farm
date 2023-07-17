import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import logging

import networkx as nx
import pydot
import time

logger = logging.get_logger(__name__)


def pad_sequences_to_left(sequences, batch_first=False, padding_value=0):
    """Add left padding so sequences have same shape"""
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, max_len-length:, ...] = tensor
        else:
            out_tensor[max_len-length:, i, ...] = tensor
    return out_tensor

def pad_sequences_to_left_states(sequences, padding_value=0, max_len=0):
    """Similar to pad_sequences_to_left function, but working on states tensor (in order to forge state for "sequential generation")"""
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    out_dims = (max_size[0], max_size[1], len(sequences), max_size[2], max_len, max_size[4])
    # print(out_dims)
    out_tensor = sequences[0].new_full(out_dims, padding_value, device=sequences[0].device)
    for i, tensor in enumerate(sequences):
        length = tensor.size()[3]
        out_tensor[:, :, i, :, max_len-length:, ...] = tensor
    return out_tensor


class BatchedMCTS():
    def __init__(
        self,
        tokenizer, policy, value_model,
        batch_size, response_len=300, num_simulations=10, num_sparse_actions=2, pb_c_init=8.0, temperature=1.0,
        init_v_with_parent=False,
        debug=False, visualize=False,
    ):
        '''
        - policy: a wrapper object with a forward_mcts() method
        - value_model: a wrapper object with a model and a regression head, has a forward_mcts() method
        '''
        self._tokenizer = tokenizer
        self._policy = policy
        self._value_model = value_model

        # Initialize parameters
        self._batch_size = batch_size
        self._response_len = response_len
        self._num_simulations = num_simulations
        self._num_actions = tokenizer.vocab_size + 1
        self._num_sparse_actions = min(num_sparse_actions, self._num_actions)
        self._pb_c_init = pb_c_init
        self._temperature = temperature

        self._init_v_with_parent = init_v_with_parent

        self._debug = debug
        self._visualize = False

        # Allocate all necessary storage.
        # For a given search associated to a batch-index, node i is the i-th node
        # to be expanded. Node 0 corresponds to the root node.
        num_nodes = num_simulations + 1
        batch_node = (batch_size, num_nodes)
        self._num_nodes = num_nodes
        self._visit_counts = np.zeros(batch_node, dtype=np.int32)
        self._values = np.zeros(batch_node, dtype=np.float32)
        self._parents = np.zeros(batch_node, dtype=np.int32)
        # action_from_parents[b, i] is the action taken to reach node i.
        # Note that action_from_parents[b, 0] will remain -1, as we do not know,
        # when doing search from the root, what action led to the root.
        self._action_from_parents = np.zeros(batch_node, dtype=np.int32)
        # The 0-indexed depth of the node. The root is the only 0-depth node.
        # The depth of node i, is the depth of its parent + 1.
        self._depth = np.zeros(batch_node, dtype=np.int32)
        self._is_terminal = np.full(batch_node, False, dtype=bool)

        # To avoid costly numpy ops, we store a sparse version of the actions.
        # We select the top k actions according to the policy, and keep a mapping
        # of indices from 0 to k-1 to the actual action indices in the
        # self._topk_mapping tensor.
        batch_node_action = (batch_size, num_nodes, self._num_sparse_actions)
        self._topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
        self._children_index = np.zeros(batch_node_action, dtype=np.int32)
        self._children_prior = np.zeros(batch_node_action, dtype=np.float32)
        self._children_values = np.zeros(batch_node_action, dtype=np.float32)
        self._children_visits = np.zeros(batch_node_action, dtype=np.int32)
        self._states_by_model = defaultdict(dict) # mapping from (b, i) to the key-value state of the policy
        self._token_ids = {}
        self._attention_mask = {}
        self._batch_range = np.arange(batch_size)
        self._sim = 0
        self._token_ix = 0

    def _reset_tree(self):
        """Resets the tree arrays."""
        self._visit_counts.fill(0)
        self._values.fill(0)
        self._parents.fill(-1)
        self._action_from_parents.fill(-1)
        self._depth.fill(0)

        self._topk_mapping.fill(-1)
        self._children_index.fill(-1)
        self._children_prior.fill(0.0)
        self._children_values.fill(0.0)
        self._children_visits.fill(0)
        self._states = defaultdict(dict)
        self._token_ids = {} # Indexed by tuples (batch index, node index)
        self._attention_mask = {}
        self._sim = 0

    def generate(self, input_ids, attention_mask):
        '''
        source: dict with input_ids and attention_mask
        Returns an expanded version of source
        '''
        original_len = len(input_ids[0])
        unfinished_sequences = torch.ones_like(input_ids[:, -1])
        for self._token_ix in range(self._response_len):
            # Logging for pre-token
            logger.warning('================================')
            logger.warning(f'Decoding token #{self._token_ix} ...')
            printed_prompt = self._tokenizer.decode(input_ids[0, :original_len], clean_up_tokenization_spaces=False).replace("\n", "\\n")
            logger.warning(f'Prompt: "{printed_prompt}"')
            printed_response = self._tokenizer.decode(input_ids[0, original_len:], clean_up_tokenization_spaces=False).replace("\n", "\\n")
            logger.warning(f'Existing response: "{printed_response}"')
            logger.warning(f'')
            start_time = time.time()

            # Search
            res_search = self.search(input_ids, attention_mask)
            new_token_ids = torch.tensor(np.argmax(res_search, axis=1), dtype=torch.long, device=input_ids.device)
            new_token_ids = new_token_ids * unfinished_sequences + self._tokenizer.pad_token_id * (1 - unfinished_sequences)
            unfinished_sequences = unfinished_sequences * (new_token_ids != self._tokenizer.eos_token_id).long()
            input_ids = torch.cat([input_ids, torch.unsqueeze(new_token_ids, dim=1)], dim=1)
            attention_mask = torch.cat((attention_mask, torch.unsqueeze(torch.ones_like(attention_mask[:, 0]), dim=1)), dim=1)

            # Logging for post-token
            logger.warning('----------------')
            logger.warning(f'Decoded token #{self._token_ix}; New token id: {new_token_ids[0]}; New token: "{self._tokenizer.convert_ids_to_tokens(new_token_ids[0].item())}"')
            printed_prompt = self._tokenizer.decode(input_ids[0, :original_len], clean_up_tokenization_spaces=False).replace("\n", "\\n")
            logger.warning(f'Prompt: "{printed_prompt}"')
            printed_response = self._tokenizer.decode(input_ids[0, original_len:], clean_up_tokenization_spaces=False).replace("\n", "\\n")
            logger.warning(f'Response so far: "{printed_response}"')
            end_time = time.time()
            logger.warning(f'Token #{self._token_ix} took {end_time - start_time} seconds.')
            logger.warning('')
            # TODO: Add records
        return input_ids, attention_mask

    def search(self, input_ids, attention_mask):
        '''
        original_input: dict with input_ids and attention_mask
        Returns an array of size (B, V)
        '''
        self._reset_tree()

        # Evaluate the root.
        outputs = self.evaluate(input_ids=input_ids, attention_mask=attention_mask)
        prior, values, states_by_model = outputs['priors'], outputs['values'], outputs['next_states_by_model']

        root_index = 0
        self.create_node(root_index, prior, values, states_by_model, input_ids, attention_mask, np.full(self._batch_size, False, dtype=bool))

        # Do simulations, expansions, and backwards.
        leaf_indices = np.zeros((self._batch_size), np.int32)
        for self._sim in range(self._num_simulations):
            logger.warning('----------------')
            logger.warning(f"Simulation #{self._sim} ...")
            start_time = time.time()
            self.print_tree()
            self.visualize_tree(stage='0-pre')

            node_indices, actions = self.select()
            next_node_index = self._sim + 1 # root is 0, therefore we offset by 1.
            self.expand(node_indices, actions, next_node_index)
            leaf_indices.fill(next_node_index)
            self.backward(leaf_indices)

            # self.visualize_tree(stage='4-post')
            end_time = time.time()
            logger.warning(f'Simulation #{self._sim} took {(end_time - start_time):.4f} seconds')
            logger.warning('')

        # Final choice: most visited, max score, max mean score
        return self.dense_visit_counts()
        # return self.dense_scores()
        # return self.dense_mean_scores()

    def select(self):
        """Goes down until all elements have reached unexplored actions."""
        logger.warning('Selecting ...')

        node_indices = np.zeros((self._batch_size), np.int32)
        depth = 0
        highlight_nodes = [] # [i]
        highlight_edges = [] # [(i, a)]
        while True:
            logger.warning(f'\tdepth={depth}, node_index={node_indices[0]}')
            depth += 1
            actions = self.uct_select_action(node_indices)
            logger.warning(f'\tselected_action={actions[0]}')
            next_node_indices = self._children_index[self._batch_range, node_indices, actions]
            highlight_nodes.append(node_indices[0])
            highlight_edges.append((node_indices[0], actions[0]))
            is_unexplored = next_node_indices == -1
            if is_unexplored.all():
                break
            else:
                node_indices = np.where(is_unexplored, node_indices, next_node_indices)
        
        logger.warning(f'\tSelected node {node_indices[0]}, action {actions[0]}')
        self.visualize_tree(stage='1-select', highlight_nodes=highlight_nodes, highlight_edges=highlight_edges, highlight_color='blue')
        return node_indices, actions

    def uct_select_action(self, node_indices):
        """Returns the action selected for a batch of node indices of shape (B)."""
        node_children_prior = self._children_prior[self._batch_range, node_indices, :] # (B, A)
        node_children_values = self._children_values[self._batch_range, node_indices, :] # (B, A)
        node_children_visits = self._children_visits[self._batch_range, node_indices, :] # (B, A)
        node_visits = self._visit_counts[self._batch_range, node_indices] # (B)
        node_policy_score = np.sqrt(node_visits[:, None]) * self._pb_c_init * node_children_prior / (node_children_visits + 1) # (B, A)

        node_value_score = node_children_values

        node_uct_score = node_value_score + node_policy_score # (B, A)
        for a in range(self._num_sparse_actions):
            logger.warning(f'\t\taction {a}: policy_score={node_policy_score[0, a]:.2f}, value_score={node_value_score[0, a]:.2f}, uct_score={node_uct_score[0, a]:.2f}')
        actions = np.argmax(node_uct_score, axis=1)
        return actions

    def expand(self, node_indices, actions, next_node_index):
        """Creates and evaluate child nodes from given nodes and unexplored actions."""
        logger.warning('Expanding ...')

        # Retrieve token ids for nodes to be evaluated.
        tokens_ids = pad_sequences_to_left([self._token_ids[(b, n)] for b, n in enumerate(node_indices)], True, self._tokenizer.eos_token_id)
        attention_masks = pad_sequences_to_left([self._attention_mask[(b, n)] for b, n in enumerate(node_indices)], True, 0)

        # Load states
        states_by_model = self.load_states(node_indices, max_len=len(tokens_ids[0]))

        # Convert sparse actions to dense actions for network computation
        dense_actions = self._topk_mapping[self._batch_range, node_indices, actions]
        # Add actions to list of tokens and extend attention mask by 1
        tokens_ids = torch.cat((tokens_ids, torch.unsqueeze(torch.cuda.LongTensor(dense_actions), 1)), dim = 1)
        attention_masks = torch.cat((attention_masks, torch.unsqueeze(torch.ones(len(dense_actions), dtype=torch.long, device=attention_masks.device), 1)), dim = 1)

        # Check if expanded nodes are terminal
        expanded_node_is_terminal = dense_actions == self._tokenizer.eos_token_id

        # Evaluate nodes.
        outputs = self.evaluate(input_ids=tokens_ids, attention_mask=attention_masks, states_by_model=states_by_model)
        prior, values, next_states_by_model = outputs['priors'], outputs['values'], outputs['next_states_by_model']

        # Create the new nodes.
        self.create_node(next_node_index, prior, values, next_states_by_model, tokens_ids, attention_masks, expanded_node_is_terminal)

        # Update tree topology.
        self._children_index[self._batch_range, node_indices, actions] = next_node_index
        self._parents[:, next_node_index] = node_indices
        self._action_from_parents[:, next_node_index] = actions
        self._depth[:, next_node_index] = self._depth[self._batch_range, node_indices] + 1

        logger.warning(f'\tCreated node {next_node_index}: n={self._visit_counts[0, next_node_index]}, v={self._values[0, next_node_index]:.2f}, parent={self._parents[0, next_node_index]}, action_from_parent={self._action_from_parents[0, next_node_index]}, depth={self._depth[0, next_node_index]}, is_terminal={self._is_terminal[0, next_node_index]}')
        for a in range(self._num_sparse_actions):
            logger.warning(f'\t\taction {a}: token_id={self._topk_mapping[0, next_node_index, a]}, child_index={self._children_index[0, next_node_index, a]}, child_p={self._children_prior[0, next_node_index, a]:.2f}, child_v={self._children_values[0, next_node_index, a]:.2f}, child_n={self._children_visits[0, next_node_index, a]}')
        self.visualize_tree(
            stage='2-expand',
            highlight_nodes=[next_node_index],
            highlight_edges=[(next_node_index, a) for a in range(self._num_sparse_actions)],
            highlight_color='red',
        )

    def evaluate(self, input_ids, attention_mask, states_by_model=None):
        '''Get score from current nodes
        Inputs:
        - input_ids: (B, L), including the new node's token
        - attention_mask: (B, L)
        - states_by_model
        Outputs:
        - priors: (B, V), numpy, on CPU, float32; the policy priors for the new node, corresponds to the P(s, a) in the AlphaGo paper
        - values: (B), numpy, on CPU, float32; this is the value of the new node
        - next_states_by_model
        '''
        next_states_by_model = {}
        policy_outputs = self._policy.forward_mcts(input_ids, attention_mask, states=states_by_model['policy'] if states_by_model is not None else None, temperature=self._temperature)
        priors = policy_outputs['priors'].float().cpu().numpy()
        next_states_by_model['policy'] = policy_outputs['next_states']
        value_outputs = self._value_model.forward_mcts(input_ids, attention_mask, states=states_by_model['value'] if states_by_model is not None else None)
        values = value_outputs['values'].float().cpu().numpy()
        next_states_by_model['value'] = value_outputs['next_states']
        return dict(priors=priors, values=values, next_states_by_model=next_states_by_model)

    def save_states(self, node_index, states_by_model):
        for model in ['policy', 'value']:
            states = states_by_model[model]
            # Transform the returned states format into tensor for easier manipulation
            key_value_tensor = torch.stack(list(
                torch.stack(list(
                    states[i]
                ), dim=0) for i in range(len(states))
            ), dim=0)  # (Y, 2, B, H, L, D/H)
            for b in range(states[0][0].size(0)):
                if node_index == 0:
                    self._states_by_model[model][(b, node_index)] = torch.clone(key_value_tensor[:, :, b])
                else:
                    self._states_by_model[model][(b, node_index)] = torch.clone(key_value_tensor[:, :, b, :, -1:])
    def get_states_from_node(self, b, n, model):
        """Forge state tensor by going backward from the node to the root (because we only store on token's part on each node to avoid duplication)"""
        state_array = []
        state_array.append(self._states_by_model[model][(b, n)])
        while n != 0:
            n = self._parents[(b, n)]
            state_array.append(self._states_by_model[model][(b, n)])
        state_array = state_array[::-1]
        result = torch.cat(state_array, 3)
        return result
    def load_states(self, node_indices, max_len):
        states_by_model = {}
        for model in ['policy', 'value']:
            states_tensor = pad_sequences_to_left_states([self.get_states_from_node(b, n, model) for b, n in enumerate(node_indices)], padding_value=0, max_len=max_len)
            states = tuple(tuple(type_of_value for type_of_value in layer) for layer in states_tensor)
            states_by_model[model] = states
        return states_by_model

    def create_node(self, node_index, prior, values, next_states_by_model, tokens_ids, attention_masks, expanded_node_is_terminal):
        """Create nodes with computed values"""
        # Truncate the prior to only keep the top k logits
        prior_topk_indices = np.argpartition(prior, -self._num_sparse_actions, axis=-1)[:, -self._num_sparse_actions:]
        prior = prior[self._batch_range[:, None], prior_topk_indices] # (B, A)

        # Store the indices of the top k logits
        self._topk_mapping[self._batch_range, node_index, :] = prior_topk_indices

        # Update prior, values and visit counts.
        self._children_prior[:, node_index, :] = prior
        self._values[:, node_index] = values
        self._visit_counts[:, node_index] = 1
        self._is_terminal[:, node_index] = expanded_node_is_terminal

        # Optionally initialize the children values with the parent value
        if self._init_v_with_parent:
            self._children_values[:, node_index, :] = values[:, np.newaxis]

        # Save states
        self.save_states(node_index, next_states_by_model)

        # Updates tokens ids and attention masks
        for b, token_ids in enumerate(tokens_ids):
            self._token_ids[(b, node_index)] = token_ids
        for b, attention_mask in enumerate(attention_masks):
            self._attention_mask[(b, node_index)] = attention_mask

    def backward(self, leaf_indices):
        """Goes up and updates the tree until all nodes reached the root."""
        logger.warning(f'Backward ...')

        node_indices = leaf_indices # (B)
        leaf_values = self._values[self._batch_range, leaf_indices]
        highlight_nodes = []
        highlight_edges = []
        while True:
            is_root = node_indices == 0
            if is_root.all():
                break
            parents = np.where(is_root, 0, self._parents[self._batch_range, node_indices])
            if parents[0] != -1:
                highlight_nodes.append(parents[0])
                a = self._action_from_parents[0, node_indices[0]]
                highlight_edges.append((parents[0], a))
            root_mask = 1.0 * is_root
            not_root_mask_int = (1 - is_root)
            not_root_mask = 1.0 - root_mask
            # Update the parent nodes iff their child is not the root.
            # We therefore mask the updates using not_root_mask and root_mask.
            self._values[self._batch_range, parents] = \
                not_root_mask * (self._values[self._batch_range, parents] * self._visit_counts[self._batch_range, parents] + leaf_values) / (self._visit_counts[self._batch_range, parents] + 1.0) + \
                root_mask * self._values[self._batch_range, parents]
            self._visit_counts[self._batch_range, parents] += not_root_mask_int
            actions = np.where(is_root, 0, self._action_from_parents[self._batch_range, node_indices])
            self._children_values[self._batch_range, parents, actions] = not_root_mask * self._values[self._batch_range,node_indices] + root_mask * self._children_values[self._batch_range, parents, actions]
            self._children_visits[self._batch_range, parents, actions] += not_root_mask_int
            # Go up
            node_indices = parents

        self.visualize_tree(
            stage='3-backward',
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges,
            highlight_color='orange',
        )

    def dense_visit_counts(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        dense_visit_counts = np.zeros((self._batch_size, self._num_actions))
        dense_visit_counts[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_visit_counts
        return dense_visit_counts

    def dense_scores(self):
        root_index = 0
        root_scores = self._children_values[:, root_index, :]
        dense_root_scores = np.zeros((self._batch_size, self._num_actions))
        dense_root_scores[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_scores
        root_visit_counts = self._children_visits[:, root_index, :]
        return dense_root_scores

    def dense_mean_scores(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        root_scores = self._children_values[:, root_index, :]
        root_mean_scores = root_scores / root_visit_counts
        dense_mean_scores = np.zeros((self._batch_size, self._num_actions))
        dense_mean_scores[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_mean_scores
        return dense_mean_scores

    def print_tree(self):
        logger.warning(f'Current tree:')
        for i in range(self._sim + 1): # sim + 1 is the current number of nodes
            logger.warning(f'\tnode {i}: n={self._visit_counts[0, i]}, v={self._values[0, i]:.2f}, parent={self._parents[0, i]}, action_from_parent={self._action_from_parents[0, i]}, depth={self._depth[0, i]}, is_terminal={self._is_terminal[0, i]}')
            for a in range(self._num_sparse_actions):
                logger.warning(f'\t\taction {a}: token_id={self._topk_mapping[0, i, a]}, child_index={self._children_index[0, i, a]}, child_p={self._children_prior[0, i, a]:.2f}, child_v={self._children_values[0, i, a]:.2f}, child_n={self._children_visits[0, i, a]}')

    def visualize_tree(self, stage, highlight_nodes=[], highlight_edges=[], highlight_color='blue'):
        '''
        stage: '0-pre', '1-select', '2-expand', '3-backup', '4-post'
        highlight_nodes: [i]
        highlight_edges: [(i, a)]
        '''
        if not self._visualize:
            return
        graph = nx.DiGraph()
        global dummy_index
        dummy_index = self._num_simulations + 1
        def visualize_node(i):
            global dummy_index
            node_label = f'{i}\nn={self._visit_counts[0, i]}\lv={self._values[0, i]:.2f}\l'#parent={self._parents[0, i]}\lafp={self._action_from_parents[0, i]}\ldepth={self._depth[0, i]}\lterm={self._is_terminal[0, i]}\l'
            graph.add_node(i, label=node_label)
            for a in range(self._num_sparse_actions):
                child_index = self._children_index[0, i, a]
                edge_label = f'{a}\ntoken="{self._tokenizer.convert_ids_to_tokens(int(self._topk_mapping[0, i, a]))}"\lprior={self._children_prior[0, i, a]:.2f}\ln={self._children_visits[0, i, a]}\lv={self._children_values[0, i, a]:.2f}\l'#index={self._children_index[0, i, a]}\l'
                if child_index != -1:
                    graph.add_edge(i, child_index, label=edge_label)
                    graph.edges[(i, child_index)]['a'] = a
                    visualize_node(child_index)
                else: # child_index == -1
                    graph.add_node(dummy_index, label='')
                    graph.add_edge(i, dummy_index, label=edge_label)
                    graph.edges[(i, dummy_index)]['a'] = a
                    dummy_index += 1
        visualize_node(0)
        dot_graph = nx.drawing.nx_pydot.to_pydot(graph)
        for node in dot_graph.get_nodes():
            node.set_shape('rectangle')
            if node.get_label() == '':
                node.set_style('filled')
                node.set_fillcolor('black')
            if int(node.get_name()) in highlight_nodes:
                node.set_color(highlight_color)
        for edge in dot_graph.get_edges():
            head_node = edge.get_source()
            tail_node = edge.get_destination()
            a = edge.get_attributes()['a']
            if (int(head_node), int(a)) in highlight_edges:
                edge.set_color(highlight_color)

        # Add a title to the image
        token_ids = self._token_ids[(0, 0)]
        prompt = self._tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
        prompt = prompt.replace('\n', '\l')
        node = pydot.Node('title', label=f'Token #{self._token_ix}, Simulation #{self._sim}, Stage: {stage}\lPrompt: {prompt}\l')
        node.set_shape('rectangle')
        node.set_color('white')
        # node.set('pos', '0,0!')
        dot_graph.add_node(node)

        dot_graph.write_png(f'trees/token-{self._token_ix:03d}_sim-{self._sim:03d}_stage-{stage}.png')
