from playfield import Playfield
from torch import nn
import torch
import pieces
import numpy as np
from collections import deque
import random
from copy import deepcopy
from torch.utils.data.dataset import Dataset
from surface_area import surface_area
from math import sqrt, log

class Gameplays(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        feature, label, _ = self.data[index]
        return feature, torch.tensor([label])

class Gameplays_with_variance(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        feature, value, variance = self.data[index]
        return feature, torch.tensor([value, variance])


class MCTS:
    class State:
        def __init__(self, value, reward, variance, terminal=False,  num_actions=4): #, speculative_pc=None):
            self.experiences = [value]
            self.value = value
            self.variance = variance
            self.variance_experience = [variance]
            self.child_nodes = [None]*num_actions
            self.n = 1
            self.terminal = terminal
            self.reward = reward
            #self.speculative_pc = speculative_pc
            if terminal:
                assert self.value == 0

        def update_value(self):
            self.value = np.mean(self.experiences)

        def update_variance(self):
            self.variance = (1/ self.n**2) * np.sum(self.variance_experience)

        def add_experience(self, value):
            #self.prev_mean = self.value
            self.experiences.append(value)
            #self.variance_experience.append(variance)
            self.n += 1
            self.update_value()
            #self.update_variance()
            #self.variance = (self.variance * (self.n - 2) + (value - self.prev_mean)*(value - self.value)) / (self.n - 1)
            
    def __init__(self, pc, gamma, model=None):
        self.model = model
        self.pc = pc
        self.gamma = gamma
        #if self.model is None:
        #    self.root = MCTS.State(self.get_rollout_value(pc))
        #else:
        #    self.root = MCTS.State(self.get_model_value(pc))
        value, variance = self.get_model_prediction(pc)
        self.root = MCTS.State(value=value, variance=variance, reward=0)

    def generate_a_game(self, num_iter=50, max_steps=500, stats_writer=None):
        data = []
        actual_data = []
        total_reward = 0
        steps = 0
        action_stats = [0,0,0,0]
        while not self.pc._game_over and steps < max_steps:
            action = self.search(self.pc, num_iter)
            #values = []
            #for node in self.root.child_nodes:
            #    values.append(len(node.experiences))
            #print(values)

            state = get_new_board_state(self.pc.gamestate())
            actual_data.append((state, self.root.value, action))
            
            action_stats[action] += 1
            new_data, reward = self.make_move_remove_and_dump_unused_branch(action, self.pc)
            data += new_data
            total_reward += reward
            steps += 1

        action_stats = np.array(action_stats)/steps
        if action_stats[2] > 0.4:
            data = []
            actual_data = []

        if stats_writer is not None:
            writer, game = stats_writer
            writer.add_scalar('Rewards', total_reward, game)
            writer.add_scalar('Steps', steps, game)
            writer.add_scalar('Move Left', action_stats[0], game)
            writer.add_scalar('Move Right', action_stats[1], game)
            writer.add_scalar('Rotate', action_stats[2], game)
            writer.add_scalar('Nothing', action_stats[3], game)
            print("%d game, %f rewards, %d num of steps" % (game, total_reward, steps))
            
        return data, actual_data, total_reward

    def make_move_remove_and_dump_unused_branch(self, action_taken, pc):
        def dump_branch(node, pc):
            state = get_new_board_state(pc.gamestate())
            value = node.value
            variance = node.variance
            data = [(state, value, variance)]
            count = 0
            for action in range(len(node.child_nodes)):
                if node.child_nodes[action] is not None:
                    count += 1
                    new_pc = deepcopy(pc)
                    if action == 0:
                        new_pc.move_left()
                    elif action == 1:
                        new_pc.move_right()
                    elif action == 2:
                        new_pc.rotate_cw()
                    new_pc.update()
                    data += dump_branch(node.child_nodes[action], new_pc)
            if not node.terminal and (count == 0 or node.n < 10):
                data = []
            return data

        def dump_speculative_branch(node, pc, new_piece_count):
            data = []
            cut = False
            count = 0
            if node.speculative_pc is not None and new_piece_count == 1 and type(pc._active_piece) == type(node.speculative_pc._active_piece):
                node.speculative_pc = None
                return [], False
            if node.speculative_pc is not None:
                cut = True
                state = get_new_board_state(pc.gamestate())
                value = node.value
                variance = node.variance
                data = [(state, value, variance)]
                for action in range(len(node.child_nodes)):
                    if node.child_nodes[action] is not None:
                        count += 1
                        new_pc = deepcopy(pc)
                        if action == 0:
                            new_pc.move_left()
                        elif action == 1:
                            new_pc.move_right()
                        elif action == 2:
                            new_pc.rotate_cw()
                        status = new_pc.update()
                        data += dump_branch(node.child_nodes[action], new_pc)
            else:
                for action in range(len(node.child_nodes)):
                    if node.child_nodes[action] is not None:
                        count += 1
                        new_pc = deepcopy(pc)
                        if action == 0:
                            new_pc.move_left()
                        elif action == 1:
                            new_pc.move_right()
                        elif action == 2:
                            new_pc.rotate_cw()
                        status = new_pc.update()
                        if status == True:
                            new_piece_count += 1
                            #assert new_piece_count != 1 or node.child_nodes[action].speculative_pc is not None
                        new_data, need_cut = dump_speculative_branch(node.child_nodes[action], new_pc, new_piece_count = new_piece_count)
                        data += new_data
                        if need_cut:
                            node.child_nodes[action] = None
            if not node.terminal and count == 0:
                data = []
            return data, cut

        state = get_new_board_state(pc.gamestate())
        value = self.root.value
        variance = self.root.variance
        data = [(state, value, variance)]
        for action in range(len(self.root.child_nodes)):
            if action == action_taken or self.root.child_nodes[action] is None:
                continue
            data += dump_branch(self.root.child_nodes[action], deepcopy(pc))
            
        prev_score = pc._score
        
        if action_taken == 0:
            pc.move_left()
        elif action_taken == 1:
            pc.move_right()
        elif action_taken == 2:
            pc.rotate_cw()
            
        pc.update()
        
        reward = pc._score - prev_score
        
        self.root = self.root.child_nodes[action_taken]

        """
        if status == True:
            
            #assert self.root.speculative_pc is None
            for action in range(len(self.root.child_nodes)):
                new_pc = deepcopy(pc)
                if self.root.child_nodes[action] is None:
                    continue
                if action_taken == 0:
                    new_pc.move_left()
                elif action_taken == 1:
                    new_pc.move_right()
                elif action_taken == 2:
                    new_pc.rotate_cw()
                    
                status = new_pc.update()
                if status:
                    piece_count = 1
                    #assert self.root.child_nodes[action].speculative_pc is not None
                else:
                    piece_count = 0
                    #assert self.root.child_nodes[action].speculative_pc is None
                new_data, need_cut = dump_speculative_branch(self.root.child_nodes[action], new_pc, piece_count)
                data += new_data
                if need_cut:
                    self.root.child_nodes[action] = None
        """
        return data, reward
        
    def get_model_value(self, pc):
        gs = pc.gamestate()
        state = get_new_board_state(gs)
        value = self.model(state.unsqueeze(0).cuda()).item()
        return value

    def get_model_prediction(self, pc):
        gs = pc.gamestate()
        state = get_new_board_state(gs)
        if self.model is None:
            value = self.get_rollout_value(pc)
        else:
            #v = []
            #for _ in range(5):
            #    result = self.model(state.unsqueeze(0).cuda())
            #    value = result.item()
            #    v.append(value)
            #value = np.mean(v)
            #variance = np.var(v, ddof=1)
            
            result = self.model(state.unsqueeze(0).cuda())
            assert result.shape[0] == 1
            value = result.item()
            #variance = result[0, 1].item()
        variance = 1.0
        return value, variance

    def get_rollout_value(self, pc):
        pc = deepcopy(pc)
        total_reward = 0
        gamma = 1
        while not pc._game_over:

            prev_score = pc._score
            #bb = pc._playfield.get_bool_board()
            #bb = np.rot90(bb, k = 1)
            #prev_sa = surface_area(bb)
            #prev_closed_spaces = get_enclosed_space(pc.gamestate())
            
            action = random.randint(0,3)
            if action == 0:
                pc.move_left()
            elif action == 1:
                pc.move_right()
            elif action == 2:
                pc.rotate_cw()
            
            pc.update()
            
            #bb = pc._playfield.get_bool_board()
            #bb = np.rot90(bb, k = 1)
            #sa = surface_area(bb)
            #closed_spaces = get_enclosed_space(pc.gamestate())
            #reward = (closed_spaces > prev_closed_spaces)*-1 + (prev_sa > sa)*1 + (pc._score > prev_score)*1
            if pc._game_over:
                #reward = -2
                reward = 0
            reward = pc._score - prev_score
            reward = [0, 40, 100, 300,  1200].index(reward)
            total_reward += gamma * reward
            gamma *= self.gamma
        return total_reward

    def search(self, pc, num_iter):
        for _ in range(num_iter):
            new_pc = deepcopy(pc)
            if self.stimulate_to_leaf(self.root, new_pc) == True:
                break
            
        best_action = 0
        best_value = self.root.child_nodes[0].value
        for action in range(1, len(self.root.child_nodes)):
            if self.root.child_nodes[action] == None:
                continue
            if self.root.child_nodes[action].value > best_value:
                best_value = self.root.child_nodes[action].value
                best_action = action
        return best_action

    def stimulate_to_leaf(self, state_node, pc, new_piece_count=0, rotate_count=0, c=1.5):
        if pc._game_over:
            assert False
        assert new_piece_count < 2
        picked_index = None
        picked_value = None
        found = False
        action_values = []
        for action in range(len(state_node.child_nodes)):
            if state_node.child_nodes[action] is not None:
                if state_node.child_nodes[action].terminal:
                    continue
                if state_node.n == 2:
                    picked_index = action
                    continue
                #q_star = 10 * log(1 - log(-log(1 - 1/(state_node.n - 1)) / log(2)) / log(22)) / log(41)
                #c = sqrt(state_node.child_nodes[action].variance) * q_star
                #q_hat = state_node.child_nodes[action].value + state_node.child_nodes[action].reward
                #v = q_hat + c
                v = (state_node.child_nodes[action].value * self.gamma + state_node.child_nodes[action].reward)/state_node.child_nodes[action].n + 1.5 * sqrt(log(state_node.n)/state_node.child_nodes[action].n)
                action_values.append((action, v))
            else:
                if action == 2 and rotate_count > 3:
                    continue
                
                prev_score = pc._score

                new_pc = deepcopy(pc)
                if action == 0:
                    illegal = new_pc.move_left()
                    if illegal == True:
                        continue
                elif action == 1:
                    illegal = new_pc.move_right()
                    if illegal == True:
                        continue
                elif action == 2:
                    new_pc.rotate_cw()
                    rotate_count += 1
                status = new_pc.update()

                

                #spec_pc = None
                if status == True:
                    if new_piece_count != 0:
                        continue #spec_pc = new_pc

                    
                        
                reward = new_pc._score - prev_score
                reward = [0, 40, 100, 300,  1200].index(reward)
                #if self.model is None:
                #    child_value = self.get_rollout_value(new_pc)
                #else:
                #    child_value = self.get_model_value(new_pc)
                child_value, child_variance = self.get_model_prediction(new_pc)   
                if new_pc._game_over:
                    reward = 0
                    child_value = 0
                    child_variance = 0
                state_node.child_nodes[action] = MCTS.State(child_value, variance=child_variance, reward=reward, terminal=new_pc._game_over) #, speculative_pc=spec_pc)
                found = True
                break
                
        if not found:
            if len(action_values) == 0:
                return True
            else:
                found = False
                action_values.sort(key=lambda x: x[1], reverse=True)
                for action, _ in action_values:
                    prev_score = pc._score
                    #if state_node.child_nodes[action].speculative_pc is not None:
                    #    new_pc = deepcopy(state_node.child_nodes[action].speculative_pc)
                    #    reward = new_pc._score - prev_score
                    #    new_piece_count += 1
                    #    rotate_count = 0
                    #    child_value = self.stimulate_to_leaf(state_node.child_nodes[action], new_pc, new_piece_count, rotate_count)
                    new_pc = deepcopy(pc)
                    
                    new_new_piece_count = new_piece_count
                    new_rotate_count = rotate_count
                    
                    if action == 0:
                        new_pc.move_left()
                    elif action == 1:
                        new_pc.move_right()
                    elif action == 2:
                        assert rotate_count <= 3
                        new_pc.rotate_cw()
                        new_rotate_count += 1
                        
                
                    status = new_pc.update()
                    
                    
                    if status == True:
                        new_new_piece_count += 1
                        new_rotate_count = 0
                    reward = new_pc._score - prev_score
                    reward = [0, 40, 100, 300,  1200].index(reward)
                    child_value = self.stimulate_to_leaf(state_node.child_nodes[action], new_pc, new_new_piece_count, new_rotate_count)
                    if child_value == True:
                        continue
                    found = True
                    break
                if not found:
                    return True

        state_node.add_experience(reward + self.gamma * child_value)
        return reward + self.gamma * child_value
    

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        bundle = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        done = []
        for s, a, r, ns, d in bundle:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            done.append(d)
        states = torch.stack(states).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        next_states = torch.stack(next_states).float()
        done = torch.tensor(done).float()
        return states, actions, rewards, next_states, done

    def __len__(self):
        return len(self.memory)

def get_inactive_board(gamestate):
    return gamestate._playfield.get_bool_board().astype(float)

def get_active_board(gamestate):
    playfield = Playfield()
    playfield.insert_piece(gamestate._active_piece, gamestate._active_piece.coords)
    return playfield.get_bool_board().astype(float)

def get_board_state(gamestate):
    return torch.stack((torch.tensor(get_inactive_board(gamestate)), torch.tensor(get_active_board(gamestate)))).float()

def get_new_board_state(gamestate):
    inactive = torch.tensor(get_inactive_board(gamestate))
    active = torch.tensor(get_active_board(gamestate))
    empty = torch.logical_not(torch.logical_and(inactive, active))
    return torch.stack([inactive, active, empty]).float()

def get_next_piece_index(gamestate):
    piece = gamestate._next_piece
    if piece is pieces.I:
        return 0
    if piece is pieces.J:
        return 1
    if piece is pieces.L:
        return 2
    if piece is pieces.O:
        return 3
    if piece is pieces.S:
        return 4
    if piece is pieces.T:
        return 5
    if piece is pieces.Z:
        return 6

def get_current_piece_index(gamestate):
    piece = gamestate._active_piece
    if isinstance(piece, pieces.I):
        return 0
    if isinstance(piece, pieces.J):
        return 1
    if isinstance(piece, pieces.L):
        return 2
    if isinstance(piece, pieces.O):
        return 3
    if isinstance(piece, pieces.S):
        return 4
    if isinstance(piece, pieces.T):
        return 5
    if isinstance(piece, pieces.Z):
        return 6

def get_current_piece(gamestate):
    result = torch.zeros(7).float()
    result[get_current_piece_index(gamestate)] = 1.0
    return result

def get_next_piece(gamestate):
    result = torch.zeros(7).float()
    result[get_next_piece_index(gamestate)] = 1.0
    return result

def get_raw_state(gamestate):
    board_state = get_inactive_board(gamestate)
    board_state = torch.tensor(board_state.reshape(200)).float()
    next_piece_state = get_next_piece(gamestate)
    current_piece_state = get_current_piece(gamestate)
    current_piece_coord = torch.tensor(gamestate._active_piece.coords).float()
    state = torch.cat([board_state, current_piece_coord, current_piece_state, next_piece_state])
    return state, board_state

def get_enclosed_space(gamestate):
    def get_open_space(board_state, x, y, visited):
        if (x,y) in visited or board_state[x,y] == 1.0:
            return 0
        visited.add((x,y))
        result = 1
        if x-1 >= 0:
            result += get_open_space(board_state, x-1, y, visited)
        if y-1 >= 0:
            result += get_open_space(board_state, x, y-1, visited)
        if x+1 < 10:
            result += get_open_space(board_state, x+1, y, visited)
        if y+1 < 20:
            result += get_open_space(board_state, x, y+1, visited)
        return result
    board_state = get_inactive_board(gamestate)
    visited = set()
    open_spaces = get_open_space(board_state, 5, 19, visited)
    total_spaces = np.sum(1.0 - board_state)
    return total_spaces - open_spaces

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

            
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2,4,3,2,1)
        self.conv1b = nn.Conv2d(4,4,3,1,1)
        self.conv2 = nn.Conv2d(4,8,3,2,1)
        self.conv2b = nn.Conv2d(8,8,3,1,1)
        self.fc1 = nn.Linear(127,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500, 4)
    def forward(self, board, piece):
        board = self.conv1(board)
        board = self.relu(board)
        board = self.conv1b(board)
        board = self.relu(board)
        board = self.conv2(board)
        board = self.relu(board)
        board = self.conv2b(board)
        board = self.relu(board)
        board = board.reshape(board.shape[0],-1)
        out = torch.cat([board,piece],1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

        
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(216, 432)
        self.fc2 = nn.Linear(432, 862)
        self.fc3 = nn.Linear(862, 7)

    def forward(self, state):
        state = self.fc1(state)
        state = self.relu(state)
        state = self.fc2(state)
        state = self.relu(state)
        state = self.fc3(state)
        return state

class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(216, 432)
        self.fc2 = nn.Linear(432, 7)

    def forward(self, state):
        state = self.fc1(state)
        state = self.relu(state)
        state = self.fc2(state)
        return state

class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,6,3,2,1)
        self.conv2 = nn.Conv2d(6,12,3,2,1)
        self.fc1 = nn.Linear(180,240)
        self.fc2 = nn.Linear(240,480)
        self.fc3 = nn.Linear(480, 4)
    def forward(self, board):
        board = self.conv1(board)
        board = self.relu(board)
        board = self.conv2(board)
        board = self.relu(board)
        out = board.reshape(board.shape[0],-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class Model5(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.conv2 = nn.Conv2d(16,16,3,1,1)
        self.fc1 = nn.Linear(3200,128)
        self.fc2 = nn.Linear(128,4)
    def forward(self, board):
        board = self.conv1(board)
        board = self.relu(board)
        board = self.conv2(board)
        board = self.relu(board)
        out = board.reshape(board.shape[0],-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Model6(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,32,5,1,2)
        self.conv2 = nn.Conv2d(32,32,3,1,1)
        self.conv3 = nn.Conv2d(32,32,3,1,1)
        self.fc1 = nn.Linear(6400,128)
        self.fc2 = nn.Linear(128,4)
    def forward(self, board):
        board = self.conv1(board)
        board = self.relu(board)
        board = board + self.relu(self.conv2(board))
        board = board + self.relu(self.conv3(board))
        out = board.reshape(board.shape[0],-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Model7(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.conv2 = nn.Conv2d(16,16,3,1,1)
        self.fc1 = nn.Linear(3200,128)
        self.fc2 = nn.Linear(128,1)
    def forward(self, board):
        board = self.conv1(board)
        board = self.relu(board)
        board = self.conv2(board)
        board = self.relu(board)
        out = board.reshape(board.shape[0],-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.sigmoid()
        return out

class Model8(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,32,5,1,2)
        self.conv2 = nn.Conv2d(32,32,3,1,1)
        self.conv3 = nn.Conv2d(32,32,3,1,1)
        self.fc1 = nn.Linear(6400,128)
        self.fc2 = nn.Linear(128,1)
    def forward(self, board):
        board = self.conv1(board)
        board = self.relu(board)
        board = board + self.relu(self.conv2(board))
        board = board + self.relu(self.conv3(board))
        out = board.reshape(board.shape[0],-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Model9(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.conv2 = nn.Conv2d(16,16,3,1,1)
        self.fc1 = nn.Linear(3200,128)
        self.fc2 = nn.Linear(128,2)
        self.sp = nn.Softplus()
    def forward(self, board):
        board = self.conv1(board)
        board = self.relu(board)
        board = self.conv2(board)
        board = self.relu(board)
        out = board.reshape(board.shape[0],-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.transpose(torch.stack([out[:, 0], self.sp(out[:, 1])]), 0, 1)
        return out

class Model10(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,16,3,1,1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(3200,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128,2)
        self.sp = nn.Softplus()
    def forward(self, board):
        board = self.bn1(self.conv1(board))
        board = self.relu(board)
        board = self.bn2(self.conv2(board))
        board = self.relu(board)
        out = board.reshape(board.shape[0],-1)
        out = self.bn3(self.fc1(out))
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.transpose(torch.stack([out[:, 0], self.sp(out[:, 1])]), 0, 1)
        return out

class Model11(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.conv2 = nn.Conv2d(16,16,3,1,1)
        self.fc1 = nn.Linear(3200,128)
        self.fc2 = nn.Linear(128,1)
    def forward(self, board):
        board = self.conv1(board)
        board = self.relu(board)
        board = self.conv2(board)
        board = self.relu(board)
        out = board.reshape(board.shape[0],-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Model12(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,32,5,1,2)
        self.conv2 = nn.Conv2d(32,32,3,1,1)
        self.conv3 = nn.Conv2d(32,32,3,1,1)
        self.fc1 = nn.Linear(6400,128)
        self.fc2 = nn.Linear(128,1)
    def forward(self, board):
        board = self.conv1(board)
        board = self.relu(board)
        board = board + self.relu(self.conv2(board))
        board = board + self.relu(self.conv3(board))
        out = board.reshape(board.shape[0],-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Model13(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.conv2 = nn.Conv2d(16,16,3,1,1)
        self.fc1 = nn.Linear(3200,128)
        self.fc2 = nn.Linear(128,2)
    def forward(self, board):
        board = self.conv1(board)
        board = self.relu(board)
        board = self.conv2(board)
        board = self.relu(board)
        out = board.reshape(board.shape[0],-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.sigmoid()
        return out

class Model14(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.drop1 = nn.Dropout2d(0.5)
        self.conv2 = nn.Conv2d(16,16,3,1,1)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(3200,128)
        self.drop3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128,1)
    def forward(self, board):
        board = self.drop1(self.conv1(board))
        board = self.relu(board)
        board = self.drop2(self.conv2(board))
        board = self.relu(board)
        out = board.reshape(board.shape[0],-1)
        out = self.drop3(self.fc1(out))
        out = self.relu(out)
        out = self.fc2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, state_size=216, action_size=7, hidden_size=216, num_hidden=2):
        super().__init__()
        self.lrelu = nn.LeakyReLU()
        self.fc_in = nn.Linear(state_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.hiddens = nn.ModuleList([])
        for _ in range(num_hidden):
            self.hiddens.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(inplace=True)))
        self.fc_out = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.bn(x)
        x = self.lrelu(x)
        for hidden in self.hiddens:
            out = hidden(x)
            x = x + out
        out = self.fc_out(x)
        return out


class ResNetWithoutBN(nn.Module):
    def __init__(self, state_size=216, action_size=4, hidden_size=1024, num_hidden=2):
        super().__init__()
        self.lrelu = nn.LeakyReLU()
        self.fc_in = nn.Linear(state_size, hidden_size)
        self.hiddens = nn.ModuleList([])
        for _ in range(num_hidden):
            self.hiddens.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(inplace=True)))
        self.fc_out = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.lrelu(x)
        for hidden in self.hiddens:
            out = hidden(x)
            x = x + out
        out = self.fc_out(x)
        return out


class ANN(nn.Module):
    def __init__(self, state_size=216, action_size=4, hidden_sizes=[1024]):
        super().__init__()
        assert len(hidden_sizes) > 0
        prev_hidden = hidden_sizes[0]
        self.lrelu = nn.LeakyReLU()
        self.fc_in = nn.Linear(state_size, prev_hidden)
        self.hiddens = nn.ModuleList([])
        for num_hidden in hidden_sizes[1:]:
            self.hiddens.append(nn.Sequential(
                nn.Linear(prev_hidden, num_hidden),
                nn.LeakyReLU(inplace=True)))
            prev_hidden = num_hidden
        self.fc_out = nn.Linear(prev_hidden, action_size)
    def forward(self, x):
        x = self.fc_in(x)
        x = self.lrelu(x)
        for hidden in self.hiddens: 
            x = hidden(x)
        out = self.fc_out(x)
        return out
