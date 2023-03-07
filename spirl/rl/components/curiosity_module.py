import torch
import torch.nn as nn

import sys


class Curiosity_Module:
    def __init__(self):
        input_size = 10 + 60 #for maze environment
        hidden_size1 = 60
        hidden_size2 = 40
        output_size = 60
        
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size1),
                              nn.ReLU(),
                              nn.Linear(hidden_size1, hidden_size2),
                              nn.ReLU(),
                              nn.Linear(hidden_size2, output_size),
                              nn.Sigmoid())
        
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.output_log = []
        self.next_step_log = []

    def experience_forward(self, experience):
        input = torch.cat((experience.observation, experience.action), 1)
        
        output = self.model.forward(input)

        intrinsic_reward = (output - experience.observation_next).pow(2).sum().sqrt() 

        self.output_log.append(output)
        self.next_step_log.append(experience.observation_next)

        experience.reward += intrinsic_reward

        return experience
    
    def train_worldModel(self):
        self.output_log = torch.stack(self.output_log)
        self.next_step_log = torch.stack(self.next_step_log)

        loss = self.loss_function(self.output_log, self.next_step_log)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.output_log = []
        self.next_step_log = []