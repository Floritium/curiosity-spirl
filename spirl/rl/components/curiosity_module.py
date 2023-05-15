import torch
import torch.nn as nn

from spirl.utils.general_utils import AttrDict

import numpy as np

import sys


class Curiosity_Module:
    def __init__(self):
        input_size = 10 + 60 #for maze environment
        hidden_size1 = 60
        output_size = 60
        
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size1),
                              nn.ReLU(),
                              nn.Linear(hidden_size1, output_size),
                              nn.ReLU(),
                              nn.Linear(hidden_size1, output_size),
                              nn.Sigmoid())

        self.model = self.model.to('cuda:0')

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self.batch_number = 1

        self.last_reward = 0

        self.output_log = []
        self.next_step_log = []

    def experience_forward(self, experience):
        input = torch.cat((experience.observation, experience.action), 1)
        input = input.to('cuda:0')
        experience_reward = torch.Tensor([])
        intrinsic_reward = torch.zeros(experience.reward.size(dim=0))

        output = self.model(input)

        self.output_log.append(output)
        self.next_step_log.append(experience.observation_next)

        for i in range(experience.reward.size(dim=0)):
                i_reward = (output[i] - experience.observation_next[i]).pow(2).sum().sqrt() 

                intrinsic_reward[i] = 0.1*(i_reward + experience.reward[i]) + 0.9*(self.last_reward)
                self.last_reward = intrinsic_reward[i]
                #print("experience.reward[i] : " + str(experience.reward[i]) + "/ i_reward : " + str(i_reward))

        #intrinsic_reward = (intrinsic_reward * (1/(self.batch_number/256))).to('cuda:0')

        reward_dict = AttrDict(observation=experience.observation, reward=intrinsic_reward, done=experience.done, action=experience.action, observation_next=experience.observation_next)
        
        return reward_dict
    
    def train_worldModel(self):
        self.output_log = torch.stack(self.output_log)
        self.next_step_log = torch.stack(self.next_step_log)

        loss = self.loss_function(self.output_log, self.next_step_log)
        self.model.zero_grad()
        loss.backward()

        self.optimizer.step()

        self.output_log = []
        self.next_step_log = []

        self.batch_number += 1
        print(self.batch_number)
