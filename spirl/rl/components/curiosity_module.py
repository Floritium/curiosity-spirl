import torch
import torch.nn as nn


class Curiosity_Module:
    def __init__(self):
        input_size = 10 + 4 #for maze environment
        hidden_size = 15
        output_size = 4
        
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                              nn.ReLU(),
                              nn.Linear(hidden_size, output_size),
                              nn.Sigmoid())
        
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.loss_log = []

    def experience_forward(self, experience):
        input = torch.cat((experience.observation, experience.action))
        
        output = self.model.forward(input)

        intrinsic_reward = 0 # TODO: compute difference between output and experience.observation_next

        loss = self.loss_function(output, experience.observation_next)
        self.loss_log.append(loss.item())

        experience.reward += intrinsic_reward

        return experience
    
    def train_worldModel(self):
        for i in range(self.loss_log.size()):
            self.model.zero_grad()
            self.loss_log[i].backward()
            self.optimizer.step
        self.loss_log = []
        