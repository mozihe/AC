import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import net.Actor as Actor
import net.Critic as Critic


class AC:
    def __init__(self, learn_table_size = 4000, epsilon_start = 0.9, epsilon_end = 0.1, batch_size = 10,  foresight = 0.7, actor_lr = 0.01, critic_lr = 0.01, action_space = 1, state_space = 4, epochs = 1000, action_size = 4):
        self.learn_table_size = learn_table_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.foresight = foresight
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.action_space = action_space
        self.state_space = state_space
        self.epochs = epochs
        self.action_size = action_size
        self.step = 0
        self.learn_count = 0

        self.actor = Actor.Actor(self.state_space, 256, 5, self.action_size)
        self.critic = Critic.Critic(1, 14, 4, 2)
        self.old_critic = Critic.Critic(1, 14, 4, 2)

        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr)

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.actor.to(self.device)
            self.critic.to(self.device)
            self.old_critic.to(self.device)

        self.epsilon = self.epsilon_start
        self.learn_table = []

    def create_tensors(self, obs, a, b):

        xList, yList = [], []

        while len(xList) < self.batch_size and len(obs) >= a + b:
                start = random.randint(0, len(obs) - a - b)
                old_state = obs[start:start + a]
                new_state = obs[start + a:start + a + b]
                xList.append(old_state)
                yList.append(new_state)

        # x = torch.tensor(xList, dtype=torch.float32)
        # y = torch.tensor(yList, dtype=torch.float32)

        return xList, yList

    def update_epsilon(self):
        decay_rate = (self.epsilon_start - self.epsilon_end) / self.epochs
        self.epsilon = max(self.epsilon - decay_rate, self.epsilon_end)
        self.step += 1

    def act(self, state):
        print(state)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(self.device)

        if random.random() < self.epsilon:
            action = torch.randint(0, self.action_size, (1,)).to(self.device)
        else:
            logits = self.actor(state)
            action = F.softmax(logits, dim=1)
            action = torch.argmax(action, dim=1)

        self.update_epsilon()
        return action

    def get_experience(self, state_sequence, action, reward):

        exp = (state_sequence, action, reward)
        if len(self.learn_table) < self.learn_table_size:
            self.learn_table.append(exp)

        else:

            self.learn_table.pop(0)
            self.learn_table.append(exp)

    def learn(self):

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        if self.learn_count % 100 == 0:
            with torch.no_grad():
                self.old_critic.load_state_dict(self.critic.state_dict())

        self.learn_count += 1
        old_exp, new_exp = self.create_tensors(self.learn_table, 10, 1)

        if len(old_exp) == 0:
            print("No experience")
            return

        old_states = []
        old_actions = []
        old_rewards = []

        for exp in old_exp:
            old_s, old_a, old_r = zip(*exp)
            old_states.append(list(old_s))
            old_actions.append(old_a)
            old_rewards.append(old_r)

        new_states, _, _ = zip(*new_exp[0])

        old_states = torch.stack([torch.stack(s) for s in old_states]).to(self.device)
        old_actions = torch.tensor(old_actions, dtype=torch.float32).unsqueeze(1).to(self.device)
        old_rewards = torch.tensor(old_rewards, dtype=torch.float32).to(self.device)
        new_states = torch.stack(new_states).unsqueeze(0).to(self.device)

        old_actions = old_actions.expand(-1, 10, -1)

        critic_input = torch.cat((old_states, old_actions), dim=2)
        critic_input = critic_input.permute(1, 0, 2)

        current_q = self.critic(critic_input).squeeze()

        with torch.no_grad():
            new_actions = self.actor(new_states).argmax(dim=1, keepdim=True).float().unsqueeze(1)
            new_actions = new_actions.expand(-1, 1, -1)
            new_critic_input = torch.cat((new_states, new_actions.float()), dim=2)
            new_critic_input = new_critic_input.permute(1, 0, 2)
            if new_critic_input.shape[-1] <= 14:
                pad_size = 14 - new_critic_input.shape[-1]
                new_critic_input = F.pad(new_critic_input, (0, pad_size))
            target_q = self.old_critic(new_critic_input).squeeze()

            discounts = torch.tensor([self.foresight ** i for i in range(len(old_rewards))], dtype=torch.float32).to(self.device)
            target_q = old_rewards + discounts * target_q

        critic_loss = F.mse_loss(current_q, target_q)
        critic_loss.backward()
        self.critic_optim.step()

        critic_input = torch.cat((old_states, old_actions), dim=2)
        actor_loss = -self.critic(critic_input).mean()
        actor_loss.backward()
        self.actor_optim.step()

        return actor_loss.item(), critic_loss.item()
