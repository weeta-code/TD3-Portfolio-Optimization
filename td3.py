import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay_buffer import ReplayBuffer

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action,
                 actor, critic,  # passed in as initialized networks
                 actor_lr=1e-3, critic_lr=1e-3,  # model learning rates
                 gamma=0.99, tau=0.005, policy_noise=0.1, noise_clip=0.25,
                 buffer_size=int(1e5), policy_delay=2):
        
        print("TD3Agent: Starting initialization...")
        
        # Initialize networks
        print("TD3Agent: Setting up actor networks...")
        self.actor = actor
        print("TD3Agent: Creating actor target network...")
        self.actor_target = Actor(state_dim, action_dim)
        print("TD3Agent: Loading actor target state dict...")
        self.actor_target.load_state_dict(self.actor.state_dict())

        print("TD3Agent: Setting up critic networks...")
        self.critic_1 = critic
        print("TD3Agent: Creating critic target network...")
        self.critic_target_1 = Critic(state_dim, action_dim)
        print("TD3Agent: Loading critic target state dict...")
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())

        # Create second critic network
        print("TD3Agent: Creating second critic network...")
        self.critic_2 = Critic(state_dim, action_dim)
        print("TD3Agent: Creating second critic target network...")
        self.critic_target_2 = Critic(state_dim, action_dim)
        print("TD3Agent: Loading second critic target state dict...")
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # Initialize optimizers
        print("TD3Agent: Setting up optimizers...")
        print("TD3Agent: Getting actor parameters...")
        actor_params = list(self.actor.parameters())
        print(f"TD3Agent: Actor has {len(actor_params)} parameter groups")
        self.actor_optimizer = optim.SGD(actor_params, lr=actor_lr)
        print("TD3Agent: Actor optimizer created")
        
        print("TD3Agent: Getting first critic parameters...")
        critic1_params = list(self.critic_1.parameters())
        print(f"TD3Agent: First critic has {len(critic1_params)} parameter groups")
        self.critic_optimizer_1 = optim.SGD(critic1_params, lr=critic_lr)
        print("TD3Agent: First critic optimizer created")
        
        print("TD3Agent: Getting second critic parameters...")
        critic2_params = list(self.critic_2.parameters())
        print(f"TD3Agent: Second critic has {len(critic2_params)} parameter groups")
        self.critic_optimizer_2 = optim.SGD(critic2_params, lr=critic_lr)
        print("TD3Agent: Second critic optimizer created")

        # Initialize replay buffer
        print("TD3Agent: Creating replay buffer...")
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)

        # Set hyperparameters
        print("TD3Agent: Setting hyperparameters...")
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0
        
        print("TD3Agent: Initialization complete!")

    def select_action(self, state, add_noise=False):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()          
        if add_noise:
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            action = (action + noise).clip(0, 1)  
        return action

    def train(self, batch_size=1000):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        # Add noise to next action for smoothing
        noise = torch.randn_like(action) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
        
        # Compute target Q-values
        target_Q1 = self.critic_target_1(next_state, next_action)
        target_Q2 = self.critic_target_2(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_loss_1 = nn.MSELoss()(current_Q1, target_Q.detach())
        critic_loss_2 = nn.MSELoss()(current_Q2, target_Q.detach())

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic_target_1, self.critic_1, self.tau)
            self.soft_update(self.critic_target_2, self.critic_2, self.tau)

        self.total_it += 1

    def soft_update(self, target_net, net, tau):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

    def save_checkpoint(self, filename):
        """Save the current state of the agent"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'critic_target_1_state_dict': self.critic_target_1.state_dict(),
            'critic_target_2_state_dict': self.critic_target_2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_1_state_dict': self.critic_optimizer_1.state_dict(),
            'critic_optimizer_2_state_dict': self.critic_optimizer_2.state_dict(),
            'total_it': self.total_it
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        """Load a saved state of the agent"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        self.critic_target_1.load_state_dict(checkpoint['critic_target_1_state_dict'])
        self.critic_target_2.load_state_dict(checkpoint['critic_target_2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer_1.load_state_dict(checkpoint['critic_optimizer_1_state_dict'])
        self.critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2_state_dict'])
        self.total_it = checkpoint['total_it']
        print(f"Checkpoint loaded from {filename}")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return torch.softmax(self.output(x), dim=1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q = torch.relu(self.l1(sa))
        q = self.l2(q)
        return q
