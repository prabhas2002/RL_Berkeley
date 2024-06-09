import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        #returning a sample from multinomial dist if descrete is true else returning normal dist for continous.
        if self.discrete:
            obs = ptu.from_numpy(obs)
            output = self.logits_net(obs)
            action_probs = nn.functional.log_softmax(output).exp()
            return torch.multinomial(action_probs, num_samples = 1).cpu().detach().numpy()[0]
        else:
            obs = ptu.from_numpy(obs)
            #samples from normal distribution
            return torch.normal(self.mean_net(obs), torch.exp(self.logstd).expand_as(self.mean_net(obs))).cpu().detach().numpy()

           

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        #if discrete==True then return logits(distribution over actions) else return probability distribution
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            action = nn.functional.log_softmax(self.logits_net(obs)).exp()       
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            action = (self.mean_net(obs), torch.exp(self.logstd))
        return action

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        
        if self.discrete:
            logits = self.forward(obs)
            dist = distributions.Categorical(logits)
            loss = -dist.log_prob(actions) * advantages 
        else:
            logits = self.forward(obs)
            dist = torch.distributions.Normal(logits[0], logits[1]).log_prob(actions).sum(-1)
            loss = -dist * advantages
           
        
        self.optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
