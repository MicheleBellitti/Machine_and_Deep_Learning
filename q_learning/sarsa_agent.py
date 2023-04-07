from agent import Agent

class SarsaAgent(Agent):
  """
  Class that models a reinforcement learning agent.
  """

  def update_Q(self, old_state, action, reward, new_state):
    """
    Update action-value function Q
    
    Parameters
    ----------
    old_state: tuple
      Previous state of the Environment
    action: int
      Action performed to go from `old_state` to `new_state`
    reward: int
      Reward got after action `action`
    new_state: tuple
      Next state of the Environment

    Returns
      The action to be executed next
    -------
    None
    """
    # TODO!
    pass