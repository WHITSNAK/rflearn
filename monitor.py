"""
Module help to monitor the history/trace between agent and ENV
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class AgentMonitor:
    def __init__(self, agent, runs, steps):
        self.agent = agent
        self.runs = runs
        self.steps = steps

        self.history = {
            'At': {run: [] for run in range(runs)},  # action taken (steps x # of actions)
            'Rt': {run: [] for run in range(runs)},  # instance rewards (steps x 1)
            'Qt': {run: [] for run in range(runs)},  # expected value (steps x # of actions)
            'OptR': {run: [] for run in range(runs)},  # Optimal reward (steps x # of actions)
            'OptA': {run: [] for run in range(runs)},  # Taken the optimal action (steps x 1)
            'truth': {run: [] for run in range(runs)},  # Ground Truth of the means
        }
        self._summary = None
    
    def record(self, run, action, reward, truth):
        """
        Record all relvant information from a step of agent

        parameter
        ---------
        run: the current run number
        action: the action took by by the agent
        reward: the reward got back from the action taken
        truth: ENV ground truth
        """
        # update, prepare for next step
        A = [0 for _ in range(self.agent.num_actions)]
        A[action] = 1
        self.history['At'][run].append(A)
        self.history['Rt'][run].append(reward)
        self.history['Qt'][run].append(self.agent.q_values.copy())

        self.history['OptR'][run].append(truth.max())
        self.history['OptA'][run].append(int(action == np.argmax(truth)))
        self.history['truth'][run].append(truth.copy())
    
    def get_summary(self):
        """Summary Statistics table off from collected hisotry records"""
        history = self.history
        nA = self.agent.num_actions

        At_tot = np.zeros(nA)
        Qt_avg = np.zeros(nA)
        truth_avg = np.zeros(nA)
        n = 0
        for i in range(self.runs):
            At_tot += np.array(history['At'][i]).sum(0)
            Qt_avg += np.array(history['Qt'][i][-1])
            truth_avg += np.array(history['truth'][i][-1])
            n += 1

        df = pd.DataFrame({
            'At total': At_tot,
            'At %': At_tot/(self.steps * self.runs) * 100,
            'Qt avg': Qt_avg/n,
            'Truth': truth_avg/n,
            'Error': (truth_avg - Qt_avg)/n
        })
        self._summary = df
        return df
    
    def plot_avg_rewards_trace(self):
        """
        Plot average rewards trace that compares
        - averaged rewards of all runs
        - to the maximal rewards based on the ENV turth
        """
        if self._summary is None:
            self.get_summary()

        sr = pd.DataFrame(self.history['Rt']).mean(1)
        ax = sr.plot(label=str(self.agent))
        ax.axhline(self._summary.Truth.max(), linestyle='--', color='black', linewidth=1)
        plt.title('Averaged Rewards')
        plt.legend()
    
    def plot_optimal_action_trace(self):
        """
        Plot the optimal action trace that compares
        - the total percentage of optimal action across all run at each step
        - to the true optimal action based on the ENV truth
        """
        sr = pd.DataFrame(self.history['OptA']).sum(1)
        ax = (sr/self.runs*100).plot(label=str(self.agent))
        ax.axhline(100, linestyle='--', color='black', linewidth=1)
        plt.title('Optimal Steps %')
        plt.legend()
