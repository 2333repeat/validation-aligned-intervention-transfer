from utils.utils import Preprocessor
import numpy as np
import math


class TradingEnv:
    """Daily-bar trading environment for Policy-Value RL agents.

    The environment maintains account state, executes discrete actions, computes
    rewards, and returns rolling-window market observations.
    """

    def __init__(self, df, startingDate, endingDate, init_money=500000):
        """Initialize market window, capital, transaction-cost, and date range."""
        self.n_actions = 3  # 0: hold, 1: buy/open long, 2: sell/close long
        self.window_size = 20  # rolling observation window length
        self.init_money = init_money  # initial capital
        self.transitionCost = 0.003  # proportional transaction cost
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.raw_df = df.copy()

        # Market data and account state are initialized lazily by init_market_data().
        self.preprocessed_market_data = None
        self.trend = None
        self.account = None
        self.input_size = None
        self.terminal_date = None
        self.t = 0

    # -------------------- Market data initialization --------------------
    def init_market_data(self):
        preprocessor = Preprocessor()
        preprocessed_data, trend_df = preprocessor.computeIndicator(
            self.raw_df, self.startingDate, self.endingDate)
        self.preprocessed_market_data = preprocessed_data
        self.trend = trend_df['Close']

        # Account table.
        self.account = trend_df.copy()
        self.account['Position'] = 0.0  # -1: short, 0: flat, 1: long
        self.account['Action'] = 0.0
        self.account['Q_Action'] = 0.0
        self.account['Holdings'] = 0.0
        self.account['Cash'] = float(self.init_money)
        self.account['Capitals'] = self.account['Holdings'] + self.account['Cash']
        self.account['Returns'] = 0.0

        self.input_size = self.preprocessed_market_data.shape[1]
        self.terminal_date = len(self.trend) - self.window_size - 1

    # -------------------- Environment reset --------------------
    def reset(self, startingPoint=1):
        if self.preprocessed_market_data is None:
            self.init_market_data()
        # Clamp the requested start point into the valid rolling-window range.
        self.t = np.clip(startingPoint, self.window_size - 1, self.terminal_date - self.window_size)
        # Reset account state.
        self.account['Position'] = 0.0  # -1: short, 0: flat, 1: long
        self.account['Action'] = 0.0  # 0: hold, 1: buy, -1: sell
        self.account['Q_Action'] = 0.0  # action recorded for Q-style agents
        self.account['Holdings'] = 0.0
        self.account['Cash'] = float(self.init_money)
        self.account['Capitals'] = float(self.init_money)
        self.account['Returns'] = 0.0

        # Position state.
        self.holdingNum = 0
        self.reward = 0.0

        return self.get_state(self.t)

    # -------------------- Observation construction --------------------
    def get_state(self, t):
        data_slice = self.preprocessed_market_data.iloc[t + 1 - self.window_size:t + 1, :]
        # Standardize each feature within the rolling window.
        state = (data_slice - data_slice.mean(axis=0)) / (data_slice.std(axis=0) + 1e-8)
        # state = (data_slice - data_slice.mean()) / (data_slice.std() + 1e-8)
        return np.ravel(state.values)

    # -------------------- Action execution --------------------
    def buy_stock(self):
        """Execute the buy action: open long or close an existing short."""
        if self.account.loc[self.t - 1, 'Position'] == 0:
            # Use all available cash to open the long position.
            max_pos = int(
                self.account.loc[self.t-1, 'Cash'] / (self.account.loc[self.t, 'Close'] * (1 + self.transitionCost)))
            self.holdingNum = max_pos
            self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash'] - self.holdingNum * \
                                               self.account.loc[self.t, 'Close'] * (1 + self.transitionCost)
            # Update position state.
            self.account.loc[self.t, 'Position'] = 1.0
            self.account.loc[self.t, 'Action'] = 1.0

        elif self.account.loc[self.t - 1, 'Position'] == -1:  # close short
            self.account.loc[self.t, 'Cash'] = self.account.loc[
                                                   self.t - 1, 'Cash'] - self.holdingNum * \
                                               self.account.loc[self.t, 'Close'] * (1 + self.transitionCost)
            self.account.loc[self.t, 'Position'] = 0
            self.account.loc[self.t, 'Action'] = 2

        else:  # already long; no-op
            self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash']
            self.account.loc[self.t, 'Position'] = self.account.loc[self.t - 1, 'Position']
            self.account.loc[self.t, 'Action'] = 0






    def sell_stock(self):
        """Execute the sell action: close an existing long or open short."""
        if self.account.loc[self.t - 1, 'Position'] == 1:  # close long
            # Update cash after transaction cost.
            self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash'] + self.holdingNum * \
                                               self.account.loc[self.t, 'Close'] * (1 - self.transitionCost)
            # Reset position state.
            self.holdingNum = 0
            self.account.loc[self.t, 'Position'] = 0
            self.account.loc[self.t, 'Action'] = -1.0


        elif self.account.loc[self.t - 1, 'Position'] == 0:  # open short from flat
            self.holdingNum = math.floor(
                self.account.loc[self.t - 1, 'Cash'] / (self.account.loc[self.t, 'Close'] * (1 + self.transitionCost)))

            self.account.loc[self.t, 'Cash'] = self.account.loc[
                                                   self.t - 1, 'Cash'] + self.holdingNum * \
                                               self.account.loc[self.t, 'Close'] * (1 - self.transitionCost)
            self.account.loc[self.t, 'Action'] = -1
            self.account.loc[self.t, 'Position'] = -1

        else:  # already short; no-op
            self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash']
            self.account.loc[self.t, 'Action'] = 0.0
            self.account.loc[self.t, 'Position'] = self.account.loc[self.t - 1, 'Position']






    def riskControl(self):
        self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash']
        self.account.loc[self.t, 'Action'] = 0
        self.account.loc[self.t, 'Position'] = self.account.loc[self.t - 1, 'Position']

    # -------------------- Environment step --------------------
    def step(self, action):

        # Execute action.
        if action == 1 and self.t < (len(self.trend) - 1):
            self.account.loc[self.t, 'Q_Action'] = 1.0
            self.buy_stock()
        elif action == 2 and self.t < (len(self.trend) - 1):
            self.account.loc[self.t, 'Q_Action'] = -1.0
            self.sell_stock()
        elif action == 0 and self.t < (len(self.trend) - 1):  # hold
            self.account.loc[self.t, 'Q_Action'] = 0
            self.riskControl()

        # print( self.account.loc[self.t-1, 'Position'],action,self.account.loc[self.t, 'Q_Action'], self.account.loc[self.t, 'Position'])
        # Reward is a short-horizon Sharpe-like signal aligned with position sign.
        chg = []
        for i in range(5):
            chg.append((self.trend[self.t + i + 1] - self.trend[self.t]) / self.trend[self.t])
        reward = np.mean(chg) / (np.std(chg) + 1e-6)
        if self.account.loc[self.t, 'Position'] == 1:
            self.reward = reward
        elif self.account.loc[self.t, 'Position'] == -1:
            self.reward = -reward
        else:
            self.reward = 0

        # Update account value.
        self.account.loc[self.t, 'Holdings'] = self.account.loc[self.t, 'Position'] * self.holdingNum * \
                                               self.account.loc[self.t, 'Close']
        self.account.loc[self.t, 'Capitals'] = self.account.loc[self.t, 'Cash'] + self.account.loc[self.t, 'Holdings']

        # Compute one-step account return.
        self.account.loc[self.t, 'Returns'] = (self.account.loc[self.t, 'Capitals'] - self.account.loc[
            self.t - 1, 'Capitals']) / self.account.loc[self.t - 1, 'Capitals']

        # Advance time.
        self.t += 1
        done = False
        if self.t == self.terminal_date:
            done = True
            # Drop rows beyond the terminal step from the saved account table.
            self.account = self.account.drop(index=(self.account.loc[(self.account.index >= self.t)].index))

        next_state = self.get_state(self.t)
        return next_state, self.reward, done

    # -------------------- Persistence --------------------
    def save_account(self, filename):
        """Save the account trajectory."""
        self.account.to_csv(filename, index=False)
