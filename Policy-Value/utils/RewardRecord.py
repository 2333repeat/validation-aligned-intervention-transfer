import os.path

from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick
from .utils import get_date
import matplotlib.dates as mdates


class RewardRecord:
    def __init__(self, saveDirName, cfg):
        self.cfg = cfg
        self.saveDirName = saveDirName
        self.rewards = []
        self.valRewards = []
        self.episode = 0
        self.episode_count = []
        self.trainSavePath = os.path.join(self.saveDirName, 'train.png')
        self.rewardcsvSavePath = os.path.join(self.saveDirName, 'trainReward.csv')

    def appendReward(self, reward):
        """Append one training reward and refresh the reward plot."""

        self.rewards.append(reward)
        self.RewardPlot(False)

    def appendRewardWithVal(self, reward, agentName):
        """Append one validation-cycle reward and refresh the reward plot."""

        self.episode += 1
        self.rewards.append(reward)
        self.episode_count.append(self.episode)
        self.RewardPlot(agentName)

    def RewardPlot(self, agentName):
        trainSavePath = os.path.join(self.saveDirName, agentName+'_train.png')
        rewardcsvSavePath = os.path.join(self.saveDirName, agentName+'_trainReward.csv')
        iters = range(len(self.rewards))
        agent_reward_pd = pd.DataFrame(
            {'episode': self.episode_count, 'accumulated reward': self.rewards})
        agent_reward_pd.to_csv(rewardcsvSavePath, sep=',')
        plt.plot(iters, self.rewards, 'red', linewidth=2, label='train Reward')
        plt.title('Accumulated Reward of {}'.format(agentName), fontsize=16)
        # plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.grid()
        plt.legend(loc="upper left")
        plt.savefig(trainSavePath)
        plt.cla()
        
        plt.close("all")

    def CapitalPlot(self, account, agentName, isTrain=False):
        capitalSavePath = os.path.join(self.saveDirName, agentName+'_testBestCapital_{}.png'.format(self.episode))
        returnRateSavePath = os.path.join(self.saveDirName, agentName+'_testBestReturnRate_{}.png'.format(self.episode))
        accountSavePath = os.path.join(self.saveDirName, agentName+'_testAccount.csv')
        test_return_rate = (account['Capitals'] - 500000) * 100 / 500000
        if 'returnRate' not in account.columns:
            account.insert(account.shape[-1], 'returnRate', test_return_rate)
        else:
            account['returnRate'] = (account['Capitals'] - 500000) * 100 / 500000
        account.to_csv(accountSavePath, sep=',')
        if not isTrain:
            date = get_date(self.cfg.dataSetName, self.cfg.time.test_startingDate, self.cfg.time.test_endingDate)
            date = date.loc[self.cfg.input.length:len(account)+(self.cfg.input.length-1)]
            tick_dates = pd.date_range(start=self.cfg.time.test_startingDate, end=self.cfg.time.test_endingDate, periods=8)
        else:
            date = get_date(self.cfg.dataSetName, self.cfg.time.train_startingDate, self.cfg.time.train_endingDate)
            date = date.loc[self.cfg.input.length:len(account) + (self.cfg.input.length - 1)]
            tick_dates = pd.date_range(start=self.cfg.time.train_startingDate, end=self.cfg.time.train_endingDate,
                                       periods=8)
        fig1, ax1 = plt.subplots(figsize=(14,10))
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(date_format)
        ax1.set_xticks(tick_dates)
        plt.xticks(rotation=45)
        ax1.plot(date, account['Capitals'], 'red', linewidth=2, label='Test Capitals')
        ax1.set_title(self.cfg.dataSetName+' '+ agentName+' Test Capitals---'+self.cfg.dataDir)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capitals')
        ax1.legend(loc="upper left")
        ax1.grid(True)
        plt.savefig(capitalSavePath)

        fig2, ax2 = plt.subplots(figsize=(14, 10))
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_formatter(date_format)
        ax2.set_xticks(tick_dates)
        plt.xticks(rotation=45)
        ax2.plot(date, account['returnRate'], 'blue', linewidth=2, label='Test Accumulated Return Rate')
        ax2.set_title(self.cfg.dataSetName+' '+ agentName+' Test Return Rate---'+self.cfg.dataDir)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Accumulated Return Rate')
        ax2.legend(loc="upper left")
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.grid(True)
        plt.savefig(returnRateSavePath)
        plt.close("all")

    def training_CapitalPlot(self, account, agentName):
        capitalSavePath = os.path.join(self.saveDirName, agentName+'_trainBestCapital_{}.png'.format(self.episode))
        returnRateSavePath = os.path.join(self.saveDirName, agentName+'_trainBestReturnRate_{}.png'.format(self.episode))
        accountSavePath = os.path.join(self.saveDirName, agentName+'_trainAccount.csv')
        test_return_rate = (account['Capitals'] - 500000) * 100 / 500000
        if 'returnRate' not in account.columns:
            account.insert(account.shape[-1], 'returnRate', test_return_rate)
        else:
            account['returnRate'] = (account['Capitals'] - 500000) * 100 / 500000
        account.to_csv(accountSavePath, sep=',')
        date = get_date(self.cfg.dataSetName, self.cfg.time.train_startingDate, self.cfg.time.train_endingDate)
        date = date.loc[self.cfg.input.length:len(account)+(self.cfg.input.length-1)]
        tick_dates = pd.date_range(start=self.cfg.time.train_startingDate, end=self.cfg.time.train_endingDate,
                                   periods=8)
        fig1, ax1 = plt.subplots(figsize=(14, 10))
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(date_format)
        ax1.set_xticks(tick_dates)
        plt.xticks(rotation=45)
        ax1.plot(date, account['Capitals'], 'red', linewidth=2, label='Train Capitals')
        ax1.set_title(self.cfg.dataSetName+' '+ agentName+' Train Capitals---'+self.cfg.dataDir)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capitals')
        ax1.legend(loc="upper left")
        ax1.grid(True)
        plt.savefig(capitalSavePath)

        fig2, ax2 = plt.subplots(figsize=(14, 10))
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_formatter(date_format)
        ax2.set_xticks(tick_dates)
        plt.xticks(rotation=45)
        ax2.plot(date, account['returnRate'], 'blue', linewidth=2, label='Train Accumulated Return Rate')
        ax2.set_title(self.cfg.dataSetName+' '+ agentName+' Train Return Rate---'+self.cfg.dataDir)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Accumulated Return Rate')
        ax2.legend(loc="upper left")
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.grid(True)
        plt.savefig(returnRateSavePath)
        plt.close("all")
    def draw_SR(self, lsttdqn_sr, dqn_sr, file_name):
        plt.clf()
        plt.plot(dqn_sr, label='DQN')
        plt.plot(lsttdqn_sr, label='E-DQN')
        plt.xlabel('Training Episode')
                # Set the y-axis label.
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.savefig(''.join(['Figures/', file_name, '_comparing_training_SR', '.eps']), format='eps', dpi=1000,
                    bbox_inches='tight')
        # Training/evaluation runs are non-interactive; avoid blocking on plt.show().
        plt.close("all")

    def draw_final(self, account):
        test = os.path.join(self.saveDirName, 'test_{}.eps'.format(self.episode))
        plt.clf()
        fig = plt.figure()
        plt.plot(account['Capitals'], label='Capitals')
        plt.grid()
        plt.xlabel('Time Step')
                # Set the y-axis label.
        plt.ylabel('Capitals')
        plt.legend()
        plt.savefig(test, format='eps', dpi=1000,
                    bbox_inches='tight')
        plt.close("all")

    def draw(self, account, agentName):
        actionSavePath = os.path.join(self.saveDirName, agentName+'_testAction_{}.png'.format(self.episode))
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(account['Close'], color='royalblue', lw=2, label="Price")
        ax1.plot(account.loc[account['Action'] == 1.0].index,
                        account['Close'][account['Action'] == 1.0],
                        '^', markersize=5, color='green', label="Long")
        ax1.plot(account.loc[account['Action'] == -1.0].index,
                        account['Close'][account['Action'] == -1.0],
                        'v', markersize=5, color='red', label="Short")
        ax1.plot(account.loc[account['Action'] == 2.0].index,
                        account['Close'][account['Action'] == 2.0],
                        'x', markersize=5, color='black', label="Close")

        ax1.set_xlabel('Days', fontsize=14)
        ax1.set_ylabel("Close Price", fontsize=14)
        ax1.tick_params(labelsize=12)

        ax1.set_xlabel('Days', fontsize=14)
        ax1.set_ylabel("Close Price", fontsize=14)
        ax1.tick_params(labelsize=12)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Close Price')
        ax1.legend(loc='upper center', ncol=3, frameon=False)
        plt.savefig(actionSavePath, dpi=1000, bbox_inches='tight')
        plt.close("all")
