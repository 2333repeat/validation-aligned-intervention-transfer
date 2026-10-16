import matplotlib.pyplot as plt
class Visualizer:
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
        plt.show()

    def draw_final(self, SARSA_test_account, save_path):
        plt.clf()
        fig = plt.figure()
        plt.plot(SARSA_test_account['Capitals'], label='DDQN')
        plt.grid()
        plt.xlabel('Time Step')
        # Set the y-axis label.
        plt.ylabel('Capitals')
        plt.legend()
        plt.savefig(''.join(['Figures/DDQN/', save_path, '_comparing_four_strategies', '.eps']), format='eps', dpi=1000,
                    bbox_inches='tight')
        plt.show()

    def draw(self, account):
        fig, ax1 = plt.subplots(figsize=(12, 5))
        # ax2 = ax1.twinx()

        self.account = account
        lns1 = ax1.plot(self.account['Close'], color='royalblue', lw=2, label="Price")
        lns2 = ax1.plot(self.account.loc[self.account['Action'] == 1.0].index,
                        self.account['Close'][self.account['Action'] == 1.0],
                        '^', markersize=5, color='green', label="Long")
        lns3 = ax1.plot(self.account.loc[self.account['Action'] == -1.0].index,
                        self.account['Close'][self.account['Action'] == -1.0],
                        'v', markersize=5, color='red', label="Short")
        lns4 = ax1.plot(self.account.loc[self.account['Action'] == 2.0].index,
                        self.account['Close'][self.account['Action'] == 2.0],
                        'x', markersize=5, color='black', label="Close")

        ax1.set_xlabel('Days', fontsize=14)
        ax1.set_ylabel("Close Price", fontsize=14)
        ax1.tick_params(labelsize=12)
        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper center', ncol=4, fontsize=14, frameon=False)
        plt.savefig(''.join(['Figures/SARSA/', 'GE_SARSA_Actions', '.eps']), format='eps', dpi=1000,
                    bbox_inches='tight')
        plt.show()
