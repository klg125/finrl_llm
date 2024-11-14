import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from datetime import timedelta

class Task2Env(gym.Env):
    """A training environment for LLM-based agents with sentiment and volatility analysis."""

    def __init__(
        self,
        model,
        tokenizer,
        stock_data,
        esg_data,
        tech_analysis_data,
        scale_range=(-10, 10),
        max_steps=252 - 4,
        threshold=3,
        lookahead=3,
        strong_positive_return=0.05,  # Hyperparameter for strong return
        moderate_positive_return=0.02,  # Moderate return threshold
        weak_positive_return=0.005,  # Weak return threshold
        low_risk_threshold=4,  # Threshold for low-risk volatility
        reward_hyperparams=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.stock_data = stock_data.reset_index(drop=True)
        self.esg_data = esg_data
        self.tech_analysis_data = tech_analysis_data
        self.threshold = threshold
        self.lookahead = lookahead
        self.strong_positive_return = strong_positive_return
        self.moderate_positive_return = moderate_positive_return
        self.weak_positive_return = weak_positive_return
        self.low_risk_threshold = low_risk_threshold
        self.max_episode_steps = max_steps
        self.date_groups = []

        # Group the data by 'Date' so that we can access all data on the same date
        for date, group in stock_data.groupby("Date"):
            self.date_groups.append((date, group))

        self.current_step = 0
        self.rewards = []
        self.states = []
        self.cumulative_returns = []

        # Default reward hyperparameters
        if reward_hyperparams is None:
            self.reward_hyperparams = {
                'great_positive_reward': 2.0,
                'moderate_positive_reward': 1.5,
                'weak_positive_reward': 1.0,
                'passive_reward': 0.0,
                'moderate_negative_reward': -1.0,
                'strong_negative_reward': -1.5,
                'esg_bonus': 0.1,
            }
        else:
            self.reward_hyperparams = reward_hyperparams

        self.eval_amt = 1e6

    def reset(self):
        self.current_step = 0
        self.rewards = []
        self.states = []
        self.cumulative_returns = []
        return self._get_state()

    def step(self, action):
        sentiment_action = action.get('sentiment')
        volatility_action = action.get('volatility')

        reward, price_return = self._calculate_reward(sentiment_action, volatility_action)
        running_eval = self._evaluate_model(sentiment_action, volatility_action)

        self.current_step += 1
        done = self.current_step >= self.max_episode_steps

        self._get_state()

        self.states.append(self.state)
        self.rewards.append(reward)

        return (
            self.state,
            reward,
            done,
            {"price change": price_return, "running eval": running_eval},
        )

    def render(self):
        pass

    def _get_state(self):
        self.state = self.date_groups[self.current_step]
        return self.state

    def _calculate_reward(self, sentiment_score, volatility_score):
        date, data = self.state
        c_price = data["Close"].values[0]
        f_price = self._get_future_price(date)
        value_change = (f_price - c_price) / c_price

        # Fetch ESG score for the date
        esg_score = self._get_esg_score(date)

        # Sentiment Reward based on return strength and risk
        if sentiment_score >= self.threshold:
            if value_change > self.strong_positive_return and volatility_score < self.low_risk_threshold:
                sentiment_reward = self.reward_hyperparams['great_positive_reward']
            elif value_change > self.moderate_positive_return:
                sentiment_reward = self.reward_hyperparams['moderate_positive_reward']
            elif value_change > self.weak_positive_return:
                sentiment_reward = self.reward_hyperparams['weak_positive_reward']
            else:
                sentiment_reward = self.reward_hyperparams['passive_reward']
        elif sentiment_score <= -self.threshold:
            if value_change < -self.strong_positive_return + 0.02:
                sentiment_reward = self.reward_hyperparams['strong_negative_reward']
            elif value_change < -self.moderate_positive_return + 0.01:
                sentiment_reward = self.reward_hyperparams['moderate_negative_reward']
            else:
                sentiment_reward = self.reward_hyperparams['passive_reward']
        else:
            sentiment_reward = self.reward_hyperparams['passive_reward']

        # Apply technical analysis "make or break"
        tech_analysis_prediction = self._get_technical_analysis_prediction(date)
        if tech_analysis_prediction == 0:
            if sentiment_reward > 0:
                sentiment_reward = 0  # Neutralize positive reward if technical analysis disagrees

        # ESG Bonus
        esg_bonus = self.reward_hyperparams['esg_bonus'] * esg_score / 10

        total_reward = sentiment_reward + esg_bonus
        return total_reward, value_change

    def _get_future_price(self, current_date):
        future_date = pd.to_datetime(current_date) + timedelta(days=self.lookahead)
        future_data = self.stock_data[self.stock_data['Date'] == future_date.strftime('%Y-%m-%d')]
        if not future_data.empty:
            return future_data["Close"].values[0]
        else:
            return self.stock_data[self.stock_data['Date'] == current_date]["Close"].values[0]

    def _get_esg_score(self, date):
        esg_row = self.esg_data[self.esg_data['Date'] == date]
        if not esg_row.empty:
            return esg_row['ESG_score'].values[0]
        else:
            return 0

    def _evaluate_model(self, sentiment_score, volatility_score):
        date, data = self.state
        c_price = data["Close"].values[0]
        f_price = self._get_future_price(date)
        print(sentiment_score)
        # Check sentiment score to decide on trading action
        if sentiment_score >= self.threshold:
            # Long position, calculate value change
            value_change = (f_price - c_price) / c_price
            
            # If volatility is low, double the value change (simulating two shares)
            if volatility_score < self.low_risk_threshold:
                value_change *= 2
        else:
            # No action taken if sentiment is below threshold
            value_change = 0

        # Update evaluation amount based on value change
        self.eval_amt = self.eval_amt * (1 + value_change)
        return self.eval_amt

    def _get_technical_analysis_prediction(self, date):
        tech_row = self.tech_analysis_data[self.tech_analysis_data['Date'] == date]
        if not tech_row.empty:
            return tech_row['buy'].values[0]
        else:
            return 0
