import os
import random

import numpy as np
import pandas as pd
import torch


def setup_seed(seed: int = 1) -> None:
    """设置随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def confirm_makedirs(path: str) -> None:
    """若目录不存在则创建。"""
    if not os.path.exists(path):
        os.makedirs(path)


class Preprocessor:
    """数据预处理：清洗 + 技术指标 + 日期截取。"""

    def computeIndicator(self, df: pd.DataFrame, start_date, end_date):
        """计算指标并返回 (features_df, trend_df)。

        参数：
        - df: 原始 OHLCV 表（至少包含 Date/Open/High/Low/Close/Volume）
        - start_date/end_date: datetime 或可比较的日期类型
        """
        # 过滤零成交量并显式 copy，避免 SettingWithCopyWarning 污染训练/评测日志
        df = df[df["Volume"] != 0].copy()

        # 类型转换
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df.loc[:, col] = df[col].astype(float)
        # pandas 的 string dtype 列不能用 .loc 直接塞 DatetimeArray，改为整列重建赋值。
        df["Date"] = pd.to_datetime(df["Date"])

        # 日期截取（兼容传入 str/datetime）
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        df = df.loc[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()
        df.reset_index(drop=True, inplace=True)

        # 技术指标（极简版本）
        df.loc[:, "MA_5"] = df["Close"].rolling(5, min_periods=1).mean()
        df.loc[:, "MA_10"] = df["Close"].rolling(10, min_periods=1).mean()
        df.ffill(inplace=True)

        features = df[["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10"]].copy()
        trend = df[["Date", "Close"]].copy()
        return features, trend


def get_date(datasetName: str, start_date: str = "2007-01-01", end_date: str = "2017-12-31"):
    """返回指定数据集在 [start_date,end_date] 的 Date 序列（用于画图刻度）。"""
    data = pd.read_csv(f"Data/{datasetName}.csv")
    data = data.dropna()
    data = data[~data["Volume"].isin([0])].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    # 兼容传入 str/datetime，避免 Timestamp 与 str 直接比较报错
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    data = data.loc[(data["Date"] >= start_ts) & (data["Date"] <= end_ts)].copy()
    data.reset_index(drop=True, inplace=True)
    return data["Date"]
