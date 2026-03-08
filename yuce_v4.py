import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
import io
import json
import os
from datetime import datetime
import zipfile
from io import BytesIO

warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 尝试导入必要的库，处理缺失情况
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    from prophet import Prophet
    import logging
    logging.getLogger('prophet').setLevel(logging.ERROR)
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
except Exception:
    # 如果 Prophet 导入时出现任何错误,将其标记为不可用
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from sklearn.svm import SVR
    SVR_AVAILABLE = True
except ImportError:
    SVR_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge
    RIDGE_AVAILABLE = True
except ImportError:
    RIDGE_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    ETS_AVAILABLE = True
except ImportError:
    ETS_AVAILABLE = False

import streamlit as st
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# ==========================================
# 1. 数据模型
# ==========================================

@dataclass
class IndexConfig:
    """训练集和测试集索引配置"""
    train_start: int
    train_end: int
    total_length: int

    @property
    def train_length(self) -> int:
        """训练集长度"""
        return self.train_end - self.train_start

    @property
    def test_length(self) -> int:
        """测试集长度"""
        return self.total_length - self.train_end

    @property
    def test_start(self) -> int:
        """测试集起始索引"""
        return self.train_end

    @property
    def test_end(self) -> int:
        """测试集结束索引"""
        return self.total_length


@dataclass
class ValidationResult:
    """索引验证结果"""
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


@dataclass
class DatasetSplit:
    """数据集划分结果"""
    train_series: pd.Series
    test_series: pd.Series
    full_series: pd.Series
    config: IndexConfig


def validate_index_config(
    train_start: int,
    train_end: int,
    total_length: int,
    min_train_length: int = 10,
    min_test_length: int = 1
) -> ValidationResult:
    """
    验证索引配置的合法性

    Args:
        train_start: 训练集起始索引
        train_end: 训练集结束索引
        total_length: 数据总长度
        min_train_length: 最小训练集长度，默认10
        min_test_length: 最小测试集长度，默认1

    Returns:
        ValidationResult: 验证结果对象
    """
    # 检查起始索引是否为负数
    if train_start < 0:
        return ValidationResult(
            is_valid=False,
            error_message="起始索引不能为负数"
        )

    # 检查结束索引是否超出数据范围
    if train_end > total_length:
        return ValidationResult(
            is_valid=False,
            error_message=f"结束索引超出数据范围（最大允许值：{total_length}）"
        )

    # 检查结束索引是否大于起始索引
    if train_end <= train_start:
        return ValidationResult(
            is_valid=False,
            error_message="结束索引必须大于起始索引"
        )

    # 检查训练集长度是否足够
    train_length = train_end - train_start
    if train_length < min_train_length:
        return ValidationResult(
            is_valid=False,
            error_message=f"训练集长度不足（当前：{train_length}，最小要求：{min_train_length}）"
        )

    # 检查测试集长度是否足够
    test_length = total_length - train_end
    if test_length < min_test_length:
        return ValidationResult(
            is_valid=False,
            error_message=f"测试集长度不足（当前：{test_length}，最小要求：{min_test_length}）"
        )

    # 所有检查通过
    return ValidationResult(is_valid=True)


def calculate_constraints(
    total_length: int,
    current_train_start: Optional[int] = None,
    current_train_end: Optional[int] = None,
    min_train_length: int = 10,
    min_test_length: int = 1
) -> Dict[str, Tuple[int, int]]:
    """
    计算输入控件的约束范围

    Args:
        total_length: 数据总长度
        current_train_start: 当前训练集起始索引（可选）
        current_train_end: 当前训练集结束索引（可选）
        min_train_length: 最小训练集长度，默认10
        min_test_length: 最小测试集长度，默认1

    Returns:
        Dict[str, Tuple[int, int]]: 包含train_start和train_end约束的字典
    """
    # 计算train_start的约束
    train_start_min = 0
    train_start_max = total_length - min_train_length - min_test_length

    # 计算train_end的约束
    if current_train_start is not None:
        train_end_min = max(current_train_start + min_train_length, min_train_length)
    else:
        train_end_min = min_train_length

    train_end_max = total_length - min_test_length

    return {
        'train_start': (train_start_min, train_start_max),
        'train_end': (train_end_min, train_end_max)
    }


# ==========================================
# 2. 核心功能函数
# ==========================================

def load_data(file):
    """加载数据并返回DataFrame和数值列信息"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file)
    elif file.name.endswith('.txt'):
        # 尝试多种分隔符
        try:
            df = pd.read_csv(file, sep='\s+', header=None)
            if df.shape[1] == 1:
                df.columns = ['value']
            else:
                df = pd.read_csv(file, header=None)
        except:
            df = pd.read_csv(file, header=None)
    else:
        raise ValueError("不支持的文件格式")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("文件中未找到数值列")
    
    return df, numeric_cols


def split_dataset(
    data_series: pd.Series,
    config: IndexConfig
) -> DatasetSplit:
    """
    根据索引配置划分数据集

    Args:
        data_series: 完整的数据序列
        config: 索引配置对象

    Returns:
        DatasetSplit: 数据集划分结果

    Raises:
        IndexError: 当索引超出数据范围时抛出
    """
    # 验证索引范围
    if config.train_start < 0 or config.train_end > config.total_length:
        raise IndexError("索引超出数据范围")

    # 切片数据
    train_series = data_series.iloc[config.train_start:config.train_end].reset_index(drop=True)
    test_series = data_series.iloc[config.train_end:].reset_index(drop=True)
    full_series = data_series.iloc[config.train_start:config.train_end + config.test_length].reset_index(drop=True)

    return DatasetSplit(
        train_series=train_series,
        test_series=test_series,
        full_series=full_series,
        config=config
    )


def detect_outliers(series, method='zscore', threshold=3):
    """异常值检测"""
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    return pd.Series([False] * len(series))

def create_features(data, lag_days=30):
    """构建 XGBoost/LightGBM 特征"""
    df = pd.DataFrame(data)
    df.columns = ['value']
    # 动态调整 lag_days，防止超过数据长度
    actual_lag = min(lag_days, len(data) - 1)
    if actual_lag < 1:
        return pd.DataFrame() # 数据太少无法构建特征
        
    for i in range(1, actual_lag + 1):
        df[f'lag_{i}'] = df['value'].shift(i)
    
    # 滚动窗口也动态调整
    for window in [7, 14, 30]:
        if window < len(data):
            df[f'rolling_mean_{window}'] = df['value'].shift(1).rolling(window=min(window, len(data)-1)).mean()
            df[f'rolling_std_{window}'] = df['value'].shift(1).rolling(window=min(window, len(data)-1)).std()
            df[f'rolling_max_{window}'] = df['value'].shift(1).rolling(window=min(window, len(data)-1)).max()
            df[f'rolling_min_{window}'] = df['value'].shift(1).rolling(window=min(window, len(data)-1)).min()
    
    # 添加趋势特征
    df['trend'] = df['value'].diff().shift(1)
    
    df = df.dropna()
    return df

def run_xgb_model(train, test, full_series):
    if not XGB_AVAILABLE:
        return None, None, None, "XGBoost 库未安装"
    
    # 动态调整滞后天数，最大不超过训练集长度的 1/3 或 30
    lag_days = min(30, max(1, len(train) // 3))
    
    df_feat = create_features(full_series, lag_days=lag_days)
    if df_feat.empty:
        return None, None, None, "训练集数据不足以构建 XGBoost 特征"
        
    feature_cols = [c for c in df_feat.columns if c != 'value']
    
    # 计算有效的训练集起始点 (减去 dropna 丢失的行)
    valid_train_count = len(train) - lag_days - 2 # 预留一点余量给 rolling window
    
    if valid_train_count <= 0:
        return None, None, None, "训练集长度不足以生成有效特征"

    X_train = df_feat.iloc[:valid_train_count][feature_cols]
    y_train = df_feat.iloc[:valid_train_count]['value']
    
    # 测试集部分
    test_start = valid_train_count
    test_end = test_start + len(test)
    
    # 确保不越界
    if test_end > len(df_feat):
        test_end = len(df_feat)
        
    if test_start >= len(df_feat):
        return None, None, None, "测试集索引超出特征范围"

    X_test = df_feat.iloc[test_start:test_end][feature_cols]
    y_test = df_feat.iloc[test_start:test_end]['value']

    if len(X_test) == 0:
        return None, None, None, "测试集为空"

    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.01, subsample=0.8, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    
    # 预测目标点 (下一个点)
    # 需要构建最后一行的特征
    last_vals = full_series.iloc[-(lag_days+5):].values # 多取一点防越界
    new_row = {}
    for i in range(1, lag_days + 1):
        if i <= len(last_vals):
            new_row[f'lag_{i}'] = last_vals[-i]
        else:
            new_row[f'lag_{i}'] = np.nan # 填充缺失
            
    for window in [7, 14, 30]:
        w = min(window, len(last_vals))
        if w > 0:
            new_row[f'rolling_mean_{window}'] = np.mean(last_vals[-w:])
            new_row[f'rolling_std_{window}'] = np.std(last_vals[-w:])
            new_row[f'rolling_max_{window}'] = np.max(last_vals[-w:])
            new_row[f'rolling_min_{window}'] = np.min(last_vals[-w:])
        else:
            new_row[f'rolling_mean_{window}'] = 0
            new_row[f'rolling_std_{window}'] = 0
            new_row[f'rolling_max_{window}'] = 0
            new_row[f'rolling_min_{window}'] = 0
    
    # 添加趋势特征
    if len(last_vals) > 1:
        new_row['trend'] = last_vals[-1] - last_vals[-2]
    else:
        new_row['trend'] = 0
            
    # 处理可能的 NaN (如果数据太短)
    X_target = pd.DataFrame([new_row])[feature_cols].fillna(method='ffill', axis=1).fillna(0)
    
    try:
        pred_target = model.predict(X_target)[0]
    except:
        pred_target = np.nan
        
    return y_pred_test, pred_target, y_test.values, None

def run_lstm_model(train, test, full_series):
    if not LSTM_AVAILABLE:
        return None, None, None, "TensorFlow 未安装"
    
    # 动态时间步长
    time_steps = min(20, len(train) // 2)
    if time_steps < 2:
        return None, None, None, "训练集太短，无法构建 LSTM 序列"

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    full_scaled = scaler.transform(full_series.values.reshape(-1, 1))

    def create_seq(data, steps):
        X, y = [], []
        for i in range(len(data) - steps):
            X.append(data[i:(i+steps), 0])
            y.append(data[i+steps, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(train_scaled, time_steps)
    if len(X_train) == 0:
        return None, None, None, "训练序列生成失败"

    X_train = X_train.reshape((X_train.shape[0], time_steps, 1))

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=3)])

    # 评估
    X_full, y_full = create_seq(full_scaled, time_steps)
    if len(X_full) == 0:
        return None, None, None, "全量序列生成失败"
        
    X_full = X_full.reshape((X_full.shape[0], time_steps, 1))
    
    start_idx = len(train) - time_steps
    end_idx = start_idx + len(test)
    
    if end_idx > len(X_full):
        end_idx = len(X_full)
    if start_idx >= len(X_full):
        return None, None, None, "索引越界"
        
    y_pred_scaled = model.predict(X_full, verbose=0)
    y_pred_test = scaler.inverse_transform(y_pred_scaled[start_idx:end_idx]).flatten()
    y_test_actual = scaler.inverse_transform(y_full[start_idx:end_idx].reshape(-1, 1)).flatten()

    # 预测目标
    last_seq = full_scaled[-time_steps:].reshape(1, time_steps, 1)
    pred_target = scaler.inverse_transform(model.predict(last_seq, verbose=0))[0, 0]

    return y_pred_test, pred_target, y_test_actual, None

def run_prophet_model(train, test):
    """Prophet 时间序列预测模型"""
    if not PROPHET_AVAILABLE:
        return None, None, None, "Prophet 未安装或不可用"

    try:
        # 检查训练数据长度
        if len(train) < 10:
            return None, None, None, "Prophet 需要至少10个训练样本"

        # 准备数据
        df_train = pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=len(train), freq='D'),
            'y': train.values
        })

        # 尝试创建最简单的模型
        try:
            # 使用最简单的参数配置
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.001  # 非常小的值,减少计算复杂度
            )
            model.fit(df_train)
        except Exception as fit_error:
            return None, None, None, f"Prophet 模型拟合失败: {str(fit_error)}"

        # 创建未来数据框
        future = model.make_future_dataframe(periods=len(test) + 1)

        # 预测
        try:
            forecast = model.predict(future)
        except Exception as pred_error:
            return None, None, None, f"Prophet 预测失败: {str(pred_error)}"

        # 提取预测结果
        y_pred_test = forecast['yhat'].iloc[len(train):len(train)+len(test)].values
        pred_target = forecast['yhat'].iloc[-1]

        # 检查是否有置信区间
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            pred_lower = forecast['yhat_lower'].iloc[-1]
            pred_upper = forecast['yhat_upper'].iloc[-1]
            return y_pred_test, pred_target, (pred_lower, pred_upper), None
        else:
            return y_pred_test, pred_target, None, None

    except Exception as e:
        # 捕获所有异常并返回错误信息
        return None, None, None, f"Prophet 执行错误: {str(e)}"

def run_arima_model(train, test):
    if not ARIMA_AVAILABLE:
        return None, None, None, "ARIMA 未安装"
    
    try:
        # 自动选择最佳参数
        best_aic = np.inf
        best_order = None
        best_model = None
        
        # 限制搜索范围以提高效率
        for p in range(0, 4):
            for d in range(0, 2):
                for q in range(0, 4):
                    try:
                        model = ARIMA(train, order=(p,d,q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p,d,q)
                            best_model = fitted_model
                    except:
                        continue
        
        if best_model is None:
            return None, None, None, "ARIMA 模型拟合失败"
        
        # 预测测试集
        forecast = best_model.forecast(steps=len(test))
        y_pred_test = forecast
        
        # 预测目标点（避免按标签索引导致 KeyError: 0）
        pred_target = best_model.forecast(steps=1).iloc[0]
        
        return y_pred_test, pred_target, None, None
    except Exception as e:
        return None, None, None, f"ARIMA 错误: {str(e)}"

def run_lightgbm_model(train, test, full_series):
    if not LGBM_AVAILABLE:
        return None, None, None, "LightGBM 未安装"
    
    # 动态调整滞后天数
    lag_days = min(30, max(1, len(train) // 3))
    
    df_feat = create_features(full_series, lag_days=lag_days)
    if df_feat.empty:
        return None, None, None, "训练集数据不足以构建 LightGBM 特征"
        
    feature_cols = [c for c in df_feat.columns if c != 'value']
    
    # 计算有效的训练集起始点
    valid_train_count = len(train) - lag_days - 2
    
    if valid_train_count <= 0:
        return None, None, None, "训练集长度不足以生成有效特征"

    X_train = df_feat.iloc[:valid_train_count][feature_cols]
    y_train = df_feat.iloc[:valid_train_count]['value']
    
    # 测试集部分
    test_start = valid_train_count
    test_end = test_start + len(test)
    
    if test_end > len(df_feat):
        test_end = len(df_feat)
        
    if test_start >= len(df_feat):
        return None, None, None, "测试集索引超出特征范围"

    X_test = df_feat.iloc[test_start:test_end][feature_cols]
    y_test = df_feat.iloc[test_start:test_end]['value']

    if len(X_test) == 0:
        return None, None, None, "测试集为空"

    model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.01, 
                             subsample=0.8, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    
    # 预测目标点
    last_vals = full_series.iloc[-(lag_days+5):].values
    new_row = {}
    for i in range(1, lag_days + 1):
        if i <= len(last_vals):
            new_row[f'lag_{i}'] = last_vals[-i]
        else:
            new_row[f'lag_{i}'] = np.nan
            
    for window in [7, 14, 30]:
        w = min(window, len(last_vals))
        if w > 0:
            new_row[f'rolling_mean_{window}'] = np.mean(last_vals[-w:])
            new_row[f'rolling_std_{window}'] = np.std(last_vals[-w:])
            new_row[f'rolling_max_{window}'] = np.max(last_vals[-w:])
            new_row[f'rolling_min_{window}'] = np.min(last_vals[-w:])
        else:
            new_row[f'rolling_mean_{window}'] = 0
            new_row[f'rolling_std_{window}'] = 0
            new_row[f'rolling_max_{window}'] = 0
            new_row[f'rolling_min_{window}'] = 0
    
    # 添加趋势特征
    if len(last_vals) > 1:
        new_row['trend'] = last_vals[-1] - last_vals[-2]
    else:
        new_row['trend'] = 0
            
    X_target = pd.DataFrame([new_row])[feature_cols].fillna(method='ffill', axis=1).fillna(0)
    
    try:
        pred_target = model.predict(X_target)[0]
    except:
        pred_target = np.nan
        
    return y_pred_test, pred_target, y_test.values, None


def run_randomforest_model(train, test, full_series):
    """RandomForest 回归模型"""
    if not RF_AVAILABLE:
        return None, None, None, "RandomForest 未安装"

    # 动态调整滞后天数
    lag_days = min(30, max(1, len(train) // 3))

    df_feat = create_features(full_series, lag_days=lag_days)
    if df_feat.empty:
        return None, None, None, "训练集数据不足以构建 RandomForest 特征"

    feature_cols = [c for c in df_feat.columns if c != 'value']

    # 计算有效的训练集起始点
    valid_train_count = len(train) - lag_days - 2

    if valid_train_count <= 0:
        return None, None, None, "训练集长度不足以生成有效特征"

    X_train = df_feat.iloc[:valid_train_count][feature_cols]
    y_train = df_feat.iloc[:valid_train_count]['value']

    # 测试集部分
    test_start = valid_train_count
    test_end = test_start + len(test)

    if test_end > len(df_feat):
        test_end = len(df_feat)

    if test_start >= len(df_feat):
        return None, None, None, "测试集索引超出特征范围"

    X_test = df_feat.iloc[test_start:test_end][feature_cols]
    y_test = df_feat.iloc[test_start:test_end]['value']

    if len(X_test) == 0:
        return None, None, None, "测试集为空"

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    # 预测目标点
    last_vals = full_series.iloc[-(lag_days+5):].values
    new_row = {}
    for i in range(1, lag_days + 1):
        if i <= len(last_vals):
            new_row[f'lag_{i}'] = last_vals[-i]
        else:
            new_row[f'lag_{i}'] = np.nan

    for window in [7, 14, 30]:
        w = min(window, len(last_vals))
        if w > 0:
            new_row[f'rolling_mean_{window}'] = np.mean(last_vals[-w:])
            new_row[f'rolling_std_{window}'] = np.std(last_vals[-w:])
            new_row[f'rolling_max_{window}'] = np.max(last_vals[-w:])
            new_row[f'rolling_min_{window}'] = np.min(last_vals[-w:])
        else:
            new_row[f'rolling_mean_{window}'] = 0
            new_row[f'rolling_std_{window}'] = 0
            new_row[f'rolling_max_{window}'] = 0
            new_row[f'rolling_min_{window}'] = 0

    if len(last_vals) > 1:
        new_row['trend'] = last_vals[-1] - last_vals[-2]
    else:
        new_row['trend'] = 0

    X_target = pd.DataFrame([new_row])[feature_cols].fillna(method='ffill', axis=1).fillna(0)

    try:
        pred_target = model.predict(X_target)[0]
    except:
        pred_target = np.nan

    return y_pred_test, pred_target, y_test.values, None


def run_catboost_model(train, test, full_series):
    """CatBoost 回归模型"""
    if not CATBOOST_AVAILABLE:
        return None, None, None, "CatBoost 未安装"

    # 动态调整滞后天数
    lag_days = min(30, max(1, len(train) // 3))

    df_feat = create_features(full_series, lag_days=lag_days)
    if df_feat.empty:
        return None, None, None, "训练集数据不足以构建 CatBoost 特征"

    feature_cols = [c for c in df_feat.columns if c != 'value']

    # 计算有效的训练集起始点
    valid_train_count = len(train) - lag_days - 2

    if valid_train_count <= 0:
        return None, None, None, "训练集长度不足以生成有效特征"

    X_train = df_feat.iloc[:valid_train_count][feature_cols]
    y_train = df_feat.iloc[:valid_train_count]['value']

    # 测试集部分
    test_start = valid_train_count
    test_end = test_start + len(test)

    if test_end > len(df_feat):
        test_end = len(df_feat)

    if test_start >= len(df_feat):
        return None, None, None, "测试集索引超出特征范围"

    X_test = df_feat.iloc[test_start:test_end][feature_cols]
    y_test = df_feat.iloc[test_start:test_end]['value']

    if len(X_test) == 0:
        return None, None, None, "测试集为空"

    model = CatBoostRegressor(
        iterations=200,
        depth=6,
        learning_rate=0.01,
        random_seed=42,
        verbose=False
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    # 预测目标点
    last_vals = full_series.iloc[-(lag_days+5):].values
    new_row = {}
    for i in range(1, lag_days + 1):
        if i <= len(last_vals):
            new_row[f'lag_{i}'] = last_vals[-i]
        else:
            new_row[f'lag_{i}'] = np.nan

    for window in [7, 14, 30]:
        w = min(window, len(last_vals))
        if w > 0:
            new_row[f'rolling_mean_{window}'] = np.mean(last_vals[-w:])
            new_row[f'rolling_std_{window}'] = np.std(last_vals[-w:])
            new_row[f'rolling_max_{window}'] = np.max(last_vals[-w:])
            new_row[f'rolling_min_{window}'] = np.min(last_vals[-w:])
        else:
            new_row[f'rolling_mean_{window}'] = 0
            new_row[f'rolling_std_{window}'] = 0
            new_row[f'rolling_max_{window}'] = 0
            new_row[f'rolling_min_{window}'] = 0

    if len(last_vals) > 1:
        new_row['trend'] = last_vals[-1] - last_vals[-2]
    else:
        new_row['trend'] = 0

    X_target = pd.DataFrame([new_row])[feature_cols].fillna(method='ffill', axis=1).fillna(0)

    try:
        pred_target = model.predict(X_target)[0]
    except:
        pred_target = np.nan

    return y_pred_test, pred_target, y_test.values, None


def run_svr_model(train, test, full_series):
    """SVR 支持向量回归模型"""
    if not SVR_AVAILABLE:
        return None, None, None, "SVR 未安装"

    # 动态调整滞后天数
    lag_days = min(30, max(1, len(train) // 3))

    df_feat = create_features(full_series, lag_days=lag_days)
    if df_feat.empty:
        return None, None, None, "训练集数据不足以构建 SVR 特征"

    feature_cols = [c for c in df_feat.columns if c != 'value']

    # 计算有效的训练集起始点
    valid_train_count = len(train) - lag_days - 2

    if valid_train_count <= 0:
        return None, None, None, "训练集长度不足以生成有效特征"

    X_train = df_feat.iloc[:valid_train_count][feature_cols]
    y_train = df_feat.iloc[:valid_train_count]['value']

    # 测试集部分
    test_start = valid_train_count
    test_end = test_start + len(test)

    if test_end > len(df_feat):
        test_end = len(df_feat)

    if test_start >= len(df_feat):
        return None, None, None, "测试集索引超出特征范围"

    X_test = df_feat.iloc[test_start:test_end][feature_cols]
    y_test = df_feat.iloc[test_start:test_end]['value']

    if len(X_test) == 0:
        return None, None, None, "测试集为空"

    # 标准化数据
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    model = SVR(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train_scaled)

    X_test_scaled = scaler_X.transform(X_test)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred_test = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # 预测目标点
    last_vals = full_series.iloc[-(lag_days+5):].values
    new_row = {}
    for i in range(1, lag_days + 1):
        if i <= len(last_vals):
            new_row[f'lag_{i}'] = last_vals[-i]
        else:
            new_row[f'lag_{i}'] = np.nan

    for window in [7, 14, 30]:
        w = min(window, len(last_vals))
        if w > 0:
            new_row[f'rolling_mean_{window}'] = np.mean(last_vals[-w:])
            new_row[f'rolling_std_{window}'] = np.std(last_vals[-w:])
            new_row[f'rolling_max_{window}'] = np.max(last_vals[-w:])
            new_row[f'rolling_min_{window}'] = np.min(last_vals[-w:])
        else:
            new_row[f'rolling_mean_{window}'] = 0
            new_row[f'rolling_std_{window}'] = 0
            new_row[f'rolling_max_{window}'] = 0
            new_row[f'rolling_min_{window}'] = 0

    if len(last_vals) > 1:
        new_row['trend'] = last_vals[-1] - last_vals[-2]
    else:
        new_row['trend'] = 0

    X_target = pd.DataFrame([new_row])[feature_cols].fillna(method='ffill', axis=1).fillna(0)
    X_target_scaled = scaler_X.transform(X_target)

    try:
        pred_target_scaled = model.predict(X_target_scaled)
        pred_target = scaler_y.inverse_transform(pred_target_scaled.reshape(-1, 1))[0, 0]
    except:
        pred_target = np.nan

    return y_pred_test, pred_target, y_test.values, None


def run_ridge_model(train, test, full_series):
    """Ridge 岭回归模型"""
    if not RIDGE_AVAILABLE:
        return None, None, None, "Ridge 未安装"

    # 动态调整滞后天数
    lag_days = min(30, max(1, len(train) // 3))

    df_feat = create_features(full_series, lag_days=lag_days)
    if df_feat.empty:
        return None, None, None, "训练集数据不足以构建 Ridge 特征"

    feature_cols = [c for c in df_feat.columns if c != 'value']

    # 计算有效的训练集起始点
    valid_train_count = len(train) - lag_days - 2

    if valid_train_count <= 0:
        return None, None, None, "训练集长度不足以生成有效特征"

    X_train = df_feat.iloc[:valid_train_count][feature_cols]
    y_train = df_feat.iloc[:valid_train_count]['value']

    # 测试集部分
    test_start = valid_train_count
    test_end = test_start + len(test)

    if test_end > len(df_feat):
        test_end = len(df_feat)

    if test_start >= len(df_feat):
        return None, None, None, "测试集索引超出特征范围"

    X_test = df_feat.iloc[test_start:test_end][feature_cols]
    y_test = df_feat.iloc[test_start:test_end]['value']

    if len(X_test) == 0:
        return None, None, None, "测试集为空"

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    # 预测目标点
    last_vals = full_series.iloc[-(lag_days+5):].values
    new_row = {}
    for i in range(1, lag_days + 1):
        if i <= len(last_vals):
            new_row[f'lag_{i}'] = last_vals[-i]
        else:
            new_row[f'lag_{i}'] = np.nan

    for window in [7, 14, 30]:
        w = min(window, len(last_vals))
        if w > 0:
            new_row[f'rolling_mean_{window}'] = np.mean(last_vals[-w:])
            new_row[f'rolling_std_{window}'] = np.std(last_vals[-w:])
            new_row[f'rolling_max_{window}'] = np.max(last_vals[-w:])
            new_row[f'rolling_min_{window}'] = np.min(last_vals[-w:])
        else:
            new_row[f'rolling_mean_{window}'] = 0
            new_row[f'rolling_std_{window}'] = 0
            new_row[f'rolling_max_{window}'] = 0
            new_row[f'rolling_min_{window}'] = 0

    if len(last_vals) > 1:
        new_row['trend'] = last_vals[-1] - last_vals[-2]
    else:
        new_row['trend'] = 0

    X_target = pd.DataFrame([new_row])[feature_cols].fillna(method='ffill', axis=1).fillna(0)

    try:
        pred_target = model.predict(X_target)[0]
    except:
        pred_target = np.nan

    return y_pred_test, pred_target, y_test.values, None


def run_ets_model(train, test):
    """指数平滑模型(ETS)"""
    if not ETS_AVAILABLE:
        return None, None, None, "ETS 未安装"

    try:
        # 检查训练数据长度
        if len(train) < 10:
            return None, None, None, "ETS 需要至少10个训练样本"

        # 尝试不同的配置,从简单到复杂
        ets_configs = [
            # 配置1: 无趋势,无季节性(最简单)
            {'trend': None, 'seasonal': None},
            # 配置2: 加性趋势,无季节性
            {'trend': 'add', 'seasonal': None, 'damped_trend': False},
            # 配置3: 加性趋势,无季节性,阻尼趋势
            {'trend': 'add', 'seasonal': None, 'damped_trend': True},
            # 配置4: 乘性趋势,无季节性
            {'trend': 'mul', 'seasonal': None, 'damped_trend': False},
        ]

        last_error = None
        for config in ets_configs:
            try:
                model = ExponentialSmoothing(train, **config)
                
                # 设置优化器参数,增加迭代次数
                fitted_model = model.fit(
                    use_brute=True,  # 使用网格搜索优化
                    optimized=True  # 使用优化器
                )
                
                forecast = fitted_model.forecast(steps=len(test) + 1)
                
                # 检查预测结果是否有效
                if np.any(np.isnan(forecast)):
                    raise ValueError("预测结果包含NaN")
                
                y_pred_test = forecast[:len(test)].values
                pred_target = forecast[-1]
                
                return y_pred_test, pred_target, None, None
                
            except Exception as e:
                last_error = str(e)
                continue  # 尝试下一个配置
        
        # 所有配置都失败,返回最后一个错误
        return None, None, None, f"ETS 模型拟合失败: {last_error}"
        
    except Exception as e:
        return None, None, None, f"ETS 执行错误: {str(e)}"


def run_polynomial_model(train, test, degree=2):
    """
    多项式曲线拟合模型
    
    Args:
        train: 训练数据
        test: 测试数据
        degree: 多项式次数 (1-9)
    
    Returns:
        y_pred_test: 测试集预测值
        pred_target: 目标预测值
        formula: 拟合公式字符串
        error: 错误信息
    """
    try:
        # 检查训练数据长度
        if len(train) < degree + 1:
            return None, None, None, f"多项式次数 {degree} 需要至少 {degree + 1} 个训练样本"
        
        # 准备数据
        x_train = np.arange(len(train))
        y_train = train.values
        
        # 多项式拟合
        coefficients = np.polyfit(x_train, y_train, degree)
        polynomial = np.poly1d(coefficients)
        
        # 生成公式字符串
        formula_parts = []
        for i, coef in enumerate(coefficients):
            power = degree - i
            if power == 0:
                formula_parts.append(f"{coef:.6f}")
            elif power == 1:
                formula_parts.append(f"{coef:.6f}x")
            else:
                formula_parts.append(f"{coef:.6f}x^{power}")
        
        formula = "y = " + " + ".join(formula_parts).replace("+ -", "- ")
        
        # 预测测试集
        x_test = np.arange(len(train), len(train) + len(test))
        y_pred_test = polynomial(x_test)
        
        # 预测目标点
        x_target = len(train) + len(test)
        pred_target = polynomial(x_target)
        
        return y_pred_test, float(pred_target), formula, None
        
    except Exception as e:
        return None, None, None, f"多项式拟合错误 (次数={degree}): {str(e)}"


def cross_validate_models(train_series, n_splits=5):
    """时间序列交叉验证"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = {}
    
    models = {
        'XGBoost': run_xgb_model,
        'LSTM': run_lstm_model,
        'Prophet': run_prophet_model,
        'ARIMA': run_arima_model,
        'LightGBM': run_lightgbm_model
    }
    
    for model_name, model_func in models.items():
        if model_name == 'Prophet':
            continue  # Prophet 需要特殊处理
            
        mae_scores = []
        try:
            for train_idx, val_idx in tscv.split(train_series):
                train_fold = train_series.iloc[train_idx]
                val_fold = train_series.iloc[val_idx]
                
                # 使用完整的训练数据构建特征
                pred_test, _, y_true, _ = model_func(train_fold, val_fold, train_series)
                
                if pred_test is not None and y_true is not None:
                    # 对齐长度
                    min_len = min(len(pred_test), len(y_true))
                    mae = mean_absolute_error(y_true[:min_len], pred_test[:min_len])
                    mae_scores.append(mae)
            
            if mae_scores:
                cv_results[model_name] = {
                    'mean_mae': np.mean(mae_scores),
                    'std_mae': np.std(mae_scores)
                }
        except:
            continue
    
    return cv_results

# ==========================================
# 配置记忆功能
# ==========================================

def save_config(config_data):
    """保存配置到文件"""
    config_file = 'prediction_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    return config_file

def load_config():
    """从文件加载配置"""
    config_file = 'prediction_config.json'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None

def save_data_cache(data_series, filename):
    """缓存数据到文件"""
    cache_file = 'data_cache.csv'
    # 保存数据
    pd.DataFrame({'value': data_series}).to_csv(cache_file, index=False)
    # 保存文件名
    with open('data_cache_info.json', 'w', encoding='utf-8') as f:
        json.dump({'filename': filename, 'length': len(data_series)}, f)
    return cache_file

def load_data_cache():
    """从缓存加载数据"""
    cache_file = 'data_cache.csv'
    info_file = 'data_cache_info.json'
    
    if os.path.exists(cache_file) and os.path.exists(info_file):
        try:
            # 加载数据
            df = pd.read_csv(cache_file)
            data_series = df['value']
            
            # 加载文件信息
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            return data_series, info.get('filename', 'cached_data')
        except:
            return None, None
    
    return None, None

def save_experiment_results(results, config, timestamp, maes, weights, ensemble_pred, y_true_final, 
                            fig_main=None, fig_models=None, train_start=None, train_end=None, 
                            test_start=None, target_index=None, data_series=None, train_series=None,
                            test_x_axis=None, colors=None):
    """保存实验结果"""
    import pandas as pd
    
    # 准备性能评估数据
    performance_data = {}
    for name, data in results.items():
        if 'test' in data:
            mae = mean_absolute_error(y_true_final, data['test'])
            rmse = np.sqrt(mean_squared_error(y_true_final, data['test']))
            performance_data[name] = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'Weight': float(weights.get(name, 0)),
                'Test_Predictions': [float(x) for x in data['test']],
                'Target_Prediction': float(data['target']) if 'target' in data else None
            }
    
    # 准备实验数据
    experiment_data = {
        'timestamp': timestamp,
        'experiment_info': {
            'description': '时间序列预测实验结果',
            'version': '1.0',
            'created_at': timestamp
        },
        'data_index_settings': {
            'train_start_index': config['train_start'],
            'train_end_index': config['train_end'],
            'train_length': config['train_length'],
            'test_length': config['test_length'],
            'total_data_length': config.get('total_length', 'N/A'),
            'selected_models': config['selected_models'],
            'cross_validation': config['do_cross_validation']
        },
        'performance_evaluation': {
            'metrics': performance_data,
            'ensemble_prediction': float(ensemble_pred),
            'best_model': min(performance_data.keys(), key=lambda x: performance_data[x]['MAE']) if performance_data else None,
            'average_mae': float(np.mean([v['MAE'] for v in performance_data.values()])) if performance_data else None,
            'average_rmse': float(np.mean([v['RMSE'] for v in performance_data.values()])) if performance_data else None
        },
        'predictions': {
            'ensemble_prediction': float(ensemble_pred),
            'model_predictions': {name: float(data['target']) if 'target' in data else None 
                                 for name, data in results.items()},
            'test_actual_values': [float(x) for x in y_true_final]
        }
    }
    
    # 保存为 JSON 文件
    filename_json = f"experiment_{timestamp.replace(':', '-').replace(' ', '_')}.json"
    with open(filename_json, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, ensure_ascii=False, indent=2)
    
    # 同时保存为 CSV 文件 (更易读)
    filename_csv = f"experiment_{timestamp.replace(':', '-').replace(' ', '_')}.csv"
    
    # 创建 DataFrame
    csv_data = {
        'Timestamp': [timestamp] * len(performance_data),
        'Model': list(performance_data.keys()),
        'MAE': [performance_data[name]['MAE'] for name in performance_data.keys()],
        'RMSE': [performance_data[name]['RMSE'] for name in performance_data.keys()],
        'Weight': [performance_data[name]['Weight'] for name in performance_data.keys()],
        'Target_Prediction': [performance_data[name]['Target_Prediction'] for name in performance_data.keys()],
        'Train_Start': [config['train_start']] * len(performance_data),
        'Train_End': [config['train_end']] * len(performance_data),
        'Train_Length': [config['train_length']] * len(performance_data),
        'Test_Length': [config['test_length']] * len(performance_data),
        'Ensemble_Prediction': [ensemble_pred] * len(performance_data)
    }
    
    df_results = pd.DataFrame(csv_data)
    df_results.to_csv(filename_csv, index=False, encoding='utf-8-sig')
    
    # 保存图表
    saved_files = [filename_json, filename_csv]
    
    if fig_main is not None:
        filename_main_chart = f"experiment_{timestamp.replace(':', '-').replace(' ', '_')}_main.png"
        fig_main.savefig(filename_main_chart, dpi=300, bbox_inches='tight')
        saved_files.append(filename_main_chart)
    
    if fig_models is not None:
        filename_models_chart = f"experiment_{timestamp.replace(':', '-').replace(' ', '_')}_models.png"
        fig_models.savefig(filename_models_chart, dpi=300, bbox_inches='tight')
        saved_files.append(filename_models_chart)
    
    return filename_json, filename_csv, saved_files

# ==========================================
# 2. Streamlit 界面逻辑
# ==========================================

st.set_page_config(page_title="高级时间序列预测", layout="wide")
st.title("📊 高级时间序列预测平台")
st.markdown("""
**功能特点：**
- 📂 **本地上传**：支持 CSV, Excel, TXT。
- 🔢 **多列支持**：支持多列数据文件，可选择任意数值列进行预测。
- 🎚️ **完全自由**：训练集可从任意位置截取，长度可设为任意正整数。
- 🤖 **多模型融合**：XGBoost, LSTM, Prophet, ARIMA, LightGBM。
- 📉 **智能适配**：自动根据数据量调整模型参数。
- 🧪 **交叉验证**：时间序列交叉验证评估模型稳定性。
- 🚨 **异常检测**：Z-Score 和 IQR 方法检测异常值。
- 💾 **结果保存**：导出预测结果和配置参数。
- 🔄 **记忆功能**：自动保存上次的数据和设置。
- ✏️ **手动输入**：支持手动添加数据点。
""")

# 加载保存的配置
saved_config = load_config()
cached_data, cached_filename = load_data_cache()

# 数据输入方式选择
st.subheader("📊 数据输入方式")
data_input_method = st.radio(
    "选择数据输入方式",
    ["上传文件", "手动输入数据", "使用缓存数据"],
    index=0,
    help="选择如何输入预测数据"
)

data_series = None
total_len = 0
filename = None

if data_input_method == "上传文件":
    uploaded_file = st.file_uploader("上传数据文件", type=['csv', 'xlsx', 'xls', 'txt'])
    
    if uploaded_file:
        try:
            df, numeric_cols = load_data(uploaded_file)
            filename = uploaded_file.name
            
            # 显示数据预览
            with st.expander("📋 查看数据预览"):
                st.dataframe(df.head(20))
                st.write(f"数据形状: {df.shape}")
                st.write(f"数值列: {list(numeric_cols)}")
            
            # 列选择功能
            if len(numeric_cols) > 1:
                st.subheader("🔢 选择数据列")
                st.info("可以选择一个或多个列进行批量实验")
                
                # 多列选择
                selected_cols = st.multiselect(
                    "选择要用于预测的数值列",
                    options=numeric_cols,
                    default=numeric_cols[:1],  # 默认选择第一列
                    key="columns_multiselect",
                    help="可以选择多个列进行批量实验"
                )
                
                if len(selected_cols) > 0:
                    st.success(f"已选择 {len(selected_cols)} 列: **{', '.join(selected_cols)}**")
                    
                    # 显示每列的统计信息
                    with st.expander("📊 查看各列统计信息"):
                        for col in selected_cols:
                            col_data = df[col].dropna()
                            st.markdown(f"**{col}**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("数据长度", f"{len(col_data)}")
                            with col2:
                                st.metric("平均值", f"{col_data.mean():.4f}")
                            with col3:
                                st.metric("标准差", f"{col_data.std():.4f}")
                            with col4:
                                st.metric("范围", f"{col_data.min():.2f}~{col_data.max():.2f}")
                            st.divider()
                else:
                    st.warning("请至少选择一列数据")
                    st.stop()
            else:
                selected_cols = [numeric_cols[0]]
                st.info(f"文件中只有一个数值列: **{numeric_cols[0]}**")
            
            # 使用第一列作为默认显示列
            selected_col = selected_cols[0]
            
            # 提取选择的数据列
            data_series = df[selected_col].dropna().reset_index(drop=True)
            total_len = len(data_series)
            st.success(f"✅ 数据加载成功！共 **{total_len}** 条记录。")
            
            # 缓存数据
            save_data_cache(data_series, filename)
            
        except Exception as e:
            st.error(f"文件加载失败: {str(e)}")

elif data_input_method == "手动输入数据":
    st.markdown("#### 手动输入数据")
    st.info("请输入数据点,每行一个数值,用逗号或换行分隔")
    
    # 文本输入
    data_input = st.text_area(
        "输入数据",
        height=200,
        placeholder="例如:\n1.2, 3.4, 5.6, 7.8\n或\n1.2\n3.4\n5.6\n7.8",
        help="支持逗号分隔或换行分隔的数据"
    )
    
    if data_input:
        try:
            # 尝试解析数据
            if ',' in data_input:
                values = [float(x.strip()) for x in data_input.split(',') if x.strip()]
            else:
                values = [float(x.strip()) for x in data_input.split('\n') if x.strip()]
            
            if len(values) > 0:
                data_series = pd.Series(values)
                total_len = len(data_series)
                filename = "手动输入数据"
                st.success(f"✅ 数据解析成功！共 **{total_len}** 条记录。")
                
                # 显示数据预览
                with st.expander("📋 查看数据预览"):
                    st.write("前10个数据点:")
                    st.write(data_series.head(10).tolist())
                    st.write(f"数据范围: {data_series.min():.4f} ~ {data_series.max():.4f}")
                    st.write(f"平均值: {data_series.mean():.4f}")
                
                # 缓存数据
                save_data_cache(data_series, filename)
            else:
                st.warning("请输入有效的数据")
        except Exception as e:
            st.error(f"数据解析失败: {str(e)}")

elif data_input_method == "使用缓存数据":
    if cached_data is not None:
        data_series = cached_data
        total_len = len(data_series)
        filename = cached_filename
        st.success(f"✅ 从缓存加载数据成功！共 **{total_len}** 条记录。")
        st.info(f"文件名: **{filename}**")
        
        # 显示数据预览
        with st.expander("📋 查看数据预览"):
            st.write("前10个数据点:")
            st.write(data_series.head(10).tolist())
            st.write(f"数据范围: {data_series.min():.4f} ~ {data_series.max():.4f}")
            st.write(f"平均值: {data_series.mean():.4f}")
    else:
        st.warning("⚠️ 未找到缓存数据,请先上传文件或手动输入数据")
        st.info("提示: 上传文件或手动输入数据后,数据会自动缓存")

# 如果有数据,继续处理
if data_series is not None and total_len > 0:
    try:
        # 异常值检测
        st.divider()
        outlier_method = st.selectbox("选择异常值检测方法", ["无", "Z-Score", "IQR"], key="outlier_method_select")
        if outlier_method != "无":
            outliers = detect_outliers(data_series, method=outlier_method.lower())
            outlier_count = outliers.sum()
            if outlier_count > 0:
                st.warning(f"🚨 检测到 {outlier_count} 个异常值")
                if st.checkbox("查看异常值位置", key="show_outliers"):
                    outlier_indices = data_series[outliers].index.tolist()
                    st.write("异常值索引:", outlier_indices)
                    st.write("异常值:", data_series[outliers].tolist())
            else:
                st.info("✅ 未检测到异常值")
        
        # 显示选择列的数据图表
        st.divider()
        with st.expander("📈 数据可视化", expanded=False):
            st.line_chart(data_series)
        
        st.divider()
        st.subheader("⚙️ 参数设置")
        
        # 参数设置模式选择
        if len(selected_cols) > 1:
            st.markdown("#### 参数设置模式")
            param_mode = st.radio(
                "选择参数设置方式",
                options=["全局设置", "独立设置"],
                index=0,
                help="全局设置: 所有列使用相同参数\n独立设置: 每列单独设置参数",
                key="param_mode_radio"
            )
            st.divider()
        else:
            param_mode = "全局设置"
        
        # 初始化列配置字典
        column_configs = {}
        
        if param_mode == "全局设置":
            # 全局参数设置
            st.markdown("#### 全局参数设置 (应用于所有列)")
            
            # 使用保存的配置作为默认值
            default_train_start = saved_config.get('train_start', 0) if saved_config else 0
            default_train_end = saved_config.get('train_end', min(total_len - 119, total_len - 1) if total_len > 120 else total_len - 1) if saved_config else (min(total_len - 119, total_len - 1) if total_len > 120 else total_len - 1)
            default_target_index = total_len
            
            col1, col2, col3 = st.columns(3)

            with col1:
                constraints = calculate_constraints(total_len)
                train_start = st.number_input(
                    "训练集起始位置 (第几个数据点)",
                    min_value=constraints['train_start'][0],
                    max_value=constraints['train_start'][1],
                    value=int(default_train_start),
                    step=1,
                    help="训练集从第几个数据点开始（从1开始计数）",
                    key="train_start_input"
                )

            with col2:
                constraints = calculate_constraints(total_len, current_train_start=train_start)
                train_end = st.number_input(
                    "训练集结束位置 (第几个数据点)",
                    min_value=constraints['train_end'][0],
                    max_value=constraints['train_end'][1],
                    value=int(default_train_end),
                    step=1,
                    help="训练集到第几个数据点结束（不包含该位置）",
                    key="train_end_input"
                )
            
            with col3:
                target_index = st.number_input(
                    "预测位置 (第几个数据点)",
                    min_value=train_end,
                    max_value=total_len + 100,
                    value=int(default_target_index),
                    step=1,
                    help="要预测第几个数据点（从1开始计数，可以超出当前数据范围预测未来）",
                    key="target_index_input"
                )
            
            # 为所有列应用相同配置
            for col_name in selected_cols:
                col_data = df[col_name].dropna().reset_index(drop=True)
                column_configs[col_name] = {
                    'train_start': train_start,
                    'train_end': train_end,
                    'target_index': target_index,
                    'data': col_data,
                    'total_len': len(col_data)
                }
        
        else:
            # 独立参数设置
            st.markdown("#### 独立参数设置 (每列单独设置)")
            st.info("为每列设置独立的训练集和预测参数")
            
            for col_name in selected_cols:
                col_data = df[col_name].dropna().reset_index(drop=True)
                col_total_len = len(col_data)
                
                with st.expander(f"📊 {col_name} (长度: {col_total_len})", expanded=False):
                    st.markdown(f"**{col_name}** 参数设置")
                    
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        constraints = calculate_constraints(col_total_len)
                        col_train_start = st.number_input(
                            f"训练集起始",
                            min_value=constraints['train_start'][0],
                            max_value=constraints['train_start'][1],
                            value=0,
                            step=1,
                            key=f"train_start_{col_name}"
                        )
                    
                    with c2:
                        constraints = calculate_constraints(col_total_len, current_train_start=col_train_start)
                        col_train_end = st.number_input(
                            f"训练集结束",
                            min_value=constraints['train_end'][0],
                            max_value=constraints['train_end'][1],
                            value=min(col_total_len - 1, col_total_len - 120) if col_total_len > 120 else col_total_len - 1,
                            step=1,
                            key=f"train_end_{col_name}"
                        )
                    
                    with c3:
                        col_target_index = st.number_input(
                            f"预测位置",
                            min_value=col_train_end,
                            max_value=col_total_len + 100,
                            value=col_total_len,
                            step=1,
                            key=f"target_index_{col_name}"
                        )
                    
                    column_configs[col_name] = {
                        'train_start': col_train_start,
                        'train_end': col_train_end,
                        'target_index': col_target_index,
                        'data': col_data,
                        'total_len': col_total_len
                    }
        
        # 显示配置信息
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            # 验证索引配置 (使用第一列的配置作为显示)
            first_col = selected_cols[0]
            first_col_config = column_configs[first_col]
            validation_result = validate_index_config(
                first_col_config['train_start'], 
                first_col_config['train_end'], 
                first_col_config['total_len']
            )

            if validation_result.is_valid:
                train_length = first_col_config['train_end'] - first_col_config['train_start']
                test_length = first_col_config['total_len'] - first_col_config['train_end']
                st.success(f"✅ 配置有效")
                st.info(f"""
                - 训练集: 第 {first_col_config['train_start']+1} 到第 {first_col_config['train_end']} 个数据点 (长度: {train_length})
                - 测试集: 第 {first_col_config['train_end']+1} 到第 {first_col_config['total_len']} 个数据点 (长度: {test_length})
                - 预测位置: 第 {first_col_config['target_index']+1} 个数据点
                """)
                if len(selected_cols) > 1:
                    st.info(f"💡 已为 {len(selected_cols)} 列设置参数")
            else:
                st.error(f"❌ {validation_result.error_message}")

        with col_info2:
            # 选择要运行的模型
            available_models = []
            if XGB_AVAILABLE:
                available_models.append("XGBoost")
            if LSTM_AVAILABLE:
                available_models.append("LSTM")
            if PROPHET_AVAILABLE:
                available_models.append("Prophet")
            if ARIMA_AVAILABLE:
                available_models.append("ARIMA")
            if LGBM_AVAILABLE:
                available_models.append("LightGBM")
            if RF_AVAILABLE:
                available_models.append("RandomForest")
            if CATBOOST_AVAILABLE:
                available_models.append("CatBoost")
            if SVR_AVAILABLE:
                available_models.append("SVR")
            if RIDGE_AVAILABLE:
                available_models.append("Ridge")
            if ETS_AVAILABLE:
                available_models.append("ETS")
            
            # 添加多项式曲线拟合模型 (1-9次)
            for degree in range(1, 10):
                available_models.append(f"Polynomial_{degree}")

            # 使用保存的模型选择作为默认值
            default_models = saved_config.get('selected_models', available_models) if saved_config else available_models
            # 确保默认模型在可用模型列表中
            default_models = [m for m in default_models if m in available_models]
            
            selected_models = st.multiselect(
                "选择要运行的模型",
                available_models,
                default=default_models,
                help="可以选择一个或多个模型进行预测。Polynomial_N 表示 N 次多项式拟合",
                key="model_multiselect"
            )
        
        # 使用第一列的配置保存
        first_col = selected_cols[0]
        first_col_config = column_configs[first_col]
        
        # 保存当前配置
        current_config = {
            'train_start': first_col_config['train_start'],
            'train_end': first_col_config['train_end'],
            'target_index': first_col_config['target_index'],
            'selected_models': selected_models,
            'total_len': first_col_config['total_len'],
            'filename': filename
        }
        save_config(current_config)
        st.info("💾 配置已自动保存")

        # 计算关键参数
        test_start = first_col_config['train_end']  # 测试集从训练集结束位置开始
        test_end = first_col_config['total_len']  # 测试集到数据末尾

        # 验证索引配置 (使用第一列的配置)
        validation_result = validate_index_config(
            first_col_config['train_start'], 
            first_col_config['train_end'], 
            first_col_config['total_len']
        )

        if validation_result.is_valid:
            # 交叉验证选项
            do_cv = st.checkbox("执行交叉验证 (耗时较长)", value=False)
            
            # 批量实验模式选择
            if len(selected_cols) > 1:
                st.markdown("#### 🔬 实验模式")
                experiment_mode = st.radio(
                    "选择实验模式",
                    options=["顺序实验", "选择实验"],
                    index=0,
                    help="顺序实验: 按列顺序依次运行\n选择实验: 选择特定列运行",
                    key="experiment_mode_radio"
                )
                
                if experiment_mode == "选择实验":
                    experiment_cols = st.multiselect(
                        "选择要实验的列",
                        options=selected_cols,
                        default=selected_cols,
                        key="experiment_cols_select"
                    )
                else:
                    experiment_cols = selected_cols
                
                # 并发设置
                st.markdown("#### ⚡ 并发设置")
                concurrent_num = st.slider(
                    "并发列数",
                    min_value=1,
                    max_value=min(len(experiment_cols), 4),
                    value=1,
                    step=1,
                    help="设置同时训练的列数，可节约运行时间\n例如: 设置2列并发，先训练第1、第2列，完成后开始第3、第4列",
                    key="concurrent_slider"
                )
                if concurrent_num > 1:
                    st.info(f"💡 将同时训练 {concurrent_num} 列数据，可节约约 {(concurrent_num-1)*100//concurrent_num}% 的运行时间")
            else:
                experiment_mode = "顺序实验"
                experiment_cols = selected_cols
                concurrent_num = 1
            
            if st.button("🚀 开始运行多模型预测"):
                # 定义单列训练函数
                def train_single_column(col_name, col_config, selected_models):
                    """训练单列数据的函数"""
                    try:
                        train_start = col_config['train_start']
                        train_end = col_config['train_end']
                        target_index = col_config['target_index']
                        data_series = col_config['data']
                        total_len = col_config['total_len']
                        
                        # 验证索引配置
                        validation_result = validate_index_config(train_start, train_end, total_len)
                        if not validation_result.is_valid:
                            return None, f"配置无效: {validation_result.error_message}"
                        
                        # 创建索引配置对象
                        config = IndexConfig(
                            train_start=train_start,
                            train_end=train_end,
                            total_length=total_len
                        )
                        
                        # 划分数据集
                        dataset_split = split_dataset(data_series, config)
                        train_series = dataset_split.train_series
                        test_series = dataset_split.test_series
                        full_series = dataset_split.full_series
                        
                        results = {}
                        y_true = test_series.values
                        
                        # 模型映射
                        model_functions = {
                            'XGBoost': run_xgb_model,
                            'LSTM': run_lstm_model,
                            'Prophet': run_prophet_model,
                            'ARIMA': run_arima_model,
                            'LightGBM': run_lightgbm_model,
                            'RandomForest': run_randomforest_model,
                            'CatBoost': run_catboost_model,
                            'SVR': run_svr_model,
                            'Ridge': run_ridge_model,
                            'ETS': run_ets_model
                        }
                        
                        # 添加多项式模型 (1-9次)
                        for degree in range(1, 10):
                            model_functions[f'Polynomial_{degree}'] = lambda train, test, d=degree: run_polynomial_model(train, test, degree=d)
                        
                        # 运行选定的模型
                        for model_name in selected_models:
                            model_func = model_functions[model_name]
                            try:
                                # 需要full_series的模型
                                if model_name in ['XGBoost', 'LSTM', 'LightGBM', 'RandomForest', 'CatBoost', 'SVR', 'Ridge']:
                                    pred_test, pred_target, y_test_actual, err = model_func(train_series, test_series, full_series)
                                elif model_name == 'Prophet':
                                    pred_test, pred_target, conf_interval, err = model_func(train_series, test_series)
                                    if conf_interval:
                                        results[model_name] = {
                                            'test': pred_test,
                                            'target': pred_target,
                                            'conf_interval': conf_interval
                                        }
                                        continue
                                elif model_name in ['ARIMA', 'ETS'] or model_name.startswith('Polynomial_'):
                                    pred_test, pred_target, extra_info, err = model_func(train_series, test_series)
                                
                                if err:
                                    pass  # 静默处理错误
                                else:
                                    result_data = {'test': pred_test, 'target': pred_target}
                                    if model_name.startswith('Polynomial_') and extra_info:
                                        result_data['formula'] = extra_info
                                    results[model_name] = result_data
                            
                            except Exception as e:
                                pass  # 静默处理错误
                        
                        # 计算权重和集成预测
                        if len(results) > 0:
                            min_common_len = min([len(v['test']) for v in results.values() if 'test' in v])
                            y_true_final = test_series.values[:min_common_len]
                            
                            weights = {}
                            maes = {}
                            total_inv = 0
                            
                            for name, data in results.items():
                                if 'test' in data:
                                    results[name]['test'] = data['test'][:min_common_len]
                                    mae = mean_absolute_error(y_true_final, data['test'])
                                    maes[name] = mae
                                    inv_mae = 1.0 / (mae + 1e-9)
                                    weights[name] = inv_mae
                                    total_inv += inv_mae
                            
                            for name in weights:
                                weights[name] /= total_inv
                            
                            # 计算集成预测
                            ensemble_pred = sum(weights[name] * data['target'] for name, data in results.items() if 'target' in data)
                            
                            # 返回结果
                            col_result = {
                                'results': results,
                                'maes': maes,
                                'weights': weights,
                                'ensemble_pred': ensemble_pred,
                                'target_index': target_index,
                                'config': {
                                    'train_start': train_start,
                                    'train_end': train_end,
                                    'total_len': total_len
                                }
                            }
                            return col_result, None
                        else:
                            return None, "所有模型运行失败"
                    
                    except Exception as e:
                        return None, str(e)
                
                # 存储所有列的实验结果
                all_results = {}
                
                # 创建进度条
                total_experiments = len(experiment_cols)
                experiment_progress = st.progress(0)
                experiment_status = st.empty()
                
                # 并发训练
                if concurrent_num > 1:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    
                    # 分批处理
                    batch_count = (total_experiments + concurrent_num - 1) // concurrent_num
                    
                    for batch_idx in range(batch_count):
                        start_idx = batch_idx * concurrent_num
                        end_idx = min(start_idx + concurrent_num, total_experiments)
                        batch_cols = experiment_cols[start_idx:end_idx]
                        
                        experiment_status.info(f"正在处理第 {start_idx + 1}-{end_idx} 列 (批次 {batch_idx + 1}/{batch_count})")
                        
                        # 使用线程池并发训练
                        with ThreadPoolExecutor(max_workers=concurrent_num) as executor:
                            future_to_col = {
                                executor.submit(train_single_column, col_name, column_configs[col_name], selected_models): col_name
                                for col_name in batch_cols
                            }
                            
                            for future in as_completed(future_to_col):
                                col_name = future_to_col[future]
                                try:
                                    col_result, error = future.result()
                                    if col_result:
                                        all_results[col_name] = col_result
                                    else:
                                        st.warning(f"⚠️ 列 '{col_name}' 训练失败: {error}")
                                except Exception as e:
                                    st.warning(f"⚠️ 列 '{col_name}' 训练出错: {str(e)}")
                        
                        # 更新进度
                        experiment_progress.progress(end_idx / total_experiments)
                
                else:
                    # 顺序训练
                    for col_idx, col_name in enumerate(experiment_cols):
                        experiment_status.info(f"正在处理第 {col_idx + 1}/{total_experiments} 列: **{col_name}**")
                        
                        col_result, error = train_single_column(col_name, column_configs[col_name], selected_models)
                        
                        if col_result:
                            all_results[col_name] = col_result
                        else:
                            st.warning(f"⚠️ 列 '{col_name}' 训练失败: {error}")
                        
                        # 更新进度
                        experiment_progress.progress((col_idx + 1) / total_experiments)
                
                experiment_status.success(f"✅ 完成! 共处理 {len(all_results)} 列数据")
                
                # 显示结果汇总
                st.divider()
                st.subheader("📊 实验结果汇总")
                
                # 创建汇总表格 (按照原始列顺序)
                summary_data = []
                for col_name in experiment_cols:  # 按照原始顺序遍历
                    if col_name in all_results:
                        col_result = all_results[col_name]
                        best_model = min(col_result['maes'].items(), key=lambda x: x[1])[0] if col_result['maes'] else 'N/A'
                        best_mae = col_result['maes'][best_model] if best_model != 'N/A' else 'N/A'
                        summary_data.append({
                            '列名': col_name,
                            '最佳模型': best_model,
                            'MAE': f"{best_mae:.9f}" if best_mae != 'N/A' else 'N/A',
                            '集成预测': f"{col_result['ensemble_pred']:.9f}",
                            '模型数量': len(col_result['results'])
                        })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # 显示每列的详细结果
                st.divider()
                st.subheader("📈 各列详细结果")
                
                # 按照原始列顺序显示详细结果
                for col_name in experiment_cols:
                    if col_name in all_results:
                        col_result = all_results[col_name]
                        with st.expander(f"📊 {col_name} - 详细结果", expanded=False):
                            # 显示预测结果
                            st.markdown(f"**列名:** {col_name}")
                            st.markdown(f"**集成预测值:** {col_result['ensemble_pred']:.9f}")
                            st.markdown(f"**最佳模型:** {min(col_result['maes'].items(), key=lambda x: x[1])[0]}")
                            
                            # 显示各模型预测值
                            st.markdown("#### 各模型预测值")
                            model_cols = st.columns(min(len(col_result['results']), 4))
                            for idx, (model_name, data) in enumerate(col_result['results'].items()):
                                with model_cols[idx % 4]:
                                    st.metric(
                                        model_name,
                                        f"{data['target']:.9f}" if 'target' in data else "N/A",
                                        f"MAE: {col_result['maes'].get(model_name, 0):.9f}"
                                    )
                        
                        # 显示各模型性能评估
                        st.markdown("#### 各模型性能评估")
                        
                        # 获取该列的数据
                        col_config = column_configs[col_name]
                        
                        # 创建性能评估表格
                        perf_data = []
                        for model_name, data in col_result['results'].items():
                            mae = col_result['maes'].get(model_name, 0)
                            weight = col_result['weights'].get(model_name, 0)
                            # 计算RMSE
                            if 'test' in data:
                                test_true = col_config['data'].values[col_config['train_end']:col_config['train_end'] + len(data['test'])]
                                rmse = np.sqrt(mean_squared_error(test_true, data['test']))
                            else:
                                rmse = 0
                            perf_data.append({
                                '模型': model_name,
                                'MAE': f"{mae:.9f}",
                                'RMSE': f"{rmse:.9f}",
                                '权重': f"{weight:.9f}",
                                '预测值': f"{data['target']:.9f}" if 'target' in data else 'N/A'
                            })
                        
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True)
                        
                        # 显示各模型趋势图
                        st.markdown("#### 各模型预测趋势")
                        
                        # 获取该列的数据
                        col_data = col_config['data']
                        col_train_start = col_config['train_start']
                        col_train_end = col_config['train_end']
                        col_total_len = col_config['total_len']
                        col_target_index = col_config['target_index']
                        
                        # 创建多个子图,每个模型一个
                        num_models = len(col_result['results'])
                        if num_models > 0:
                            cols_per_row = 2
                            num_rows = (num_models + cols_per_row - 1) // cols_per_row
                            
                            fig_models, axes = plt.subplots(num_rows, cols_per_row, figsize=(14, 4*num_rows))
                            if num_models == 1:
                                axes = [axes]
                            elif num_rows == 1:
                                axes = axes.reshape(1, -1)
                            
                            axes = axes.flatten()
                            
                            # 测试集x轴
                            test_x_axis = range(col_train_end, col_train_end + len(list(col_result['results'].values())[0]['test']))
                            test_true = col_data.values[col_train_end:col_train_end + len(list(col_result['results'].values())[0]['test'])]
                            
                            idx = 0
                            for model_name, data in col_result['results'].items():
                                ax_model = axes[idx]
                                
                                # 绘制测试集真实值
                                ax_model.plot(test_x_axis, test_true,
                                            label='真实值', color='black', linewidth=2, marker='o', markersize=3)
                                
                                # 绘制该模型的预测值
                                if 'test' in data:
                                    ax_model.plot(test_x_axis, data['test'],
                                                label=f'{model_name} 预测', 
                                                color='blue',
                                                linewidth=2, linestyle='--', marker='s', markersize=3)
                                
                                # 标记目标预测点
                                if 'target' in data:
                                    ax_model.scatter([col_target_index], [data['target']],
                                                   color='red', s=100, marker='*', zorder=10,
                                                   label=f'预测: {data["target"]:.4f}')
                                
                                # 添加性能指标
                                mae = col_result['maes'].get(model_name, 0)
                                weight = col_result['weights'].get(model_name, 0)
                                
                                ax_model.set_title(f'{model_name}\nMAE: {mae:.9f} | 权重: {weight:.9f}')
                                ax_model.set_xlabel('时间步索引')
                                ax_model.set_ylabel('数值')
                                ax_model.legend(loc='best', fontsize='small')
                                ax_model.grid(True, alpha=0.3)
                                
                                idx += 1
                            
                            # 隐藏多余的子图
                            for i in range(idx, len(axes)):
                                axes[i].set_visible(False)
                            
                            plt.tight_layout()
                            st.pyplot(fig_models, use_container_width=True)
                        
                        st.divider()
                
                # 保存所有结果
                st.divider()
                st.subheader("💾 保存实验结果")
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 准备ZIP文件
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # 为每列保存结果 (按照原始列顺序)
                    for col_name in experiment_cols:
                        if col_name in all_results:
                            col_result = all_results[col_name]
                            # 保存JSON
                            json_filename = f"{col_name}_experiment_{timestamp.replace(':', '-').replace(' ', '_')}.json"
                            
                            # 计算各模型的RMSE
                            col_config = column_configs[col_name]
                        model_details = {}
                        for name, data in col_result['results'].items():
                            mae = col_result['maes'].get(name, 0)
                            weight = col_result['weights'].get(name, 0)
                            # 计算RMSE
                            if 'test' in data:
                                test_true = col_config['data'].values[col_config['train_end']:col_config['train_end'] + len(data['test'])]
                                rmse = np.sqrt(mean_squared_error(test_true, data['test']))
                            else:
                                rmse = 0
                            
                            model_details[name] = {
                                'mae': float(mae),
                                'rmse': float(rmse),
                                'weight': float(weight),
                                'target_prediction': float(data['target']) if 'target' in data else None
                            }
                        
                        json_data = json.dumps({
                            'column': col_name,
                            'timestamp': timestamp,
                            'config': col_result['config'],
                            'ensemble_prediction': float(col_result['ensemble_pred']),
                            'models': model_details
                        }, ensure_ascii=False, indent=2)
                        zipf.writestr(json_filename, json_data)
                    
                    # 保存详细汇总CSV (包含各模型性能)
                    detailed_summary_data = []
                    # 按照原始列顺序保存
                    for col_name in experiment_cols:
                        if col_name in all_results:
                            col_result = all_results[col_name]
                            col_config = column_configs[col_name]
                        
                        # 为每个模型创建一行
                        for model_name, data in col_result['results'].items():
                            mae = col_result['maes'].get(model_name, 0)
                            weight = col_result['weights'].get(model_name, 0)
                            
                            # 计算RMSE
                            if 'test' in data:
                                test_true = col_config['data'].values[col_config['train_end']:col_config['train_end'] + len(data['test'])]
                                rmse = np.sqrt(mean_squared_error(test_true, data['test']))
                            else:
                                rmse = 0
                            
                            detailed_summary_data.append({
                                '列名': col_name,
                                '模型': model_name,
                                'MAE': f"{mae:.9f}",
                                'RMSE': f"{rmse:.9f}",
                                '权重': f"{weight:.9f}",
                                '预测值': f"{data['target']:.9f}" if 'target' in data else 'N/A',
                                '集成预测值': f"{col_result['ensemble_pred']:.9f}",
                                '训练集长度': col_config['train_end'] - col_config['train_start'],
                                '测试集长度': len(list(col_result['results'].values())[0]['test']) if col_result['results'] else 0
                            })
                    
                    detailed_summary_df = pd.DataFrame(detailed_summary_data)
                    detailed_csv_filename = f"detailed_summary_{timestamp.replace(':', '-').replace(' ', '_')}.csv"
                    # 使用BytesIO来正确处理UTF-8-SIG编码
                    csv_buffer = BytesIO()
                    detailed_summary_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    csv_buffer.seek(0)
                    zipf.writestr(detailed_csv_filename, csv_buffer.read())
                    
                    # 保存简要汇总CSV
                    summary_csv_filename = f"summary_{timestamp.replace(':', '-').replace(' ', '_')}.csv"
                    csv_buffer2 = BytesIO()
                    summary_df.to_csv(csv_buffer2, index=False, encoding='utf-8-sig')
                    csv_buffer2.seek(0)
                    zipf.writestr(summary_csv_filename, csv_buffer2.read())
                
                # 下载按钮
                zip_filename = f"batch_experiment_{timestamp.replace(':', '-').replace(' ', '_')}.zip"
                st.download_button(
                    label=f"📥 下载所有实验结果 ({len(all_results)} 列)",
                    data=zip_buffer.getvalue(),
                    file_name=zip_filename,
                    mime="application/zip",
                    key="download_batch_btn",
                    use_container_width=True
                )
                
                st.info("💡 ZIP压缩包包含: 每列的JSON结果文件、详细汇总CSV文件、简要汇总CSV文件")

    except Exception as e:
        st.error(f"发生错误：{str(e)}")
        st.exception(e)

else:
    st.info("👆 请上传文件以开始分析。支持 CSV, Excel, TXT 格式。")
