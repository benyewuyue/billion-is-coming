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
import time
import pickle

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


# ==========================================
# 2. 参数搜索数据模型
# ==========================================

@dataclass
class ParameterSearchConfig:
    """参数搜索配置"""
    initial_train_start: int  # 起始训练集索引，默认200
    train_length: int  # 训练集长度，固定200
    train_start_step: int  # 训练集起始索引步进值，默认200
    test_length_start: int  # 测试集长度起始值，默认10
    test_length_end: int  # 测试集长度结束值，默认200
    test_length_step: int  # 测试集长度步进值，默认10
    mae_threshold: float  # MAE阈值，默认0.01
    selected_models: list  # 选择的模型列表
    max_experiments: int  # 最大实验次数限制，防止资源耗尽，默认1000


@dataclass
class ParameterCombination:
    """单个参数组合"""
    train_start: int  # 训练集起始索引
    train_end: int  # 训练集结束索引
    train_length: int  # 训练集长度
    test_length: int  # 测试集长度
    is_valid: bool  # 是否有效
    skip_reason: Optional[str] = None  # 跳过原因（如果无效）


@dataclass
class ExperimentResult:
    """单个实验结果"""
    combination: ParameterCombination  # 参数组合
    mae: Optional[float]  # MAE值
    status: str  # 状态：success/failed/skipped
    error_message: Optional[str] = None  # 错误信息
    timestamp: str = ""  # 时间戳
    execution_time: float = 0.0  # 执行时间（秒）


@dataclass
class SearchSummary:
    """搜索结果汇总"""
    search_config: ParameterSearchConfig  # 搜索配置
    total_experiments: int  # 总实验次数
    successful_experiments: int  # 成功实验次数
    failed_experiments: int  # 失败实验次数
    skipped_experiments: int  # 跳过实验次数
    qualified_experiments: int  # 满足MAE阈值的实验次数
    best_mae: Optional[float]  # 最优MAE值
    best_combination: Optional[ParameterCombination]  # 最优参数组合
    start_time: str  # 搜索开始时间
    end_time: str  # 搜索结束时间
    total_execution_time: float  # 总执行时间（秒）


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
# 3. 参数搜索核心函数
# ==========================================

def validate_search_config(
    config: ParameterSearchConfig,
    total_data_length: int
) -> ValidationResult:
    """
    验证参数搜索配置的合法性

    Args:
        config: 参数搜索配置
        total_data_length: 数据总长度

    Returns:
        ValidationResult: 验证结果
    """
    # 检查起始训练集索引
    if config.initial_train_start < 0:
        return ValidationResult(
            is_valid=False,
            error_message="起始训练集索引不能小于0"
        )

    # 检查训练集长度步进值
    if config.train_start_step < 1:
        return ValidationResult(
            is_valid=False,
            error_message="训练集长度步进值不能小于1"
        )

    # 检查测试集长度起始值
    if config.test_length_start < 1:
        return ValidationResult(
            is_valid=False,
            error_message="测试集长度起始值不能小于1"
        )

    # 检查测试集长度结束值
    if config.test_length_end < config.test_length_start:
        return ValidationResult(
            is_valid=False,
            error_message="测试集长度结束值不能小于起始值"
        )

    # 检查测试集长度步进值
    if config.test_length_step < 1:
        return ValidationResult(
            is_valid=False,
            error_message="测试集长度步进值不能小于1"
        )

    # 检查MAE阈值
    if config.mae_threshold < 0:
        return ValidationResult(
            is_valid=False,
            error_message="MAE阈值不能小于0"
        )

    # 检查最大实验次数
    if config.max_experiments < 1:
        return ValidationResult(
            is_valid=False,
            error_message="最大实验次数不能小于1"
        )

    # 检查数据长度是否足够
    min_required_length = config.initial_train_start + config.train_length + config.test_length_start
    if total_data_length < min_required_length:
        return ValidationResult(
            is_valid=False,
            error_message=f"数据长度不足。当前数据长度：{total_data_length}，最小要求：{min_required_length}"
        )

    # 检查是否有选择的模型
    if not config.selected_models:
        return ValidationResult(
            is_valid=False,
            error_message="必须至少选择一个模型"
        )

    # 所有检查通过
    return ValidationResult(is_valid=True)


def generate_parameter_combinations(
    config: ParameterSearchConfig,
    total_data_length: int
):
    """
    生成所有可能的参数组合

    Args:
        config: 参数搜索配置
        total_data_length: 数据总长度

    Yields:
        ParameterCombination: 参数组合
    """
    train_start = config.initial_train_start
    while train_start + config.train_length + config.test_length_start <= total_data_length:
        train_end = train_start + config.train_length
        for test_length in range(
            config.test_length_start,
            config.test_length_end + 1,
            config.test_length_step
        ):
            if train_end + test_length > total_data_length:
                # 生成无效组合
                yield ParameterCombination(
                    train_start=train_start,
                    train_end=train_end,
                    train_length=config.train_length,
                    test_length=test_length,
                    is_valid=False,
                    skip_reason=f"超出数据范围: {train_end} + {test_length} > {total_data_length}"
                )
            else:
                # 生成有效组合
                yield ParameterCombination(
                    train_start=train_start,
                    train_end=train_end,
                    train_length=config.train_length,
                    test_length=test_length,
                    is_valid=True
                )
        train_start += config.train_start_step


def extract_mae_from_results(results: dict) -> float:
    """
    从训练结果中提取MAE值

    Args:
        results: 训练结果字典

    Returns:
        float: MAE值
    """
    # 根据v5版本的返回结果结构提取MAE
    # 假设MAE存储在results中
    if results and 'ensemble_mae' in results:
        return float(results['ensemble_mae'])
    elif results and 'mae' in results:
        return float(results['mae'])
    else:
        return float('inf')


def execute_single_experiment(
    combination: ParameterCombination,
    data_series: pd.Series,
    selected_models: list,
    column_name: str
) -> ExperimentResult:
    """
    执行单个参数组合的训练实验

    Args:
        combination: 参数组合
        data_series: 数据序列
        selected_models: 选择的模型列表
        column_name: 列名称

    Returns:
        ExperimentResult: 实验结果
    """
    start_time = time.time()

    try:
        # 创建索引配置
        config = IndexConfig(
            train_start=combination.train_start,
            train_end=combination.train_end,
            total_length=len(data_series)
        )

        # 创建列配置
        col_config = {
            'train_start': combination.train_start,
            'train_end': combination.train_end,
            'data': data_series,
            'total_len': len(data_series)
        }

        # 调用v5版本的训练函数
        results, error = train_single_column(column_name, col_config, selected_models)

        if error:
            return ExperimentResult(
                combination=combination,
                mae=None,
                status='failed',
                error_message=error,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                execution_time=time.time() - start_time
            )

        # 计算MAE（从results中提取）
        mae = extract_mae_from_results(results)

        return ExperimentResult(
            combination=combination,
            mae=mae,
            status='success',
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            execution_time=time.time() - start_time
        )

    except Exception as e:
        return ExperimentResult(
            combination=combination,
            mae=None,
            status='failed',
            error_message=str(e),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            execution_time=time.time() - start_time
        )


def execute_parameter_search(
    config: ParameterSearchConfig,
    data_series: pd.Series,
    progress_callback
):
    """
    执行参数搜索

    Args:
        config: 参数搜索配置
        data_series: 数据序列
        progress_callback: 进度回调函数 (current, total, result)

    Returns:
        Tuple[SearchSummary, List[ExperimentResult]]: 搜索结果汇总和所有实验结果
    """
    start_time = time.time()
    start_datetime = datetime.now()

    results = []
    qualified_results = []

    # 初始化统计
    total_experiments = 0
    successful_experiments = 0
    failed_experiments = 0
    skipped_experiments = 0

    # 生成参数组合
    generator = generate_parameter_combinations(config, len(data_series))

    # 预计算总实验次数
    all_combinations = list(generator)
    total_combinations = len(all_combinations)
    total_combinations = min(total_combinations, config.max_experiments)

    # 执行实验
    for i, combination in enumerate(all_combinations[:config.max_experiments]):
        total_experiments += 1

        if not combination.is_valid:
            # 记录跳过结果
            result = ExperimentResult(
                combination=combination,
                mae=None,
                status='skipped',
                error_message=combination.skip_reason,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                execution_time=0.0
            )
            skipped_experiments += 1
        else:
            # 执行训练
            result = execute_single_experiment(
                combination,
                data_series,
                config.selected_models,
                column_name='value'
            )

            if result.status == 'success':
                successful_experiments += 1
                if result.mae and result.mae < config.mae_threshold:
                    qualified_results.append(result)
            elif result.status == 'failed':
                failed_experiments += 1

        results.append(result)

        # 调用进度回调
        if progress_callback:
            progress_callback(i + 1, total_combinations, result)

    # 计算最优结果
    best_mae = None
    best_combination = None
    successful_results = [r for r in results if r.status == 'success']
    if successful_results:
        best_result = min(successful_results, key=lambda x: x.mae if x.mae else float('inf'))
        best_mae = best_result.mae
        best_combination = best_result.combination

    # 创建汇总
    summary = SearchSummary(
        search_config=config,
        total_experiments=total_experiments,
        successful_experiments=successful_experiments,
        failed_experiments=failed_experiments,
        skipped_experiments=skipped_experiments,
        qualified_experiments=len(qualified_results),
        best_mae=best_mae,
        best_combination=best_combination,
        start_time=start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        end_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        total_execution_time=time.time() - start_time
    )

    return summary, results


def filter_qualified_results(
    results: list,
    mae_threshold: float
) -> list:
    """
    筛选满足MAE阈值的实验结果

    Args:
        results: 所有实验结果列表
        mae_threshold: MAE阈值

    Returns:
        List[ExperimentResult]: 满足条件的实验结果列表（按MAE升序排序）
    """
    qualified = [
        r for r in results
        if r.status == 'success' and r.mae is not None and r.mae < mae_threshold
    ]
    return sorted(qualified, key=lambda x: x.mae)


def save_search_results(
    summary: SearchSummary,
    results: list,
    output_dir: str = "."
):
    """
    保存搜索结果到文件

    Args:
        summary: 搜索结果汇总
        results: 所有实验结果列表
        output_dir: 输出目录

    Returns:
        Tuple[str, str]: JSON文件路径和CSV文件路径
    """
    timestamp = summary.start_time.replace(' ', '_').replace(':', '-')

    # 保存JSON文件
    json_data = {
        'search_config': {
            'initial_train_start': summary.search_config.initial_train_start,
            'train_length': summary.search_config.train_length,
            'train_start_step': summary.search_config.train_start_step,
            'test_length_start': summary.search_config.test_length_start,
            'test_length_end': summary.search_config.test_length_end,
            'test_length_step': summary.search_config.test_length_step,
            'mae_threshold': summary.search_config.mae_threshold,
            'selected_models': summary.search_config.selected_models,
            'max_experiments': summary.search_config.max_experiments
        },
        'total_experiments': summary.total_experiments,
        'successful_experiments': summary.successful_experiments,
        'failed_experiments': summary.failed_experiments,
        'skipped_experiments': summary.skipped_experiments,
        'qualified_experiments': summary.qualified_experiments,
        'best_mae': summary.best_mae,
        'best_combination': {
            'train_start': summary.best_combination.train_start,
            'train_end': summary.best_combination.train_end,
            'train_length': summary.best_combination.train_length,
            'test_length': summary.best_combination.test_length
        } if summary.best_combination else None,
        'start_time': summary.start_time,
        'end_time': summary.end_time,
        'total_execution_time': summary.total_execution_time,
        'results': [
            {
                'train_start': r.combination.train_start,
                'train_end': r.combination.train_end,
                'train_length': r.combination.train_length,
                'test_length': r.combination.test_length,
                'mae': r.mae,
                'status': r.status,
                'error_message': r.error_message,
                'timestamp': r.timestamp,
                'execution_time': r.execution_time
            }
            for r in results
        ]
    }

    json_filename = f"parameter_search_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # 保存CSV文件
    csv_data = []
    for r in results:
        csv_data.append({
            '训练集起始索引': r.combination.train_start,
            '训练集结束索引': r.combination.train_end,
            '训练集长度': r.combination.train_length,
            '测试集长度': r.combination.test_length,
            'MAE值': r.mae,
            '状态': r.status,
            '错误信息': r.error_message,
            '训练时间': r.timestamp,
            '执行时间(秒)': r.execution_time
        })

    df = pd.DataFrame(csv_data)
    csv_filename = f"parameter_search_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return json_path, csv_path


# ==========================================
# 4. 核心功能函数
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

st.set_page_config(page_title="智能参数优化时间序列预测", layout="wide")
st.title("📊 智能参数优化时间序列预测平台 v5")
st.markdown("""
**功能特点：**
- 📂 **本地上传**：支持 CSV, Excel, TXT。
- 🔢 **多列支持**：支持多列数据文件，可选择任意数值列进行预测。
- 🎯 **智能优化**：自动寻找最佳训练集和测试集长度，达到MAE < 0.001的目标。
- ⚡ **快速搜索**：使用智能搜索策略，快速找到最优参数。
- 💾 **断点续传**：支持暂停和恢复，避免长时间运行丢失进度。
- 🤖 **多模型融合**：XGBoost, LSTM, Prophet, ARIMA, LightGBM。
- 📉 **智能适配**：自动根据数据量调整模型参数。
- 🧪 **交叉验证**：时间序列交叉验证评估模型稳定性。
- 🚨 **异常检测**：Z-Score 和 IQR 方法检测异常值。
- 💾 **结果保存**：导出预测结果和配置参数。
- 🔄 **记忆功能**：自动保存上次的数据和设置。
- ✏️ **手动输入**：支持手动添加数据点。
- 🔍 **参数搜索**：自动化训练参数搜索，批量实验并筛选最优配置。
""")

# 添加Tab结构
tab1, tab2, tab3 = st.tabs(["单次训练", "参数搜索", "历史记录"])

# Tab1: 单次训练（现有功能）
with tab1:
    # 加载保存的配置
    saved_config = load_config()
    cached_data, cached_filename = load_data_cache()

    # 数据输入方式选择
    st.subheader("📊 数据输入方式")
    data_input_method = st.radio(
        "选择数据输入方式",
        ["上传文件", "手动输入数据", "使用缓存数据"],
        index=0,
        help="选择如何输入预测数据",
        key="tab1_data_input_method"
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
            except Exception as e:
                st.error(f"文件加载失败: {str(e)}")
            else:
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
    else:
        st.info("👆 请上传文件以开始分析。支持 CSV, Excel, TXT 格式。")

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
                    options=["全局设置", "独立设置", "自动优化"],
                    index=0,
                    help="全局设置: 所有列使用相同参数\n独立设置: 每列单独设置参数\n自动优化: 自动寻找最佳训练集和测试集长度",
                    key="param_mode_radio"
                )
                st.divider()
            else:
                param_mode = st.radio(
                    "选择参数设置方式",
                    options=["手动设置", "自动优化"],
                    index=0,
                    help="手动设置: 手动设置训练集和测试集参数\n自动优化: 自动寻找最佳训练集和测试集长度",
                    key="param_mode_radio_single"
                )
                if param_mode == "手动设置":
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
        
            elif param_mode == "独立设置":
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
        

        except Exception as e:
            st.error(f"数据处理失败: {str(e)}")
            st.exception(e)
        else:
            pass
        elif param_mode == "自动优化":
            # 自动优化模式
            st.markdown("#### 🎯 自动参数优化设置")
            
            # 优化目标设置
            st.markdown("**优化目标**")
            target_mae = st.number_input(
            "目标MAE阈值",
            min_value=0.0001,
            max_value=1.0,
            value=0.001,
            step=0.0001,
            format="%.4f",
            help="程序将自动寻找MAE小于此值的最佳参数组合",
            key="target_mae_input"
            )
            
            # 搜索范围设置
            st.markdown("**搜索范围**")
            col_range1, col_range2 = st.columns(2)

            with col_range1:
                min_train_ratio = st.slider(
                    "最小训练集比例",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.5,
                    step=0.05,
                    help="训练集占数据的最小比例",
                    key="min_train_ratio"
                )
                max_train_ratio = st.slider(
                    "最大训练集比例",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.8,
                    step=0.05,
                    help="训练集占数据的最大比例",
                    key="max_train_ratio"
                )

            with col_range2:
                min_test_len = st.number_input(
                    "最小测试集长度",
                    min_value=5,
                    max_value=100,
                    value=10,
                    step=5,
                    help="测试集的最小数据点数",
                    key="min_test_len"
                )
                max_test_len = st.number_input(
                    "最大测试集长度",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="测试集的最大数据点数",
                    key="max_test_len"
                )
            
            # 搜索策略设置
            st.markdown("**搜索策略**")
            search_strategy = st.radio(
            "选择搜索策略",
            options=["快速搜索", "精细搜索", "全面搜索"],
            index=0,
            help="快速搜索: 使用大步长快速找到近似解\n精细搜索: 使用中等步长平衡速度和精度\n全面搜索: 使用小步长找到最优解",
            key="search_strategy"
            )
            
            # 断点续传设置
            st.markdown("**断点续传**")
            enable_checkpoint = st.checkbox(
                "启用断点续传",
                value=True,
                help="保存优化进度,支持暂停和恢复",
                key="enable_checkpoint"
            )
            
            if enable_checkpoint:
                checkpoint_interval = st.number_input(
                    "保存间隔(尝试次数)",
                    min_value=10,
                    max_value=100,
                    value=20,
                    step=10,
                    help="每尝试多少组参数保存一次进度",
                    key="checkpoint_interval"
                )
            
            st.info(f"💡 程序将自动寻找MAE < {target_mae} 的最佳训练集和测试集长度")
            
            # 为每列设置默认配置(将在优化时更新)
            for col_name in selected_cols:
                col_data = df[col_name].dropna().reset_index(drop=True)
                col_total_len = len(col_data)
                column_configs[col_name] = {
                    'train_start': 0,
                    'train_end': int(col_total_len * 0.8),
                    'target_index': col_total_len,
                    'data': col_data,
                    'total_len': col_total_len,
                    'auto_optimize': True  # 标记为自动优化模式
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


# ==========================================
# 5. 参数搜索Streamlit界面组件
# ==========================================

    def render_parameter_search_config(default_config: ParameterSearchConfig, data_length: int) -> ParameterSearchConfig:
        """
        在Streamlit界面中渲染参数搜索配置表单

        Args:
        default_config: 默认配置
        data_length: 数据总长度

        Returns:
        ParameterSearchConfig: 用户配置的参数
        """
        st.subheader("🔍 参数搜索配置")

        # 计算训练集长度的最大值（数据总长度 - 最小测试集长度）
        max_train_length = data_length - default_config.test_length_start

        col1, col2 = st.columns(2)

        with col1:
            initial_train_start = st.number_input(
                "起始训练集索引",
                min_value=0,
                value=default_config.initial_train_start,
                help="训练集的起始索引位置"
            )

            train_length = st.number_input(
                "训练集长度",
                min_value=10,
                max_value=max_train_length,
                value=min(default_config.train_length, max_train_length),
                help=f"训练集长度（数据总长度: {data_length}）"
            )

            train_start_step = st.number_input(
                "训练集起始索引步进值",
                min_value=1,
                value=default_config.train_start_step,
                help="每次增加训练集起始索引的步长"
            )

            test_length_start = st.number_input(
                "测试集长度起始值",
                min_value=1,
                value=default_config.test_length_start,
                help="测试集长度的最小值"
            )

        with col2:
            test_length_end = st.number_input(
                "测试集长度结束值",
                min_value=1,
                max_value=data_length - initial_train_start,
                value=min(default_config.test_length_end, data_length - initial_train_start),
                help="测试集长度的最大值"
            )

            test_length_step = st.number_input(
                "测试集长度步进值",
                min_value=1,
                value=default_config.test_length_step,
                help="每次增加测试集长度的步长"
            )

            mae_threshold = st.number_input(
                "MAE阈值",
                min_value=0.0,
                value=default_config.mae_threshold,
                step=0.001,
                help="筛选满足条件的MAE上限"
            )

            max_experiments = st.number_input(
                "最大实验次数",
                min_value=1,
                max_value=10000,
                value=default_config.max_experiments,
                help="限制最大实验次数，防止资源耗尽"
            )

        available_models = [
            'XGBoost', 'LSTM', 'Prophet', 'ARIMA', 'LightGBM',
            'RandomForest', 'CatBoost', 'SVR', 'Ridge', 'ETS'
        ]

        selected_models = st.multiselect(
            "选择模型",
            available_models,
            default=default_config.selected_models,
            help="选择用于训练的模型列表"
        )

        return ParameterSearchConfig(
            initial_train_start=initial_train_start,
            train_length=train_length,
            train_start_step=train_start_step,
            test_length_start=test_length_start,
            test_length_end=test_length_end,
            test_length_step=test_length_step,
            mae_threshold=mae_threshold,
            selected_models=selected_models,
            max_experiments=max_experiments
        )


    def render_search_control(
        is_running: bool,
        progress: float,
        current_result: ExperimentResult,
        completed_count: int,
        qualified_count: int,
        best_mae: float
    ) -> str:
        """
        在Streamlit界面中渲染搜索控制组件

        Args:
        is_running: 搜索是否正在进行
        progress: 进度百分比（0-100）
        current_result: 当前实验结果
        completed_count: 已完成实验数量
        qualified_count: 满足条件的实验数量
        best_mae: 最优MAE值

        Returns:
        str: 用户操作指令
        """
        st.subheader("⚙️ 搜索控制")

        col1, col2 = st.columns(2)

        with col1:
            if not is_running:
                start_button = st.button("🚀 开始搜索", type="primary")
                if start_button:
                    return "start"
            else:
                pause_button = st.button("⏸️ 暂停")
                if pause_button:
                    return "pause"

        with col2:
            st.metric("已完成实验", completed_count)
            st.metric("满足条件", qualified_count)

        # 进度条
        st.progress(progress / 100)

        # 当前实验信息
        if current_result:
            st.info(f"当前: 训练集[{current_result.combination.train_start}:{current_result.combination.train_end}], "
                    f"测试集长度={current_result.combination.test_length}")

            if current_result.status == 'success' and current_result.mae is not None:
                st.success(f"MAE: {current_result.mae:.6f}")
            elif current_result.status == 'failed':
                st.error(f"训练失败: {current_result.error_message}")

        # 最优结果
        if best_mae is not None:
            st.success(f"🏆 最优MAE: {best_mae:.6f}")

        return "continue"


    def render_search_results(summary: SearchSummary, qualified_results: list) -> None:
        """
        在Streamlit界面中渲染搜索结果

        Args:
        summary: 搜索结果汇总
        qualified_results: 满足条件的实验结果列表
        """
        st.subheader("📊 搜索结果")

        # 汇总信息
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总实验数", summary.total_experiments)
        col2.metric("成功实验", summary.successful_experiments)
        col3.metric("失败实验", summary.failed_experiments)
        col4.metric("满足条件", summary.qualified_experiments)

        # 最优结果
        if summary.best_mae is not None and summary.best_combination is not None:
            st.success(f"🏆 最优MAE: {summary.best_mae:.6f} "
                      f"(训练集[{summary.best_combination.train_start}:{summary.best_combination.train_end}], "
                      f"测试集长度={summary.best_combination.test_length})")

        # 满足条件的结果表格
        if qualified_results:
            st.subheader("满足条件的参数组合")

            table_data = []
            for r in qualified_results:
                table_data.append({
                    '训练集起始索引': r.combination.train_start,
                    '训练集结束索引': r.combination.train_end,
                    '训练集长度': r.combination.train_length,
                    '测试集长度': r.combination.test_length,
                    'MAE值': f"{r.mae:.6f}",
                    '训练时间': r.timestamp
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

            # 下载按钮
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 下载CSV",
                data=csv,
                file_name=f"qualified_results_{summary.start_time.replace(' ', '_').replace(':', '-')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("没有找到满足条件的参数组合")

        # 完整结果（可折叠）
        with st.expander("查看完整搜索结果"):
            st.json({
                'search_config': summary.search_config.__dict__,
                'total_experiments': summary.total_experiments,
                'successful_experiments': summary.successful_experiments,
                'failed_experiments': summary.failed_experiments,
                'skipped_experiments': summary.skipped_experiments,
                'qualified_experiments': summary.qualified_experiments,
                'best_mae': summary.best_mae,
                'best_combination': summary.best_combination.__dict__ if summary.best_combination else None,
                'start_time': summary.start_time,
                'end_time': summary.end_time,
                'total_execution_time': summary.total_execution_time
            })

# Tab2: 参数搜索（新功能）
with tab2:
    st.title("🔍 自动化训练参数搜索")
    st.markdown("""
    自动化训练参数搜索功能可以帮助您：
    - 自动遍历不同的训练集和测试集配置
    - 批量执行训练实验
    - 筛选满足MAE阈值条件的参数组合
    - 找到最优的训练参数配置
    """)

    # 数据输入
    st.subheader("📊 数据输入")
    data_input_method_search = st.radio(
        "选择数据输入方式",
        ["上传文件", "使用缓存数据"],
        index=0,
        key="tab2_data_input_method"
    )

    df_search = None
    numeric_cols_search = None

    if data_input_method_search == "上传文件":
        uploaded_file_search = st.file_uploader("上传数据文件", type=['csv', 'xlsx', 'xls', 'txt'], key="search_upload")
        if uploaded_file_search:
            try:
                df_search, numeric_cols_search = load_data(uploaded_file_search)
                st.success(f"✅ 数据加载成功！共{len(df_search)}行，{len(numeric_cols_search)}个数值列")

                with st.expander("📋 查看数据预览"):
                    st.dataframe(df_search.head(20))
                    st.write(f"数值列: {list(numeric_cols_search)}")

                if len(numeric_cols_search) > 1:
                    selected_col_search = st.selectbox(
                        "选择要用于参数搜索的数值列",
                        options=numeric_cols_search,
                        index=0
                    )
                else:
                    selected_col_search = numeric_cols_search[0]
                    st.info(f"只有一个数值列: {selected_col_search}")

                data_series_search = df_search[selected_col_search].dropna().reset_index(drop=True)
            except Exception as e:
                st.error(f"数据加载失败: {str(e)}")
    else:
        if cached_data is not None:
            data_series_search = cached_data
            st.success(f"✅ 使用缓存数据！共{len(data_series_search)}条记录")
        else:
            st.warning("没有可用的缓存数据")
            st.stop()

    # 参数配置
    if 'data_series_search' in locals() and data_series_search is not None:
        st.divider()
        st.subheader("⚙️ 参数配置")

        default_config = ParameterSearchConfig(
            initial_train_start=200,
            train_length=200,
            train_start_step=200,
            test_length_start=10,
            test_length_end=200,
            test_length_step=10,
            mae_threshold=0.01,
            selected_models=['XGBoost', 'LightGBM'],
            max_experiments=1000
        )

        config = render_parameter_search_config(default_config, len(data_series_search))

        # 验证配置
        if st.button("验证配置"):
            validation_result = validate_search_config(config, len(data_series_search))
            if validation_result.is_valid:
                st.success("✅ 配置验证通过")
            else:
                st.error(f"❌ 配置验证失败: {validation_result.error_message}")

        # 搜索控制
        if 'search_state' not in st.session_state:
            st.session_state['search_state'] = {
                'is_running': False,
                'progress': 0,
                'completed_count': 0,
                'qualified_count': 0,
                'best_mae': None,
                'current_result': None,
                'summary': None,
                'qualified_results': []
            }

        command = render_search_control(
            st.session_state['search_state']['is_running'],
            st.session_state['search_state']['progress'],
            st.session_state['search_state']['current_result'],
            st.session_state['search_state']['completed_count'],
            st.session_state['search_state']['qualified_count'],
            st.session_state['search_state']['best_mae']
        )

        # 执行搜索
        if command == "start":
            st.session_state['search_state']['is_running'] = True
            st.session_state['search_state']['progress'] = 0
            st.session_state['search_state']['completed_count'] = 0
            st.session_state['search_state']['qualified_count'] = 0
            st.session_state['search_state']['best_mae'] = None

            try:
                # 执行搜索
                def progress_callback(current, total, result):
                    st.session_state['search_state']['progress'] = (current / total) * 100
                    st.session_state['search_state']['completed_count'] = current
                    st.session_state['search_state']['current_result'] = result

                    if result.status == 'success' and result.mae is not None:
                        if result.mae < config.mae_threshold:
                            st.session_state['search_state']['qualified_count'] += 1
                        if st.session_state['search_state']['best_mae'] is None or result.mae < st.session_state['search_state']['best_mae']:
                            st.session_state['search_state']['best_mae'] = result.mae

                summary, results = execute_parameter_search(config, data_series_search, progress_callback)

                # 筛选满足条件的结果
                qualified_results = filter_qualified_results(results, config.mae_threshold)

                # 保存结果
                json_path, csv_path = save_search_results(summary, results)

                # 更新状态
                st.session_state['search_state']['is_running'] = False
                st.session_state['search_state']['summary'] = summary
                st.session_state['search_state']['qualified_results'] = qualified_results

                st.success(f"✅ 搜索完成！结果已保存到 {json_path} 和 {csv_path}")

            except Exception as e:
                st.session_state['search_state']['is_running'] = False
                st.error(f"❌ 搜索过程中发生错误: {str(e)}")
                st.exception(e)

        # 展示结果
        if st.session_state['search_state']['summary'] is not None:
            st.divider()
            render_search_results(
                st.session_state['search_state']['summary'],
                st.session_state['search_state']['qualified_results']
            )

# Tab3: 历史记录（占位符）
with tab3:
    st.title("📜 历史记录")
    st.info("历史记录功能开发中...")
