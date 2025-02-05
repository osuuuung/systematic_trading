from freqtrade.strategy import IStrategy,IntParameter,DecimalParameter
# from freqtrade.optimize.space import SKDecimal
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import numpy as np
import logging

class MergeStrategy(IStrategy):
    """
    병합된 전략의 기본 설정
    """
    # 최적화
    # class HyperOpt:
    #     def stoploss_space():
    #         return [SKDecimal(-0.0125, -0.007, decimals=4, name='stoploss')]
        
    # 공통 설정 
    can_short = True  # 숏 포지션 허용 여부
    timeframe = '1m'
    adjusted_roi = 0.02
    # 최적화 시 무시 
    stoploss = -0.0171  #-0.0085
    trend_roi_0 = 0.0182  
    ema_roi_0 = 0.0146  
    ema_threshold = 10 
    vpvr_threshold = 103 


    # 전략별 고유 설정
    # TrendlineWithBreaksStrategy
    lookback_length = 11    
    mult = 1.0
    slope_method = 'atr'
    atr_period = 11
    # trend_roi_0 = DecimalParameter(0.0033, 0.0077, default=0.0005, decimals=4, optimize=True, space='sell') 

    # EMACrossStrategy
    ema_trailing_stop = False
    # ema_threshold = IntParameter(8, 15, default=10, optimize=True, space='buy')
    # vpvr_threshold = IntParameter(95, 105, default=100, optimize=True, space='buy')
    # ema_roi_0 = DecimalParameter(0.004, 0.0125, default=0.01, decimals=4, optimize=True, space = 'sell')
    
    # VolumeProfileStrategy
    vpvr_informative_timeframe_5m = '5m'
    vpvr_informative_timeframe_4h = '4h'

    # 로거 설정
    logger = logging.getLogger(__name__)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        병합된 populate_indicators 함수.
        동일한 시간 프레임을 공유하며, EMA는 15m, 30m, 1h, 4h에서만 계산.
        """
        # 캔들 부족 경고
        required_candles = max(self.lookback_length, self.atr_period)
        if len(dataframe) < required_candles:
            self.logger.warning(f"Not enough candles for indicators. Need at least {required_candles} candles.")
            return dataframe

        # ---------------------------
        # 공유 시간 프레임 지표 계산
        # ---------------------------
        # trendline 1h 계산(trendline)
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
        informative_1h['pivot_high'] = informative_1h['high'].rolling(window=self.lookback_length, center=False).apply(self.is_pivot_high, raw=True)
        informative_1h['pivot_low'] = informative_1h['low'].rolling(window=self.lookback_length, center=False).apply(self.is_pivot_low, raw=True)
        informative_1h['upper_trendline'], informative_1h['lower_trendline'] = self.calculate_trendlines(informative_1h)

        # 병합
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, '1h', ffill=True)

        # VPVR 5m 계산 (매물대)
        informative_5m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='5m')
        informative_5m = self.calculate_vpvr_levels(informative_5m, '5m', lookback_period=576, top_levels=3, accumulate_levels=False)

        # 병합
        dataframe = merge_informative_pair(dataframe, informative_5m, self.timeframe, '5m', ffill=True)

        # ---------------------------
        # EMA 전략: 여러 시간 프레임에 대해 각각 EMA 계산
        # ---------------------------
        ema_timeframes = ['15m', '30m', '1h', '4h']
        for tf in ema_timeframes:
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=tf)
            informative['ema_7'] = ta.EMA(informative['close'], timeperiod=7)
            informative['ema_14'] = ta.EMA(informative['close'], timeperiod=14)
            informative['ema_28'] = ta.EMA(informative['close'], timeperiod=28)
            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, tf, ffill=True)

        # ---------------------------
        # VPVR 전략: 4h 매물대 및 볼륨 EMA 계산
        # ---------------------------
        # 4시간봉 데이터프레임 생성
        informative_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='4h')
        informative_4h['volume_ema_14'] = ta.EMA(informative_4h['volume'], timeperiod=14)

        # 4h VPVR 매물대 계산
        informative_4h = self.calculate_vpvr_levels(informative_4h, '4h', lookback_period=50, accumulate_levels=True)

        # 병합
        dataframe = merge_informative_pair(dataframe, informative_4h, self.timeframe, '4h', ffill=True)

        # volatility : 과거 20개와 5개 캔들의 high-low 차이의 평균 계산
        dataframe['volatility_20'] = (dataframe['high'] - dataframe['low']).rolling(window=20).mean()
        dataframe['volatility_5'] = (dataframe['high'] - dataframe['low']).rolling(window=5).mean()

        # print('------------------------------------------------')
        # # print(dataframe.loc[31000:31100, ['ema_7_15m','ema_7_30m','ema_7_1h','ema_7_4h']])
        # print(dataframe.loc[31000:31100, ['ema_14_15m','ema_14_30m','ema_14_1h','ema_14_4h']])
        # # print(dataframe.loc[31000:31100, ['upper_trendline_5m','lower_trendline_5m']])
        # # print(dataframe.loc[31000:31100, ['vpvr_level_1_4h','vpvr_level_2_4h','vpvr_level_3_4h']])
        # print(dataframe['vpvr_level_1_4h'].unique())
        # # print(dataframe.loc[31000:31100, ['vpvr_level_1_5m','vpvr_level_2_5m','vpvr_level_3_5m']])
        # print('-------------------------------------------------')

        vpvr_columns = [col for col in dataframe.columns if 'vpvr' in col]
        # 해당 컬럼들만 선택하여 출력
        print(dataframe[vpvr_columns])

        return dataframe
    
    # 지표 정의에 필요한 함수들 정의
    # VPVR에 필요한 함수
    def calculate_vpvr_levels(self, informative_df: pd.DataFrame, timeframe: str, num_bins: int = 30, top_levels: int = 15, lookback_period: int = 30, accumulate_levels: bool = False) -> pd.DataFrame:
        """
        각 캔들 위치에서 슬라이딩 윈도우 방식으로 가격 구간별 거래량을 계산하여 상위 매물대(top_levels)를 정의합니다.
        `accumulate_levels=True`인 경우 1일봉(1d)에서 상위 10개의 매물대를 누적하여 저장합니다.
        `accumulate_levels=False`인 경우 5분봉(5m)에서 매물대 컬럼을 매번 덮어씁니다.
        """
        
        # 4h 매물대인 경우 상위 10개의 매물대를 저장할 컬럼 초기화
        if accumulate_levels:
            # 매물대 상위 10개를 저장할 컬럼을 초기화
            for idx in range(1, top_levels + 1):
                column_name = f'vpvr_level_{idx}'
                informative_df[column_name] = np.nan
            
            # 누적할 매물대 리스트 초기화
            vpvr_levels = [{"price": np.nan, "volume": 0} for _ in range(top_levels)]

            for end_idx in range(lookback_period, len(informative_df)):
                start_idx = end_idx - lookback_period
                window_df = informative_df.iloc[start_idx:end_idx]

                # 가격 구간 생성
                min_price, max_price = window_df['close'].min(), window_df['close'].max()
                price_bins = np.linspace(min_price, max_price, num_bins)

                # 각 가격 구간별 거래량을 저장할 배열
                volume_profile = np.zeros(num_bins-1)

                # 각 구간의 거래량을 계산
                for i in range(len(price_bins) - 1):
                    mask = (window_df['close'] >= price_bins[i]) & (window_df['close'] < price_bins[i + 1])
                    volume_profile[i] = window_df.loc[mask, 'volume'].sum()

                # 거래량이 많은 구간을 top_levels 개수만큼 찾습니다.
                top_indices = np.flip(np.argsort(volume_profile))[0]
                idx = top_indices
                # 누적 방식: 새로운 매물대를 리스트에 추가하고 상위 10개만 유지
                new_level_price = (price_bins[idx] + price_bins[idx + 1]) / 2
                new_level_volume = volume_profile[idx]
                new_level = {"price": new_level_price, "volume": new_level_volume}
                
                # 매물대 리스트에 추가 후, 거래량 기준으로 정렬하여 상위 10개 유지
                vpvr_levels.append(new_level)
                vpvr_levels = sorted(vpvr_levels, key=lambda x: x["volume"], reverse=True)

                # 중복되는 가격 존재 시 volume 큰 객체만 남기기 
                unique_levels = {}
                for level in vpvr_levels:
                    price = level["price"]
                    if price not in unique_levels or level["volume"] > unique_levels[price]["volume"]:
                        unique_levels[price] = level
                
                # unique_levels에서 상위 10개만 남기기
                vpvr_levels = list(unique_levels.values())
                vpvr_levels = sorted(vpvr_levels, key=lambda x: x["volume"], reverse=True)[:top_levels]

                # 상위 10개의 매물대를 level_1 ~ level_10 컬럼에 저장
                for idx, level in enumerate(vpvr_levels):
                    column_name = f'vpvr_level_{idx + 1}'
                    informative_df.at[informative_df.index[end_idx], column_name] = level["price"]

        # 5m
        else:
            # 5분봉 매물대인 경우 매번 갱신할 컬럼 초기화
            for idx in range(1, 4):
                column_name = f'vpvr_level_{idx}'
                informative_df[column_name] = np.nan

            vpvr_levels = []

            for end_idx in range(lookback_period, len(informative_df)):
                start_idx = end_idx - lookback_period
                window_df = informative_df.iloc[start_idx:end_idx]

                # 가격 구간 생성
                min_price, max_price = window_df['close'].min(), window_df['close'].max()
                price_bins = np.linspace(min_price, max_price, num_bins)

                # 각 가격 구간별 거래량을 저장할 배열
                volume_profile = np.zeros(num_bins - 1)

                # 각 구간의 거래량을 계산
                for i in range(len(price_bins) - 1):
                    mask = (window_df['close'] >= price_bins[i]) & (window_df['close'] < price_bins[i + 1])
                    volume_profile[i] = window_df.loc[mask, 'volume'].sum()

                # 거래량이 많은 구간을 top_levels 개수만큼 찾습니다.
                top_indices = np.flip(np.argsort(volume_profile))

                # 비누적 방식: 각 lookback_period 주기마다 새로 계산한 상위 매물대만 저장
                for idx, top_index in enumerate(top_indices[:top_levels]):
                    column_name = f'vpvr_level_{idx + 1}'
                    central_value = (price_bins[top_index] + price_bins[top_index + 1]) / 2
                    informative_df.at[informative_df.index[end_idx], column_name] = central_value       
        return informative_df
    
    # Trendline 전략에 필요한 함수
    def is_pivot_high(self, highs: np.ndarray) -> float:
        if highs[self.lookback_length - 1] == np.max(highs):
            return highs[self.lookback_length - 1]
        return np.nan

    def is_pivot_low(self, lows: np.ndarray) -> float:
        if lows[self.lookback_length - 1] == np.min(lows):
            return lows[self.lookback_length - 1]
        return np.nan

    def calculate_slope(self, dataframe: DataFrame, idx: int, trend_type: str) -> float:
        if trend_type == 'upper':
            if self.slope_method == 'atr':
                atr = ta.ATR(dataframe, timeperiod=self.atr_period)
                return atr.iloc[idx] / self.lookback_length * self.mult
            elif self.slope_method == 'stdev':
                stdev = dataframe['close'].rolling(window=self.lookback_length).std()
                return stdev.iloc[idx] / self.lookback_length * self.mult
            elif self.slope_method == 'linreg':
                x = np.arange(self.lookback_length)
                y = dataframe['close'][idx - self.lookback_length:idx]
                slope, _ = np.polyfit(x, y, 1)
                return slope * self.mult
        elif trend_type == 'lower':
            if self.slope_method == 'atr':
                atr = ta.ATR(dataframe, timeperiod=self.atr_period)
                return atr.iloc[idx] / self.lookback_length * self.mult
            elif self.slope_method == 'stdev':
                stdev = dataframe['close'].rolling(window=self.lookback_length).std()
                return stdev.iloc[idx] / self.lookback_length * self.mult
            elif self.slope_method == 'linreg':
                x = np.arange(self.lookback_length)
                y = dataframe['close'][idx - self.lookback_length:idx]
                slope, _ = np.polyfit(x, y, 1)
                return slope * self.mult
        return 0

    def calculate_trendlines(self, dataframe: DataFrame):
        dataframe['upper_trendline'] = np.nan
        dataframe['lower_trendline'] = np.nan
        prev_upper_trendline = prev_lower_trendline = np.nan
        prev_upper_slope = prev_lower_slope = np.nan

        for i in range(self.lookback_length, len(dataframe)):
            if not np.isnan(dataframe['pivot_high'].iloc[i]):
                dataframe.at[i, 'upper_trendline'] = dataframe['pivot_high'].iloc[i]
                prev_upper_trendline = dataframe['upper_trendline'].iloc[i]
                prev_upper_slope = self.calculate_slope(dataframe, i, 'upper')
            elif not np.isnan(prev_upper_trendline):
                dataframe.at[i, 'upper_trendline'] = prev_upper_trendline - prev_upper_slope
                prev_upper_trendline = dataframe['upper_trendline'].iloc[i]

            if not np.isnan(dataframe['pivot_low'].iloc[i]):
                dataframe.at[i, 'lower_trendline'] = dataframe['pivot_low'].iloc[i]
                prev_lower_trendline = dataframe['lower_trendline'].iloc[i]
                prev_lower_slope = self.calculate_slope(dataframe, i, 'lower')
            elif not np.isnan(prev_lower_trendline):
                dataframe.at[i, 'lower_trendline'] = prev_lower_trendline + prev_lower_slope
                prev_lower_trendline = dataframe['lower_trendline'].iloc[i]

        return dataframe['upper_trendline'], dataframe['lower_trendline']
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        세 가지 전략의 진입 조건을 병합하여 구현.
        진입 시 enter_tag에 해당 전략 정보를 포함.
        """
        # 초기화
        dataframe['enter_long'] = False
        dataframe['enter_short'] = False
        dataframe['enter_tag'] = None

        # ---------------------------
        # Trendline 전략
        if len(dataframe) >= self.lookback_length:
            long_condition = (
                (dataframe['close'] >= dataframe['upper_trendline_1h']) & 
                (dataframe['close'].shift(1) < dataframe['upper_trendline_1h'].shift(1))
            )
            dataframe.loc[long_condition, 'enter_long'] = True
            dataframe.loc[long_condition, 'enter_tag'] = 'trend_long'

            short_condition = (
                (dataframe['close'] <= dataframe['lower_trendline_1h']) & 
                (dataframe['close'].shift(1) > dataframe['lower_trendline_1h'].shift(1))
            )
            dataframe.loc[short_condition, 'enter_short'] = True
            dataframe.loc[short_condition, 'enter_tag'] = 'trend_short'

        # ---------------------------
        # EMA 전략
        # ---------------------------
        threshold = self.ema_threshold
        ema_columns = [col for col in dataframe.columns if 'ema' in col]  # EMA 컬럼 필터링

        for index, row in dataframe.iterrows():
            close_emas = []
            # 롱 포지션 대전제: close > ema_14_4h
            if pd.notna(row['ema_14_4h']):
                if row['close'] > row['ema_14_4h']:
                    for i, ema1 in enumerate(ema_columns):
                        for j, ema2 in enumerate(ema_columns):
                            if i < j and abs(row[ema1] - row[ema2]) < threshold:
                                close_emas.extend([ema1, ema2])
                    if len(set(close_emas)) >= 2 and row['close'] > max(row[ema] for ema in set(close_emas)):
                        dataframe.at[index, 'enter_long'] = True
                        dataframe.at[index, 'enter_tag'] = 'ema_long'

                # 숏 포지션 대전제: close < ema_14_4h
                elif row['close'] < row['ema_14_4h']:
                    for i, ema1 in enumerate(ema_columns):
                        for j, ema2 in enumerate(ema_columns):
                            if i < j and abs(row[ema1] - row[ema2]) < threshold:
                                close_emas.extend([ema1, ema2])
                    if len(set(close_emas)) >= 2 and row['close'] < min(row[ema] for ema in set(close_emas)):
                        dataframe.at[index, 'enter_short'] = True
                        dataframe.at[index, 'enter_tag'] = 'ema_short'

        # ---------------------------
        # VPVR 전략
        # ---------------------------
        enter_long_condition = pd.Series(False, index=dataframe.index)
        long_breakout_level = pd.Series(np.nan, index=dataframe.index)

        for label_suffix in ['5m', '4h']:
            for i in range(1, 4):  # VPVR 매물대 레벨
                level_column = f'vpvr_level_{i}_{label_suffix}'
                condition = (
                    (dataframe['close'] > dataframe[level_column]) & 
                    (dataframe['close'].shift(1) <= dataframe[level_column].shift(1)) &
                    (dataframe['volume_ema_14_4h']/240 <= dataframe['volume']) 
                )
                enter_long_condition |= condition
                long_breakout_level.loc[condition] = dataframe[level_column]

        dataframe.loc[enter_long_condition, 'enter_long'] = True
        dataframe.loc[enter_long_condition, 'enter_tag'] = (
            'vpvr_long_' + long_breakout_level.astype(str)
        )

        enter_short_condition = pd.Series(False, index=dataframe.index)
        short_breakout_level = pd.Series(np.nan, index=dataframe.index)

        for label_suffix in ['5m', '4h']:
            for i in range(1, 4):
                level_column = f'vpvr_level_{i}_{label_suffix}'
                condition = (
                    (dataframe['close'] < dataframe[level_column]) & 
                    (dataframe['close'].shift(1) >= dataframe[level_column].shift(1)) &
                    (dataframe['volume_ema_14_4h']/240 <= dataframe['volume'])
                )
                enter_short_condition |= condition
                short_breakout_level.loc[condition] = dataframe[level_column]

        dataframe.loc[enter_short_condition, 'enter_short'] = True
        dataframe.loc[enter_short_condition, 'enter_tag'] = (
            'vpvr_short_' + short_breakout_level.astype(str)
        )

        return dataframe
    
    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        진입 전략에 따라 동적으로 청산 조건을 정의.
        - Trendline, EMA: 변동성 조건 만족 시 minimal_roi 상향 조정.
        - VPVR: 기존 로직 유지.
        """
        enter_tag = trade.enter_tag
        if not enter_tag:
            return None  # 진입 태그가 없으면 청산 조건 없음

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        # 변동성 조건 확인
        last_row = dataframe.iloc[-1]
        volatility_condition = (
            'volatility_20' in last_row and
            'volatility_5' in last_row and
            last_row['volatility_5'] > last_row['volatility_20'] * 1.5
        )

        # ROI 조정
        adjusted_roi_targets = {
            "0": self.adjusted_roi,  # 0.47% 수익
        }

        # ---------------------------
        # Trendline 전략 청산
        # ---------------------------
        if enter_tag.startswith('trend'):
            # Trendline 전략은 minimal_roi로 처리
            # (Freqtrade 기본 minimal_roi를 사용하는 대신 ROI 조건을 명시적으로 설정 가능)
            
            roi_targets = adjusted_roi_targets if volatility_condition else {
                "0": self.trend_roi_0,    # 0.25% 수익
            }
            trade_duration = (current_time - trade.open_date_utc).seconds // 60  # 거래 기간(분)
            
            for duration, roi in sorted(roi_targets.items(), key=lambda x: int(x[0])):
                if isinstance(roi,float):
                    if trade_duration >= int(duration) and current_profit >= roi:
                        return "exit_trend_roi"
                else : 
                    if trade_duration >= int(duration) and current_profit >= roi:
                        return "exit_trend_roi"

        # ---------------------------
        # EMA 전략 청산
        # ---------------------------
        elif enter_tag.startswith('ema'):
            # EMA 전략은 minimal_roi로 처리 (다른 ROI 조건 적용 가능)
            roi_targets = {
                "0": self.ema_roi_0,    # 0.0025% 수익
            }
            trade_duration = (current_time - trade.open_date_utc).seconds // 60  # 거래 기간(분)
            
            for duration, roi in sorted(roi_targets.items(), key=lambda x: int(x[0])):
                if isinstance(roi,float):
                    if trade_duration >= int(duration) and current_profit >= roi:
                        return "exit_ema_roi"
                else : 
                    if trade_duration >= int(duration) and current_profit >= roi:
                        return "exit_ema_roi"
        # ---------------------------
        # VPVR 전략 청산
        # ---------------------------
        elif enter_tag.startswith('vpvr'):
            # VPVR 매물대 기반 동적 청산
            parts = enter_tag.split('_')
            position = parts[1]  # 'long' 또는 'short'
            breakout_level = float(parts[2])  # 진입 시 돌파한 매물대 값

            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            vpvr_levels = [col for col in dataframe.columns if 'vpvr_level' in col]

            # 현재 캔들 정보 가져오기
            current_row = dataframe.iloc[-1]
            levels = [current_row[col] for col in vpvr_levels if not pd.isna(current_row[col])]

            # 매물대 근접 허용 범위 (threshold)
            threshold = self.vpvr_threshold # $1 근처로 정의

            # 롱 포지션: 위쪽 매물대 탐색
            if position == 'long':
                upper_levels = [lvl for lvl in levels if lvl > breakout_level]
                if upper_levels:
                    closest_upper = min(upper_levels)
                    if abs(current_rate - closest_upper) <= threshold:
                        return f"exit_vpvr_long_near_{closest_upper}"
                else:
                    # 위쪽 매물대가 없으면 고정 수익률로 청산
                    if current_profit >= 0.05:
                        return "exit_vpvr_long_roi"

            # 숏 포지션: 아래쪽 매물대 탐색
            elif position == 'short':
                lower_levels = [lvl for lvl in levels if lvl < breakout_level]
                if lower_levels:
                    closest_lower = max(lower_levels)
                    if abs(current_rate - closest_lower) <= threshold:
                        return f"exit_vpvr_short_near_{closest_lower}"
                else:
                    # 아래쪽 매물대가 없으면 고정 수익률로 청산
                    if current_profit >= 0.05:
                        return "exit_vpvr_short_roi"

        return None
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 명시적인 exit 조건은 없음 - minimal_roi와 stoploss로 관리
        return dataframe
    
    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake,
        max_stake: float,
        leverage: float,
        entry_tag,
        side: str,
        **kwargs
    ) -> float:
        """
        Dynamic stake allocation based on `entry_tag` keywords.

        - pair: 거래 페어 (e.g., BTC/USDT)
        - current_time: 현재 시간
        - current_rate: 현재 거래 페어의 가격
        - proposed_stake: 기본으로 제안된 스테이크 금액
        - min_stake: 거래소에서 요구하는 최소 거래 금액
        - max_stake: 잔고 기반으로 가능한 최대 거래 금액
        - leverage: 레버리지 값
        - entry_tag: 거래 진입 시 설정된 태그 (진입 전략 정보를 담고 있음)
        - side: 거래 방향 ('long' 또는 'short')
        """

        # **데이터프레임에서 현재 캔들 정보 추출**
        # dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        # current_candle = dataframe.iloc[-1].squeeze()

        # **특정 전략 태그에 따른 스테이크 결정**
        # 'ema', 'vpvr', 'trend' 키워드를 기반으로 동적 할당
        if entry_tag:
            if "ema" in entry_tag:
                # 'ema' 태그가 포함된 경우 전체 최대 잔고의 80%
                return max(min(max_stake * 0.8, max_stake), min_stake)

            elif "vpvr" in entry_tag:
                # 'vpvr' 태그가 포함된 경우 전체 최대 잔고의 35%
                return max(min(max_stake * 0.35, max_stake), min_stake)

            elif "trend" in entry_tag:
                # 'trend' 태그가 포함된 경우 전체 최대 잔고의 60%
                return max(min(max_stake * 0.6, max_stake), min_stake)

        # **기본 스테이크 사용 (proposed_stake)**
        return proposed_stake







