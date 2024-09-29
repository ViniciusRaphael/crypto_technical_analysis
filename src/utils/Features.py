# Third-party libraries
import pandas as pd
import pandas_ta as ta


class Features():

    def __init__(self) -> None:
        pass


    def classify_adx_value(self, value):
        """
        Checks the ADX value against predefined ranges and returns the corresponding trend category.

        Args:
            value (int): The ADX value to be categorized.

        Returns:
            str or None: The category name for the provided ADX value.
        """
        NM_ADX_GROUP = {
            '0-25': '1. Absent or Weak Trend',
            '25-50': '2. Strong Trend',
            '50-75': '3. Very Strong Trend',
            '75-100': '4. Extremely Strong Trend'
        }

        for key, name in NM_ADX_GROUP.items():
            range_start, range_end = map(int, key.split('-'))
            if range_start <= value <= range_end:
                return name
        return None


    def classify_rsi_value(self, value):
        """
        Checks the RSI value against predefined ranges and returns the corresponding category.

        Args:
            value (float): The RSI value to be categorized.

        Returns:
            str or None: The category name for the provided RSI value.
        """
        RSI_GROUP = {
            '0-10': 'rsi_0_to_10',
            '11-20': 'rsi_11_to_20',
            '21-30': 'rsi_21_to_30',
            '31-40': 'rsi_31_to_40',
            '41-50': 'rsi_41_to_50',
            '51-60': 'rsi_51_to_60',
            '61-70': 'rsi_61_to_70',
            '71-80': 'rsi_71_to_80',
            '81-90': 'rsi_81_to_90',
            '91-100': 'rsi_91_to_100'
        }

        for key, name in RSI_GROUP.items():
            range_start, range_end = map(int, key.split('-'))
            if range_start <= value < range_end:
                return name
        return None


    def count_positive_reset(self, df_column):
        """
        Counts consecutive positive values in a DataFrame column and resets count on encountering negative values.

        Args:
            df_column (pandas.Series): The DataFrame column to be processed.

        Returns:
            list: A list containing counts of consecutive positive values.
        """
        count = 0
        counts = []

        for value in df_column:
            if value > 0:
                count += 1
            else:
                count = 0  # Reset count if a negative value is encountered
            counts.append(count)

        return counts


    def calculate_adx(self, grupo):
        adx_values = ta.adx(grupo['High'], grupo['Low'], grupo['Close'], length=14)
        if adx_values is not None and not adx_values.empty:
            # Adiciona as colunas calculadas ao grupo original
            grupo['vl_adx'] = adx_values['ADX_14'].values
            grupo['vl_dmp'] = adx_values['DMP_14'].values
            grupo['vl_dmn'] = adx_values['DMN_14'].values
        else:
            # Se o cálculo falhar, preenche com NaN
            grupo['vl_adx'] = pd.NA
            grupo['vl_dmp'] = pd.NA
            grupo['vl_dmn'] = pd.NA
        return grupo
    
    
    def apply_indicator(self, dataframe, indicator_function, symbol_column='Symbol', **kwargs):
        """
        Aplica um indicador técnico que usa High, Low e Close, adicionando as colunas geradas ao dataframe.
        
        Args:
        - dataframe (pd.DataFrame): O dataframe com os dados históricos.
        - indicator_function (function): Função do indicador a ser aplicada (por exemplo, ta.supertrend).
        - symbol_column (str): Nome da coluna de símbolos.
        - high_column (str): Nome da coluna de preços "High".
        - low_column (str): Nome da coluna de preços "Low".
        - close_column (str): Nome da coluna de preços "Close".
        - **kwargs: Parâmetros adicionais da função do indicador (ex. `length`, `mult`).
        
        Returns:
        - pd.DataFrame: DataFrame original com as colunas do indicador adicionadas.
        """
        print(f'Executing: ', str(indicator_function))

        dataframe = dataframe.reset_index(drop=True)

        # Aplica o indicador por grupo de símbolos
        indicator_columns = dataframe.groupby(symbol_column, group_keys=False).apply(
            lambda df: indicator_function(high=df['High'], 
                                            low=df['Low'], 
                                            close=df['Close'], 
                                            volume=df['Volume'], 
                                            open_=df['Open'],
                                            **kwargs)
        ).reset_index(drop=True)
        
        # Concatena as novas colunas ao dataframe original
        dataframe_concat = pd.concat([dataframe, indicator_columns], axis=1)

        # Removing duplicate cols
        dataframe_cleaned = dataframe_concat.loc[:, ~dataframe_concat.columns.duplicated()] 

        return dataframe_cleaned


    def add_indicators(self, dataframe):
        """
        Add technical indicators to the DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing financial data.

        Returns:
        - pd.DataFrame: Updated DataFrame with added technical indicators.
        """

        # Sort DataFrame by 'Symbol' and 'Date', convert Date column to Datetime
        dataframe = dataframe.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)

        # Some Overlap indicators
        
        dataframe = self.apply_indicator(dataframe, ta.hilo) # Gann HiLo (HiLo)
        dataframe = self.apply_indicator(dataframe, ta.hlc3) # HLC3
        dataframe = self.apply_indicator(dataframe, ta.ichimoku) # Ichimoku Kinkō Hyō (Ichimoku) ## Erro
        dataframe = self.apply_indicator(dataframe, ta.ohlc4) # OHLC4
        dataframe = self.apply_indicator(dataframe, ta.vwma) # Volume Weighted Moving Average (VWMA) 
        dataframe = self.apply_indicator(dataframe, ta.wcp) # Weighted Closing Price (WCP) # maybe cause memory problem 
        # dataframe = self.apply_indicator(dataframe, ta.vwap) # Volume Weighted Average Price (VWAP) ## Problema no calculo to_period não é mais usado

        # Volume indicators
        dataframe['pvr'] =  ta.pvr(dataframe['Close'], dataframe['Volume']) # Price Volume Rank 
        dataframe = self.apply_indicator(dataframe, ta.ad) # Accumulation/Distribution (AD)
        dataframe = self.apply_indicator(dataframe, ta.adosc) # Accumulation/Distribution Oscillator #
        dataframe = self.apply_indicator(dataframe, ta.aobv) #Archer On Balance Volume (AOBV)
        dataframe = self.apply_indicator(dataframe, ta.cmf) # Chaikin Money Flow (CMF) 
        dataframe = self.apply_indicator(dataframe, ta.efi) # Elder's Force Index (EFI)
        dataframe = self.apply_indicator(dataframe, ta.eom) # Ease of Movement (EOM)
        # dataframe = self.apply_indicator(dataframe, ta.kvo) # Klinger Volume Oscillator (KVO) #### erro com o dado todo
        dataframe = self.apply_indicator(dataframe, ta.pvol) # Price-Volume (PVOL)
        dataframe = self.apply_indicator(dataframe, ta.pvt) # Price-Volume Trend (PVT) ## Já calculado por outro indicador 
        # dataframe = self.apply_indicator(dataframe, ta.obv) # On Balance Volume (OBV) ## É chamado tmb no AOBV
        # dataframe = self.apply_indicator(dataframe, ta.vp) #Volume Profile (VP) ### Problema na função

        # Momentum Indicators
        dataframe = self.apply_indicator(dataframe, ta.ao) # Awesome Oscillator (AO)
        dataframe = self.apply_indicator(dataframe, ta.apo) #  Absolute Price Oscillator (APO)
        dataframe = self.apply_indicator(dataframe, ta.bop) # Balance of Power (BOP)
        dataframe = self.apply_indicator(dataframe, ta.brar) #  BRAR (BRAR)
        dataframe = self.apply_indicator(dataframe, ta.kst) # 'Know Sure Thing' (KST)
        # dataframe = self.apply_indicator(dataframe, ta.macd) # Moving Average, Convergence/Divergence (MACD) #### erro com o dado todo
        dataframe = self.apply_indicator(dataframe, ta.pvo) # Percentage Volume Oscillator (PVO)
        dataframe = self.apply_indicator(dataframe, ta.qqe) # Quantitative Qualitative Estimation (QQE)
        dataframe = self.apply_indicator(dataframe, ta.slope) # Slope
        dataframe = self.apply_indicator(dataframe, ta.squeeze_pro) #  Squeeze Momentum (SQZ) PRO
        dataframe = self.apply_indicator(dataframe, ta.squeeze) #  Squeeze Momentum (SQZ)
        dataframe = self.apply_indicator(dataframe, ta.stoch) # Stochastic Oscillator (STOCH)
        dataframe = self.apply_indicator(dataframe, ta.tsi) # True Strength Index (TSI)
        dataframe = self.apply_indicator(dataframe, ta.uo) #  Ultimate Oscillator (UO)
        dataframe = self.apply_indicator(dataframe, ta.ppo) # Percentage Price Oscillator (PPO)
        # dataframe = self.apply_indicator(dataframe, ta.td_seq) # Tom Demark Sequential (TD_SEQ)  # erro na chamada do índice 
        # dataframe = self.apply_indicator(dataframe, ta.stc) # Schaff Trend Cycle (STC) #### Erro na função

        # Cyclo Indicators
        # dataframe = self.apply_indicator(dataframe, ta.ebsw, length=window) #  Even Better SineWave (EBSW) # Erro na chamada da função

        # Performance Indicators
        dataframe = self.apply_indicator(dataframe, ta.drawdown) #  "Indicator: Drawdown (DD)
        
        # Trend Indicators
        dataframe = self.apply_indicator(dataframe, ta.amat) #  Archer Moving Averages Trends (AMAT)
        ## long_run, short_run, t_signals, xsignals cant be used right now (Differente parameters)

        # Volatility Indicators
        dataframe = self.apply_indicator(dataframe, ta.pdist) #Indicator:Price Distance (PDIST)
        # dataframe = self.apply_indicator(dataframe, ta.hwc) #Indicator: Holt-Winter Channel ## Erro na função
        # dataframe = self.apply_indicator(dataframe, ta.massi) #Indicator: Mass Index (MASSI) ## Multiplos Close na função

        # Candles Indicators
        dataframe = self.apply_indicator(dataframe, ta.cdl_inside) #  Candle Type: Inside Bar
        # dataframe = self.apply_indicator(dataframe, ta.ha) #  Candle Type: Heikin Ashi

        windows = [5, 12, 26, 50, 100, 200]

        # Calculate and add Exponential Moving Averages (EMAs)
        for window in windows:

            print('Executing: ', window, ' window')

            dataframe = self.apply_indicator(dataframe, ta.supertrend, length=window) # supertrend       

            # Indicadores de volume 
            dataframe = self.apply_indicator(dataframe, ta.mfi, length=window) #  Money Flow Index (MFI)
            dataframe = self.apply_indicator(dataframe, ta.nvi, length=window) #  Negative Volume Index (NVI)
            dataframe = self.apply_indicator(dataframe, ta.pvi, length=window) # Positive Volume Index (PVI)

            # Momentum Indicators
            dataframe = self.apply_indicator(dataframe, ta.bias, length=window) # Bias (BIAS)
            dataframe = self.apply_indicator(dataframe, ta.cci, length=window) # Commodity Channel Index (CCI)
            dataframe = self.apply_indicator(dataframe, ta.cfo, length=window) # Chande Forcast Oscillator (CFO)
            dataframe = self.apply_indicator(dataframe, ta.cg, length=window) # Center of Gravity (CG)
            dataframe = self.apply_indicator(dataframe, ta.cmo, length=window) # Chande Momentum Oscillator (CMO)
            dataframe = self.apply_indicator(dataframe, ta.coppock, length=window) # Coppock Curve (COPC)
            dataframe = self.apply_indicator(dataframe, ta.cti, length=window) # Correlation Trend Indicator
            # dataframe = self.apply_indicator(dataframe, ta.dm, length=window) # DM ## erro ao rodar o arquivo todo
            dataframe = self.apply_indicator(dataframe, ta.er, length=window) # Efficiency Ratio (ER)
            dataframe = self.apply_indicator(dataframe, ta.eri, length=window) #  Elder Ray Index (ERI)
            dataframe = self.apply_indicator(dataframe, ta.fisher, length=window) # Fisher Transform (FISHT)
            dataframe = self.apply_indicator(dataframe, ta.inertia, length=window) # Inertia (INERTIA)
            dataframe = self.apply_indicator(dataframe, ta.kdj, length=window) # KDJ (KDJ)
            dataframe = self.apply_indicator(dataframe, ta.mom, length=window) # Momentum (MOM)
            dataframe = self.apply_indicator(dataframe, ta.pgo, length=window) # Pretty Good Oscillator (PGO)
            dataframe = self.apply_indicator(dataframe, ta.psl, length=window) #  Psychological Line (PSL)
            dataframe = self.apply_indicator(dataframe, ta.roc, length=window) # Rate of Change (ROC)
            dataframe = self.apply_indicator(dataframe, ta.rsi, length=window) # Relative Strength Index (RSI) ############## RSI
            dataframe = self.apply_indicator(dataframe, ta.rsx, length=window) #  Relative Strength Xtra (inspired by Jurik RSX)
            dataframe = self.apply_indicator(dataframe, ta.rvgi, length=window) #  Relative Vigor Index (RVGI)
            dataframe = self.apply_indicator(dataframe, ta.trix, length=window) # Trix (TRIX)
            dataframe = self.apply_indicator(dataframe, ta.willr, length=window) #  William's Percent R (WILLR)
            dataframe = self.apply_indicator(dataframe, ta.stochrsi, length=window) # Stochastic RSI Oscillator (STOCHRSI)

            # Performance Indicators
            dataframe = self.apply_indicator(dataframe, ta.log_return, length=window) #  "Indicator:  Log Return
            dataframe = self.apply_indicator(dataframe, ta.percent_return, length=window) #   Percent Return  

            # Trend Indicators
            dataframe = self.apply_indicator(dataframe, ta.adx, length=window) #  Indicator: ADX
            dataframe = self.apply_indicator(dataframe, ta.aroon, length=window) #  Indicator: Aroon & Aroon Oscillator
            dataframe = self.apply_indicator(dataframe, ta.chop, length=window) # Indicator: Choppiness Index (CHOP)
            dataframe = self.apply_indicator(dataframe, ta.cksp, length=window) #  Indicator: Chande Kroll Stop (CKSP)
            # dataframe = self.apply_indicator(dataframe, ta.decay, length=window) #  Indicator: Decay ### Erro na função
            dataframe = self.apply_indicator(dataframe, ta.decreasing, length=window) #  Indicator: Decreasing
            dataframe = self.apply_indicator(dataframe, ta.dpo, length=window) #  Indicator: Detrend Price Oscillator (DPO)
            dataframe = self.apply_indicator(dataframe, ta.increasing, length=window) #  Indicator: Increasing
            dataframe = self.apply_indicator(dataframe, ta.psar, length=window) #  Indicator: Parabolic Stop and Reverse (PSAR)
            dataframe = self.apply_indicator(dataframe, ta.qstick, length=window) #  Indicator: Q Stick
            dataframe = self.apply_indicator(dataframe, ta.ttm_trend, length=window) #  Indicator: TTM Trend (TTM_TRND)
            dataframe = self.apply_indicator(dataframe, ta.vhf, length=window) #  Indicator: Vertical Horizontal Filter (VHF)
            dataframe = self.apply_indicator(dataframe, ta.vortex, length=window) #  Indicator: Vortex   

            # Statistics Indicators
            dataframe = self.apply_indicator(dataframe, ta.entropy, length=window) #  Entropy (ENTP)
            dataframe = self.apply_indicator(dataframe, ta.kurtosis, length=window) #  Indicator: Kurtosis
            dataframe = self.apply_indicator(dataframe, ta.mad, length=window) # Mean Absolute Deviation
            dataframe = self.apply_indicator(dataframe, ta.median, length=window) #  Indicator: median
            dataframe = self.apply_indicator(dataframe, ta.quantile, length=window) # Quantile
            dataframe = self.apply_indicator(dataframe, ta.skew, length=window) #  Skew
            dataframe = self.apply_indicator(dataframe, ta.stdev, length=window) #  Indicator: Standard Deviation
            # dataframe = self.apply_indicator(dataframe, ta.tos_stdevall, length=window) #  TD Ameritrade's Think or Swim Standard Deviation All  #### Erro na função
            dataframe = self.apply_indicator(dataframe, ta.variance, length=window) #  Indicator: Variance
            dataframe = self.apply_indicator(dataframe, ta.zscore, length=window) #  Z Score

            # Volatility Indicators
            dataframe = self.apply_indicator(dataframe, ta.aberration, length=window) #Indicator: Aberration (ABER)
            dataframe = self.apply_indicator(dataframe, ta.accbands, length=window) #Indicator: Acceleration Bands (ACCBANDS)
            dataframe = self.apply_indicator(dataframe, ta.atr, length=window) #Indicator: Average True Range (ATR)"
            dataframe = self.apply_indicator(dataframe, ta.bbands, length=window) #Indicator: Indicator Bollinger Bands (BBANDS)
            dataframe = self.apply_indicator(dataframe, ta.donchian, length=window) #Indicator: Donchian Channels (DC)
            dataframe = self.apply_indicator(dataframe, ta.kc, length=window) #Indicator: Keltner Channels (KC)"
            dataframe = self.apply_indicator(dataframe, ta.natr, length=window) #Indicator:Normalized Average True Range (NATR)
            dataframe = self.apply_indicator(dataframe, ta.thermo, length=window) #Indicator:Elders Thermometer (THERMO)
            dataframe = self.apply_indicator(dataframe, ta.rvi, length=window) #Indicator:Relative Volatility Index (RVI)
            dataframe = self.apply_indicator(dataframe, ta.true_range, length=window) #Indicator:True Range
            dataframe = self.apply_indicator(dataframe, ta.ui, length=window) #Indicator:Ulcer Index (UI)
            
            # Candles Indicators
            dataframe = self.apply_indicator(dataframe, ta.cdl_doji, length=window) #  Candle Type: Doji
            dataframe = self.apply_indicator(dataframe, ta.cdl_z, length=window) #  Candle Type: Z Score

            # Moving Average (Overlap indicators)
            print(f'Executing: Overlap Indicators {window} window')

            dataframe[f'ema_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.ema(x, length=window)) # Exponential Moving Average (EMA)
            dataframe[f'sma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.sma(x, length=window)) # Weighted Moving Average (WMA)
            dataframe[f'wma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.wma(x, length=window)) # Simple Moving Average (SMA)
            dataframe[f'alma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.alma(x, length=window)) # Média Móvel Arnaud Legoux (ALMA)
            dataframe[f'dema_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.dema(x, length=window)) # Double Exponential Moving Average (DEMA)
            dataframe[f'fwma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.fwma(x, length=window)) # Fibonacci's Weighted Moving Average (FWMA)
            dataframe[f'hma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.hma(x, length=window)) # Hull Moving Average (HMA)
            dataframe[f'linreg_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.linreg(x, length=window)) # Linear Regression Moving Average (LINREG)
            dataframe[f't3_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.t3(x, length=window)) # Tim Tillson's T3 Moving Average (T3)
            dataframe[f'swma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.swma(x, length=window)) # Symmetric Weighted Moving Average (SWMA)
            dataframe[f'sinwma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.sinwma(x, length=window)) # Sine Weighted Moving Average (SINWMA) by Everget of TradingView
            dataframe[f'zlma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.zlma(x, length=window)) # Zero Lag Moving Average (ZLMA)
            dataframe[f'vidya_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.vidya(x, length=window)) # Variable Index Dynamic Average (VIDYA)
            dataframe[f'trima_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.trima(x, length=window)) # Triangular Moving Average (TRIMA)
            dataframe[f'tema_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.tema(x, length=window)) # Triple Exponential Moving Average (TEMA)
            dataframe[f'midpoint_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.midpoint(x, length=window)) # Midpoint
            dataframe[f'pwma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.pwma(x, length=window)) # Pascal's Weighted Moving Average (PWMA)
            dataframe[f'rma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.rma(x, length=window)) # wildeR's Moving Average (RMA)
            dataframe[f'ssf_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.ssf(x, length=window)) # Ehler's Super Smoother Filter (SSF)
            dataframe[f'kama_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.kama(x, length=window)) # Kaufman's Adaptive Moving Average (KAMA)

            # dataframe[f'mcgd_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.mcgd(x, length=window)) # McGinley Dynamic Indicator (MCGD) ## Erro na função
            # dataframe[f'jma{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.jma(x, length=window)) # Jurik Moving Average (JMA) # Erro na função

        for ind in ['ema', 'sma', 'wma', 'alma','dema', 'fwma', #'mcgd', 'jma', 
                    'hma', 'linreg', 't3', 'swma', 'sinwma', 'zlma', 'vidya', 'trima',
                    'tema', 'midpoint', 'pwma', 'rma', 'kama', 'ssf']:
            
            for base_window in windows:
                for compare_window in windows:
                    if base_window < compare_window:
                        col_name = f'{ind}_{base_window}_above_{ind}_{compare_window}'
                        dataframe[col_name] = dataframe[f'{ind}_{base_window}'] > dataframe[f'{ind}_{compare_window}']
        
        # Calculate Average Directional Index (ADX)
        dataframe = dataframe.groupby('Symbol', group_keys=False).apply(self.calculate_adx).reset_index(drop=True)
        dataframe['nm_adx_trend'] = dataframe['vl_adx'].transform(self.classify_adx_value)
        print('created nm_adx_trend')

        # Calculate Relative Strength Index (RSI)
        dataframe['rsi'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.rsi(x))
        dataframe['nm_rsi_trend'] = dataframe['rsi'].transform(self.classify_rsi_value)

        print('created nm_rsi_trend')


        #Calculate the MACD and Signal Line
        dataframe['vl_macd'] = dataframe['ema_12'] - dataframe['ema_26']
        dataframe['vl_macd_signal'] = dataframe.groupby('Symbol')['vl_macd'].transform(lambda x: x.ewm(span=9).mean())

        print('created vl_macd_signal')

        dataframe['vl_macd_delta'] = dataframe['vl_macd'] - dataframe['vl_macd_signal']
        dataframe['qt_days_macd_delta_positive'] = self.count_positive_reset(dataframe['vl_macd_delta'])

        print('created qt_days_macd_delta_positive')

        # Cols that have more than 80% of NaN
        _remove_features = ['0', 'SUPERTs_200_3.0', 'HILOl_13_21', 'SUPERTl_50_3.0', 'SUPERTl_200_3.0', 'PSARl_0.02_0.2', 'SUPERTs_100_3.0', 'QQEs_14_5_4.236', 
                'SUPERTl_12_3.0', 'PSARs_0.02_0.2', 'SUPERTs_26_3.0', 'SUPERTl_26_3.0', 'QQEl_14_5_4.236', 'SUPERTl_5_3.0', 
                'SUPERTs_5_3.0', 'HILOs_13_21', 'SUPERTs_50_3.0', 'SUPERTl_100_3.0', 'SUPERTs_12_3.0',
                'ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26']
            

        dataframe = dataframe.drop(columns=_remove_features, errors='ignore')
        
        # Financial Result
        targets = [10, 15, 20, 25]
        timeframes = [7, 15, 30]

        for target in targets:
            for timeframe in timeframes:
                dataframe[f'target_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ((x.shift(-timeframe) - x)/x))
                dataframe[f'bl_target_{target}P_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ((x.shift(-timeframe) - x)/x) >= target / 100).astype(int)
                dataframe[f'bl_target_{target}N_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ((x.shift(-timeframe) - x)/x) <= -target / 100).astype(int)
        print('created target')

        duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
        print('duplicated_cols', duplicate_columns)

        return dataframe
