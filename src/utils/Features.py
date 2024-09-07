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
        # Aplica o indicador por grupo de símbolos
        indicator_columns = dataframe.groupby(symbol_column, group_keys=False).apply(
            lambda df: indicator_function(high=df['High'], 
                                            low=df['Low'], 
                                            close=df['Close'], 
                                            volume=df['Volume'], 
                                            open_=df['Open'],
                                            **kwargs)
        )
        
        # Concatena as novas colunas ao dataframe original
        dataframe_concat = pd.concat([dataframe, indicator_columns], axis=1)
        
        return dataframe_concat


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
        dataframe = self.apply_indicator(dataframe, ta.ad) # Accumulation/Distribution (AD
        dataframe = self.apply_indicator(dataframe, ta.adosc) # Accumulation/Distribution Oscillator
        dataframe = self.apply_indicator(dataframe, ta.aobv) #Archer On Balance Volume (AOBV)
        dataframe = self.apply_indicator(dataframe, ta.cmf) # Chaikin Money Flow (CMF)
        dataframe = self.apply_indicator(dataframe, ta.efi) # Elder's Force Index (EFI)
        dataframe = self.apply_indicator(dataframe, ta.eom) # Ease of Movement (EOM)
        dataframe = self.apply_indicator(dataframe, ta.kvo) # Klinger Volume Oscillator (KVO)
        dataframe = self.apply_indicator(dataframe, ta.pvol) # Price-Volume (PVOL)
        dataframe = self.apply_indicator(dataframe, ta.pvr) # Price Volume Rank
        dataframe = self.apply_indicator(dataframe, ta.pvt) # Price-Volume Trend (PVT)
        dataframe = self.apply_indicator(dataframe, ta.vp) #Volume Profile (VP)

        windows = [5, 12, 26, 50, 100, 200]

        # Calculate and add Exponential Moving Averages (EMAs)
        for window in windows:

            dataframe = self.apply_indicator(dataframe, ta.supertrend, length=window) # supertrend            

            # Moving Average (Overlap indicators)
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

            # Indicadores de volume 
            dataframe = self.apply_indicator(dataframe, ta.mfi, length=window) #  Money Flow Index (MFI)
            dataframe = self.apply_indicator(dataframe, ta.nvi, length=window) #  Negative Volume Index (NVI)
            dataframe = self.apply_indicator(dataframe, ta.obv, length=window) # On Balance Volume (OBV)
            dataframe = self.apply_indicator(dataframe, ta.pvi, length=window) # Positive Volume Index (PVI)

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
        print('passou nm_adx_trend')

        # Calculate Relative Strength Index (RSI)
        dataframe['rsi'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.rsi(x))
        dataframe['nm_rsi_trend'] = dataframe['rsi'].transform(self.classify_rsi_value)

        print('passou nm_rsi_trend')


        #Calculate the MACD and Signal Line
        dataframe['vl_macd'] = dataframe['ema_12'] - dataframe['ema_26']
        dataframe['vl_macd_signal'] = dataframe.groupby('Symbol')['vl_macd'].transform(lambda x: x.ewm(span=9).mean())

        print('passou vl_macd_signal')

        dataframe['vl_macd_delta'] = dataframe['vl_macd'] - dataframe['vl_macd_signal']
        dataframe['qt_days_macd_delta_positive'] = self.count_positive_reset(dataframe['vl_macd_delta'])

        print('passou qt_days_macd_delta_positive')
        
        # Financial Result
        targets = [10, 15, 20, 25]
        timeframes = [7, 15, 30]

        for target in targets:
            for timeframe in timeframes:
                dataframe[f'target_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ((x.shift(-timeframe) - x)/x))
                dataframe[f'bl_target_{target}P_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ((x.shift(-timeframe) - x)/x) >= target / 100).astype(int)
                dataframe[f'bl_target_{target}N_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ((x.shift(-timeframe) - x)/x) <= -target / 100).astype(int)
        print('passou target')

        return dataframe
