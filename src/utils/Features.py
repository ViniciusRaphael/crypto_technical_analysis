# Third-party libraries
import pandas as pd
import pandas_ta as ta
from pathlib import Path


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
        print('Executing: ', str(indicator_function.__name__), '| cols: ', len(dataframe.columns))

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

        # print(len(dataframe_cleaned.columns))
        l1 = len(dataframe_concat.columns)
        l2 = len(set((dataframe_concat.columns)))
        ll = l1 == l2
        
        if ll == False:
            print(l1, '--', l2)

        return dataframe_concat

    # def apply_indicator(
    #     self, 
    #     dataframe, 
    #     indicator_function, 
    #     symbol_column='Symbol', 
    #     high_column='High',
    #     low_column='Low',
    #     close_column='Close',
    #     open_column='Open',
    #     volume_column='Volume',
    #     **kwargs
    # ):
    #     """
    #     Aplica um indicador técnico que usa High, Low e Close, adicionando as colunas geradas ao dataframe.
        
    #     Args:
    #     - dataframe (pd.DataFrame): O dataframe com os dados históricos.
    #     - indicator_function (function): Função do indicador a ser aplicada (por exemplo, ta.supertrend).
    #     - symbol_column (str): Nome da coluna de símbolos.
    #     - high_column (str): Nome da coluna de preços "High".
    #     - low_column (str): Nome da coluna de preços "Low".
    #     - close_column (str): Nome da coluna de preços "Close".
    #     - open_column (str): Nome da coluna de preços "Open".
    #     - volume_column (str): Nome da coluna de volume.
    #     - **kwargs: Parâmetros adicionais da função do indicador (ex. `length`, `mult`).
        
    #     Returns:
    #     - pd.DataFrame: DataFrame original com as colunas do indicador adicionadas.
    #     """
    #     print(f'Executing: {indicator_function.__name__} | Columns before: {len(dataframe.columns)}')

    #     dataframe = dataframe.reset_index(drop=True)

    #     # Validação de colunas necessárias
    #     required_columns = [symbol_column, high_column, low_column, close_column, volume_column, open_column]
    #     missing_columns = [col for col in required_columns if col not in dataframe.columns]
    #     if missing_columns:
    #         raise ValueError(f"Missing columns in dataframe: {missing_columns}")

    #     # Aplica o indicador por grupo de símbolos
    #     indicator_columns = dataframe.groupby(symbol_column, group_keys=False).apply(
    #         lambda df: indicator_function(
    #             high=df[high_column], 
    #             low=df[low_column], 
    #             close=df[close_column], 
    #             volume=df[volume_column], 
    #             open_=df[open_column],
    #             **kwargs
    #         )
    #     ).reset_index(drop=True)

    #     # Verifica se o indicador retornou um DataFrame ou Series
    #     if isinstance(indicator_columns, pd.Series):
    #         indicator_columns = indicator_columns.to_frame()
        
    #     # Adiciona um prefixo para evitar duplicatas
    #     indicator_columns = indicator_columns.add_prefix(f"{indicator_function.__name__}_")

    #     # Reseta o índice para concatenar corretamente
    #     indicator_columns = indicator_columns.reset_index(drop=True)

    #     # Concatena as novas colunas ao dataframe original
    #     dataframe_concat = pd.concat([dataframe, indicator_columns], axis=1)

    #     return dataframe_concat



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

        indicators = [
            ta.hilo, ta.hlc3, ta.ohlc4, ta.vwma, ta.wcp
            # Volume indicators
            # dataframe['pvr'] =  ta.pvr(dataframe['Close'], dataframe['Volume']) # Price Volume Rank 
            ,ta.ad # Accumulation/Distribution (AD)
            ,ta.adosc # Accumulation/Distribution Oscillator 
            ,ta.aobv #Archer On Balance Volume (AOBV)
            ,ta.cmf # Chaikin Money Flow (CMF) 
            ,ta.eom # Ease of Movement (EOM)
            ,ta.pvol # Price-Volume (PVOL)
            ,ta.pvt # Price-Volume Trend (PVT) ## Já calculado por outro indicador 
            # ,ta.kvo) # Klinger Volume Oscillator (KVO) erro no nome da função kvos.name erro ao executar todas as criptos
            # ,ta.obv) # On Balance Volume (OBV) ## É chamado tmb no AOBV
            # ,ta.vp) #Volume Profile (VP) ### Problema na função

            # Momentum Indicators
            ,ta.ao # Awesome Oscillator (AO)
            ,ta.apo #  Absolute Price Oscillator (APO)
            ,ta.bop # Balance of Power (BOP)
            ,ta.brar #  BRAR (BRAR)
            ,ta.kst # 'Know Sure Thing' (KST)
            ,ta.pvo # Percentage Volume Oscillator (PVO)
            ,ta.qqe # Quantitative Qualitative Estimation (QQE)
            ,ta.slope # Slope
            ,ta.squeeze_pro #  Squeeze Momentum (SQZ) PRO
            ,ta.squeeze #  Squeeze Momentum (SQZ)
            ,ta.stoch # Stochastic Oscillator (STOCH)
            ,ta.tsi # True Strength Index (TSI)
            ,ta.uo #  Ultimate Oscillator (UO)
            ,ta.ppo # Percentage Price Oscillator (PPO)
            # ,ta.macd) # Moving Average, Convergence/Divergence (MACD) erro ao executar todas as criptos
            # ,ta.td_seq) # Tom Demark Sequential (TD_SEQ)  # erro na chamada do índice 
            # ,ta.stc) # Schaff Trend Cycle (STC) #### Erro na função

            # Cyclo Indicators
            # ,ta.ebsw, length=window) #  Even Better SineWave (EBSW) # Erro na chamada da função

            # Performance Indicators
            ,ta.drawdown #  "Indicator: Drawdown (DD)
            
            # Trend Indicators
            ,ta.amat #  Archer Moving Averages Trends (AMAT)
            ,ta.cksp #  Indicator: Chande Kroll Stop (CKSP) #################################### começou aqui os erros #### não tem window
            ,ta.psar #  Indicator: Parabolic Stop and Reverse (PSAR) ########### não tem window
            ## long_run, short_run, t_signals, xsignals cant be used right now (Differente parameters)

            # Volatility Indicators
            ,ta.pdist #Indicator:Price Distance (PDIST)
            ,ta.donchian #Indicator: Donchian Channels (DC)
            ,ta.true_range #Indicator:True Range


            # ,ta.hwc) #Indicator: Holt-Winter Channel ## Erro na função
            # ,ta.massi) #Indicator: Mass Index (MASSI) ## Multiplos Close na função

            # Candles Indicators
            ,ta.cdl_inside #  Candle Type: Inside Bar
            # ,ta.ha) #  Candle Type: Heikin Ashi
        ]

        
        indicators_window = [

            ta.supertrend # supertrend       

        #     # Indicadores de volume 
            ,ta.mfi #  Money Flow Index (MFI)
            ,ta.nvi #  Negative Volume Index (NVI)
            ,ta.pvi # Positive Volume Index (PVI)

        #     # Momentum Indicators
            ,ta.bias # Bias (BIAS)
            ,ta.cci # Commodity Channel Index (CCI)
            ,ta.cfo # Chande Forcast Oscillator (CFO)
            ,ta.cg # Center of Gravity (CG)
            ,ta.cmo # Chande Momentum Oscillator (CMO)
            ,ta.coppock # Coppock Curve (COPC)
            ,ta.cti # Correlation Trend Indicator
            ,ta.er # Efficiency Ratio (ER)
            ,ta.eri #  Elder Ray Index (ERI)
            ,ta.fisher # Fisher Transform (FISHT)
            ,ta.inertia # Inertia (INERTIA)
            ,ta.kdj # KDJ (KDJ)
            ,ta.mom # Momentum (MOM)
            ,ta.pgo # Pretty Good Oscillator (PGO)
            ,ta.psl #  Psychological Line (PSL)
            ,ta.roc # Rate of Change (ROC)
            ,ta.rsi # Relative Strength Index (RSI) ############## RSI
            ,ta.rsx #  Relative Strength Xtra (inspired by Jurik RSX)
            ,ta.rvgi #  Relative Vigor Index (RVGI)
            ,ta.trix # Trix (TRIX)
            ,ta.willr #  William's Percent R (WILLR)
            ,ta.stochrsi # Stochastic RSI Oscillator (STOCHRSI)
            # ,ta.dm # DM  # erro ao utilizar todas as criptos

        #     # Performance Indicators
            ,ta.log_return #  "Indicator:  Log Return
            ,ta.percent_return #   Percent Return  

        #     # Trend Indicators
            ,ta.adx #  Indicator: ADX
            ,ta.aroon #  Indicator: Aroon & Aroon Oscillator
            ,ta.chop # Indicator: Choppiness Index (CHOP)
            # ,ta.decay #  Indicator: Decay ### Erro na função
            ,ta.decreasing #  Indicator: Decreasing
            # ,ta.dpo #  Indicator: Detrend Price Oscillator (DPO) ######## mismatch em número de indicators
            ,ta.increasing #  Indicator: Increasing
            ,ta.qstick #  Indicator: Q Stick
            ,ta.ttm_trend #  Indicator: TTM Trend (TTM_TRND)
            ,ta.vhf #  Indicator: Vertical Horizontal Filter (VHF)
            ,ta.vortex #  Indicator: Vortex   

        #     # Statistics Indicators
            ,ta.entropy #  Entropy (ENTP)
            ,ta.kurtosis #  Indicator: Kurtosis
            ,ta.mad # Mean Absolute Deviation
            ,ta.median #  Indicator: median
            ,ta.quantile # Quantile
            ,ta.skew #  Skew
            ,ta.stdev #  Indicator: Standard Deviation
            # ,ta.tos_stdevall #  TD Ameritrade's Think or Swim Standard Deviation All  #### Erro na função
            ,ta.variance #  Indicator: Variance
            ,ta.zscore #  Z Score

        #     # Volatility Indicators
            ,ta.aberration #Indicator: Aberration (ABER)
            ,ta.accbands #Indicator: Acceleration Bands (ACCBANDS)
            ,ta.atr #Indicator: Average True Range (ATR)"
            ,ta.bbands #Indicator: Indicator Bollinger Bands (BBANDS)
            ,ta.kc #Indicator: Keltner Channels (KC)"
            ,ta.natr #Indicator:Normalized Average True Range (NATR)
            ,ta.thermo #Indicator:Elders Thermometer (THERMO)
            ,ta.rvi #Indicator:Relative Volatility Index (RVI)
            ,ta.ui #Indicator:Ulcer Index (UI)
            
        #     # Candles Indicators
            ,ta.cdl_doji #  Candle Type: Doji
            ,ta.cdl_z #  Candle Type: Z Score

        #     # Moving Average (Overlap indicators)
        #     print(f'Executing: Overlap Indicators {window} window')
        ]
        

        for ind in indicators:
            dataframe = self.apply_indicator(dataframe, ind)

        for ind in indicators_window:
            
            windows = [5, 12, 26, 50, 100]

            for window in windows:

                dataframe = self.apply_indicator(dataframe, ind, length = window)


        # # Some Overlap indicators
        # dataframe = self.apply_indicator(dataframe, ta.hilo) # Gann HiLo (HiLo)
        # dataframe = self.apply_indicator(dataframe, ta.hlc3) # HLC3
        # dataframe = self.apply_indicator(dataframe, ta.ohlc4) # OHLC4
        # dataframe = self.apply_indicator(dataframe, ta.vwma) # Volume Weighted Moving Average (VWMA)   
        # dataframe = self.apply_indicator(dataframe, ta.wcp) # Weighted Closing Price (WCP) # maybe cause memory problem 
        # # dataframe = self.apply_indicator(dataframe, ta.vwap) # Volume Weighted Average Price (VWAP) ## Erro na função
        # # dataframe = self.apply_indicator(dataframe, ta.ichimoku) # Ichimoku Kinkō Hyō (Ichimoku) #### Erro na função, se usar, vai dar erro no índice


        # # Volume indicators
        # dataframe['pvr'] =  ta.pvr(dataframe['Close'], dataframe['Volume']) # Price Volume Rank 
        # dataframe = self.apply_indicator(dataframe, ta.ad) # Accumulation/Distribution (AD)
        # dataframe = self.apply_indicator(dataframe, ta.adosc) # Accumulation/Distribution Oscillator 
        # dataframe = self.apply_indicator(dataframe, ta.aobv) #Archer On Balance Volume (AOBV)
        # dataframe = self.apply_indicator(dataframe, ta.cmf) # Chaikin Money Flow (CMF) 
        # dataframe = self.apply_indicator(dataframe, ta.eom) # Ease of Movement (EOM)
        # dataframe = self.apply_indicator(dataframe, ta.pvol) # Price-Volume (PVOL)
        # dataframe = self.apply_indicator(dataframe, ta.pvt) # Price-Volume Trend (PVT) ## Já calculado por outro indicador 
        # # dataframe = self.apply_indicator(dataframe, ta.kvo) # Klinger Volume Oscillator (KVO) erro no nome da função kvos.name erro ao executar todas as criptos
        # # dataframe = self.apply_indicator(dataframe, ta.obv) # On Balance Volume (OBV) ## É chamado tmb no AOBV
        # # dataframe = self.apply_indicator(dataframe, ta.vp) #Volume Profile (VP) ### Problema na função

        # # Momentum Indicators
        # dataframe = self.apply_indicator(dataframe, ta.ao) # Awesome Oscillator (AO)
        # dataframe = self.apply_indicator(dataframe, ta.apo) #  Absolute Price Oscillator (APO)
        # dataframe = self.apply_indicator(dataframe, ta.bop) # Balance of Power (BOP)
        # dataframe = self.apply_indicator(dataframe, ta.brar) #  BRAR (BRAR)
        # dataframe = self.apply_indicator(dataframe, ta.kst) # 'Know Sure Thing' (KST)
        # dataframe = self.apply_indicator(dataframe, ta.pvo) # Percentage Volume Oscillator (PVO)
        # dataframe = self.apply_indicator(dataframe, ta.qqe) # Quantitative Qualitative Estimation (QQE)
        # dataframe = self.apply_indicator(dataframe, ta.slope) # Slope
        # dataframe = self.apply_indicator(dataframe, ta.squeeze_pro) #  Squeeze Momentum (SQZ) PRO
        # dataframe = self.apply_indicator(dataframe, ta.squeeze) #  Squeeze Momentum (SQZ)
        # dataframe = self.apply_indicator(dataframe, ta.stoch) # Stochastic Oscillator (STOCH)
        # dataframe = self.apply_indicator(dataframe, ta.tsi) # True Strength Index (TSI)
        # dataframe = self.apply_indicator(dataframe, ta.uo) #  Ultimate Oscillator (UO)
        # dataframe = self.apply_indicator(dataframe, ta.ppo) # Percentage Price Oscillator (PPO)
        # # dataframe = self.apply_indicator(dataframe, ta.macd) # Moving Average, Convergence/Divergence (MACD) erro ao executar todas as criptos
        # # dataframe = self.apply_indicator(dataframe, ta.td_seq) # Tom Demark Sequential (TD_SEQ)  # erro na chamada do índice 
        # # dataframe = self.apply_indicator(dataframe, ta.stc) # Schaff Trend Cycle (STC) #### Erro na função

        # # Cyclo Indicators
        # # dataframe = self.apply_indicator(dataframe, ta.ebsw, length=window) #  Even Better SineWave (EBSW) # Erro na chamada da função

        # # Performance Indicators
        # dataframe = self.apply_indicator(dataframe, ta.drawdown) #  "Indicator: Drawdown (DD)
        
        # # Trend Indicators
        # dataframe = self.apply_indicator(dataframe, ta.amat) #  Archer Moving Averages Trends (AMAT)
        # ## long_run, short_run, t_signals, xsignals cant be used right now (Differente parameters)

        # # Volatility Indicators
        # dataframe = self.apply_indicator(dataframe, ta.pdist) #Indicator:Price Distance (PDIST)
        # # dataframe = self.apply_indicator(dataframe, ta.hwc) #Indicator: Holt-Winter Channel ## Erro na função
        # # dataframe = self.apply_indicator(dataframe, ta.massi) #Indicator: Mass Index (MASSI) ## Multiplos Close na função

        # # Candles Indicators
        # dataframe = self.apply_indicator(dataframe, ta.cdl_inside) #  Candle Type: Inside Bar
        # # dataframe = self.apply_indicator(dataframe, ta.ha) #  Candle Type: Heikin Ashi

        # windows = [5, 12, 26, 50, 100, 
        #         # 200
        #         ]

        # # Calculate and add Exponential Moving Averages (EMAs)
        # for window in windows:

        #     print('Executing: ', window, ' window')

        #     dataframe = self.apply_indicator(dataframe, ta.supertrend, length=window) # supertrend       

        #     # Indicadores de volume 
        #     dataframe = self.apply_indicator(dataframe, ta.mfi, length=window) #  Money Flow Index (MFI)
        #     dataframe = self.apply_indicator(dataframe, ta.nvi, length=window) #  Negative Volume Index (NVI)
        #     dataframe = self.apply_indicator(dataframe, ta.pvi, length=window) # Positive Volume Index (PVI)

        #     # Momentum Indicators
        #     dataframe = self.apply_indicator(dataframe, ta.bias, length=window) # Bias (BIAS)
        #     dataframe = self.apply_indicator(dataframe, ta.cci, length=window) # Commodity Channel Index (CCI)
        #     dataframe = self.apply_indicator(dataframe, ta.cfo, length=window) # Chande Forcast Oscillator (CFO)
        #     dataframe = self.apply_indicator(dataframe, ta.cg, length=window) # Center of Gravity (CG)
        #     dataframe = self.apply_indicator(dataframe, ta.cmo, length=window) # Chande Momentum Oscillator (CMO)
        #     dataframe = self.apply_indicator(dataframe, ta.coppock, length=window) # Coppock Curve (COPC)
        #     dataframe = self.apply_indicator(dataframe, ta.cti, length=window) # Correlation Trend Indicator
        #     dataframe = self.apply_indicator(dataframe, ta.er, length=window) # Efficiency Ratio (ER)
        #     dataframe = self.apply_indicator(dataframe, ta.eri, length=window) #  Elder Ray Index (ERI)
        #     dataframe = self.apply_indicator(dataframe, ta.fisher, length=window) # Fisher Transform (FISHT)
        #     dataframe = self.apply_indicator(dataframe, ta.inertia, length=window) # Inertia (INERTIA)
        #     dataframe = self.apply_indicator(dataframe, ta.kdj, length=window) # KDJ (KDJ)
        #     dataframe = self.apply_indicator(dataframe, ta.mom, length=window) # Momentum (MOM)
        #     dataframe = self.apply_indicator(dataframe, ta.pgo, length=window) # Pretty Good Oscillator (PGO)
        #     dataframe = self.apply_indicator(dataframe, ta.psl, length=window) #  Psychological Line (PSL)
        #     dataframe = self.apply_indicator(dataframe, ta.roc, length=window) # Rate of Change (ROC)
        #     dataframe = self.apply_indicator(dataframe, ta.rsi, length=window) # Relative Strength Index (RSI) ############## RSI
        #     dataframe = self.apply_indicator(dataframe, ta.rsx, length=window) #  Relative Strength Xtra (inspired by Jurik RSX)
        #     dataframe = self.apply_indicator(dataframe, ta.rvgi, length=window) #  Relative Vigor Index (RVGI)
        #     dataframe = self.apply_indicator(dataframe, ta.trix, length=window) # Trix (TRIX)
        #     dataframe = self.apply_indicator(dataframe, ta.willr, length=window) #  William's Percent R (WILLR)
        #     dataframe = self.apply_indicator(dataframe, ta.stochrsi, length=window) # Stochastic RSI Oscillator (STOCHRSI)
        #     # dataframe = self.apply_indicator(dataframe, ta.dm, length=window) # DM  # erro ao utilizar todas as criptos

        #     # Performance Indicators
        #     dataframe = self.apply_indicator(dataframe, ta.log_return, length=window) #  "Indicator:  Log Return
        #     dataframe = self.apply_indicator(dataframe, ta.percent_return, length=window) #   Percent Return  

        #     # Trend Indicators
        #     dataframe = self.apply_indicator(dataframe, ta.adx, length=window) #  Indicator: ADX
        #     dataframe = self.apply_indicator(dataframe, ta.aroon, length=window) #  Indicator: Aroon & Aroon Oscillator
        #     dataframe = self.apply_indicator(dataframe, ta.chop, length=window) # Indicator: Choppiness Index (CHOP)
        #     dataframe = self.apply_indicator(dataframe, ta.cksp, length=window) #  Indicator: Chande Kroll Stop (CKSP)
        #     # dataframe = self.apply_indicator(dataframe, ta.decay, length=window) #  Indicator: Decay ### Erro na função
        #     dataframe = self.apply_indicator(dataframe, ta.decreasing, length=window) #  Indicator: Decreasing
        #     # dataframe = self.apply_indicator(dataframe, ta.dpo, length=window) #  Indicator: Detrend Price Oscillator (DPO) ######## mismatch em número de indicators
        #     dataframe = self.apply_indicator(dataframe, ta.increasing, length=window) #  Indicator: Increasing
        #     dataframe = self.apply_indicator(dataframe, ta.psar, length=window) #  Indicator: Parabolic Stop and Reverse (PSAR)
        #     dataframe = self.apply_indicator(dataframe, ta.qstick, length=window) #  Indicator: Q Stick
        #     dataframe = self.apply_indicator(dataframe, ta.ttm_trend, length=window) #  Indicator: TTM Trend (TTM_TRND)
        #     dataframe = self.apply_indicator(dataframe, ta.vhf, length=window) #  Indicator: Vertical Horizontal Filter (VHF)
        #     dataframe = self.apply_indicator(dataframe, ta.vortex, length=window) #  Indicator: Vortex   

        #     # Statistics Indicators
        #     dataframe = self.apply_indicator(dataframe, ta.entropy, length=window) #  Entropy (ENTP)
        #     dataframe = self.apply_indicator(dataframe, ta.kurtosis, length=window) #  Indicator: Kurtosis
        #     dataframe = self.apply_indicator(dataframe, ta.mad, length=window) # Mean Absolute Deviation
        #     dataframe = self.apply_indicator(dataframe, ta.median, length=window) #  Indicator: median
        #     dataframe = self.apply_indicator(dataframe, ta.quantile, length=window) # Quantile
        #     dataframe = self.apply_indicator(dataframe, ta.skew, length=window) #  Skew
        #     dataframe = self.apply_indicator(dataframe, ta.stdev, length=window) #  Indicator: Standard Deviation
        #     # dataframe = self.apply_indicator(dataframe, ta.tos_stdevall, length=window) #  TD Ameritrade's Think or Swim Standard Deviation All  #### Erro na função
        #     dataframe = self.apply_indicator(dataframe, ta.variance, length=window) #  Indicator: Variance
        #     dataframe = self.apply_indicator(dataframe, ta.zscore, length=window) #  Z Score

        #     # Volatility Indicators
        #     dataframe = self.apply_indicator(dataframe, ta.aberration, length=window) #Indicator: Aberration (ABER)
        #     dataframe = self.apply_indicator(dataframe, ta.accbands, length=window) #Indicator: Acceleration Bands (ACCBANDS)
        #     dataframe = self.apply_indicator(dataframe, ta.atr, length=window) #Indicator: Average True Range (ATR)"
        #     dataframe = self.apply_indicator(dataframe, ta.bbands, length=window) #Indicator: Indicator Bollinger Bands (BBANDS)
        #     dataframe = self.apply_indicator(dataframe, ta.donchian, length=window) #Indicator: Donchian Channels (DC)
        #     dataframe = self.apply_indicator(dataframe, ta.kc, length=window) #Indicator: Keltner Channels (KC)"
        #     dataframe = self.apply_indicator(dataframe, ta.natr, length=window) #Indicator:Normalized Average True Range (NATR)
        #     dataframe = self.apply_indicator(dataframe, ta.thermo, length=window) #Indicator:Elders Thermometer (THERMO)
        #     dataframe = self.apply_indicator(dataframe, ta.rvi, length=window) #Indicator:Relative Volatility Index (RVI)
        #     dataframe = self.apply_indicator(dataframe, ta.true_range, length=window) #Indicator:True Range
        #     dataframe = self.apply_indicator(dataframe, ta.ui, length=window) #Indicator:Ulcer Index (UI)
            
        #     # Candles Indicators
        #     dataframe = self.apply_indicator(dataframe, ta.cdl_doji, length=window) #  Candle Type: Doji
        #     dataframe = self.apply_indicator(dataframe, ta.cdl_z, length=window) #  Candle Type: Z Score

        #     # Moving Average (Overlap indicators)
        #     print(f'Executing: Overlap Indicators {window} window')

        #     dataframe[f'ema_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.ema(x, length=window)) # Exponential Moving Average (EMA)
        #     dataframe[f'sma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.sma(x, length=window)) # Weighted Moving Average (WMA)
        #     dataframe[f'wma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.wma(x, length=window)) # Simple Moving Average (SMA)
        #     dataframe[f'alma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.alma(x, length=window)) # Média Móvel Arnaud Legoux (ALMA)
        #     dataframe[f'dema_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.dema(x, length=window)) # Double Exponential Moving Average (DEMA)
        #     dataframe[f'fwma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.fwma(x, length=window)) # Fibonacci's Weighted Moving Average (FWMA)
        #     dataframe[f'hma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.hma(x, length=window)) # Hull Moving Average (HMA)
        #     dataframe[f'linreg_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.linreg(x, length=window)) # Linear Regression Moving Average (LINREG)
        #     dataframe[f't3_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.t3(x, length=window)) # Tim Tillson's T3 Moving Average (T3)
        #     dataframe[f'swma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.swma(x, length=window)) # Symmetric Weighted Moving Average (SWMA)
        #     dataframe[f'sinwma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.sinwma(x, length=window)) # Sine Weighted Moving Average (SINWMA) by Everget of TradingView
        #     dataframe[f'zlma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.zlma(x, length=window)) # Zero Lag Moving Average (ZLMA)
        #     dataframe[f'vidya_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.vidya(x, length=window)) # Variable Index Dynamic Average (VIDYA)
        #     dataframe[f'trima_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.trima(x, length=window)) # Triangular Moving Average (TRIMA)
        #     dataframe[f'tema_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.tema(x, length=window)) # Triple Exponential Moving Average (TEMA)
        #     dataframe[f'midpoint_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.midpoint(x, length=window)) # Midpoint
        #     dataframe[f'pwma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.pwma(x, length=window)) # Pascal's Weighted Moving Average (PWMA)
        #     dataframe[f'rma_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.rma(x, length=window)) # wildeR's Moving Average (RMA)
        #     dataframe[f'ssf_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.ssf(x, length=window)) # Ehler's Super Smoother Filter (SSF)
        #     dataframe[f'kama_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.kama(x, length=window)) # Kaufman's Adaptive Moving Average (KAMA)

        #     # print(len(dataframe.columns))

        #     # dataframe[f'mcgd_{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.mcgd(x, length=window)) # McGinley Dynamic Indicator (MCGD) ## Erro na função
        #     # dataframe[f'jma{window}'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.jma(x, length=window)) # Jurik Moving Average (JMA) # Erro na função

        # for ind in ['ema', 'sma', 'wma', 'alma','dema', 'fwma', #'mcgd', 'jma', 
        #             'hma', 'linreg', 't3', 'swma', 'sinwma', 'zlma', 'vidya', 'trima',
        #             'tema', 'midpoint', 'pwma', 'rma', 'kama', 'ssf']:
            
        #     for base_window in windows:
        #         for compare_window in windows:
        #             if base_window < compare_window:
        #                 col_name = f'{ind}_{base_window}_above_{ind}_{compare_window}'
        #                 dataframe[col_name] = dataframe[f'{ind}_{base_window}'] > dataframe[f'{ind}_{compare_window}']

        # print(len(dataframe.columns))

        
        # # Calculate Average Directional Index (ADX)
        # dataframe = dataframe.groupby('Symbol', group_keys=False).apply(self.calculate_adx).reset_index(drop=True)
        # dataframe['nm_adx_trend'] = dataframe['vl_adx'].transform(self.classify_adx_value)
        # print('created nm_adx_trend')

        # # Calculate Relative Strength Index (RSI)
        # dataframe['rsi'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ta.rsi(x))
        # dataframe['nm_rsi_trend'] = dataframe['rsi'].transform(self.classify_rsi_value)

        # print('created nm_rsi_trend')


        # #Calculate the MACD and Signal Line
        # dataframe['vl_macd'] = dataframe['ema_12'] - dataframe['ema_26']
        # dataframe['vl_macd_signal'] = dataframe.groupby('Symbol')['vl_macd'].transform(lambda x: x.ewm(span=9).mean())

        # print('created vl_macd_signal')

        # dataframe['vl_macd_delta'] = dataframe['vl_macd'] - dataframe['vl_macd_signal']
        # dataframe['qt_days_macd_delta_positive'] = self.count_positive_reset(dataframe['vl_macd_delta'])

        # print('created qt_days_macd_delta_positive')

        # Cols that have more than 80% of NaN
        _remove_features = ['0', 'SUPERTs_200_3.0', 'HILOl_13_21', 'SUPERTl_50_3.0', 'SUPERTl_200_3.0', 'PSARl_0.02_0.2', 'SUPERTs_100_3.0', 'QQEs_14_5_4.236', 
                'SUPERTl_12_3.0', 'PSARs_0.02_0.2', 'SUPERTs_26_3.0', 'SUPERTl_26_3.0', 'QQEl_14_5_4.236', 'SUPERTl_5_3.0', 
                'SUPERTs_5_3.0', 'HILOs_13_21', 'SUPERTs_50_3.0', 'SUPERTl_100_3.0', 'SUPERTs_12_3.0',
                'DPO_5', 'DPO_12', 'DPO_26', 'DPO_50', 'DPO_100', 'DPO_200',
                #'ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26', '0'
                ]
            
        print('colunas totais arquivo ', len(dataframe.columns))

        dataframe = dataframe.drop(columns=_remove_features, axis = 1, errors='ignore')     

        print('colunas utilizadas salvar', len(dataframe.columns))

        
        # Financial Result
        targets = [10, 15, 20, 25]
        timeframes = [7, 15, 30]

        for target in targets:
            for timeframe in timeframes:
                dataframe[f'target_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ((x.shift(-timeframe) - x)/x))
                dataframe[f'bl_target_{target}P_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ((x.shift(-timeframe) - x)/x) >= target / 100)#astype(int)
                dataframe[f'bl_target_{target}N_{timeframe}d'] = dataframe.groupby('Symbol')['Close'].transform(lambda x: ((x.shift(-timeframe) - x)/x) <= -target / 100)#
        print('created target')

        duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
        print('duplicated_cols', duplicate_columns)

        return dataframe
    
    
    def clean_date(self, dados_date):

        dados_date['Date'] = pd.to_datetime(dados_date['Date'])
        dados_date['Date'] = dados_date['Date'].dt.strftime('%Y-%m-%d')

        return dados_date
    

    def build_crypto_indicators(self, parameters):

        # Read Parquet file into a Pandas DataFrame
        df_input_raw = parameters.cls_FileHandling.read_file(parameters.files_folder, parameters.file_ingestion)

        # Remove companies with discrepant numbers
        remove_symbols = 'MYRIA-USD'

        df_input_raw = df_input_raw[df_input_raw['Symbol'] != remove_symbols]

        # Add indicators to the DataFrame and cleaning the date format
        crypto_indicators_dataframe = self.add_indicators(df_input_raw)

        crypto_indicators_dataframe = self.clean_date(crypto_indicators_dataframe)

        if crypto_indicators_dataframe is not None:
            # Specify the folder and file name for saving the Parquet file
            output_path = Path(parameters.files_folder, parameters.file_w_indicators)

            # Create the output folder if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print('Converting file into parquet')

            crypto_indicators_dataframe.to_parquet(output_path, compression = 'snappy')

            print(f"Parquet file saved to {output_path} with {len(crypto_indicators_dataframe)} rows")
            
            parameters.cls_FileHandling.wait_for_file(output_path)

        else:
            print("No data fetched.")

