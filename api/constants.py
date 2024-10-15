from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

class Constants():

    def __init__(self) -> None:
        pass


    def _get_classifiers(self): 
        return {
            'v1.0': {
                'lr': LogisticRegression(class_weight='balanced',random_state=0,max_iter=1000),
                'rf': RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=5, min_samples_leaf=5),
                'Xv': XGBClassifier(),
                'Sv': SVC(probability=True, kernel='linear', C=0.7, max_iter=1000)
            },
            'v2.0': {
                'lr': LogisticRegression(class_weight='balanced',random_state=0),
                'rf': RandomForestClassifier(random_state=0),
                'Xv': XGBClassifier(random_state=0),
                'Sv': SVC(probability=True, kernel='linear', C=0.7, max_iter=1000, random_state=0)
            }
        }
    

    def _get_configs_train(self):
        return {
            'v1.0': {
                'removing_cols_for_train': ['Date', 'Symbol', 'Dividends', 'Stock Splits'],  # Removing cols when training and predict
                'min_volume_prep_models': 250_000, # Define the minimum daily volume that must be considered when training
                'clean_targets_prep_models': True  # If True, remove outliers when training (beta)
            },
            'v2.0': {
                'removing_cols_for_train': ['Date', 'Dividends', 'Stock Splits'], # Removing cols when training and predict
                'min_volume_prep_models': 250_000, # Define the minimum daily volume that must be considered when training
                'clean_targets_prep_models': True  # If True, remove outliers when training (beta)
            },
            'v2.1.20': {
                'removing_cols_for_train': ['Date', 'Symbol', 'Dividends', 'Stock Splits'],  # Removing cols when training and predict
                'min_volume_prep_models': 250_000, # Define the minimum daily volume that must be considered when training
                'clean_targets_prep_models': True  # If True, remove outliers when training (beta)
            }
        }
    

    ### atualmente as colunas são retiradas diretamente
    def _remove_features(self):
        return {
            'v2.2.20': [
                '0', 'SUPERTs_200_3.0', 'HILOl_13_21', 'SUPERTl_50_3.0', 'SUPERTl_200_3.0', 'PSARl_0.02_0.2', 'SUPERTs_100_3.0', 'QQEs_14_5_4.236', 
                'SUPERTl_12_3.0', 'PSARs_0.02_0.2', 'SUPERTs_26_3.0', 'SUPERTl_26_3.0', 'QQEl_14_5_4.236', 'SUPERTl_5_3.0', 
                'SUPERTs_5_3.0', 'HILOs_13_21', 'SUPERTs_50_3.0', 'SUPERTl_100_3.0', 'SUPERTs_12_3.0'
            ]
        }
    

    def _get_stable_symbols(self):
        '''
        Gets the valid symbols based in the default parameters configs. 
            active_date_symbol = '2024-10-01'
            recent_date_symbol = '2022-10-01'

        This is interessing due we don't request the API to the data that we won't need.
        '''
        return ['1INCH-USD', 'AAVE-USD', 'ABBC-USD', 'ACH-USD', 'ACQ-USD',
            'ADA-USD', 'ADS-USD', 'ADX-USD', 'AERGO-USD', 'AERO-USD',
            'AGLD-USD', 'AIOZ-USD', 'AKRO-USD', 'AKT-USD', 'ALEPH-USD',
            'ALGO-USD', 'ALICE-USD', 'ALPHA-USD', 'ALPINE-USD', 'AMB-USD',
            'AMP-USD', 'AMPL-USD', 'ANKR-USD', 'AOG-USD', 'APE-USD',
            'API3-USD', 'APT-USD', 'AR-USD', 'ARB-USD', 'ARPA-USD', 'ARX-USD',
            'ASTR-USD', 'ATA-USD', 'ATOM-USD', 'AUCTION-USD', 'AUDIO-USD',
            'AURY-USD', 'AVA-USD', 'AVAX-USD', 'AXS-USD', 'AZERO-USD',
            'BABYDOGE-USD', 'BAL-USD', 'BAND-USD', 'BAT-USD', 'BAX-USD',
            'BCH-USD', 'BDX-USD', 'BEPRO-USD', 'BICO-USD', 'BIFI-USD',
            'BLOCK-USD', 'BLOK-USD', 'BLZ-USD', 'BMX-USD', 'BNB-USD',
            'BOBA-USD', 'BOND-USD', 'BONDLY-USD', 'BOSON-USD', 'BRISE-USD',
            'BRWL-USD', 'BSW-USD', 'BTC-USD', 'BTT-USD', 'BURGER-USD',
            'C98-USD', 'CAKE-USD', 'CAS-USD', 'CAT-USD', 'CATS-USD', 'CCD-USD',
            'CEEK-USD', 'CELO-USD', 'CELR-USD', 'CERE-USD', 'CFG-USD',
            'CFX-USD', 'CHMB-USD', 'CHO-USD', 'CHR-USD', 'CHZ-USD',
            'CIRUS-USD', 'CKB-USD', 'CLH-USD', 'CLV-USD', 'COMBO-USD',
            'COTI-USD', 'CPOOL-USD', 'CREAM-USD', 'CREDI-USD', 'CRO-USD',
            'CRV-USD', 'CSPR-USD', 'CTC-USD', 'CTI-USD', 'CTSI-USD',
            'CULT-USD', 'CVX-USD', 'CWAR-USD', 'CWEB-USD', 'CWS-USD',
            'CXT-USD', 'DAG-USD', 'DAO-USD', 'DAPP-USD', 'DAPPX-USD',
            'DAR-USD', 'DASH-USD', 'DATA-USD', 'DCR-USD', 'DEGO-USD',
            'DENT-USD', 'DEXE-USD', 'DFI-USD', 'DFYN-USD', 'DGB-USD',
            'DIA-USD', 'DMTR-USD', 'DOCK-USD', 'DODO-USD', 'DOGE-USD',
            'DOT-USD', 'DPET-USD', 'DPR-USD', 'DREAMS-USD', 'DUSK-USD',
            'DVPN-USD', 'DYDX-USD', 'DYP-USD', 'EGLD-USD', 'ELA-USD',
            'ELF-USD', 'ELON-USD', 'ENJ-USD', 'ENS-USD', 'EOS-USD', 'EPIK-USD',
            'EPX-USD', 'EQX-USD', 'ERG-USD', 'ERN-USD', 'ERTHA-USD', 'ETC-USD',
            'ETH-USD', 'ETHW-USD', 'ETN-USD', 'EUL-USD', 'EVER-USD', 'EWT-USD',
            'FCON-USD', 'FEAR-USD', 'FET-USD', 'FIDA-USD', 'FIL-USD',
            'FITFI-USD', 'FLAME-USD', 'FLIP-USD', 'FLOKI-USD', 'FLOW-USD',
            'FLUX-USD', 'FLY-USD', 'FORTH-USD', 'FRM-USD', 'FTM-USD',
            'FTT-USD', 'FURY-USD', 'FX-USD', 'FXS-USD', 'G-USD', 'GAFI-USD',
            'GAME-USD', 'GARI-USD', 'GAS-USD', 'GEEQ-USD', 'GFT-USD',
            'GGG-USD', 'GHX-USD', 'GLM-USD', 'GLMR-USD', 'GLQ-USD', 'GMEE-USD',
            'GMM-USD', 'GNS-USD', 'GOAL-USD', 'GODS-USD', 'GRAPE-USD',
            'GTC-USD', 'HAI-USD', 'HAPI-USD', 'HARD-USD', 'HBAR-USD',
            'HBB-USD', 'HEART-USD', 'HIGH-USD', 'HNT-USD', 'HOTCROSS-USD',
            'HTR-USD', 'HYDRA-USD', 'HYVE-USD', 'ICE-USD', 'ICP-USD',
            'ICX-USD', 'ID-USD', 'IDEA-USD', 'ILV-USD', 'INJ-USD', 'IOI-USD',
            'IOST-USD', 'IOTX-USD', 'ISP-USD', 'IZI-USD', 'JAM-USD',
            'JASMY-USD', 'JST-USD', 'JUP-USD', 'KAI-USD', 'KAS-USD',
            'KAVA-USD', 'KCS-USD', 'KDA-USD', 'KLAY-USD', 'KLV-USD', 'KMD-USD',
            'KNC-USD', 'KRL-USD', 'KSM-USD', 'LAYER-USD', 'LBP-USD', 'LBR-USD',
            'LDO-USD', 'LIKE-USD', 'LINK-USD', 'LIT-USD', 'LITH-USD',
            'LMR-USD', 'LOCUS-USD', 'LOKA-USD', 'LOOKS-USD', 'LOOM-USD',
            'LPOOL-USD', 'LPT-USD', 'LQTY-USD', 'LRC-USD', 'LSK-USD',
            'LSS-USD', 'LTC-USD', 'LTO-USD', 'LUNC-USD', 'LYM-USD', 'MAHA-USD',
            'MAN-USD', 'MANA-USD', 'MARS4-USD', 'MARSH-USD', 'MAV-USD',
            'MAX-USD', 'MBL-USD', 'MELOS-USD', 'MEME-USD', 'METIS-USD',
            'MINA-USD', 'MJT-USD', 'MKR-USD', 'MLK-USD', 'MNDE-USD', 'MNW-USD',
            'MON-USD', 'MONI-USD', 'MOVR-USD', 'MPLX-USD', 'MTL-USD',
            'MTRG-USD', 'MTV-USD', 'MV-USD', 'MXC-USD', 'NAKA-USD', 'NAVI-USD',
            'NEAR-USD', 'NEO-USD', 'NGC-USD', 'NGL-USD', 'NIM-USD', 'NKN-USD',
            'NMR-USD', 'NORD-USD', 'NTRN-USD', 'NUM-USD', 'NWC-USD', 'NYM-USD',
            'ODDZ-USD', 'OGN-USD', 'OLE-USD', 'OLT-USD', 'OM-USD', 'OMG-USD',
            'OMNI-USD', 'ONE-USD', 'ONT-USD', 'OOE-USD', 'OP-USD', 'OPEN-USD',
            'OPUL-USD', 'ORAI-USD', 'ORBS-USD', 'ORCA-USD', 'ORN-USD',
            'OSMO-USD', 'OUSD-USD', 'OVR-USD', 'OXT-USD', 'PAXG-USD',
            'PBR-USD', 'PBX-USD', 'PENDLE-USD', 'PEOPLE-USD', 'PERP-USD',
            'PHA-USD', 'PIT-USD', 'PIX-USD', 'PLU-USD', 'PMG-USD', 'POKT-USD',
            'POLC-USD', 'POLK-USD', 'POLS-USD', 'POLYX-USD', 'POND-USD',
            'PRE-USD', 'PROM-USD', 'PRQ-USD', 'PSTAKE-USD', 'PUNDIX-USD',
            'PUSH-USD', 'PYR-USD', 'QI-USD', 'QKC-USD', 'QNT-USD', 'QUICK-USD',
            'RACA-USD', 'RATS-USD', 'RAY-USD', 'RDNT-USD', 'REEF-USD',
            'REN-USD', 'REQ-USD', 'REV3L-USD', 'REVU-USD', 'REVV-USD',
            'RIO-USD', 'RLC-USD', 'ROOBEE-USD', 'ROSE-USD', 'ROUTE-USD',
            'RPL-USD', 'RSR-USD', 'RUNE-USD', 'RVN-USD', 'SAFE-USD',
            'SAND-USD', 'SCA-USD', 'SCLP-USD', 'SCRT-USD', 'SD-USD',
            'SDAO-USD', 'SENSO-USD', 'SFP-USD', 'SFUND-USD', 'SHIB-USD',
            'SHR-USD', 'SIDUS-USD', 'SKEY-USD', 'SKL-USD', 'SKY-USD',
            'SLIM-USD', 'SLP-USD', 'SNS-USD', 'SNX-USD', 'SOL-USD',
            'SOLVE-USD', 'SOUL-USD', 'SPA-USD', 'SSV-USD', 'STND-USD',
            'STORE-USD', 'STORJ-USD', 'STRAX-USD', 'STRK-USD', 'STX-USD',
            'SUKU-USD', 'SUN-USD', 'SUSHI-USD', 'SUTER-USD', 'SWASH-USD',
            'SWEAT-USD', 'SWFTC-USD', 'SXP-USD', 'SYLO-USD', 'SYS-USD',
            'T-USD', 'TARA-USD', 'TEL-USD', 'TFUEL-USD', 'THETA-USD',
            'TIA-USD', 'TIDAL-USD', 'TIME-USD', 'TLM-USD', 'TLOS-USD',
            'TOKO-USD', 'TON-USD', 'TOWER-USD', 'TRAC-USD', 'TRB-USD',
            'TRIAS-USD', 'TRU-USD', 'TRVL-USD', 'TRX-USD', 'TT-USD',
            'TUSD-USD', 'TWT-USD', 'UFO-USD', 'UMA-USD', 'UNB-USD', 'UNFI-USD',
            'UNI-USD', 'UNO-USD', 'UOS-USD', 'UPO-USD', 'UQC-USD', 'USDC-USD',
            'USDD-USD', 'USDJ-USD', 'USDP-USD', 'USTC-USD', 'UTK-USD',
            'VAI-USD', 'VELO-USD', 'VEMP-USD', 'VERSE-USD', 'VET-USD',
            'VINU-USD', 'VOLT-USD', 'VOXEL-USD', 'VR-USD', 'VRA-USD',
            'VSYS-USD', 'VXV-USD', 'WAVES-USD', 'WBTC-USD', 'WEMIX-USD',
            'WEST-USD', 'WHALE-USD', 'WILD-USD', 'WIN-USD', 'WLD-USD',
            'WLKN-USD', 'WOO-USD', 'WOOP-USD', 'WRX-USD', 'XAI-USD',
            'XAVA-USD', 'XCAD-USD', 'XCH-USD', 'XCN-USD', 'XCUR-USD',
            'XCV-USD', 'XDB-USD', 'XDC-USD', 'XDEFI-USD', 'XEC-USD', 'XEM-USD',
            'XLM-USD', 'XMR-USD', 'XNL-USD', 'XNO-USD', 'XPR-USD', 'XPRT-USD',
            'XRD-USD', 'XRP-USD', 'XTAG-USD', 'XTM-USD', 'XTZ-USD', 'XWG-USD',
            'XYM-USD', 'XYO-USD', 'YFI-USD', 'YGG-USD', 'ZCX-USD', 'ZEC-USD',
            'ZEE-USD', 'ZEN-USD', 'ZIL-USD', 'ZPAY-USD', 'ZRX-USD']