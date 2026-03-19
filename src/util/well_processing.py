import pandas as pd
import numpy as np


def filter_wells_by_lithology(las_dfs):
    """
    Filter all wells into separate dictionaries based on lithology and NAS markers.
    
    Returns:
    --------
    tuple of 4 dictionaries: (shale, brine_sand, oil_sand, gas_sand)
    """
    shale = {}
    brine_sand = {}
    oil_sand = {}
    gas_sand = {}
    
    for well_name, df in las_dfs.items():
        # shale - LITH == 0
        shale[well_name] = df[df['LITH'].isin([0])]
        
        # brine-sand - LITH in [1, 3] and NAS == 1
        brine_sand[well_name] = df[df['LITH'].isin([1, 3]) & df['NAS'].isin([1])]
        
        # oil-sand - LITH in [1, 3] and NAS in [2, 5, 9]
        oil_sand[well_name] = df[df['LITH'].isin([1, 3]) & df['NAS'].isin([2, 5, 9])]
        
        # gas-sand - LITH in [1, 3] and NAS in [3, 4, 9]
        gas_sand[well_name] = df[df['LITH'].isin([1, 3]) & df['NAS'].isin([3, 4, 9])]
        
        print(f"{well_name}: shale={len(shale[well_name])}, brine={len(brine_sand[well_name])}, "
              f"oil={len(oil_sand[well_name])}, gas={len(gas_sand[well_name])}")
    
    return shale, brine_sand, oil_sand, gas_sand

def prepare_valid_tagilsk(las_dfs_o):
    las_dfs = las_dfs_o.copy()
    las_dfs_val = {}
    for well in las_dfs.keys():
        print(f'Processing well {well}')
        pl_gg = ['PL_GG_BH', 'PL_GG', 'PL_GGNORM']
        dtp = ['DTP_BH', 'DTP']
        dts = ['DTS_BH', 'DTS']

        # get valid column name density
        for colname in pl_gg:
            if colname in las_dfs[well].columns:
                pl_gg = colname
                break
        # get valid column name DTP
        for colname in dtp:
            if colname in las_dfs[well].columns:
                dtp = colname
                break
        # get valid column name DTS
        for colname in dts:
            if colname in las_dfs[well].columns:
                dts = colname
                break
        if type(dts) is list:
            dts = 'DTS'
            las_dfs[well][dts] = np.nan


        las_dfs_val[well] = pd.concat([las_dfs[well]['DEPTH'],
                                       las_dfs[well]['LITH'],
                                       las_dfs[well]['NAS'],
                                       las_dfs[well][pl_gg],
                                       las_dfs[well][dtp],
                                       las_dfs[well][dts]],
                                       keys = ['DEPTH', 'LITH', 'NAS', 'PL_GG', 'DTP', 'DTS'],
                                       axis=1)
        _create_velocities(las_dfs_val[well])
    
    return las_dfs_val

def _create_velocities(df):
    df['VP'] = 1e6/df['DTP']
    df['VS'] = 1e6/df['DTS']
            
