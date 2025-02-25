import numpy as np
import pandas as pd

class Diagnostics:
    """
    Remove truncated aborts by urself

    data frame with column names
    rt: RT wrt fixation
    t_stim: stimulus onset
    choice: 1 or -1
    ABL, ILD
    correct: 1 or 0
    """
    def __init__(self, data):
        self.data = data

    def plot_rtd_wrt_fix(self,bins):
        """
        return x and y for plotting
        """
        df = self.data.copy()
        rt_wrt_fix = df['rt']
        rt_wrt_fix_hist, _ = np.histogram(rt_wrt_fix, bins=bins, density=True)
        
        bin_centers = bins[:-1] + 0.5*(bins[1] - bins[0])

        return bin_centers, rt_wrt_fix_hist
    

    def plot_rtd_wrt_stim(self,bins):
        df = self.data.copy()
        rt_wrt_stim = df['rt'] - df['t_stim']
        rt_wrt_stim_hist, _ = np.histogram(rt_wrt_stim, bins=bins, density=True)
        
        bin_centers = bins[:-1] + 0.5*(bins[1] - bins[0])

        return bin_centers, rt_wrt_stim_hist

    def plot_tacho(self, bins):
        # prob of correct vs RT
        df = self.data.copy()
        df['RT_bin'] = pd.cut(df['rt'] - df['t_stim'], bins=bins, include_lowest=True)
        grouped_by_rt_bin = df.groupby('RT_bin', observed=False)['correct'].agg(['mean', 'count'])
        grouped_by_rt_bin['bin_mid'] = grouped_by_rt_bin.index.map(lambda x: x.mid)
        return grouped_by_rt_bin['bin_mid'], grouped_by_rt_bin['mean']
    
    def plot_chrono(self):
        # mean rt vs abs ILD for each ABL
        df = self.data.copy()
        all_ABL = np.sort(df['ABL'].unique())
        all_ILD = np.sort(df['ILD'].unique())
        all_ILD = all_ILD[all_ILD > 0] 

        abl_rt_dict = {}

        for ABL in all_ABL:
            per_ILD_rt = np.zeros_like(all_ILD)
            for idx, ILD in enumerate(all_ILD):
                filtered_df = df[ (df['ABL'] == ABL) \
                                            & (df['ILD'].isin([ILD, -ILD])) ]
                mean_rt = (filtered_df['rt'] - filtered_df['t_stim']).replace([np.nan, np.inf, -np.inf], np.nan).dropna().mean()
                per_ILD_rt[idx] = mean_rt
            abl_rt_dict[ABL] = per_ILD_rt
        
        return all_ILD, abl_rt_dict

    def plot_chrono_median(self):
        # median rt vs abs ILD for each ABL
        df = self.data.copy()
        all_ABL = np.sort(df['ABL'].unique())
        all_ILD = np.sort(df['ILD'].unique())
        all_ILD = all_ILD[all_ILD > 0] 

        abl_rt_dict = {}

        for ABL in all_ABL:
            per_ILD_rt = np.zeros_like(all_ILD)
            for idx, ILD in enumerate(all_ILD):
                filtered_df = df[ (df['ABL'] == ABL) \
                                            & (df['ILD'].isin([ILD, -ILD])) ]
                mean_rt = (filtered_df['rt'] - filtered_df['t_stim']).replace([np.nan, np.inf, -np.inf], np.nan).dropna().median()
                per_ILD_rt[idx] = mean_rt
            abl_rt_dict[ABL] = per_ILD_rt
        
        return all_ILD, abl_rt_dict

    def plot_quantile(self):
        # 10 - 90 percentiles in steps of 20
        df = self.data.copy()
        df['rt_wrt_stim'] = df['rt'] - df['t_stim']

        abl_ild_quantiles = {}
        quantile_levels = [0.25, 0.5, 0.75]

        all_ABL = np.sort(df['ABL'].unique())
        all_ILD = np.sort(df['ILD'].unique())
        all_ILD = all_ILD[all_ILD > 0] 

        for abl in all_ABL:
            abl_ild_quantiles[abl] = {}
            for ild in all_ILD:
                filtered_df = df[(df['ABL'] == abl) & (df['ILD'].isin([ild, -ild]))]
                quantiles = filtered_df['rt_wrt_stim'].replace([np.nan, np.inf, -np.inf], np.nan).dropna().quantile(quantile_levels).tolist()
                abl_ild_quantiles[abl][ild] = quantiles

        return abl_ild_quantiles
    
    def plot_psycho(self):
        df = self.data.copy()
        prob_choice_dict = {}

        all_ABL = np.sort(df['ABL'].unique())
        all_ILD = np.sort(df['ILD'].unique())

        for abl in all_ABL:
            filtered_df = df[df['ABL'] == abl]
            prob_choice_dict[abl] = [sum(filtered_df[filtered_df['ILD'] == ild]['choice'] == 1) / len(filtered_df[filtered_df['ILD'] == ild]) for ild in all_ILD]

        return prob_choice_dict
    
    def plot_correct_vs_abs_ILD(self):
        df = self.data.copy()
        df['abs_ILD'] = np.abs(df['ILD'])
        
        prob_correct_dict = {}
        
        all_ABL = np.sort(df['ABL'].unique())
        all_abs_ILD = np.sort(df['abs_ILD'].unique())
        
        for abl in all_ABL:
            filtered_df = df[df['ABL'] == abl]
            prob_correct_dict[abl] = [
                filtered_df[filtered_df['abs_ILD'] == abs_ILD]['correct'].mean()
                for abs_ILD in all_abs_ILD
            ]
        
        return prob_correct_dict


