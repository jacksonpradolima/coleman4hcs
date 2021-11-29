import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# rpy2
import rpy2.robjects as ro
import seaborn as sns
import textwrap
import traceback
from pathlib import Path
from roses.effect_size import vargha_delaney
# roses
from roses.statistical_test.kruskal_wallis import kruskal_wallis
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

MAX_XTICK_WIDTH = 10

# For a beautiful plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
sns.set(palette="pastel")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def exception_to_string(excp):
    stack = traceback.extract_stack(
    )[:-3] + traceback.extract_tb(excp.__traceback__)  # add limit=??
    pretty = traceback.format_list(stack)
    return ''.join(pretty) + '\n  {} {}'.format(excp.__class__, excp)


class Analisys(object):
    """
    This class provide function to manage the result provide by COLEMAN
    """

    def __init__(self, project_dir, results_dir, variant_folder=None, font_size_plots=25,
                 sched_time_ratio=[0.1, 0.5, 0.8], replace_names=[], columns_metrics=[]):
        self.project_dir = project_dir
        self.project = project_dir.split('/')[-1]
        self.results_dir = results_dir
        self.columns_metrics = columns_metrics

        self.is_variant_test = variant_folder is not None

        if self.is_variant_test:
            self.figure_dir = f"{self.results_dir}_plots/{variant_folder}/{self.project}"
        else:
            self.figure_dir = f"{self.results_dir}_plots/{self.project}"

        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

        self.sched_time_ratio = sched_time_ratio
        self.sched_time_ratio_names = [
            str(int(tr * 100)) for tr in sched_time_ratio]

        self.reward_names = {
            'Time-ranked Reward': 'TimeRank',
            'timerank': 'TimeRank',  # From RETECS definition
            'Reward Based on Failures': 'RNFail'
        }

        # Load the information about the system
        self.df_system = self._get_df_system()

        # Load the results from system
        self.datasets = {}
        self._load_datasets(variant_folder, replace_names)

        self.font_size_plots = font_size_plots
        self._update_rc_params()

    def update_project(self, project_dir, variant_folder=None, replace_names=[]):
        self.project_dir = project_dir
        self.project = project_dir.split('/')[-1]

        if self.is_variant_test:
            self.figure_dir = f"{self.results_dir}_plots/{variant_folder}/{self.project}"
        else:
            self.figure_dir = f"{self.results_dir}_plots/{self.project}"

        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

        # Load the information about the system
        self.df_system = self._get_df_system()

        # Load the results from system
        self.datasets = {}
        self._load_datasets(variant_folder, replace_names)

    def update_font_size(self, font_size_plots):
        self.font_size_plots = font_size_plots
        self._update_rc_params()

    def _update_rc_params(self):
        plt.rcParams.update({
            'font.size': self.font_size_plots,
            'xtick.labelsize': self.font_size_plots,
            'ytick.labelsize': self.font_size_plots,
            'legend.fontsize': self.font_size_plots,
            'axes.titlesize': self.font_size_plots,
            'axes.labelsize': self.font_size_plots,
            'figure.max_open_warning': 0,
            'pdf.fonttype': 42
        })

    def _get_df_system(self):
        # Dataset Info
        df = pd.read_csv(f'{self.project_dir}/features-engineered.csv', sep=';', thousands=',')
        df = df.groupby(['BuildId'], as_index=False).agg({'Duration': np.sum})
        df.rename(columns={'BuildId': 'step',
                           'Duration': 'duration'}, inplace=True)

        return df

    def _load_datasets(self, variant_folder=None, replace_names=[]):
        for tr in self.sched_time_ratio_names:

            if variant_folder is not None:
                df_path = f"{self.results_dir}/time_ratio_{tr}/{variant_folder}"
            else:
                df_path = f"{self.results_dir}/time_ratio_{tr}"

            df = pd.read_csv(f'{df_path}/{self.project}.csv', sep=';', thousands=',', low_memory=False)

            df = df[['experiment', 'step', 'policy', 'reward_function', 'prioritization_time', 'time_reduction', 'ttf',
                     'fitness', 'avg_precision', 'cost', 'rewards']]

            df['reward_function'] = df['reward_function'].apply(
                lambda x: x if x == '-' else x.replace(x, self.reward_names[x]))

            df['policy'] = df['policy'].apply(
                lambda x: replace_names[x] if x in replace_names.keys() else x)

            df = df[df.policy.isin(self.columns_metrics)]
            df = df[df.reward_function != 'RNFail']

            # df['name'] = df.apply(lambda row: f"{row['policy']} ({row['reward_function']})" 
            # if 'Deterministic' not in row['policy'] else 'Deterministic', axis=1)
            # df['name'] = df['name'].apply(
            #   lambda x: 'Random' if 'Random' in x else x)
            df['name'] = df['policy']

            n_builds = len(df['step'].unique())

            # Find the deterministic
            dt = df[df['name'] == 'Deterministic']

            # As we have only one experiment run (deterministic), we increase to have 30 independent runs
            # This allow us to calculate the values without problems :D
            dt = dt.append([dt] * 29, ignore_index=True)
            dt['experiment'] = np.repeat(list(range(1, 31)), n_builds)

            # Clean
            df = df[df['name'] != 'Deterministic']

            df = df.append(dt)

            df.sort_values(by=['name'], inplace=True)

            df.drop_duplicates(inplace=True)

            self.datasets[tr] = df

    def _replace_names(self, df, names: dict, column='policy'):
        return df[column].apply(lambda x: names[x] if x in names.keys() else x)

    def replace_names(self, names: dict, column='policy'):
        for key in self.datasets.keys():
            self.datasets[key][column] = self._replace_names(
                self.datasets[key], names, column)

    def _get_metric_ylabel(self, column, rw=None):
        metric = 'NAPFD'
        ylabel = metric
        if column == 'cost':
            metric = 'APFDc'
            ylabel = metric
        elif column == 'ttf':
            metric = 'RFTC'
            ylabel = 'Rank of the Failing Test Cases'
        elif column == 'prioritization_time':
            metric = 'PrioritizationTime'
            ylabel = 'Prioritization Time (sec.)'
        elif column == "rewards":
            metric = rw
            ylabel = rw

        return metric, ylabel

    def _get_rewards(self):
        if len(self.datasets.keys()) > 0:
            return self.datasets[list(self.datasets.keys())[0]][
                'reward_function'].unique()
        else:
            return []

    def _get_policies(self):
        if len(self.datasets.keys()) > 0:
            return self.datasets[list(self.datasets.keys())[0]]['name'].unique()
        else:
            return []

    def print_mean(self, df, column, direction='max'):
        mean = df.groupby(['name'], as_index=False).agg(
            {column: ['mean', 'std', 'max', 'min']})
        mean.columns = ['name', 'mean', 'std', 'max', 'min']

        # sort_df(mean)

        # Round values (to be used in the article)
        mean = mean.round({'mean': 4, 'std': 3, 'max': 4, 'min': 4})
        mean = mean.infer_objects()

        # minimum = mean[mean['mean'] > 0]
        # minimum = minimum['mean'].idxmin()
        minimum = mean['mean'].idxmin()

        bestp = mean.loc[mean['mean'].idxmax() if direction ==
                                                  'max' else minimum]

        val = 'Highest' if direction == 'max' else 'Lowest'

        print(f"\n{val} Value found by {bestp['name']}: {bestp['mean']:.4f}")
        print("\nMeans:")
        print(mean)

        return mean, bestp['name']

    def print_mean_latex(self, x, column):
        policies = self._get_policies()

        print(*policies, sep="\t")
        cont = len(policies)

        for policy in policies:
            df_temp = x[x.name == policy]
            print(f"{df_temp[column].mean():.4f} $\pm$ {df_temp[column].std():.3f} ", end="")

            cont -= 1
            if cont != 0:
                print("& ", end="")
        print()

    def _define_axies(self, ax, tr, column, ylabel=None):
        metric, ylabel_temp = self._get_metric_ylabel(column)

        ax.set_xlabel('CI Cycle', fontsize=self.font_size_plots)
        ax.set_ylabel(ylabel + " " + metric if ylabel is not None else metric,
                      fontsize=self.font_size_plots)
        ax.set_title(f"Time Budget: {tr}%", fontsize=self.font_size_plots)

    def _plot_accumulative(self, df, ax, tr, column='fitness'):
        df = df[['step', 'name', column]]
        df.groupby(['step', 'name']).mean()[
            column].unstack().cumsum().plot(ax=ax, linewidth=3)

        self._define_axies(ax, tr, column, ylabel='Accumulative')

    def plot_accumulative(self, figname, column='fitness'):
        policies = self._get_policies()

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(25, 8))
        # Todo try a generic way
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            self._plot_accumulative(self.datasets[df_k], ax, tr, column)

        handles, labels = ax1.get_legend_handles_labels()
        # lgd = fig.legend(handles, labels, bbox_to_anchor=(
        #    0, -0.03, 1, 0.2), loc='lower center', ncol=len(policies))
        # lgd = ax1.legend(handles, labels, bbox_to_anchor=(-0.02,
        # 1.05, 1, 0.2), loc='lower left', ncol=len(policies))        
        lgd = fig.legend(handles, labels, bbox_to_anchor=(
            0.98, 0.1), loc='lower right')

        ax1.get_legend().remove()
        ax2.get_legend().remove()
        ax3.get_legend().remove()

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{figname}.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

    def _plot_lines(self, df, ax, tr, column='fitness'):
        df = df[['step', 'name', column]]
        df.groupby(['step', 'name']).mean()[
            column].unstack().plot(ax=ax, linewidth=3)

        self._define_axies(ax, tr, column)

    def plot_lines(self, figname, column='fitness'):
        policies = self._get_policies()

        fig, axes = plt.subplots(
            nrows=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(25, 20))
        # Todo try a generic way
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            self._plot_lines(self.datasets[df_k], ax, tr, column)

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, bbox_to_anchor=(-0.005, 1.05, 1, 0.2), loc='lower left',
                         ncol=len(policies))

        ax2.get_legend().remove()
        ax3.get_legend().remove()

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{figname}.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

    def _visualize_ntr(self, df, tr, ax, total_time_spent):
        policies = self.columns_metrics

        # Only the commits which failed
        x = df[['experiment', 'name', 'time_reduction']
        ][(df.avg_precision == 123)]

        df_ntr = pd.DataFrame(columns=['experiment', 'name', 'n_reduction'])

        # print(*policies, sep="\t")
        # cont = len(policies)

        row = [tr]
        means = []
        for policy in policies:
            df_ntr_temp = x[x.name == policy]

            # sum all differences (time_reduction column) in all cycles for
            # each experiment
            df_ntr_temp = df_ntr_temp.groupby(['experiment'], as_index=False).agg({
                'time_reduction': np.sum})

            # Evaluate for each experiment
            df_ntr_temp['n_reduction'] = df_ntr_temp['time_reduction'].apply(
                lambda x: x / (total_time_spent))

            df_ntr_temp['name'] = policy

            df_ntr_temp = df_ntr_temp[['experiment', 'name', 'n_reduction']]

            df_ntr = df_ntr.append(df_ntr_temp)

            means.append(df_ntr_temp['n_reduction'].mean())
            if means[-1] <= 0:
                means[-1] = 0.0
            text = f"{means[-1]:.4f} $\pm$ {df_ntr_temp['n_reduction'].std():.3f}"

            row.append(text)

        if len(df_ntr) > 0:
            df_ntr.sort_values(by=['name', 'experiment'], inplace=True)
            sns.boxplot(x='name', y='n_reduction', data=df_ntr, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('Normalized Time Reduction' if tr ==
                                                         '10' else '')  # Workaround
            ax.set_title(f"Time Budget: {tr}%")
            ax.set_xticklabels(textwrap.fill(x.get_text(), MAX_XTICK_WIDTH)
                               for x in ax.get_xticklabels())

        best_idx = [i for i, x in enumerate(means) if x == max(means)]

        if len(best_idx) == 1:
            best_i = best_idx[0]
            row[best_i + 1] = f"\\cellbold{{{row[best_i + 1]}}}"
        else:
            for best_i in best_idx:
                row[best_i + 1] = f"\\cellgray{{{row[best_i + 1]}}}"

        return row

    def visualize_ntr(self):
        stat_columns = ['TimeBudget'] + self.columns_metrics
        df_stats = pd.DataFrame(columns=stat_columns)

        # Total time spent in each Cycle
        total_time_spent = self.df_system['duration'].sum()
        policies = self._get_policies()

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(int(8.3 * 3 * (len(policies) / 3)), 8))
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            df = self.datasets[df_k]
            row = self._visualize_ntr(df, tr, ax, total_time_spent)
            df_stats = df_stats.append(
                pd.DataFrame([row], columns=stat_columns))

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/NTR.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

        df_stats['Metric'] = 'NTR'

        return df_stats

    def _visualize_duration(self, df):
        dd = df[['name', 'prioritization_time']]
        # sort_df(dd)
        self.print_mean(dd, 'prioritization_time', direction='min')
        self.print_mean_latex(dd, 'prioritization_time')

    def visualize_duration(self):
        rewards = self._get_rewards()

        for rw in rewards:
            print(f"\n\n======{rw}======")
            for df_k, tr in zip(self.datasets.keys(), self.sched_time_ratio_names):
                print(f"\nTime Budget {tr}%")
                df = self.datasets[df_k]
                self._visualize_duration(df[df.reward_function == rw])

    def _transpose_df(self, df, column='name'):
        df_tras = df.copy()
        df_tras.index = df_tras[column]
        return df_tras.transpose()

    def _rmse_calculation(self, df, column='fitness'):
        def get_rmse_symbol(mean):
            """
            very near     if RMSE < 0.15
            near          if 0.15 <= RMSE < 0.23
            reasonable    if 0.23 <= RMSE < 0.30
            far           if 0.30 <= RMSE < 0.35
            very far      if 0.35 <= RMSE
            """
            if mean < 0.15:
                # very near
                return "$\\bigstar$"
            elif mean < 0.23:
                # near
                return "$\\blacktriangledown$"
            elif mean < 0.30:
                # reasonable
                return "$\\triangledown$"
            elif mean < 0.35:
                # far
                return "$\\vartriangle$"
            else:
                # very far
                return "$\\blacktriangle$"

        def get_mean_std_rmse(df_rmse, column, n_builds):
            df_f = df_rmse.groupby(['experiment'], as_index=False).agg(
                {column: lambda x: np.sqrt(sum(x) / n_builds)})

            # Get column values and provide a beautiful output
            mean, std = round(df_f[column].mean(), 4), round(
                df_f[column].std(), 4)

            return [mean, std, f"{mean:.4f} $\\pm$ {std:.3f} {get_rmse_symbol(mean)}".strip()]

        def get_config_latex(row, best_rmse, contains_equivalent):
            """
            Latex commands used:
            - Best algorithm: \newcommand{\cellgray}[1]{\cellcolor{gray!30}{#1}}
            - Equivalent to the best one: \newcommand{\cellbold}[1]{\cellcolor{gray!30}{\textbf{#1}}}
            """
            if contains_equivalent and row['mean'] == best_rmse:
                return f"\\cellgray{{{row['output']}}}"

            if row['mean'] == best_rmse:
                return f"\\cellbold{{{row['output']}}}"

            return row['output']

        columns = [column, 'experiment', 'step']

        n_builds = len(df['step'].unique())

        # Get only the required columns
        df = df[['experiment', 'step', 'name', column]]

        # Orderby to guarantee the right value
        df.sort_values(by=['experiment', 'step'], inplace=True)

        df_rmse = pd.DataFrame(
            columns=['experiment', 'step', 'Deterministic'])

        dt = df.loc[df['name'] == 'Deterministic', columns]

        df_rmse['Deterministic'] = dt[column]
        df_rmse['experiment'] = dt['experiment']
        df_rmse['step'] = dt['step']

        policies = list(self._get_policies())
        policies.remove('Deterministic')

        for pol in policies:
            df_rmse[pol] = df.loc[df['name']
                                  == pol, columns][column].tolist()

        df_rmse = df_rmse.reset_index()

        for pol in policies:
            df_rmse[f'RMSE_{pol}'] = df_rmse.apply(lambda x: (x[pol] - x['Deterministic']) ** 2, axis=1)

        df_rmse_rows = []
        for pol in policies:
            rmse = get_mean_std_rmse(df_rmse, f'RMSE_{pol}', n_builds)

            df_rmse_rows.append([pol] + rmse)

        df_rmse_results = pd.DataFrame(
            df_rmse_rows, columns=['name', 'mean', 'std', 'output'])

        best_rmse = df_rmse_results['mean'].min()
        contains_equivalent = len(
            df_rmse_results[df_rmse_results['mean'] == best_rmse]) > 1

        df_rmse_results['latex_format'] = df_rmse_results.apply(
            lambda row: get_config_latex(row, best_rmse, contains_equivalent), axis=1)

        # Select the main information
        rmse = df_rmse_results[['name', 'latex_format']]

        # Return only the values
        return self._transpose_df(rmse).values[1]

    def rmse_calculation(self, column='fitness'):
        policies = list(self._get_policies())
        policies.remove('Deterministic')

        rmse_cols = ['TimeBudget'] + policies
        df_rmse = pd.DataFrame(columns=rmse_cols)

        for df_k, tr in zip(self.datasets.keys(), self.sched_time_ratio_names):
            df = self.datasets[df_k]
            row = [tr] + list(self._rmse_calculation(df, column))
            df_rmse = df_rmse.append(pd.DataFrame([row], columns=rmse_cols))

        metric, ylabel = self._get_metric_ylabel(column)

        df_rmse['Metric'] = 'RMSE_' + metric

        return df_rmse

    def _statistical_test_kruskal(self, df, ax, column):
        if column == 'ttf':
            df = df[df.ttf > 0]

        if (len(df)) > 0:
            # Get the mean of fitness in each experiment
            x = df[['experiment', 'name', column]]

            policies = self._get_policies()
            diff_pol = list(set(policies) - set(x['name'].unique()))

            x = x.groupby(['experiment', 'name'], as_index=False).agg(
                {column: np.mean})

            # Remove unnecessary columns
            x = x[['name', column]]

            mean, best = self.print_mean(x, column, 'min' if column in [
                'ttf', 'prioritization_time'] else 'max')

            mean['eff_symbol'] = " "

            posthoc_df = None
            all_equivalent = False

            try:
                k = kruskal_wallis(x, column, 'name')
                kruskal, posthoc = k.apply(ax)
                print(f"\n{kruskal}")  # Kruskal results

                all_equivalent = 'p-unc' not in kruskal.columns or kruskal[
                    'p-unc'][0] >= 0.05

                if posthoc is not None:
                    print("\n--- POST-HOC TESTS ---")
                    print("\np-values:")
                    print(posthoc[0])

                    # Get the posthoc
                    df_eff = vargha_delaney.reduce(posthoc[1], best)

                    print(df_eff)

                    def get_eff_symbol(x, best, df_eff):
                        if x['name'] == best:
                            return "$\\bigstar$"
                        elif len(df_eff.loc[df_eff.compared_with == x['name'], 'effect_size_symbol'].values) > 0:
                            return df_eff.loc[df_eff.compared_with == x['name'], 'effect_size_symbol'].values[0]
                        else:
                            return df_eff.loc[df_eff.base == x['name'], 'effect_size_symbol'].values[0]

                    mean['eff_symbol'] = mean.apply(
                        lambda x: get_eff_symbol(x, best, df_eff), axis=1)

                    # Parse the posthoc to a dataframe in R because allows us
                    # to parse to pandas in Py
                    ro.r.assign('posthoc', posthoc[0])
                    ro.r('posthoc_table <- t(as.matrix(posthoc$p.value))')
                    ro.r('df_posthoc <- as.data.frame(t(posthoc_table))')

                    # Convert the dataframe from R to pandas
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        posthoc_df = ro.conversion.rpy2py(ro.r('df_posthoc'))

            except Exception as e:
                print("\nError in statistical test:", exception_to_string(e))

            # Concat the values to a unique columns
            mean['avg_std_effect'] = mean.apply(
                lambda row: f"{row['mean']:.4f} $\\pm$ {row['std']:.4f} {row['eff_symbol']}".strip(), axis=1)

            def get_config_latex(row, best, posthoc_df, all_equivalent):
                """
                Latex commands used:
                - Best algorithm: \newcommand{\cellgray}[1]{\cellcolor{gray!30}{#1}}
                - Equivalent to the best one: \newcommand{\cellbold}[1]{\cellcolor{gray!30}{\textbf{#1}}}
                """
                current_name = row['name']

                if all_equivalent:
                    return f"\\cellgray{{{row['avg_std_effect']}}}"

                if row['name'] == best:
                    return f"\\cellbold{{{row['avg_std_effect']}}}"

                is_equivalent = False

                # If the posthoc was applied
                if posthoc_df is not None:
                    if best in posthoc_df.columns and current_name in posthoc_df.index and not np.isnan(
                            posthoc_df.loc[current_name][best]):
                        # They are equivalent
                        is_equivalent = posthoc_df.loc[
                                            current_name][best] >= 0.05
                    elif current_name in posthoc_df.columns and best in posthoc_df.index and not np.isnan(
                            posthoc_df.loc[best][current_name]):
                        # They are equivalent
                        is_equivalent = posthoc_df.loc[
                                            best][current_name] >= 0.05
                    else:
                        raise Exception(
                            "Problem found when we tried to find the post-hoc p-value")

                if is_equivalent:
                    return f"\\cellgray{{{row['avg_std_effect']}}}"

                return row['avg_std_effect']

            # Insert the latex commands
            mean['latex_format'] = mean.apply(lambda row: get_config_latex(
                row, best, posthoc_df, all_equivalent), axis=1)

            # Select the main information
            mean = mean[['name', 'latex_format']]

            mean_trans = mean.copy()
            mean_trans.index = mean['name']
            mean_trans = mean_trans.transpose()

            if len(diff_pol) > 0:
                # We remove the value from the policies that do not have result
                for dp in diff_pol:
                    mean_trans[dp] = '-'

            # Return only the values
            return mean_trans[self.columns_metrics].values[1]
        else:
            return ['-', '-', '-']

    def statistical_test_kruskal(self, column='fitness'):
        stat_columns = ['TimeBudget'] + self.columns_metrics
        df_stats = pd.DataFrame(columns=stat_columns)

        metric, ylabel = self._get_metric_ylabel(column)

        print(
            f"\n\n\n\n||||||||||||||||||||||||||||||| STATISTICAL TEST - KRUSKAL WALLIS - {metric} |||||||||||||||||||||||||||||||\n")

        policies = self._get_policies()

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(int(8.3 * 3 * (len(policies) / 3)), 8))
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            print(f"~~~~ Time Budget {tr}% ~~~~")
            row = self._statistical_test_kruskal(
                self.datasets[df_k], ax, column)
            row = np.insert(row, 0, tr)
            df_stats = df_stats.append(
                pd.DataFrame([row], columns=stat_columns))
            ax.set_title(f"Time Budget: {tr}%")
            ax.set_ylabel(ylabel if tr == '10' else '')  # Workaround
            ax.set_xticklabels(textwrap.fill(x.get_text(), MAX_XTICK_WIDTH)
                               for x in ax.get_xticklabels())

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{metric}_Kruskal.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

        df_stats['Metric'] = metric

        return df_stats
