import argparse
import os

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from tabulate import tabulate
from pathlib import Path
from analysis_EMSE import Analisys

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def run_complete_analysis(ana, dataset, RUN_WITH_DETERMINISTIC):
    ana.plot_accumulative("ACC_NAPFD")
    ana.plot_accumulative("ACC_APFDc", 'cost')  # APFDc

    # Variation Visualization along the CI Cycles
    ana.plot_lines("NAPFD_Variation")
    ana.plot_lines("APFDc_Variation", 'cost')  # APFDc

    # Normalized time reduction
    df_stats = ana.visualize_ntr()

    # Apply the Kruskal-Wallis Test in the Data
    df_stats = df_stats.append(ana.statistical_test_kruskal())  # NAPFD
    df_stats = df_stats.append(ana.statistical_test_kruskal('cost'))  # APFDc
    # df_stats = df_stats.append(ana.statistical_test_kruskal('ttf'))  # RFTC
    df_stats = df_stats.append(
        ana.statistical_test_kruskal('prioritization_time'))  #
    # Prioritization Time

    # Update the current dataset used
    df_stats['Dataset'] = dataset

    if RUN_WITH_DETERMINISTIC:
        # RMSE
        df_rmse = ana.rmse_calculation()
        df_rmse = df_rmse.append(ana.rmse_calculation('cost'))

        df_rmse['Dataset'] = dataset

        return df_stats, df_rmse

    return df_stats, None


def export_df(df, filename, caption):
    # print(f"Exporting {filename}")
    with open(f'{filename}.txt', 'w') as tf:
        tf.write(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

    latex = df.to_latex(index=False)

    # Remove special characters provided by pandas
    latex = latex.replace("\\textbackslash ", "\\").replace(
        "\$", "$").replace("\{", "{").replace("\}", "}")

    # split lines into a list
    latex_list = latex.splitlines()

    # Insert new LaTeX commands
    latex_list.insert(0, '\\begin{table*}[!ht]')
    latex_list.insert(1, f'\\caption{{{caption}}}')
    latex_list.insert(2, '\\resizebox{\\linewidth}{!}{')
    latex_list.append('}')
    latex_list.append('\\end{table*}')

    # join split lines to get the modified latex output string
    latex_new = '\n'.join(latex_list)

    # Save in a file
    with open(f'{filename}.tex', 'w') as tf:
        tf.write(latex_new)


def print_dataset(dataset):
    print(f"====================================================\n\t\t{dataset}\n"
          f"====================================================")


def get_best_equiv(df, column):
    equiv = len(df[df[column].str.contains("cellgray")])
    best = len(df[df[column].str.contains("cellbold")])

    return best, equiv


def get_magnitude(df, column):
    very_near = len(df[df[column].str.contains("bigstar")])
    near = len(df[df[column].str.contains("blacktriangledown")])
    far = len(df[df[column].str.contains("vartriangle")])

    df_temp = df[~df[column].str.contains("blacktriangledown")]

    reasonable = len(df_temp[df_temp[column].str.contains("triangledown")])
    very_far = len(df_temp[df_temp[column].str.contains("blacktriangle$")])

    return very_near, near, reasonable, far, very_far


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Analysis for EMSE')

    ap.add_argument('--results_dir', default="merge_results")
    ap.add_argument('--system_name', required=True)
    ap.add_argument('--project_dir', required=True)
    ap.add_argument('--datasets', nargs='+', default=[],
                    help='Datasets to analyse. Ex: \'deeplearning4j@deeplearning4j\'')

    ap.add_argument('--considers_variants', default=False,
                    type=lambda x: (str(x).lower() == 'true'))

    args = ap.parse_args()

    table_results_dir = f"{args.results_dir}_plots{os.sep}tables_{args.system_name}"

    variant_folder_result = f"WTS" if args.considers_variants else "VTS"
    variant_folder_result = f"{table_results_dir}{os.sep}{variant_folder_result}"
    Path(variant_folder_result).mkdir(parents=True, exist_ok=True)

    variant_folder = f"{args.system_name}@total_variants" if args.considers_variants else None

    replace_names = {
        'ε-greedy (ε=0.3)': 'ε-Greedy',
        'ε-greedy (ε=0.5)': 'ε-Greedy',
        'UCB (C=0.5)': 'UCB',
        'UCB (C=0.3)': 'UCB',
        'FRRMAB (C=0.3, D=1, SW=100)': 'FRRMAB',
        'mlpclassifier': 'ANN'
    }
    columns_metrics = ['FRRMAB', 'ANN']
    #columns_metrics = ['Random', 'FRRMAB', 'ANN']
    #columns_metrics = ['Random', 'Deterministic', 'FRRMAB', 'ANN']

    print_dataset(args.datasets[0])

    ana = Analisys(f"{args.project_dir}/{args.datasets[0]}",
                   args.results_dir,
                   variant_folder=variant_folder,
                   replace_names=replace_names,
                   columns_metrics=columns_metrics)

    policies = list(ana._get_policies())

    RUN_WITH_DETERMINISTIC = 'Deterministic' in policies

    df_stats_main = pd.DataFrame(columns=['Dataset', 'Metric', 'TimeBudget'] + policies)

    if RUN_WITH_DETERMINISTIC:
        policies.remove('Deterministic')

    df_distances_main = pd.DataFrame(columns=['Dataset', 'Metric', 'TimeBudget'] + policies)

    df_stats, df_distances = run_complete_analysis(ana, args.datasets[0], RUN_WITH_DETERMINISTIC)
    df_stats_main = df_stats_main.append(df_stats)

    if RUN_WITH_DETERMINISTIC:
        df_distances_main = df_distances_main.append(df_distances)

    for dataset in args.datasets[1:]:
        print_dataset(dataset)

        ana.update_project(f"{args.project_dir}/{dataset}",
                           variant_folder=variant_folder,
                           replace_names=replace_names)
        
        df_stats, df_distances = run_complete_analysis(ana, dataset, RUN_WITH_DETERMINISTIC)
        df_stats_main = df_stats_main.append(df_stats)

        if RUN_WITH_DETERMINISTIC:
            df_distances_main = df_distances_main.append(df_distances)

    print("\n\n\n\n\n\n===========================================================")
    print(f"\t\tSummary Results -", "WTS" if args.considers_variants else "VTS")
    print("===========================================================")
    if RUN_WITH_DETERMINISTIC:
        df_distances_main.sort_values(by=['TimeBudget', 'Dataset'], inplace=True)
        metrics_dist = df_distances_main['Metric'].unique()
        df_distances_main['TimeBudget'] = pd.to_numeric(df_distances_main['TimeBudget'])

        print("\n\n\n== RMSE")
        policies = columns_metrics[:] 
                        
        policies.remove('Deterministic')

        for tr in [10, 50, 80]:
            print(f"\n\nTime Budget {tr}%")
            for m in metrics_dist:
                print("\n\n~~", m)
                df_temp = df_distances_main[(df_distances_main.TimeBudget == tr) & (df_distances_main.Metric == m)]

                df_temp.sort_values(by=['Dataset'], inplace=True)

                del df_temp['TimeBudget']
                del df_temp['Metric']
                
                filename = f"{variant_folder_result}{os.sep}{m}_{tr}"
                caption = f"{m} values - Time Budget {tr}\\%"

                export_df(df_temp, filename, caption)

                for pol in policies:
                    best, equiv = get_best_equiv(df_temp, pol)
                    print(f"{pol}: {best} ({equiv}) ")

                if 'RMSE' in m:
                    print(f"\n{m} Magnitudes:")

                    for pol in policies:
                        print("\n", pol)
                        very_near, near, reasonable, far, very_far = get_magnitude(
                            df_temp, pol)

                        print(f"Very Near: {very_near}({round(very_near * 100 / len(args.datasets))})")
                        print(f"Near: {near}({round(near * 100 / len(args.datasets))})")
                        print(f"Reasonable: {reasonable}({round(reasonable * 100 / len(args.datasets))})")
                        print(f"Far: {far}({round(far * 100 / len(args.datasets))})")
                        print(f"Very Far: {very_far}({round(very_far * 100 / len(args.datasets))})")

    df_stats_main.sort_values(by=['TimeBudget', 'Dataset'], inplace=True)
    metrics = df_stats_main['Metric'].unique()
    df_stats_main['TimeBudget'] = pd.to_numeric(df_stats_main['TimeBudget'])

    print("\n\n\n== Each Metric")
    policies = columns_metrics 
    if 'Dataset' in policies:
        policies.remove('Dataset')
        
    for tr in [10, 50, 80]:
        print(f"\n\nTime Budget {tr}%")
        for m in metrics:
            print("\n~~", m)
            df_temp = df_stats_main[(df_stats_main.TimeBudget == tr) & (df_stats_main.Metric == m)]

            del df_temp['TimeBudget']
            del df_temp['Metric']

            df_temp.sort_values(by=['Dataset'], inplace=True)

           

            filename = f"{variant_folder_result}{os.sep}NTR_{tr}" if "NTR" in m \
                else f"{variant_folder_result}{os.sep}stats_{tr}_{m}"

            caption = f"NTR values - Time budget {tr}\\%" if "NTR" in m else f"{m} values - Time Budget {tr}\\%"
            export_df(df_temp, filename, caption)
                
            print(' & '.join(policies))
            for i, col in enumerate(policies):
                best, equiv = get_best_equiv(df_temp, col)

                if i == len(policies) - 1:
                    print(f"{best} ({equiv})")
                else:
                    print(f"{best} ({equiv}) & ", end="")