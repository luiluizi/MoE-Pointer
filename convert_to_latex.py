import pandas as pd
import argparse
import numpy as np
from run_all import datasets, eval_algorithms

def format_float(x: float):
    if abs(x) >= 1e4:
        return f'{x:.5g}'
    else:
        return f'{x:.1f}'
    
def convert_to_latex(df: pd.DataFrame):
    latex = ""
    for algorithm_name, row in df.iterrows():
        line = [algorithm_name]
        for scenario in df.columns.get_level_values(0)[::3][:4]:
            obj = format_float(df.loc[algorithm_name, (scenario, "Obj")])
            comp = f'{df.loc[algorithm_name, (scenario, "Comp")]:.2f}'
            time = f'{df.loc[algorithm_name, (scenario, "Time")]:.3f}'
            
            if not args.only_time:
                if algorithm_name == "mapt":
                    obj = "\\textbf{" + obj + "}"
                    comp = "\\textbf{" + comp + "}"
                line.extend([obj, comp])
            if not args.no_time:
                if time is not None:
                    line.append(time)
        latex += " & ".join(line) + " \\\\\n"
    latex = latex.replace("nan", "N/A").replace("_", "\_")
    return latex

def convert_to_latex2(df: pd.DataFrame):
    latex = ""
    for scenario1 in df.index:
        line = [scenario1]
        for scenario2 in df.index:
            obj = format_float(df.loc[scenario1, (scenario2, "Obj")])
            comp = f'{df.loc[scenario1, (scenario2, "Comp")]:.2f}'
            if scenario1 == scenario2:
                obj = "\\textbf{" + obj + "}"
                comp = "\\textbf{" + comp + "}"
            line.extend([obj, comp])
        latex += " & ".join(line) + " \\\\\n"
    latex = latex.replace("nan", "N/A").replace("_", "\_")
    return latex

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="eval_results.txt")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--no_time", action="store_true")
    parser.add_argument("--only_time", action="store_true")
    args = parser.parse_args()
    df = pd.read_csv(args.file, header=[0,1], sep="\t", index_col=0)
    if not args.transfer:
        latex = convert_to_latex(df)
    else:
        latex = convert_to_latex2(df)
    print(latex)