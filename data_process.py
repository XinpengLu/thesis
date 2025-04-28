import pandas as pd


def process_hopper_excel(input_excel_path, output_excel_path=None):
    df = pd.read_excel(input_excel_path)

    columns_to_keep = [
        col for col in df.columns
        if not (col.endswith('__MIN') or col.endswith('__MAX'))
    ]
    df = df[columns_to_keep]

    new_columns = []
    for col in df.columns:
        if col == "Step":
            new_columns.append(col)
            continue

        parts = col.split('-')
        # print(parts, len(parts))

        if len(parts) == 10 and parts[0] != 'av':
            new_col = '-'.join(parts[1:7])
            new_columns.append(new_col)
        elif len(parts) == 10 and parts[0] == 'av':
            new_col = '-'.join(parts[0:7])
            new_columns.append(new_col)
        elif len(parts) == 9:
            new_col = '-'.join(parts[1:6])
            new_columns.append(new_col)
        elif len(parts) == 11:
            new_col = '-'.join(parts[0:8])
            new_columns.append(new_col)
        else:
            new_columns.append(col)

    df.columns = new_columns

    if output_excel_path:
        df.to_excel(output_excel_path, index=False)

    return df


if __name__ == "__main__":
    algs = ['dbc', 'pbc', 'av_pbc']
    envs = ['hopper']
    for alg in algs:
        for env in envs:
            input_file = f"{alg}_{env}.xlsx"
            output_file = f"processed_{alg}_{env}.xlsx"
            processed_data = process_hopper_excel(input_file, output_file)
            print(f"{alg}_{env} 数据处理完成")
