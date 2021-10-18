def make_yaml_dict(train_df, target_col):
    return {
        'class_names': train_df[target_col].unique().tolist(),
        'input': list(train_df.drop(target_col, axis=1).keys())
    }
