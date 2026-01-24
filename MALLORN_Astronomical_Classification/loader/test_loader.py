def build_test():
    import numpy as np
    import pandas as pd

    test_df = pd.read_csv("data/test_log.csv")

    for col in ["English Translation", "SpecType", "Z_err"]:
        if col in test_df.columns:
            test_df.drop(columns=[col], inplace=True)

    light_curve_cache = {}
    idx_cache = {}

    for s in test_df["split"].unique():
        path = f"data/{s}/test_full_lightcurves.csv"
        lc = pd.read_csv(path)
        groups = lc.groupby("object_id").indices

        light_curve_cache[s] = lc
        idx_cache[s] = groups

    def get_lightcurve(split, object_id):
        df_ = light_curve_cache[split]
        idx = idx_cache[split].get(object_id)
        if idx is None:
            return None
        return df_.iloc[idx]

    filters = ["u", "g", "r", "i", "z", "y"]

    base_cols = [
        "total_time", "n_obs", "median_flux", "mean_flux", "std_flux", "min_flux",
        "max_flux", "range_flux", "median_err", "median_snr", "max_snr", "neg_flux_frac"
    ]

    for c in base_cols:
        test_df[c] = np.nan

    for band in filters:
        test_df[f"n_obs_{band}"] = 0
        test_df[f"total_time_{band}"] = 0.0
        test_df[f"median_flux_{band}"] = 0.0
        test_df[f"std_flux_{band}"] = 0.0
        test_df[f"amp_{band}"] = 0.0
        test_df[f"median_err_{band}"] = 0.0
        test_df[f"median_snr_{band}"] = 0.0
        test_df[f"max_snr_{band}"] = 0.0
        test_df[f"neg_flux_frac_{band}"] = 0.0

    test_df["n_filters_present"] = 0
    test_df["total_obs"] = 0

    for i in range(test_df.shape[0]):
        x = test_df.iloc[i]
        lc = get_lightcurve(x["split"], x["object_id"])

        if lc is None or lc.shape[0] == 0:
            continue

        t = lc["Time (MJD)"].to_numpy()
        f = lc["Flux"].to_numpy()
        e = lc["Flux_err"].to_numpy()

        t_rel = t - t.min()

        test_df.loc[i, "total_time"] = float(t_rel.max() - t_rel.min())
        test_df.loc[i, "n_obs"] = int(lc.shape[0])

        test_df.loc[i, "median_flux"] = float(np.median(f))
        test_df.loc[i, "mean_flux"]   = float(np.mean(f))
        test_df.loc[i, "std_flux"]    = float(np.std(f))
        test_df.loc[i, "min_flux"]    = float(np.min(f))
        test_df.loc[i, "max_flux"]    = float(np.max(f))
        test_df.loc[i, "range_flux"]  = float(np.max(f) - np.min(f))

        test_df.loc[i, "median_err"] = float(np.median(e))
        snr = np.abs(f) / (e + 1e-8)
        test_df.loc[i, "median_snr"] = float(np.median(snr))
        test_df.loc[i, "max_snr"]    = float(np.max(snr))

        test_df.loc[i, "neg_flux_frac"] = float(np.mean(f < 0))

        present = 0
        total_obs = 0

        for band in filters:
            sub = lc[lc["Filter"] == band]
            n = int(sub.shape[0])
            test_df.loc[i, f"n_obs_{band}"] = n
            total_obs += n

            if n == 0:
                continue

            present += 1

            tb = sub["Time (MJD)"].to_numpy()
            fb = sub["Flux"].to_numpy()
            eb = sub["Flux_err"].to_numpy()

            tb_rel = tb - tb.min()

            test_df.loc[i, f"total_time_{band}"] = float(tb_rel.max() - tb_rel.min())
            test_df.loc[i, f"median_flux_{band}"] = float(np.median(fb))
            test_df.loc[i, f"std_flux_{band}"] = float(np.std(fb))
            test_df.loc[i, f"amp_{band}"] = float(np.max(fb) - np.median(fb))

            test_df.loc[i, f"median_err_{band}"] = float(np.median(eb))
            snr_b = np.abs(fb) / (eb + 1e-8)
            test_df.loc[i, f"median_snr_{band}"] = float(np.median(snr_b))
            test_df.loc[i, f"max_snr_{band}"] = float(np.max(snr_b))

            test_df.loc[i, f"neg_flux_frac_{band}"] = float(np.mean(fb < 0))

        test_df.loc[i, "n_filters_present"] = int(present)
        test_df.loc[i, "total_obs"] = int(total_obs)

    drop_cols = ["object_id", "split", "target", "SpecType", "English Translation"]
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    return X_test, test_df