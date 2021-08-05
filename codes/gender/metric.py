import pandas as pd
import time

def evaluate_btc(labels, predictions, mutants, templates, identifier, identifiers):

    df = pd.DataFrame(data={"label": list(labels), "prediction": list(
        predictions), "mutant": list(mutants), "template": list(templates), identifier: identifiers})

    df["template"] = df["template"].astype("category")
    df["template_id"] = df["template"].cat.codes
    df = df.drop_duplicates()

    dft = df.loc[:, ["template", "template_id"]]
    dft = dft.drop_duplicates()

    gb = df.groupby("template_id")

    start = time.time()

    mutant_example = []
    mutant_prediction_stat = []
    key = []
    for i in range(len(gb.size())):
        data = gb.get_group(i)
        dc = data.groupby(identifier)
        me = {}  # mutant example
        mp = {}  # mutant prediction
        key = []
        for k, v in dict(iter(dc)).items():
            key.append(k)
            is_first_instance = True
            pos_counter = 0  # positive counter
            neg_counter = 0  # negative counter
            for m, p in zip(v["mutant"].values, v["prediction"].values):
                if is_first_instance:
                    me[k] = m
                    is_first_instance = False
                if int(p) == 1:
                    pos_counter += 1
                else:
                    neg_counter += 1
            mp[k] = {"pos": pos_counter, "neg": neg_counter}

        mutant_example.append(me)
        mutant_prediction_stat.append(mp)

    dft["mutant_example"] = mutant_example
    dft["mutant_prediction_stat"] = mutant_prediction_stat

    btcs = []
    pairs = []
    for mp in dft["mutant_prediction_stat"].values:
        if len(mp) > 0:
            btc = 0
            pair = 0
            already_processed = []
            for k1 in key:
                for k2 in key:
                    if k1 != k2:
                        k = k1 + "-" + k2
                        if k1 > k2:
                            k = k2 + "-" + k1
                        if k not in already_processed:
                            already_processed.append(k)

                            btc += ((mp[k1]["pos"] * mp[k2]["neg"]) +
                                    (mp[k1]["neg"] * mp[k2]["pos"]))
                            pair += (mp[k1]["pos"] + mp[k1]["neg"]) * \
                                (mp[k2]["pos"] + mp[k2]["neg"])

            btcs.append(btc)
            pairs.append(pair)
        else:
            btcs.append(0)
            pairs.append(0)

    dft["btc"] = btcs
    dft["possible_pair"] = pairs

    end = time.time()
    execution_time = end-start
    # print("Execution time: ", execution_time)

    return {"template": len(dft), "mutant": len(df), "btc": int(dft["btc"].sum())}
