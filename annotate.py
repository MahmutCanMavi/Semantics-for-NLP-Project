import pandas as pd
import numpy as np
d = pd.read_csv("tweet_samp_060522_annotate.csv")
d["annotate_person"].value_counts()
philipps_rows = d["annotate_person"].isin(["P","P_T_E"])

for idx in np.where(philipps_rows)[0]:
    print("-- Tweet", idx, "--")
    print("")
    
    d.loc[idx, "tweet"]
    pass_check = False
    while (not pass_check):
        print("")
        sent = input("Enter your sentiment regarding emotion: ")
        if (sent == "end"): break
        pass_check = any(val == sent for val in ["-1","0","1"])
        if (not pass_check): print("WRONG INPUT, enter again:")
    
    if (sent == "end"): break
    print("")
    print("-------------")
    d.loc[idx,"annotate_sent"] = sent
    print(d.loc[idx,:])
