#!/usr/bin/env python3

import pandas as pd


def pl(args):
    for e in args:
        print("- " + e)


def getVbs(outputAllSolvers=True, removePlanning=False):
    pResults = "../../downloads/results.csv"
    pNames = "../solver_names_final.txt"

    d = pd.read_csv(pResults)
    d["n"] = d["solver"] + "__" + d["configuration"]
    o = d
    print("\n" + str(len(o.n.unique())) + " original solvers")

    # withdrawn
    withdrawn = d[d["solver"] == "Relaxed_LCMDCBDL_noTimePara"].n.unique()
    d = d[~d["n"].isin(withdrawn)]
    print("Removed withdrawn solver:")
    pl(withdrawn)

    # disqualified
    dis = d[d["verifier-result"].isin(["SAT-INCORRECT", "UNSAT-INCORRECT"])][
        "n"
    ].unique()
    d = d[~d["n"].isin(dis)]
    print("Removed disqualified solver:")
    pl(dis)

    # competed in no limits
    np = d[d["drat"] == "NOPROOF"]["n"].unique()
    d = d[~d["n"].isin(np)]
    print("Removed NoLimits solver:")
    pl(np)

    # demoted to no limits
    dem = d[d["drat"] == "NOT_VERIFIED"]["n"].unique()
    d = d[~d["n"].isin(dem)]
    print("Removed demoted solver:")
    pl(dem)

    solverInMain = len(d["n"].unique())
    print(str(solverInMain) + " solvers participated in main track")

    d = d.set_index(["n", "benchmark"])
    s = d[d["result"].isin(["SAT", "UNSAT"])]
    print(str(len(s.reset_index().benchmark.unique())) + " benchmarks have been solved")
    vbsRows = s.groupby("benchmark").time.idxmin()

    if outputAllSolvers:
        d = o.set_index(["n", "benchmark"])

    d["VBS_Origin"] = False
    d.loc[vbsRows, "VBS_Origin"] = True
    d = d.reset_index()

    # adding short names
    nameDict = pd.read_csv(pNames, sep=" ", names=["n", "name"])
    nameDict["name"] = nameDict["name"].str.replace("\\", "")
    d = d.merge(nameDict, on="n", how="left")
    d["name"] = d["name"].fillna("UNDEF")

    v = d[d["VBS_Origin"]].copy()
    v["VBS_Origin"] = False
    v["n"] = v["name"] = v["solver"] = "VBS"
    v["configuration"] = "default"
    d = d.append(v)

    if removePlanning:
        print("removed " + str(len(d[d["benchmark"].apply(lambda x: "ddl_" in x)].benchmark.unique())) + " planning bechmarks")
        d = d[~d["benchmark"].apply(lambda x: "ddl_" in x)]
        print(str(len(d[d.result.isin(["SAT", "UNSAT"])].benchmark.unique())) + " solved benchmarks are left")
    d = d.drop("n", axis=1)
    return d


vbs = getVbs()
validSolversOnly = getVbs(outputAllSolvers=False)
withoutPlanning = getVbs(outputAllSolvers=False, removePlanning=True)
vbs.to_csv("results_vbs.csv", index=False)
validSolversOnly.to_csv("results_vbs_main.csv", index=False)
withoutPlanning.to_csv("results_vbs_main_noPlan.csv", index=False)
