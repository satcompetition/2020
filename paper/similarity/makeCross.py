#!/usr/bin/env python3

from tabulate import tabulate
import pandas as pd


def pl(args):
    for e in args:
        print("- " + e)


pResults = "../../downloads/results.csv"
pNames = "../solver_names_final.txt"
numSolvers = 30

d = pd.read_csv(pResults)
d["n"] = d["solver"] + "__" + d["configuration"]
print(str(len(d.n.unique())) + " original solvers")

# withdrawn
withdrawn = d[d["solver"] == "Relaxed_LCMDCBDL_noTimePara"].n.unique()
d = d[~d["n"].isin(withdrawn)]
print("Removed withdrawn solver:")
pl(withdrawn)

# disqualified
dis = d[d["verifier-result"].isin(["SAT-INCORRECT", "UNSAT-INCORRECT"])]["n"].unique()
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

# only main track (-> no planning)
print(
    "removed "
    + str(len(d[d["benchmark"].apply(lambda x: "ddl_" in x)].benchmark.unique()))
    + " planning bechmarks"
)
d = d[~d["benchmark"].apply(lambda x: "ddl_" in x)]
print(
    str(len(d[d.result.isin(["SAT", "UNSAT"])].benchmark.unique()))
    + " solved benchmarks are left"
)

# remove fully unsolved benchmarks
d["s"] = d["result"].isin(["SAT", "UNSAT"])
count = d.groupby("benchmark").sum()["s"].reset_index()
unsolved = count[count["s"] == 0]["benchmark"].unique()
d = d[~d["benchmark"].isin(unsolved)]

# PAR2
d.loc[d["s"] == 0, "time"] = 10000
d = d[["n", "benchmark", "time"]]

# shorter names
nameDict = pd.read_csv(pNames, sep=" ", names=["n", "name"])
nameDict["name"] = nameDict["name"].str.replace("\\", "")
d = d.merge(nameDict, on="n")[["name", "benchmark", "time"]]
assert len(d.name.unique()) == solverInMain

top = d.groupby("name").sum().reset_index().sort_values(by="time")


def sim(a, b):
    r = 0
    n = len(a)
    for x, y in zip(a, b):
        r += abs(x - y)
    return 1 - (r / (10000 * n))


s = {}
for n, b, t in d.values.tolist():
    if n not in s:
        s[n] = []
    s[n].append((b, t))

top = top[:numSolvers]
print(tabulate(top, headers=top.columns, tablefmt="orgtbl", showindex=False))
l = []
names = []
for n in top["name"]:
    l.append([t for _, t in sorted(s[n])])
    names.append(n)
m = [[0 for _ in range(len(l))] for _ in range(len(l))]
for i in range(len(l)):
    for j in range(len(l)):
        m[i][j] = sim(l[i], l[j])
d = pd.DataFrame(m)
d.columns = names
d.insert(0, "with", "")
d["with"] = names
d = d.set_index("with")
d.to_csv("cross.csv")
