#initial import

import pandas as pd

marsis_raw = pd.read_csv("MARSISdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv")
marsis_deadly = marsis_raw[marsis_raw["TotalDeaths"] > 0]

import matplotlib.pyplot as plt

marsis_deadly["ImoClassLevelID"] = marsis_deadly["ImoClassLevelID"].replace([0], 1)
marsis_deadly["ImoClassLevelID"] = marsis_deadly["ImoClassLevelID"].fillna(1)

names = ["Very Serious Casuality", "Casualty", "Incident"]
values = [
    len(marsis_deadly[marsis_deadly["ImoClassLevelID"] == 1]),
    len(marsis_deadly[marsis_deadly["ImoClassLevelID"] == 2]),
    len(marsis_deadly[marsis_deadly["ImoClassLevelID"] == 3])
]

#visualize distribution of marine occurrence severity classifications
plt.bar(names, values)
plt.yscale("log", base=10)
plt.xlabel("IMO Marine Occurrence Severity")
plt.ylabel("Number of Occurrences")
plt.title("Marine Accident Occurrences Classified by Severity")
plt.grid(True)
plt.show()

#group dataset into relevant attributes based off of marine occurrence severity
marsis_deadly_table = marsis_deadly.filter(["AccIncTypeDisplayEng", "ImoClassLevelID", "TotalDeaths"], axis = "columns")
marsis_deadly_table = pd.DataFrame(marsis_deadly_table.groupby(["AccIncTypeDisplayEng", "ImoClassLevelID"])['TotalDeaths'].sum())
marsis_deadly_table.to_csv("df.csv")

sum(marsis_deadly_table["TotalDeaths"])

#plot correlation matrix
marsis_corr = marsis_deadly.filter(["OccID", "AccIncTypeDisplayEng", "TotalDeaths", "TotalMinorInjuries", "TotalSeriousInjuries", "TotalMissingIndividuals", "TotalPeopleInTheWater"])
marsis_corr = marsis_corr.drop_duplicates(subset = "OccID")
marsis_corr = marsis_corr.groupby(["OccID"])["TotalDeaths", "TotalMinorInjuries", "TotalSeriousInjuries", "TotalMissingIndividuals", "TotalPeopleInTheWater"].sum()
marsis_corr_matrix = marsis_corr.corr(method='pearson')
marsis_corr_matrix = pd.DataFrame(marsis_corr_matrix)
marsis_corr_matrix.style.format('{:.2f}').background_gradient(axis=0)
marsis_corr_matrix.to_csv("corr.csv")

summary = marsis_corr.describe(include = "all")
summary.to_csv("summary.csv")
