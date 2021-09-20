import csv
import time
import os
import pandas as pd

"""
Running this file will load in the knowledge graph - assertion list (downloadable at 
https://github.com/commonsense/conceptnet5/wiki/Downloads) and convert it to a cropped version that only contains the
English subset and filters out all negative relations (NotDesires, NotHasProperty, NotCapableOf, NotUsedFor, Antonym,
DistinctFrom and ObstructedBy) and self-loops. Additionally, each concept is converted to an integer to speed up
computation of the RWR, and stored in a look-up file. The resulting graph's format is: concept 1, concept 2,
concept 1 (int), concept 2 (int), relation, weight. It is stored in a new csv file, which will be read out to determine
the semantic consistency.
"""

path_project = os.path.abspath(os.path.join(__file__, "../.."))
# change these 3 lines for preparing a different knowledge graph:
file_path = os.path.join(path_project, "Data raw/ConceptNet/assertions55.csv")
lookup_filepath = os.path.join(path_project, "Semantic Consistency/KG_lookup_55.csv")
cropped_filepath = os.path.join(path_project, "Semantic Consistency/KG_crop_55.csv")

start = time.time()

counter = 0
eng_counter = 0
rel_counter = 0
con_counter = 0
ext_counter = 0
assertion_list = []

# Read through the cvs file containing all assertions and filter out only english concepts and non-negative relations.
with open(file_path, newline='', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        counter = counter + 1
        if counter % 1000000 == 0:
            print(counter)
        part1 = row[0].split(",/")[0]
        part2 = row[0].split(",/")[1]
        part3 = row[0].split(",/")[2].split("\t")[0]
        relation = part1.split("/")[4]
        language1 = part2.split("/")[1]
        concept1 = part2.split("/")[2]
        language2 = part3.split("/")[1]
        concept2 = part3.split("/")[2]
        weight = row[-1][0:-1]

        if language1 == "en" and language2 == "en":
            eng_counter = eng_counter + 1
            if (relation == 'Antonym' or relation == 'NotDesires' or relation == 'NotHasProperty' or
                    relation == 'NotCapableOf' or relation == 'NotUsedFor' or relation == 'DistinctFrom' or
                    relation == 'ObstructedBy'):
                rel_counter = rel_counter + 1
                continue
            elif concept1 == concept2:
                con_counter = con_counter + 1
                continue
            else:
                entry = (concept1, concept2, relation, weight)
                assertion_list.append(entry)
            continue

print("total relations left Eng - relations - selfloops: ", eng_counter - rel_counter - con_counter)

print("assertion_list length: ", len(assertion_list))
assertion_list = sorted(assertion_list, key=lambda x: x[0])

concepts = []
c1s = []
c2s = []
relations = []
weights = []

# Create lists of all (unique) concepts, relations and weights.
for unit in range(len(assertion_list)):
    c_1 = assertion_list[unit][0]
    c_2 = assertion_list[unit][1]
    rel = assertion_list[unit][2]
    w = assertion_list[unit][3]
    concepts.append(c_1)
    concepts.append(c_2)
    c1s.append(c_1)
    c2s.append(c_2)
    relations.append(rel)
    weights.append(w)

concepts_uni = sorted(list(set(concepts)))
c1s_uni = list(set(c1s))
c2s_uni = list(set(c2s))
relations_uni = list(set(relations))
weights_uni = list(set(weights))

print("number of unique concepts in the graph: ", len(concepts_uni))

integers = range(len(concepts_uni))

df_lookup = pd.DataFrame(list(zip(concepts_uni, integers)),
               columns =['Concept', 'Index'])

df_assertions = pd.DataFrame(list(zip(c1s, c2s, relations, weights)),
               columns =['c1s', 'c2s', 'relations', 'weights'])

with open(lookup_filepath, 'w', encoding="utf8") as file:
    writer = csv.writer(file, delimiter='\t', lineterminator='\n',)

    for unit in range(df_lookup.shape[0]):
        if unit % 100000 == 0:
            print('Checked: [{0}/{1}]'.format(unit, len(relations)))
        concept = df_lookup.loc[unit, 'Concept']
        number = df_lookup.loc[unit, 'Index']

        writer.writerow([concept, number])


df_complete = pd.merge(df_assertions, df_lookup, left_on='c1s', right_on='Concept')
df_complete2 = pd.merge(df_complete, df_lookup, left_on='c2s', right_on='Concept')
print("df_complete2 shape", df_complete2.shape)
print(df_complete2.loc[[10000]])
df_complete3 = df_complete2.drop(columns=['Concept_x', 'Concept_y'])
df_complete3 = df_complete3.rename(columns={'Index_x': 'c1idx', 'Index_y': 'c2idx'})


print("df_complete3 shape", df_complete3.shape)
print(df_complete3.loc[[10]])
print(df_complete3.loc[[10000]])
print(df_complete3.loc[10000, 'c1s'])


# Create a new csv file that contains the cropped list of assertions (only english and non-negative) in both string
# format as its unique integer companion format.  (Note: this takes around 70 hours to run)
with open(cropped_filepath, 'w', encoding="utf8") as file:
    writer = csv.writer(file, delimiter='\t', lineterminator='\n',)

    for unit in range(df_complete3.shape[0]):
        if unit % 100000 == 0:
            print('Checked: [{0}/{1}]'.format(unit, len(relations)))
        c1 = df_complete3.loc[unit, 'c1s']
        c2 = df_complete3.loc[unit, 'c2s']
        c1idx = df_complete3.loc[unit, 'c1idx']
        c2idx = df_complete3.loc[unit, 'c2idx']
        r = df_complete3.loc[unit, 'relations']
        w = df_complete3.loc[unit, 'weights']

        writer.writerow([c1, c2, r, w, c1idx, c2idx])

end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("time it took to create KG_Crop: ")
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))