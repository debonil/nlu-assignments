# %% [markdown]
# ## Question 2 & 3
# #### Data filter, process, and load into Neo4j Knowledge Graph

# %%
from neo4j import GraphDatabase
import pandas as pd
import json

data = []
with open('data/train.json', 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))


# %%
def save_json(data_to_save, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)


# %%


def extract_relation_types(data_src):
    merged = []
    for d in data_src:
        for p in d['passages']:
            merged.extend(p['exhaustivelyAnnotatedProperties'])

    factes_df = pd.DataFrame(merged)
    factes_df = factes_df.drop_duplicates().sort_values('propertyId')
    factes_df.to_csv('data/relations.csv', index=False)
    return factes_df


extract_relation_types(data)


# %% [markdown]
# sample document

# %%
data[2]


# %%

relations_df = pd.read_csv('data/relations.csv')
relations_id_map = {k[0]: k for k in relations_df.values}
relations_name_map = {k[1]: k for k in relations_df.values}


# %% [markdown]
# relations given in this assignment

# %%
relations_scope = [relations_name_map[r.split(" ")[0]][0] for r in """DATE_OF_BIRTH (PER–DATE)
PLACE_OF_RESIDENCE (PER–LOC) 
PLACE_OF_BIRTH (PER–LOC)
NATIONALITY (PER–LOC)
EMPLOYEE_OR_MEMBER_OF (PER–ORG) 
EDUCATED_AT (PER–ORG) """.split('\n')]
# relations_scope.sort()
relations_scope


# %% [markdown]
# FIltering data

# %%
data_with_facts = []
all_facts = []
print(f'Original document count = {len(data)}')
for d in data:
    passages_with_facts = []
    for p in d['passages']:
        facts = []
        for f in p['facts']:
            if int(f['propertyId']) in relations_scope and str(f['subjectText']).lower() not in ['i', 'my', 'he', 'she', 'his', 'her', 'who', 'myself'] and str(f['objectText']).lower() not in ['where', 'them', 'they', 'when', 'its']:
                facts.append(f)
                all_facts.append(f)
        p['facts'] = facts
        # print(len(p['facts']))
        if len(p['facts']) > 0:
            passages_with_facts.append(p)
    d['passages'] = passages_with_facts
    if len(d['passages']) > 0:
        data_with_facts.append(d)
print(f'Document with given relations count = {len(data_with_facts)}')
print(f'Total fact count = {len(all_facts)}')
save_json(data_with_facts, 'data/filtered_train.json')
save_json(all_facts, 'data/filtered_all_facts.json')


# %% [markdown]
# trying to see data

# %%
for d in data_with_facts[1000:1010]:
    print(f'{d["documentId"]}:')
    # print(f'{d["documentText"]}:')
    for p in d['passages']:
        print(f'\t{p["passageId"]}:')
        for f in p['facts']:
            print(f'\t\t{f["humanReadable"]}:')


# %% [markdown]
# ### Generate query and load data into Neo4J Knowledge Graph

# %%

# Connect to Neo4j
uri = "bolt://localhost:7687"
username = "etl_user"
password = "etl_user123"
driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session(database="knowledgenet")

# Iterate through the folds
for fact in all_facts:
    try:
        subjectText = str(fact['subjectText']).replace("'", "\\'")
        objectText = str(fact['objectText']).replace("'", "\\'")
        relation = relations_id_map[int(fact['propertyId'])][1]

        query = f"MERGE (p:Person {{name: '{subjectText}'}}) \
                                ON CREATE SET p.name = '{subjectText}'"

        print(query)
        session.run(query)

        if fact['propertyId'] in ['10', '11', '12']:
            # Loc
            query = f"MERGE (p:Location {{name: '{objectText}'}}) \
                                ON CREATE SET p.name = '{objectText}'"
            session.run(query)
            # Create the  relationship
            query = f"MATCH (d:Person {{name: '{subjectText}'}}), (p:Location {{name: '{objectText}'}}) MERGE (d)-[:{relation}]->(p)"

            print(query)
            session.run(query)

        if fact['propertyId'] in ['3', '9']:
            # Org
            query = f"MERGE (p:Organization {{name: '{objectText}'}}) \
                                ON CREATE SET p.name = '{objectText}'"
            print(query)
            session.run(query)
            # Create the  relationship
            query = f"MATCH (d:Person {{name: '{subjectText}'}}), (p:Organization {{name: '{objectText}'}}) MERGE (d)-[:{relation}]->(p)"
            print(query)
            session.run(query)

        if fact['propertyId'] in ['15']:
            # Org
            query = f"MERGE (p:Date {{name: '{objectText}'}}) \
                                ON CREATE SET p.name = '{objectText}'"
            print(query)
            session.run(query)
            # Create the  relationship
            query = f"MATCH (d:Person {{name: '{subjectText}'}}), (p:Date {{name: '{objectText}'}}) MERGE (d)-[:{relation}]->(p)"
            print(query)
            session.run(query)
    except:
        print('failed to insert')
