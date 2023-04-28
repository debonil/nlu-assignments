from neo4j import GraphDatabase


def get_subject(subject_name="Heraclio", properrty_name="NATIONALITY"):
    response = 'unknown'
    # Connection details
    uri = "bolt://localhost:7687"
    username = "debonil"
    password = "test1234"
    database = "knowledgenet"   # database name

    # Connect to the Neo4j database
    driver = GraphDatabase.driver(uri, auth=(
        username, password), encrypted=False)
    with driver.session(database=database) as session:

        query1 = f"MATCH (p:Passage)-[:HAS_FACT]-(f:Fact),(f:Fact)-[:HAS_SUBJECT]-(s:Subject),(f:Fact)-[:HAS_OBJECT]-(o:Object), (p:Passage)-[:HAS_PROPERTY]-(pr:Property) \
        WHERE pr.propertyName='{properrty_name}' and s.subjectText='{subject_name}' and f.humanReadable contains '{properrty_name}' \
        RETURN o.objectText"

        query2 = f"MATCH(d:Document)-[:HAS_PASSAGE]-(p:Passage),(p:Passage)-[:HAS_FACT]-(f:Fact),(f:Fact)-[:HAS_SUBJECT]-(s:Subject) \
        WHERE s.subjectText='{subject_name}' \
        RETURN d.documentId limit 1"
        doc_id = ""

        info = session.run(query1)
        list_values = info.values()
        # print(type(list_values))
        # print(len(list_values))
        if len(list_values) > 0:
            for item in list_values:
                print(item[0])
        if len(list_values) == 0:
            #print("I am in if")
            docs = session.run(query2)
            for items in docs:
                doc_id = items.values()[0]
                # print(doc_id)
                query3 = f"MATCH(d:Document)-[:HAS_PASSAGE]-(p:Passage),(p:Passage)-[:HAS_PROPERTY]-(pr:Property),(p:Passage)-[:HAS_FACT]-(f:Fact),(f:Fact)-[:HAS_SUBJECT]-(s:Subject),(f:Fact)-[:HAS_OBJECT]-(o:Object)\
                    WHERE (s.subjectText='He' or s.subjectText='She' or s.subjectText='Who') and d.documentId= '{doc_id}' \
                    and pr.propertyName = '{properrty_name}' and f.humanReadable contains '{properrty_name}' \
                    RETURN o.objectText"
                mila = session.run(query3)
                if len(mila) > 0:
                    response = mila[0].values()[0]
    return response
