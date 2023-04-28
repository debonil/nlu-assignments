# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
from neo4j import GraphDatabase
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet


print('loading actions:')


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
            return list_values[0][0]
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


class ActionDescPerson(Action):

    def name(self) -> Text:
        return "action_desc_person"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        intent = tracker.latest_message['intent'].get('name')
        person = tracker.get_slot('person')
        if len(tracker.latest_message['entities']) > 0:
            person = tracker.latest_message['entities'][0]['value']

        query = intent[13:]

        print(f'intent={intent}, entity_value={person}, query={query}')

        try:
            response = get_subject(person, query)
        except:
            print('failed to fetch from database!')
        print(f'response ==> {response}')

        if person != None:
            dispatcher.utter_message(
                response=f"desc_person/{query}",
                person=person,
                nationality=response,
                city=response,
                dob=response,
                organization=response
            )
        else:
            dispatcher.utter_message(
                text="Sorry could not get whome you are refering.. can you refrase your query with person name? ")
        return [SlotSet("person", person)]
