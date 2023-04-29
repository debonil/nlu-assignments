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
    response = None
    # Connection details
    uri = "bolt://localhost:7687"
    username = "etl_user"
    password = "etl_user123"
    database = "knowledgenet"   # database name

    # Connect to the Neo4j database
    driver = GraphDatabase.driver(uri, auth=(
        username, password), encrypted=False)
    with driver.session(database=database) as session:
        query = f" MATCH (n:Person{{name:'{subject_name}'}})-[:{properrty_name}] ->(r) RETURN r.name "
        print(query)
        info = session.run(query)
        list_values = info.values()
        #print(type(list_values))
        print(list_values)
        if len(list_values) > 0:
            response = list_values[0][0]

    return response


class ActionDescPerson(Action):

    def name(self) -> Text:
        return "action_desc_person"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        intent = tracker.latest_message['intent'].get('name')
        person = tracker.get_slot('PERSON')
        if len(tracker.latest_message['entities']) > 0:
            person = tracker.latest_message['entities'][0]['value']

        query = intent[13:]

        print(f'intent={intent}, entity_value={person}, query={query}')
        result = get_subject(person, query)
        print(f'response ==> {result}')

        if person != None:
            if result != None:
                dispatcher.utter_message(
                    response=f"desc_person/{query}",
                    person=person,
                    nationality=result,
                    city=result,
                    dob=result,
                    organization=result
                )
            else:
                dispatcher.utter_message(
                    text="Sorry, it is not in my knowledge!")
        else:
            dispatcher.utter_message(
                text="Sorry could not get whome you are refering.. can you refrase your query with person name? ")
        return [SlotSet("PERSON", person)]
