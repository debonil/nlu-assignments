# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

from database.database import get_subject


print('loading actions:')


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
