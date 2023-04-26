# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import json
from difflib import get_close_matches

#f = open('data/readyData.json', "r")
#finalData = json.loads(f.read())
org_map = {}
parent_org_map = {}
org_list = []
parent_org_list = []
# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#


class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []


class ActionDescPerson(Action):

    def name(self) -> Text:
        return "action_desc_person"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print(tracker.latest_message['intent'])
        print(tracker.latest_message['entities'])
        intent = tracker.latest_message['intent'].get('name')
        org = 'xyz'
        for e in tracker.latest_message['entities']:
            org = e['value']

        print(intent, ',ORG is :', org)
        dispatcher.utter_message(
            template="desc_person/date_of_birth",
            person=org,
            nationality="Indian",
            city="Kolkata",
            dob="1970-01-01"
        )

        return []
