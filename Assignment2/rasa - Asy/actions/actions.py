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

f = open ('data/readyData.json', "r")
finalData = json.loads(f.read())
org_map = {}
parent_org_map = {}
org_list = []
parent_org_list = []

for fd in finalData.values():
    for org in fd['org']:
        org_map[org.lower()] = fd['documentId']
    if len(fd['parrent_org']) > 0:
        for p_org in fd['parrent_org']:
            parent_org_map[p_org.lower()] = fd['documentId']

org_list = org_map.keys()
parent_org_list = parent_org_map.keys()

def processData(org, propertyId):
    org = org.lower()
    msg = ''
    if org not in org_map:
        print('Full name not matched for org: ', org)
        t = get_close_matches(org, org_list)
        if len(t) > 0:
            print('showing information about :', t[0], ' in place of :', org)
            org = t[0]
            msg = f'Did you mean: {org}?\n\n'
    
    if org in org_map:
        data = finalData[org_map[org]]
        if propertyId in data['passages']:
            return msg + data['passages'][propertyId]
        return 'Sorry!!! I don\'t have information about your query'
    return 'Sorry!!! I don\'t have information about your query organization'

def processDataParent(org, propertyId):
    org = org.lower()
    if org not in parent_org_map:
        print('Full name not matched for parent org: ', org)
        t = get_close_matches(org, parent_org_list)
        if len(t) > 0:
            print('showing information about :', t[0], ' in place of parent org :', org)
            org = t[0]
    
    if org in parent_org_map:
        data = finalData[parent_org_map[org]]
        if propertyId in data['passages']:
            return data['passages'][propertyId]
        return 'Sorry!!! I don\'t have information about your query'
    return 'Sorry!!! I don\'t have information about your query organization'

class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []

class ActionCEO(Action):

    def name(self) -> Text:
        return "action_ceo"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        intent = tracker.latest_message['intent'].get('name')
        org = 'xyz'
        for e in tracker.latest_message['entities']:
            org = e['value']

        msg = processData(org, '4')
        print(intent, ',ORG is :', org)
        dispatcher.utter_message(text=msg)

        return []

class ActionFOUNDER(Action):

    def name(self) -> Text:
        return "action_founder"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        intent = tracker.latest_message['intent'].get('name')
        org = 'xyz'
        for e in tracker.latest_message['entities']:
            org = e['value']

        msg = processData(org, '2')
        print(intent, ',ORG is :', org)
        dispatcher.utter_message(text=msg)
        
        return []

class ActionLocation(Action):

    def name(self) -> Text:
        return "action_location"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        intent = tracker.latest_message['intent'].get('name')
        org = 'xyz'
        for e in tracker.latest_message['entities']:
            org = e['value']

        msg = processData(org, '6')
        print(intent, ',ORG is :', org)
        dispatcher.utter_message(text=msg)

        return []

class ActionSubsidiary(Action):

    def name(self) -> Text:
        return "action_subsidiary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        intent = tracker.latest_message['intent'].get('name')
        org = 'xyz'
        for e in tracker.latest_message['entities']:
            org = e['value']

        msg = processData(org, '1')
        print(intent, ',ORG is :', org)
        dispatcher.utter_message(text=msg)

        return []

class ActionParent(Action):

    def name(self) -> Text:
        return "action_parent"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        intent = tracker.latest_message['intent'].get('name')
        org = 'xyz'
        for e in tracker.latest_message['entities']:
            org = e['value']

        msg = processDataParent(org, '1')
        print(intent, ',ORG is :', org)
        dispatcher.utter_message(text=msg)

        return []

class ActionFoundedIn(Action):

    def name(self) -> Text:
        return "action_founder_in"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        intent = tracker.latest_message['intent'].get('name')
        org = 'xyz'
        for e in tracker.latest_message['entities']:
            org = e['value']

        msg = processData(org, '5')
        print(intent, ',ORG is :', org)
        dispatcher.utter_message(text=msg)

        return []
