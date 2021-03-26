import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Describe Accident':'During the activation of a sodium sulphide pump, the piping was uncoupled and the sulfide solution was designed in the area to reach the maid. Immediately she made use of the emergency shower and was directed to the ambulatory doctor and later to the hospital. Note: of sulphide solution = 48 grams / liter.'})

print(r.json())