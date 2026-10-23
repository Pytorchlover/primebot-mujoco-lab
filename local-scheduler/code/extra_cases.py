"""Additional wording checks; the expected actions are never shown to the model."""
import json
from bootstrap import ROOT
from local_model import LocalChat
from run import run_case
client=LocalChat()
cases=[
 ({'id':'extra_capabilities','text':'你都会些什么？'},[]),
 ({'id':'extra_no_action','text':'我只是问你会不会跳舞，先别做动作。'},[]),
 ({'id':'extra_heart','text':'给大家比个心。'},['gesture.heart']),
 ({'id':'extra_order','text':'给大家挥手致意，结束后再比心。'},['gesture.wave','gesture.heart']),
]
results=[]
for case,expected in cases:
    r=run_case(client,case)
    observed=[e['action'] for e in r['backend_events'] if e['state']=='start']
    results.append({'trace':r,'expected':expected,'observed':observed,'passed':observed==expected and r['outcome']=='complete'})
(ROOT/'results/extra-cases.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
assert all(r['passed'] for r in results),[(r['trace']['id'],r['observed']) for r in results if not r['passed']]
print('4 additional wording checks passed')
