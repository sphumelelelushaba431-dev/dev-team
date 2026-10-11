"""Generate the bundled synthetic complaint dataset (data/sample_complaints.csv).

This is a stand-in with the same structure as the Kaggle "University Students
Complaints & Reports" dataset (omarsobhy14), so the system runs out of the
box. For the study, download the real CSV and save it as
data/university_complaints.csv; it is then used automatically.

The text is deliberately noisy (typos, slang, informal phrasing, multi-part
complaints) so that the models are tested on realistic language.
"""
import csv
import random
import sys
from pathlib import Path

random.seed(7)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = Path(__file__).resolve().parent.parent / "data" / "sample_complaints.csv"

SLOTS = {
    "module": ["Database Systems", "Accounting 2", "Software Engineering", "Business Statistics", "Networks 3",
               "Financial Management", "Programming 1", "Research Methods", "Economics 1B", "Information Systems 3"],
    "place": ["Block A", "the library", "S4 building", "the engineering building", "Steve Biko campus",
              "the main lecture hall", "ML Sultan campus", "the student centre", "lab 12"],
    "res": ["Residence 3", "Block C res", "the new residence", "Emahlathini res", "my res block", "Corlo residence"],
    "weeks": ["two weeks", "three weeks", "a month", "since February", "the whole semester"],
}

T = {
    "Academic": [
        "The lecturer for {module} never shows up for class and we have a test next week",
        "My marks for the {module} assignment are still not on the portal after a month",
        "I want my exam script for {module} remarked, I think it was marked unfairly",
        "There is a clash between two of my tests on the same day and nobody is helping",
        "We were never given the assessment criteria for the {module} project",
        "I missed my exam because I was sick, how do I get a special exam",
        "The tutor for {module} is rude and refuses to explain the work",
        "My results show incomplete even though I wrote everything",
        "How do I qualify for a supplementary exam, I got 47 percent",
        "The timetable puts {module} and another module at the same time",
        "Our class has had no lecturer for {weeks} and the work is piling up",
        "I was excluded academically and I want to appeal the decision",
        "The lecturer uploaded the wrong memo and now everyone failed the test",
        "Is it possible to get an extension, I was in hospital when the project was due",
        "The marking of the practical was inconsistent between groups",
        "We have not received feedback on our first test and the second one is tomorrow",
        "Classes for {module} are always cancelled at the last minute",
        "the notes for {module} dont match what was asked in the test at all",
    ],
    "Finance": [
        "My NSFAS allowance has not been paid for {weeks} and I can't buy food",
        "I paid my fees but the statement still shows I owe money",
        "I was charged twice for registration, I need a refund",
        "My results are blocked because of outstanding fees, I need a payment plan",
        "My bursary was approved but the money is not reflecting on my account",
        "How do I get a refund of my credit balance",
        "The accommodation allowance came late again this month",
        "I can't register because the system says I have debt from last year",
        "Why am I being charged for a module I cancelled",
        "My book allowance still has not arrived and classes started {weeks} ago",
        "I need a proof of payment letter for my sponsor urgently",
        "My funding agreement was signed but they still say it is outstanding",
        "the money for meals hasnt come through and im struggling",
        "I can't afford the minimum initial payment, what are my options",
        "Finance office keeps sending me from one queue to another about my statement",
        "the deposit I paid in January is not showing anywhere",
    ],
    "ICT": [
        "The wifi in {place} keeps disconnecting every few minutes",
        "I forgot my password and the reset portal does not send the code",
        "Moodle is down and my assignment is due tonight",
        "I can't log in to my student email since yesterday",
        "The computers in {place} are very slow and some don't switch on",
        "I am not able to see {module} on the learning platform",
        "The printer in the lab keeps jamming and eats my printing credit",
        "My account got locked after I typed the wrong password",
        "The online test crashed halfway and it submitted my answers",
        "connection in {res} is terrible we cant stream lectures",
        "I can't access the student portal from my phone",
        "The network is so slow in {place} that videos don't load",
        "my laptop wont connect to the campus network even with the right details",
        "The lab software we need for {module} is not installed",
        "I didn't receive the email with my login details",
        "the system kicked me out while I was uploading my project",
    ],
    "Facilities": [
        "The toilets in {place} are dirty and there is no toilet paper",
        "The lights in the lecture hall are flickering and some are broken",
        "The aircon in {place} has not worked for {weeks}",
        "The lift in {place} is broken and I use a wheelchair",
        "There are not enough chairs in the venue, people sit on the floor",
        "The projector in our venue has been broken for {weeks}",
        "Water is leaking from the ceiling in {place} when it rains",
        "There is exposed wiring near the stairs in {place}",
        "The shuttle is always late and I miss my first lecture",
        "There is no parking for students near {place}",
        "The windows in the lecture room are broken and it is freezing",
        "the bathroom taps in {place} have been running non stop",
        "Someone stole my bag in {place} and there was no security guard",
        "The ramp at {place} is blocked by construction material",
        "There is a strong smell of gas in the corridor of {place}",
        "the venue is too small for our class and it is very hot inside",
    ],
    "Hostel": [
        "The geyser in {res} has been broken for {weeks}, no hot water",
        "My roommate brings visitors every night and I can't study",
        "I applied for residence but I am still on the waiting list",
        "The door lock on my room in {res} is broken",
        "There has been no water in {res} since yesterday",
        "I want to change rooms because of conflict with my roommate",
        "The windows in my room don't close and it is cold at night",
        "The warden ignores our complaints about the noise in {res}",
        "Pipes in the bathroom of {res} are leaking and the floor is flooded",
        "There are cockroaches in the kitchen of {res}",
        "I lost my access card and now I can't get into my block",
        "The electricity in our floor of {res} keeps tripping",
        "Our private accommodation landlord is not fixing anything",
        "There is no security at the gate of {res} at night",
        "the beds in {res} are broken and the mattress is torn",
        "I was allocated a room very far from campus, can I be moved",
    ],
}

PREFIX = ["", "", "", "Hi, ", "Good day, ", "Please help, ", "Hello admin, ", "Urgent: ", "Sorry to bother, "]
SUFFIX = ["", "", "", " please help", " this is really frustrating", " nobody is responding",
          " I have reported it before", " what must I do?", " it is affecting my studies"]
CROSS = {  # second clause from another department makes complaints multi-part
    "Academic": " and I also cannot pay for the remark fee",
    "Finance": " and now the portal will not let me see my results",
    "ICT": " so I will miss the assignment deadline",
    "Facilities": " and the wifi there is also bad",
    "Hostel": " and the toilets on campus are just as bad",
}
CROSS_LABEL = {"Academic": "Finance", "Finance": "Academic", "ICT": "Academic",
               "Facilities": "ICT", "Hostel": "Facilities"}
# Vague complaints carry little department signal, like many real reports.
VAGUE = ["I have been waiting for help for weeks and nobody responds",
         "Nobody at the office answers the phone, this is unacceptable",
         "I was sent from one office to another the whole day",
         "The staff at the counter were very rude to me",
         "I submitted a form last month and heard nothing back",
         "the queue was so long that I left without being helped",
         "My problem from last semester is still not solved",
         "I need someone to call me back about my issue"]

TYPO = {"the": ["teh", "th"], "because": ["coz", "bcoz"], "please": ["pls", "plz"], "you": ["u"],
        "password": ["pasword"], "residence": ["rez", "residance"], "lecturer": ["lectuer"],
        "assignment": ["assignmnt"], "wifi": ["wify", "wi fi"], "fees": ["feez"], "results": ["resluts"]}


def noisy(text):
    words = text.split()
    for i, w in enumerate(words):
        key = w.lower().strip(",.?")
        if key in TYPO and random.random() < 0.25:
            words[i] = random.choice(TYPO[key])
    text = " ".join(words)
    if random.random() < 0.3:
        text = text.lower()
    return text


rows = []
cats = list(T)
for i in range(N):
    dept = random.choice(cats)
    if random.random() < 0.05:  # vague report: department is effectively unpredictable
        text = random.choice(VAGUE)
    else:
        text = random.choice(T[dept]).format(**{k: random.choice(v) for k, v in SLOTS.items()})
        if random.random() < 0.12:  # multi-part complaint; sometimes filed under the second office
            text += CROSS[dept]
            if random.random() < 0.35:
                dept = CROSS_LABEL[dept]
    text = noisy(random.choice(PREFIX) + text + random.choice(SUFFIX))
    rows.append({
        "Department": dept, "Report": text,
        "Academic_Year": random.choice(["First year", "Second year", "Third year", "Fourth year", "Postgraduate"]),
        "Nationality": random.choices(["South African", "Zimbabwean", "Nigerian", "Eswatini", "Lesotho", "Other"],
                                      [75, 6, 4, 5, 5, 5])[0],
        "Age": random.randint(18, 34), "GPA": round(random.uniform(1.6, 3.9), 2),
        "Gender": random.choice(["Female", "Male"]), "Channel": random.choice(["Web", "Web", "LINE", "Email"]),
    })

OUT.parent.mkdir(exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {len(rows)} synthetic complaints to {OUT}")
