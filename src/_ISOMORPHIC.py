"""
Isomorphic versions of the 50 GSM8K questions (seed=42 order).

Each isomorphic problem:
  - Uses the SAME numbers and mathematical operations
  - Has a completely DIFFERENT narrative domain (factory, sports, electronics, etc.)
  - Produces the SAME answer
  - Forces the model to process different surface tokens while extracting
    the same abstract mathematical structure
"""

_ISOMORPHIC_QUESTIONS = [
    # Q0: 430 + 750 + 300 + 700 = 2180
    "A logistics hub tracks incoming shipments across four docks. Dock A received 430 pallets this morning. Dock B received 320 more pallets than Dock A. Dock C handled 300 pallets, while Dock D processed 400 more than Dock C. What is the combined total number of pallets across all four docks?",

    # Q1: 360/(4+2) = 60 min = 1 hour
    "A factory needs to assemble 360 microchips. Machine A solders 4 chips per minute. Machine B runs at half of Machine A's speed. Working simultaneously, how many hours will it take to complete the batch?",

    # Q2: (10×3)/6 = 5 hours
    "A cargo truck travels at 10 miles per hour for 3 hours delivering goods. On the return trip, the truck moves at 6 miles per hour due to heavier load. How many hours does the return journey take?",

    # Q3: (12+8)/5 × 3 = 12
    "A contractor needs to purchase screws for two construction projects. The first project requires 12 boxes and the second needs 4 fewer boxes. Screws are sold in packs of 5 boxes for $3. How much does the contractor spend in total?",

    # Q4: 364/4 + 364/2 = 273
    "Two machines draw from a 364-liter fuel tank. Machine Alpha consumes one-quarter of the tank. Machine Beta uses one-half of the tank. How many liters of fuel did they consume in total?",

    # Q5: 50 + 25 - 30 = 45
    "A network monitoring system detected 50 active connections on a server. Twenty minutes later, half as many new connections were established. After another ten minutes, 30 connections were terminated. How many active connections remain?",

    # Q6: 9×2 + 3 = 21
    "A thermometer reads 9 degrees Celsius. A second thermometer in a different location reads twice that temperature. If the second thermometer rises by 3 more degrees, what will it read?",

    # Q7: (6×6000 - 1200)/240 = 145
    "Twenty dozen small monitors cost $1,200 less than the total price of six large displays sold at $6,000 each. What is the price of each individual small monitor?",

    # Q8: 50 × 1.2 = 60
    "A warehouse processed 50 orders last quarter. This quarter, order volume grew by 20%. How many orders were processed this quarter?",

    # Q9: 40 + 36 + 46 = 122
    "Server Rack Alpha contains 40 hard drives. Rack Beta holds 4 fewer drives than Alpha. Rack Gamma has 10 more drives than Beta. How many hard drives are there across all three racks?",

    # Q10: 3×1 + 3×2 + 5×4 = 29
    "A water treatment plant operates three filtration modes: basic removes 1 ton of sediment per day, standard removes twice as much as basic, and heavy-duty removes twice as much as standard. The plant ran basic for 3 days, standard for 3 days, and heavy-duty for 5 days. How many total tons of sediment were removed?",

    # Q11: 600 - 500 - 20 = 80
    "A supervisor ordered 600 circuit boards for an electronics workshop. 50 technicians each took 10 boards for their projects, and the supervisor used 20 boards for testing. How many circuit boards were left over?",

    # Q12: (2+1)×3×4 = 36
    "A cooling system runs for 2 hours, then switches to heating for half that duration. This cycle repeats 3 times per day. Over 4 days of operation, how many total hours does the system spend on both cooling and heating combined?",

    # Q13: 500 + 800 + 130 = 1430
    "An architect pays $500 for blueprints and $800 for engineering consultation. A building permit costs 10% of these combined fees. What is the architect's total expenditure?",

    # Q14: 3.5×10/7 = 5
    "Each robot on an assembly line consumes 3.5 kilograms of raw material per week. If there are 10 robots operating, how many kilograms of material do they need per day?",

    # Q15: (214+26) - (225+10) = 5
    "Team Red starts with 214 ranking points and earns 26 more. Team Blue has 225 points and gains 10. By how many points is Team Red now ahead?",

    # Q16: 300/15/4 = 5
    "A laboratory ordered $300 worth of chemical reagents at $15 per vial. The vials are distributed equally among 4 research stations. How many vials does each station receive?",

    # Q17: 3×16 + 3×6 = 66
    "A mechanic buys 3 tires priced at $16 each and 3 spark plugs at $6 each. What is the total amount spent?",

    # Q18: 20 × (1 - 1/4) = 15
    "A storage tank holds 20 gallons of oil. After transferring some to another facility, only one-quarter of the original volume remains. How many gallons were transferred out?",

    # Q19: 7 + 21 + 12 = 40
    "A communication relay receives 7 encrypted channels from an external source. Internal generator Alpha produces 3 times as many channels as the external source, while generator Beta outputs 12 channels. How many channels does the relay handle in total?",

    # Q20: 3 × 1 × 31 = 93
    "A data center runs 3 servers. Each server costs $1 per day in electricity. What is the total electricity cost for the month of May?",

    # Q21: 10000 - 5000 - 1000 - 2000 = 2000
    "A battery has a capacity of 10,000 mAh. The screen consumes half the capacity. Bluetooth uses another 1,000 mAh. After the WiFi module runs, only 2,000 mAh remains unused. How many mAh did the WiFi module consume?",

    # Q22: 800 + 720 = 1520
    "At a manufacturing plant, Machine Alpha processed 40 batches, generating $20 in revenue per batch. Machine Beta's revenue was 10% less than Alpha's. What is the combined revenue from both machines?",

    # Q23: (900/20 + 20) × 170 = 11050
    "Producing 20 custom chips costs $900 in total. Two days later, the per-chip production cost rises by $20. If a client orders 170 chips at the new rate, how much will it cost?",

    # Q24: 30×10×0.8 - 30×10×0.5 = 90
    "Thirty warehouse workers each packed 10 boxes. Each box sells for $0.80. The packing materials cost $0.50 per box. What was the net profit from selling all the packed boxes?",

    # Q25: 250000 × 0.4 × 0.4 = 40000
    "A company's intellectual property portfolio is valued at $250,000. An investor acquires a 40% stake. Of that stake, 60% is allocated to paying off outstanding debts. How much cash remains from the investor's portion after debt repayment?",

    # Q26: 21×4 - 63 = 21
    "Competitor A generated $63 in ad revenue last quarter. Competitor B ran 21 ad campaigns at $4 revenue each during the same period. By how many dollars did Competitor B outperform Competitor A?",

    # Q27: solve: 5 = final remaining → started with 18
    "A database system allocates one-third of its memory to the primary index, then 2 GB to a cache layer, and half of what remains to a temporary buffer. If 5 GB of memory is still unallocated, how much total memory did the system start with?",

    # Q28: 20 - 5 - 1 = 14
    "A reservoir starts with an unknown water level. At checkpoint one, 5 liters flow in. At checkpoint two, 7 liters drain out and 8 liters flow in. If the final level reads 20 liters, what was the initial water level?",

    # Q29: 75 - 12 - 10 - 30 = 23
    "A particle accelerator has 12 red lasers. It has 2 fewer green lasers than red ones. The number of blue lasers is 3 times the count of green lasers. There are also yellow lasers. If the accelerator contains 75 lasers in total, how many are yellow?",

    # Q30: 400 - (50+65+35+105) = 145
    "Four thermometers in a climate chamber report readings contributing to a safety score. Thermometer A registers 50 points. Thermometer B scores 30% more than A. Thermometer D reads 30 points below B. Thermometer C records 3 times D's score. If the safety threshold is 400 points, how many more points are needed?",

    # Q31: 30 + 90 + 3 = 123
    "In a data center, the number of Linux servers is 3 times the number of Windows servers. There are one-tenth as many BSD servers as Windows servers. Given 30 Windows servers, what is the total number of servers?",

    # Q32: 2 × (9 + 40) = 98
    "A solar panel array covers a rectangular area of 360 square feet. If the width measures 3 yards, what is the perimeter of the array in feet?",

    # Q33: 10 × 0.40 + 10 × 0.30 = 7.00
    "An electronics store sold 5 capacitors and 4 resistors for $3.20. Each capacitor costs $0.40. How much would 10 capacitors and 10 resistors cost?",

    # Q34: (20.25 + 15.75 + 66) / 3 = 34
    "Three branch offices share cloud infrastructure costs. Server hosting totals $20.25 thousand. Database costs are $4.50 thousand less than hosting. Additional API calls cost $33 thousand for each of 2 services. Split evenly, what does each branch pay (in thousands)?",

    # Q35: 6×2 + 8×1 + 6×3 = 38
    "A construction project uses copper pipes at $2 per foot, PVC pipes at $1 per foot, and steel pipes at $3 per foot. The project requires 6 feet of copper, 8 feet of PVC, and 6 feet of steel. What is the total material cost?",

    # Q36: 20 + 60 + 40 + 200 = 320
    "A server log recorded 20 events on day zero. On day one, the event count tripled. On day two, there were 20 fewer events than day one. On day three, the event count was double the combined total of days one and two. How many events have been logged in total?",

    # Q37: 100 - 20 - 30 = 50
    "In a resource allocation system with 100 units total, Process A claims 20% of the resources. Process B takes 50% more than Process A. Process C receives whatever remains. How many units does Process C get?",

    # Q38: 100×3/5 - 10 = 50
    "Two departments divide a $100 budget in a 3:2 ratio. The larger department spends $10 on office supplies. How much budget does the larger department have left?",

    # Q39: 6 + 30 + 48 = 84
    "Process Alpha uses 6 CPU cores. Process Beta requires 5 times as many cores as Alpha. Process Gamma takes all remaining cores and splits them into 4 equal subtasks of 12 cores each. How many CPU cores were available initially?",

    # Q40: 1125/(15+30) × 30 = ... wait, let me recalculate
    # 15×X + 15×2X = 1125 → 45X = 1125 → X = 25 → 2X = 50
    "A manufacturing line has a monthly output target of 1,125 units. During the second half of the 30-day month, daily output is doubled compared to the first half to meet the target. What is the required daily output during the second half?",

    # Q41: 88000/11 = 8000
    "Satellite A transmitted 10 times as many packets as Satellite B. Together they sent 88,000 packets. How many packets did Satellite B transmit?",

    # Q42: 8 × 7 × 5 = 280
    "A computing grid arranges nodes in groups of 8 per rack. Each cluster has 7 racks, and there are 5 clusters total. How many compute nodes are in the entire grid?",

    # Q43: (24-4) + 10 = 30
    "Router A has a latency 10 ms higher than Router B. Router B's latency is 4 ms lower than Router C, which operates at 24 ms. What is Router A's latency?",

    # Q44: 8 × (20+2) = 176
    "Ten storage arrays each hold 20 terabytes of data. An additional 40 terabytes are distributed equally across 20 arrays. How much data is stored in 8 of the arrays?",

    # Q45: 650 - 57×4 - 73×2 = 276
    "A workshop needs 57 chairs (4 legs each) and 73 stools (2 legs each). If 650 legs are available in inventory, how many legs will remain unused?",

    # Q46: (10+6+3) + (5+3+5) = 32
    "Scanner Alpha detected 10 trojans, 6 worms, and 3 rootkits. Scanner Beta found 5 fewer trojans, 3 fewer worms, and 2 more rootkits than Alpha. How many malware instances were detected in total?",

    # Q47: (200/8×8 + 200×1) / 16 = 25
    "A shipping company packs 200 parcels into crates of 8. Each empty crate weighs 8 ounces and each parcel weighs 1 ounce. What is the total combined weight in pounds? (16 ounces = 1 pound)",

    # Q48: 8×1.5 - 2 = 10
    "A motor runs at 8 RPM. The speed is increased by 50%, but this causes vibration, so it is reduced by 2 RPM. What is the final operating speed?",

    # Q49: 3×16 + 12 + 24 = 84
    "In a factory, 3 identical assembly lines each run at 16 units per hour. A fourth line runs at 12 units per hour. A fifth line runs at 24 units per hour (the fourth line's rate is half of the fifth). What is the combined hourly output of all five lines?",
]
