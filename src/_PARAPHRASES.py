"""
Paraphrased versions of the 50 GSM8K questions (seed=42 order).

Each paraphrase preserves the exact same mathematical structure,
numbers, and answer — only the surface wording and sentence structure differ.
"""

_PARAPHRASED_QUESTIONS = [
    # Q0: raise money carnival
    "A group of girls is fundraising for a carnival. Alexandra collected $430, while Kim collected $320 more than Alexandra. Sarah brought in $300, and Maryam collected $400 more than Sarah. What is the total amount of money, in dollars, that all of them raised together?",

    # Q1: puzzle 360 pieces
    "Kalinda and her mother are putting together a jigsaw puzzle that has 360 pieces. Kalinda can fit 4 pieces each minute, and her mother works at half of Kalinda's speed. How many hours will the puzzle take them to finish?",

    # Q2: Tom's ship
    "Tom sails his boat at a speed of 10 mph. He departs at 1 PM and sails until 4 PM. On the return trip, his speed drops to 6 mph. How many hours does the return journey take?",

    # Q3: birthday candles
    "James wants to purchase birthday candles for both of his sons. The older son is turning 12, and the younger one is 4 years younger than that. Candles come in packs of 5, with each pack priced at $3. What is the total amount James spends on the candles?",

    # Q4: knitting yarn
    "While learning to knit from her grandmother, Mariah used up one-quarter of a skein. Her grandmother used one-half of a skein. Given that each skein contains 364 yards of yarn, what is the combined number of yards they both used?",

    # Q5: fairies
    "Katelyn spotted 50 fairies flying over the forest near the school playground. About 20 minutes later, a friend noticed that half the number of fairies Katelyn had seen arrived from the east and joined the group. After another 10 minutes, 30 of the fairies departed. How many fairies stayed behind?",

    # Q6: Ann's age
    "Ann is currently 9 years of age. Her brother is twice as old as Ann. What will her brother's age be 3 years from now?",

    # Q7: cups and plates
    "The cost of twenty dozen cups is $1,200 less than the price of half a dozen plates, each priced at $6,000. Determine the price of a single cup.",

    # Q8: calligraphy enrollment
    "A calligraphy course had 50 enrolled students last year. This year, enrollment went up by 20%. What is the number of students enrolled in the calligraphy course this year?",

    # Q9: crabs
    "Bo owns 40 crabs. Monic has 4 fewer crabs than Bo does. Rani has 10 more crabs than Monic. How many crabs do Bo, Monic, and Rani have in total?",

    # Q10: dehumidifier
    "To fix the dampness in his basement, Brian purchased a dehumidifier that offers low, medium, and high settings. Testing showed the low setting extracts 1 liter of water daily, medium removes double what low does, and high removes double what medium does. Over 3 days on low, 3 days on medium, and 5 days on high, how many total liters of water were removed from the air?",

    # Q11: burritos picnic
    "Mr. George, a sixth-grade teacher, placed an order for 600 burritos for the class picnic. At the event, 50 students each received 10 burritos, and Mr. George ate 20 himself. Determine the number of burritos remaining after the picnic.",

    # Q12: TV and reading
    "Each time Jim does his routine, he watches television for 2 hours and then reads in bed for half that duration. He follows this routine 3 times per week. Over the course of 4 weeks, how many total hours does Jim spend on television and reading combined?",

    # Q13: brooch insurance
    "Janet purchased a brooch for her daughter. The materials cost $500, and the jeweler charged $800 to craft it. She then paid 10% of the combined cost for insurance. What was the total amount Janet paid?",

    # Q14: ducks insects
    "Each duck requires 3.5 pounds of insects every week to stay alive. A flock consists of 10 ducks. How many pounds of insects does the entire flock need each day?",

    # Q15: Scrabble points
    "Before his turn, Joey had 214 points in Scrabble. He then scored 26 more points. Marcy had 225 points and added 10 to her total. By what margin is Joey now ahead?",

    # Q16: books for kids
    "Sarah purchased books from a bookstore, spending a total of $300. Each book was priced at $15. She then distributed the books equally among her 4 children. How many books did each child receive?",

    # Q17: Ted's purchases
    "Ted has $200 to spend. He purchases 3 books priced at $16 each along with 3 pencils at $6 each. What is the total amount he spent?",

    # Q18: pomelos
    "Eve started with 20 pomelos. After sharing some with a friend, she was left with one-fourth of her original pomelos. How many pomelos did Eve give to her friend?",

    # Q19: kittens
    "On their way home from the animal shelter with 7 newly adopted kittens, the Doubtfire sisters got a call from their mother saying their two house cats had given birth. The first cat, Patchy, had three times as many kittens as the number that were adopted, and the second cat, Trixie, had 12 kittens. What is the total number of kittens the Doubtfire family now has?",

    # Q20: fish food May
    "Jen purchased 3 fish. Feeding each fish costs $1 per day. How much does Jen spend on fish food during the month of May?",

    # Q21: Elliott's steps
    "Elliott's daily goal is 10,000 steps. He completed half of them while walking to and from school, then took another 1,000 steps while walking with a friend. He also jogged around the block and found that he still needed 2,000 more steps to reach his goal. How many steps did the jog account for?",

    # Q22: restaurant tips
    "During her shift at the restaurant, Rafaela received $20 tips from each of the 40 customers. Julieta earned 10% less in tips than Rafaela. What is the combined tip total for both Julieta and Rafaela?",

    # Q23: helium balloons
    "Filling 20 helium balloons cost a total of $900 on a particular day. Two days later, the per-balloon filling price went up by $20. If Bentley comes in after the price increase to fill 170 balloons, how much will it cost him?",

    # Q24: lollipops fundraiser
    "Thirty students in one class participated in a fundraiser by selling lollipops. Each lollipop was sold for $0.80, and each student sold an average of 10 lollipops. The lollipops had been purchased at $0.50 each. What was the total profit the class made from selling lollipops?",

    # Q25: mortgage loan
    "John takes out a mortgage loan on his house, which is valued at $250,000. He borrows 40% of the home's value, then uses 60% of the loan to settle his debts. How much cash does John have remaining after paying off the debts?",

    # Q26: lemonade stands
    "Liam and Mitchell run rival lemonade stands on opposite sides of the street. Liam boasted about earning $63 over the weekend. Mitchell responded that he sold 21 cups of lemonade at $4 each during the same weekend. How much more money did Mitchell earn than Liam?",

    # Q27: vacuum cleaners
    "Melanie sells vacuum cleaners door to door. At the green house she sold a third of her stock. At the red house, she sold 2 additional units. At the orange house, she sold half of her remaining inventory. She now has 5 vacuum cleaners left. How many did she begin with?",

    # Q28: bus terminal
    "At the main terminal, an unknown number of passengers boarded a bus. At the first stop, 5 additional passengers got on. At the second stop, 7 passengers got off and 8 more got on. If there were 20 passengers on board heading toward the third stop, how many boarded at the terminal originally?",

    # Q29: bumper cars
    "A bumper car rink contains 12 red cars. The number of green cars is 2 fewer than the red ones. There are 3 times as many blue cars as green cars. The rink also has yellow cars, bringing the total fleet to 75 cars. How many yellow cars are there?",

    # Q30: class points
    "Class 3B is collecting behavior points to earn a class trip. The class consists of Adam, Martha, Betty, and Tom. Adam has earned 50 points. Betty earned 30% more than Adam. Martha collected 3 times as many points as Tom, and Tom has 30 fewer points than Betty. If the trip requires 400 points minimum, how many more points does the class still need?",

    # Q31: classroom children
    "In a classroom, the number of girls is 3 times the number of boys, and the number of nongendered children is one-tenth the number of boys. Given that there are 30 boys in the room, what is the total number of children present?",

    # Q32: bedroom perimeter
    "Billie's bedroom is rectangular with an area of 360 square feet. The room's length measures 3 yards. What is the perimeter of the room expressed in feet?",

    # Q33: lollipops and candies
    "Manolo spent $3.20 on five lollipops and four candies. Given that each lollipop costs $0.40, determine the total cost of buying 10 lollipops and 10 candies.",

    # Q34: fair trip
    "Three friends bought admission tickets to the fair for a total of $20.25. Their food expenses were $4.50 less than the ticket cost. They also each paid $33 for two different rides. If they split all expenses equally, how much did each person pay?",

    # Q35: fruit orchard
    "At a pick-your-own orchard, peaches run $2.00 per pound, plums are $1.00 per pound, and apricots cost $3.00 per pound. Winston harvested 6 pounds of peaches, 8 pounds of plums, and 6 pounds of apricots. What was his total bill for the fruit?",

    # Q36: Pokemon cards
    "Elaine began with 20 Pokemon cards. After the first month, her collection tripled. In the second month, she acquired 20 fewer cards than she did in the first month. During the third month, she collected twice the total number of cards from the first two months combined. How many Pokemon cards does Elaine have now?",

    # Q37: student council election
    "During the student council election with 100 voters, candidate A received 20% of the votes. Candidate B earned 50% more votes than candidate A. The remaining votes went to candidate C. How many votes did candidate C receive?",

    # Q38: dividing money
    "Gerald and Julia split $100 between them in a 3:2 ratio. Afterward, Gerald used $10 of his share to buy a book. How much money does Gerald have remaining?",

    # Q39: grapes
    "Madeline consumed 6 grapes. Her brother used 5 times that many to produce a full glass of grape juice. Their mother then took all the remaining grapes and baked 4 pies, using 12 grapes per pie. How many grapes were there originally?",

    # Q40: savings target
    "Elvis aims to save $1,125 each month. In April, he plans to save twice as much per day during the second half of the month compared to the first half in order to meet his goal. What is his required daily savings amount during the second half of April?",

    # Q41: record sales
    "Marilyn's debut record outsold Harald's by a factor of 10. Together they sold 88,000 copies. How many copies did Harald sell?",

    # Q42: children's day drill
    "For a Children's Day celebration, students will perform a mass drill before the President. They are arranged with 8 students per row, 7 rows per school, and 5 schools are participating. What is the total number of children in the drill?",

    # Q43: Jeff's age
    "Jeff is 10 years older than his sister Martha. Martha is 4 years younger than her boyfriend Mike, who is 24 years old. What is Jeff's age?",

    # Q44: cows in stalls
    "There are 10 stalls with 20 cows in each one. Mr. Sylas purchases 40 additional cows and distributes them evenly across 20 stalls. How many cows are now in 8 of the stalls?",

    # Q45: toy wheels
    "Henry must put together 57 toy cars and 73 toy motorcycles. Each car requires 4 wheels and each motorcycle needs 2 wheels. If Henry has 650 wheels available, how many wheels will be unused after assembly?",

    # Q46: beach catch
    "Anakin and Locsin visited the beach. Anakin caught 10 starfish, 6 seahorses, and 3 clownfish. Locsin's catch consisted of 5 fewer starfish, 3 fewer seahorses, and 2 more clownfish than Anakin's. How many creatures did they catch in total?",

    # Q47: crayons and boxes
    "Nik possesses 200 crayons and plans to pack them into boxes, placing 8 crayons in each one. Each empty box weighs 8 ounces, and each crayon weighs 1 ounce. With 16 ounces equaling 1 pound, what is the combined weight in pounds of all the crayons and boxes once packed?",

    # Q48: gym weights
    "Jamaal has been lifting an 8-pound weight at the gym. He raises it by 50%, but finds it too heavy, so he switches to one that is 2 pounds lighter. What is the weight, in pounds, that Jamaal ends up using?",

    # Q49: siblings' ages
    "A household includes 2 brothers and 3 sisters. Each of the 3 sisters is 16 years old. One brother is 12, which is half the age of the older brother. What is the sum of all the siblings' ages?",
]
