"""
5 euros kept every day
How much money would you have after X year if you spared that money?

Y = 5 * 365 more money per year

Spared money after 55 years: sum (i = 0 .. 55) Y * rate ^ i
"""


rate = 1.02
per_day = 5
per_year = per_day * 365
nb_year = 55
print(sum(per_year * rate ** i for i in range(nb_year)))
