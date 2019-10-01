"""
Generate entries with records
- city
- rent
- state

Use Bayes rule to fill the missing inputs.
Then use a neural net to do the same task.
"""

from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass()
class Entry:
    city: str
    rent: int
    surface: int
    state: str


def generate_inputs(size: int) -> List[Entry]:
    @dataclass()
    class CityInfo:
        name: str
        rent_by_m2: int
        state: str
        population: int

    city_knowledge = [
        CityInfo(state='Ile-de-France', name='Paris', rent_by_m2=35, population=2_190_000),
        CityInfo(state='Ile-de-France', name='Nanterre', rent_by_m2=21, population=94_000),
        CityInfo(state='Ile-de-France', name='Puteaux', rent_by_m2=24, population=45_000),
        CityInfo(state='Ile-de-France', name='Suresnes', rent_by_m2=23, population=48_000),
        CityInfo(state='Auvergne Rhone-Alpes', name='Lyon', rent_by_m2=18, population=513_000),
        CityInfo(state='Auvergne Rhone-Alpes', name='Clermont Ferrand', rent_by_m2=11, population=142_000),
        CityInfo(state='Auvergne Rhone-Alpes', name='Grenoble', rent_by_m2=9, population=161_000),
        CityInfo(state='Auvergne Rhone-Alpes', name='Valence', rent_by_m2=9, population=63_000),
        CityInfo(state='Nouvelle Acquitaine', name='Bordeaux', rent_by_m2=12, population=250_000),
        CityInfo(state='Nouvelle Acquitaine', name='Poitiers', rent_by_m2=8, population=88_000),
        CityInfo(state='Nouvelle Acquitaine', name='Limoges', rent_by_m2=9, population=133_000),
        CityInfo(state='Nouvelle Acquitaine', name='Cognac', rent_by_m2=10, population=20_000)
    ]
    total_population = sum(city.population for city in city_knowledge)
    probabilities = [city.population / total_population for city in city_knowledge]
    cities = np.random.choice(city_knowledge, replace=True, size=size, p=probabilities)
    surfaces = np.random.uniform(low=30, high=45, size=size)
    rents = np.random.normal(loc=1, scale=0.2, size=size)
    return [Entry(city=city.name,
                  state=city.state,
                  surface=surfaces[i],
                  rent=surfaces[i] * rents[i] * city.rent_by_m2)
            for i, city in enumerate(cities)]


# TODO - generate a ton of inputs
# TODO - compute the probabilities
# TODO - delete some inputs in some new generated entries - try to deduce them, and check if it works


"""
Tests
"""

print(generate_inputs(10))
