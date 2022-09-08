from collections import defaultdict
from dataclasses import dataclass
import random
from string import ascii_letters
from typing import Callable, Generic, TypeVar

K1 = TypeVar("K1")
V1 = TypeVar("V1")
K2 = TypeVar("K2")
V2 = TypeVar("V2")


@dataclass
class Mapper(Generic[K1, V1, K2, V2]):
    map_func: Callable[[tuple[K1, V1]], tuple[K2, V2]]

    def run(self, input: list[tuple[K1, V1]]) -> list[tuple[K2, V2]]:
        return [self.map_func(pair) for pair in input]


@dataclass
class Reducer(Generic[K2, V2]):
    reduce_func: Callable[[tuple[K2, list[V2]]], tuple[K2, V2]]

    def run(self, input: tuple[K2, list[V2]]) -> tuple[K2, V2]:
        return self.reduce_func(input)


@dataclass
class Runner(Generic[K1, V1, K2, V2]):
    map_func: Callable[[tuple[K1, V1]], tuple[K2, V2]]
    reduce_func: Callable[[tuple[K2, list[V2]]], tuple[K2, V2]]
    num_workers: int = 3

    def split_input(self, input: list[tuple[K1, V1]]) -> list[list[tuple[K1, V1]]]:
        blocks = [[] for _ in range(self.num_workers)]
        for i, pair in enumerate(input):
            blocks[i % self.num_workers].append(pair)

        return blocks

    def shuffle(self, map_out: list[list[tuple[K2, V2]]]) -> dict[K2, list[V2]]:
        shuffled = defaultdict[K2, V2](list)
        for out in map_out:
            for k, v in out:
                shuffled[k].append(v)
        return shuffled

    def run(self, input: list[tuple[K1, V1]]) -> list[tuple[K2, V2]]:
        mappers = [Mapper(self.map_func) for _ in range(self.num_workers)]
        blocks = self.split_input(input)
        map_out = [mapper.run(block) for block, mapper in zip(blocks, mappers)]
        shuffled = self.shuffle(map_out)
        reducers = [Reducer(self.reduce_func) for _ in shuffled.keys()]

        return [reducer.run(pair) for pair, reducer in zip(shuffled.items(), reducers)]


if __name__ == "__main__":
    input = [(letter, None) for letter in random.choices(population=ascii_letters, k=100)]
    runner = Runner(map_func=lambda t: (t[0], 1), reduce_func=lambda t: (t[0], sum(t[1])))
    output = runner.run(input)
    print(output)
