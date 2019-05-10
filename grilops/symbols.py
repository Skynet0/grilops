"""This module supports defining symbols that may be filled into grid cells."""

from typing import Dict, List, Optional


class Symbol:
  """A marking that may be filled into a single #SymbolGrid cell.

  # Arguments
  index (int): The index value assigned to the symbol.
  name (str, None): The Python-safe name of the symbol.
  label (str, None): The printable label of the symbol.
  """
  def __init__(
      self,
      index: int,
      name: Optional[str] = None,
      label: Optional[str] = None
  ):
    self.__index = index
    self.__name = name
    self.__label = label

  @property
  def index(self) -> int:
    """(int): The index value assigned to the symbol."""
    return self.__index

  @property
  def name(self) -> str:
    """(str): The Python-safe name of the symbol."""
    if self.__name:
      return self.__name
    if self.__label:
      return self.__label
    return str(self.__index)

  @property
  def label(self) -> str:
    """(str): The printable label of the symbol."""
    if self.__label:
      return self.__label
    if self.__name:
      return self.__name
    return str(self.__index)

  def __repr__(self):
    return self.label


class SymbolSet:
  """The complete set of markings allowed to be filled into a #SymbolGrid.

  # Arguments
  names (List[str], None): A list of Python-safe names for the symbols.
  labels (List[str], None): A list of printable labels for the symbols.
  """
  def __init__(
      self,
      names: List[str] = None,
      labels: List[str] = None
  ):
    self.__symbols: List[Symbol] = []
    self.__label_to_symbol_index: Dict[str, int] = {}

    if names and labels:
      if len(names) != len(labels):
        raise Exception("names and labels must have the same length")
      for i, (name, label) in enumerate(zip(names, labels)):
        self.__symbols.append(Symbol(i, name, label))
    elif names:
      for i, name in enumerate(names):
        self.__symbols.append(Symbol(i, name))
    elif labels:
      for i, label in enumerate(labels):
        self.__symbols.append(Symbol(i, label=label))

    for symbol in self.__symbols:
      self.__dict__[symbol.name] = symbol.index
      self.__label_to_symbol_index[symbol.label] = symbol.index

  def append(self, name: str = None, label: str = None):
    """Appends an additional symbol to this symbol set.

    # Arguments
    name (str, None): The Python-safe name of the symbol.
    label (str, None): The printable label of the symbol.
    """
    if self.__symbols:
      index = self.__symbols[-1].index + 1
    else:
      index = 0
    symbol = Symbol(index, name, label)
    self.__symbols.append(symbol)
    self.__dict__[symbol.name] = symbol.index
    self.__label_to_symbol_index[symbol.label] = symbol.index

  @property
  def symbols(self) -> List[Symbol]:
    """(List[Symbol]): The list of all symbols."""
    return self.__symbols

  def __getitem__(self, index):
    return self.__label_to_symbol_index[str(index)]

  def __repr__(self):
    return self.symbols.__repr__()


def make_letter_range_symbol_set(
    min_letter: str,
    max_letter: str
) -> SymbolSet:
  """Returns a #SymbolSet consisting of consecutive letters.

  # Arguments
  min_letter (str): The lowest letter to include in the set.
  max_letter (str): The highest letter to include in the set.

  # Returns
  (SymbolSet): A #SymbolSet consisting of consecutive letters.
  """
  return SymbolSet(
      [chr(v) for v in range(ord(min_letter), ord(max_letter) + 1)]
  )


def make_number_range_symbol_set(
    min_number: int,
    max_number: int
) -> SymbolSet:
  """Returns a #SymbolSet consisting of consecutive numbers.

  The names of the symbols will be prefixed with S so that they may be
  referred to directly in Python code.

  # Arguments
  min_number (int): The lowest number to include in the set.
  max_number (int): The highest number to include in the set.

  # Returns
  (SymbolSet): A #SymbolSet consisting of consecutive numbers.
  """
  return SymbolSet(
      [f"S{v}" for v in range(min_number, max_number + 1)],
      [str(v) for v in range(min_number, max_number + 1)]
  )