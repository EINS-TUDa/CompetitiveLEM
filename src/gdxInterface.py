# --------------------------------------------------------------------------------
#  Brief API to interface with GAMS GDX files, built on top of the GAMS Python API.
#  Barbosa, J. (2021)
#  Original code: git@github.com:JP-Barbosa/gdxpy.git
#  mail@juliabarbosa.net
# ---------------------------------------------------------------------------------- 

from gams import *
import re  # # for splitting with multiple deliminators
import pandas as pd
from enum import Enum

class GdxException(Exception):
    # Brief API class for throwing execeptions
    pass

class SymbolType(Enum):
    # Brief API class for GAMS symbol types
    SET = 1
    PARAMETER = 2
    VARIABLE = 3
    EQUATION = 4
    UNKNOWN = 5
    
class GdxInterface:

    """
    General Class to interface with gdx files.
    Methods work for all gdx files.
    """

    def __init__(self, fname, ws=''):
        self.ws = GamsWorkspace(ws)
        self.file = self.ws.add_database_from_gdx(gdx_file_name=fname)

    def load_column_to_dict(self, entry_name, attr, integer_index=True) -> dict:
        """
        Load data from one column to a dictionary, where the keys are the indexes.
        Only for 1D indexed entries!!

        :param entry_name: <str>,
            GDX entry name to be loaded.
        :param attr: <str>,
            column from the entry to be loaded
        :param integer_index: <bool>,
            if the index must be converted to 0-indexed integer. Works only for 1D index
        :return: <dict>,
            dict with the index as keys and respective attr.
        """
        # todo: implement 2D general solution
        assert self.file[entry_name].dimension == 1, "Entry index dimension is greater than 1!"

        if integer_index:
            # -1 since python is 0-base indexed and GAMS 1-based index
            ret = {int(rec.keys[0][1:]) - 1: getattr(rec, attr) for rec in self.file[entry_name]}
        else:
            ret = {rec.keys: getattr(rec, attr) for rec in self.file[entry_name]}
        return ret

    def _get_symbol_attr(self, symbol_name, attributes, filter=None) -> pd.DataFrame:
        ret = []

        # get symbol
        try:
            s = self.file.get_symbol(symbol_identifier=symbol_name)
        except GamsException as e:
            raise GdxException("GAMS did not found the symbol %s" % symbol_name)
        
        dom_names = s.get_domains_as_strings()

        # validate attributes

        # Validate filter
        if filter is not None:
            if len(filter) != s.dimension:
                raise GdxException(
                    "Filter dim does not match symbol dimension. Expected %d, got %d" % (s.dimension, len(filter)))

        # Get pointer
        try:
            pointer = s.first_record(slice=filter)
        except GamsException as e:
            raise GdxException("GAMS did not found any entry with the provided filter %s" %filter)

        # get values for first entry
        line = pointer.keys
        for att in attributes:
            line.append(getattr(pointer, att))
        ret.append(line)

        # Iterate until end (False)
        while pointer.move_next():
            line = pointer.keys
            for att in attributes:
                line.append(getattr(pointer, att))
            ret.append(line)

        # convert to Dataframe and return
        return pd.DataFrame(data=ret, columns=dom_names + attributes)


    @staticmethod
    def split_keys(input_dict) -> dict:
        """
        Aux function to split equation and variable strings.

        :param input_dict: <dict>, dict to be split
        :return: <dict> , dict with the same keys as the input_dict, but entries are also dictionaries. See example!

        Example:
            input_dict = {43: "Pin('Electric Heating',Electricity,Heat,'2007-01-01 01:00:00',Point,2025)"}
            return = {43:{"name": "Pin",
                          "keys": ["Electric Heating", "Electricity", "Heat", "2007-01-01 01:00:00", "Point", "2025"],
                          }
        """
        ret = dict()
        for key in input_dict.keys():
            # separate name from index and remove white space in the end
            aux = re.split('[()]', input_dict[key])
            assert (len(aux) < 4), "Split function did not work how it was supposed to"
            if len(aux) == 1:
                ret[key] = {"name": aux[0], "keys": []}
            elif len(aux) == 3:
                aux = aux[0:-1]  # exclude white space in the end
                # remove '' from key names with space!
                k = [x.replace("\'", "") for x in aux[1].split(",")]  # split keys
                ret[key] = {"name": aux[0], "keys": k}
        return ret

class GdxSolution(GdxInterface):
    def __init__(self, fname, ws = ''):
        super().__init__(fname=fname, ws=ws)

    def get_parameter_values(self, pname, filter=None) -> pd.DataFrame:
        valid_attr = ["value"]
        return self._get_symbol_attr(pname, valid_attr, filter=filter)

    def get_variable_attr(self, varname, attributes=None, filter=None, as_single_value=False) -> pd.DataFrame: 
        """
        varname: name of the variable 
        attributes can be level, marginal, etc. 
        """

        varattr = ["level", "marginal", "upper", "lower", "scale"]

        if attributes is None:
            attributes = ["level", "marginal", "upper", "lower", "scale"]
        elif isinstance(attributes, str):
            if attributes in varattr:
                attributes = [attributes]
            else:
                raise ValueError("%s is not a variable attribute" % attributes)
        elif isinstance(attributes, list):
            fun = lambda x: x in varattr
            valid_attr = [fun(a) for a in attributes]
            if not all(valid_attr):
 
                raise ValueError("not all given attributes are variable attributes")

        empty = False
        try:
            ret = self._get_symbol_attr(varname, attributes, filter)
        except GdxException as e :
            if not as_single_value:
                raise GdxException(e)
            empty = True
        finally:
            if as_single_value:
                if len(attributes) > 1:
                    raise GdxException("Returning multiple attributes as single value not possible!")
                if not empty:
                    if len(ret) > 1:
                        raise GdxException("Request returned data frame with more than one line. Returning as single value not possible!")
                    ret = ret[attributes[0]].values[0]
                if empty:
                    ret = 0

        return ret

    def get_set(self, setname) -> list:
        s = self.file.get_set(setname)
        dim = s.dimension
        if dim == 1:
            ret = [x.keys[0] for x in s]
        else:
            ret = [x.keys for x in s]
        return ret

