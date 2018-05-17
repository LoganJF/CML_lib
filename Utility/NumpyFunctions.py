"""Purpose of this script is to hold functions used primarily with numpy arrays used in CML"""

# Add some dimension to your life
import numpy as np  # I say numpy like 'lumpy', no I don't mean num-pie

# -------> UTILITY FUNCTIONS FOR BEHAVIORAL EVENT MANIPULATIONS
def append_fields(old_array, list_of_tuples_field_type):
    """Return a new array that is like "old_array", but has additional fields.

    The contents of "old_array" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    *This is necessary to do use than using the np.lib.recfunction.append_fields
    function b/c the json loaded events use a dictionary for stim_params in the events*
    -----
    INPUTS
    old_array: a structured numpy array, the behavioral events from ptsa
    list_of_tuples_field_type: a numpy type description of the new fields
    -----
    OUTPUTS
    new_array: a structured numpy array, a copy of old_array with the new fields
    ------
    EXAMPLE USE
    >>> events = BaseEventReader(filename = logans_file_path).read()
    >>> events = append_field_workaround(events, [('inclusion', '<i8'), ('foobar', float)])
    >>> sa = np.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == np.dtype([('id', int), ('name', 'S3')])
    True
    """
    if old_array.dtype.fields is None:
        raise ValueError("'old_array' must be a structured numpy array")
    new_dtype = old_array.dtype.descr + list_of_tuples_field_type

    # Try to add the new field to the array, should work if it's not already a field
    try:
        new_array = np.empty(old_array.shape, dtype=new_dtype).view(np.recarray)
        for name in old_array.dtype.names:
            new_array[name] = old_array[name]
        return new_array
    # If user accidentally tried to add a field already there, then return the old array
    except ValueError as e:
        print(sys.exc_info()[0])
        error = FieldAlreadyExistsException(e)
        return old_array

# -------> UTILITY OBJECTS FOR INFORMING USER OF ERRORS
class BaseException(Exception):
    """Base Exception Object for handling"""

    def __init__(self, *args):
        self.args = args
        self._set_error_message()

    def _set_error_message(self):
        self.message = self.args[0] if self.args else None


class FieldAlreadyExistsException(BaseException):
    """Utility for handling excepts of appending fields"""

    def __init__(self, msg):
        self.msg = msg
        warning = '{}\nSo, the field is already in the array, returning inputted array'
        super(FieldAlreadyExistsException, self).__init__('ValueError: {}'.format(msg))
        print(warning.format(self.message))