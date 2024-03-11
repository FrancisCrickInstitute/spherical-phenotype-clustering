
FILEPATH_COLUMN = 'filepath'

CELL_TYPE_COLUMN = 'cell_type'
PLATE_COLUMN = 'plate'
WELL_COLUMN = 'well'
MOA_COLUMN = 'moa'
TREATMENT_COLUMN = 'treatment'
COMPOUND_NAME_COLUMN = 'compound_name'
COMPOUND_UM_COLUMN = 'compound_uM'

# these are used for the hover plots
META_DF_COLUMNS = [CELL_TYPE_COLUMN, PLATE_COLUMN, WELL_COLUMN, MOA_COLUMN, TREATMENT_COLUMN, COMPOUND_NAME_COLUMN, COMPOUND_UM_COLUMN]

# the code inspects the moa column for controls according to this value (for batch correction)
CONTROL_MOA_NAME = 'DMSO'


