#!/usr/bin/python


from xlrd import open_workbook
import constants
import sys


#Open excel book
print 'opening Machine Learning Training Workbook...(this can take a while)'
try:
    book = open_workbook(constants.training_workbook_name)
except:
    print 'unable to find or open training workbook...'
    print 'program is exiting...'
    sys.exit(0)

print 'extracting classified training and test data from Machine Learning Workbook...'

#Extract training and test data from the Machine Learning Summary Excel Spreadsheet
#Open excel sheet
sheet = book.sheet_by_name(constants.training_sheet_name)

#Read header values into the list
keys = [sheet.cell(0, col_index).value for col_index in xrange(sheet.ncols)]

#Add values to dictionary and append to list
dict_list = []
for row_index in xrange(1, sheet.nrows):
    d = {keys[col_index]: sheet.cell(row_index, col_index).value
         for col_index in xrange(sheet.ncols)}
    dict_list.append(d)


