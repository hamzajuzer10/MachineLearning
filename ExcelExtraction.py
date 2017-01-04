#!/usr/bin/python


from xlrd import open_workbook
import constants


#Open excel book
print 'opening MachineLearningSummaryWorkbook...(this can take a while)'
book = open_workbook(constants.workbook_name)

print 'extracting classified training and test data from MachineLearningSummaryWorkbook...'

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

#Extract actual data from Machine Learning Summary Excel Spreadsheet - TODO

print 'extracting unclassified data from MachineLearningSummaryWorkbook...'
