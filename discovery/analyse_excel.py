import pandas as pd


def main():
    # filepath = "../data/excel_spreadsheets/2014.xls"
    filepath = "../data/Bangladesh_PCV_onlyStudyPatients.xlsx"
    print("File: {}".format(filepath))
    s = pd.read_excel(filepath, engine='openpyxl')
    print(s['PEP1'].value_counts())
    # print(s.stack().value_counts()[:4])
    # print("Nb of elements: {}".format(s.stack().value_counts()[:4].sum()))

if __name__ == "__main__":
    main()