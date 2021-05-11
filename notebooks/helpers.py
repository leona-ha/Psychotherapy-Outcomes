import pandas as pd

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Type of missing column
        answer_type = [df[column].unique() if df[column].nunique() < 15 else "Free Text or Numeric Input" for column in df.columns]
        answer_df = pd.DataFrame(answer_type, index = df.columns)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent, answer_df], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Answers'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print ("The dataset has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def show_answers(df):
    amount_answers = []
    answer_type = []
    column_list = df.columns.tolist()
    for column in column_list:
        amount_answers.append(df[column].nunique())
        if df[column].nunique() < 10:
            answer_type.append(df[column].unique())
        else:
            answer_type.append("Free Input")

    answer_table = pd.DataFrame({"Item": column_list, "Amount of Answers": amount_answers, "Anwers": answer_type}, index=None)
    answer_table = answer_table.set_index("Item")
    answer_table = answer_table.sort_values("Amount of Answers", ascending = False)

    return answer_table
