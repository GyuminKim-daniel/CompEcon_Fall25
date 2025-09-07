!pip install xlrd

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Load the Excel file
# ----------------------------------------------------------------------
fitting = pd.read_excel(r"C:\Users\Gyumin Kim\Desktop\fitting room.xls",
                        header=0, index_col=0)

plt.style.use("ggplot")

# ----------------------------------------------------------------------
# Use functions to reduce redundant lines in my code
# ----------------------------------------------------------------------
def save_barplot(series, title, xlabel, ylabel, filename, color="skyblue", figsize=(12,6), dpi=120):
    """
    Create and save a barplot from a pandas Series.

    Arguments:
        series (pd.Series): Indexed data to plot (x=series.index, y=series.values).
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        filename (str): Path to save the figure.
        color (str, optional): Bar color. Default = "skyblue".
        figsize (tuple, optional): Figure size. Default = (12,6).
        dpi (int, optional): Figure resolution. Default = 120.
    """
    ax = series.plot(kind="bar", color=color, figsize=figsize,
                     title=title, xlabel=xlabel, ylabel=ylabel)
    ax.get_figure().savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.show()


def save_scatter(x, y, title, xlabel, ylabel, filename, color="blue", figsize=(8,6), dpi=100):
    """
    Create and save a scatter plot.

    Arguments:
        x (array-like): X-axis values.
        y (array-like): Y-axis values.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        filename (str): Path to save the figure.
        color (str, optional): Marker color. Default = "blue".
        figsize (tuple, optional): Figure size. Default = (8,6).
        dpi (int, optional): Figure resolution. Default = 100.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, alpha=0.5, color=color)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.show()


# ----------------------------------------------------------------------
# 1. Daily average fitting count per item
# ----------------------------------------------------------------------
average_fit = (
    fitting.groupby(["item","storecode","eventday"])["fit_in"]
           .mean()                            
           .groupby(["item"]).mean() 
           .sort_values(ascending=False)
)

"""
As shown in the dataset, each observation contains information on whether an item
was brought into the fitting room. 

To analyze this, I first calculate the daily average fitting count at the store–item level 
(i.e., how many times each item is tried on in each store per day). 

Then I compute the mean across days for each item, resulting in the 
average daily fitting count for each item. Finally, I sort the results in descending order.
"""

save_barplot(
    series=average_fit,
    title="Daily Average Fitting Count by Item",
    xlabel="Item",
    ylabel="Daily Average Fitting Count",
    filename="Fittingcount.png"
)

# ----------------------------------------------------------------------
# 2. Sales vs fitting count for item "TJ"
# ----------------------------------------------------------------------
tj_df = fitting[fitting["item"] == "TJ"]

"""
This subset contains only rows where the item is 'TJ' (Jeans).
It will be used to analyze the relationship between fitting room counts (fit_in)
and sales specifically for jeans.
"""

save_scatter(
    x=tj_df["fit_in"],
    y=tj_df["sales"],
    title="Relationship Between Sales and Fitting Count for Jeans (TJ)",
    xlabel="Fitting Room Counts",
    ylabel="Sales (Million KRW)",
    filename="Fittingtj.png",
    color="blue"
)

# ----------------------------------------------------------------------
# 3. Daily traffic vs sales per store-day
# ----------------------------------------------------------------------
daily = fitting.groupby(["eventday", "storecode"]).agg({
    "traffic": "mean",
    "sales": "sum"
}).reset_index()

"""
Each observation corresponds to an item–store–day. 
- Traffic is repeated across items within the same store-day (all items share the same store traffic), 
  so we take the mean to avoid double-counting.
- Sales vary across items within the same store-day (some items sold, others not), 
  so we sum sales to get the store’s total daily sales.

"""

save_scatter(
    x=daily["traffic"],
    y=daily["sales"],
    title="Relationship Between Daily Traffic and Daily Sales",
    xlabel="Store Traffic",
    ylabel="Sales (Million KRW)",
    filename="Traffic.png",
    color="green"
)
