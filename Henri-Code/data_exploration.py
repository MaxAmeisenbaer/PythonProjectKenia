import matplotlib.pyplot as plt

import pandas as pd
import os
import seaborn as sns

"""
Just getting to know the data and visualizing some basic correlations.
"""


def load_csv(filename, directory="Data", time_span=144):
    """
    :param filename: -
    :param directory: -
    :param time_span: set to 1 if you want no average values
    :return:
    """

    data_dataframe = pd.read_csv(os.path.join(directory, filename))

    # linear interpolation to fill NA values
    data = data_dataframe["value"].interpolate("linear", limit_direction="both").tolist()
    time = data_dataframe["date"].tolist()

    start = time.index("2015-04-21 11:00:00")
    # reduce number of measurements by taking the average value of a given time span

    if time_span > 1:
        new_data = []
        new_time = []
        for i in range(start, len(data), time_span):
            d = data[i:i + time_span]

            new_data.append(sum(d) / time_span)
            if i + time_span // 2 < len(data):
                new_time.append(time[i + time_span // 2])
            else:
                new_time.append(time[i])
        return new_time[:-2], new_data[:-2]

    return time[start:], data[start:]


def process_data(file1, file2, time_span=144):
    time, data = load_csv(file1, time_span=time_span)
    time1, data1 = load_csv(file2, time_span=time_span)

    if len(time) != len(time1):
        print("equalizing time length")

        missing = list(set(time1).difference(set(time)))
        if len(missing) * 100 / len(time1) > 3:
            raise "Not compatible"

        for inde, missing_time in enumerate(missing):
            ind = time1.index(missing_time)
            time1.pop(ind)
            data1.pop(ind)

        missing = list(set(time).difference(set(time1)))
        if len(missing) * 100 / len(time) > 3:
            raise "Not compatible"

        for missing_time in missing:
            ind = time.index(missing_time)
            time.pop(ind)
            data.pop(ind)

    return time, data, time1, data1


def visualize_one_data(time, data, label, filename=None, title=None):
    sns.set_theme(color_codes=sns.color_palette("pastel"))
    font = {'weight': 'bold', }

    plt.rc('font', **font)

    if title is not None:
        plt.title(title, fontsize=30)
    p, = plt.plot(time, data, label=label, color=(0.3, 0.3, 0.3))
    plt.xticks([i for i in range(0, len(data), len(data) // 10)])
    plt.legend(fontsize=20)
    fig = plt.gcf()

    fig.set_size_inches(25.6, 14.4)

    plt.setp(p, linewidth=1.5)

    if filename is not None:
        plt.savefig(fname=filename, dpi=100)

    plt.show()
    plt.cla()


def visualize_data_correlation(time1, data1, label1, time2, data2, label2, filename=None, title=None):
    spec_to_einheit = {"prec": "mm", "disch": "m³/s", "doc": "mg/L", "elc": "µS/cm", "tcd": "°C", "toc": "mg/L",
                       "tsp": "°C", "tur": "FTU", "wl": "m"}

    sns.set_theme(color_codes=sns.color_palette("pastel"))
    fig, ax = plt.subplots()
    font = {'weight': 'bold', }

    plt.rc('font', **font)

    if title is not None:
        plt.title(title, fontsize=30)

    ax2 = ax.twinx()
    ax.set_ylabel("Nitrat in mg/L", fontsize=20)
    ax2.set_ylabel(f"{label2} {spec_to_einheit[label2]}", fontsize=20)
    ax, = ax.plot(time1, data1, color=(0.3, 0.3, 0.3))
    ax2, = ax2.plot(time2, data2, color="#fca903")

    plt.figlegend((ax, ax2), (label1, label2), fontsize=15)
    plt.xticks([i for i in range(0, len(data1), len(data1) // 10)])

    fig = plt.gcf()

    fig.set_size_inches(25.6, 14.4)

    plt.setp(ax, linewidth=1.0)
    plt.setp(ax2, linewidth=1.8)
    plt.grid(False)
    if filename is not None:
        plt.savefig(fname=filename, dpi=100)

    plt.show()
    plt.cla()


def visualize_all():
    categories = ["doc", "disch", "elc", "nit", "prec", "tcd", "toc", "tsp", "tur", "wl"]
    stations = ["SHA", "NF", "OUT"]

    for station in stations:
        print(f"Starting with : {station}")
        for category in categories:
            print(f"Visualizing {station} - {category}")

            time, data = load_csv(f"{station}-{category}.csv", time_span=1)
            visualize_one_data(time, data, category,
                               title=f"{station}-{category}.csv")


def visualize_sha_nit_relation():
    categories = ["disch", "doc", "elc", "prec", "tcd", "toc", "tsp", "tur", "wl"]
    stations = ["SHA"]

    time_nit, data_nit = load_csv(f"SHA-nit.csv", time_span=1)

    for station in stations:
        print(f"Starting with : {station}")
        for category in categories:
            print(f"Visualizing {station} - {category}")
            time, data = load_csv(f"{station}-{category}.csv", time_span=1)

            visualize_data_correlation(time_nit, data_nit, "Nitrat", time, data, category,
                                       title=f"SHA-nit-{category}")


if __name__ == '__main__':
    #visualize_all()
    visualize_sha_nit_relation()


