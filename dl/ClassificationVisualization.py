from collections import defaultdict
import matplotlib.pyplot as plot


def show_result(points, expected, predicted):
    by_categories_x = defaultdict(list)
    by_categories_y = defaultdict(list)
    for i, (x, y) in enumerate(points):
        if expected[i] != predicted[i]:
            color = "r"
        elif expected[i] == 1:
            color = "b"
        else:
            color = "g"
        marker = '+' if predicted[i] == 1 else '_' # https://matplotlib.org/api/markers_api.html
        by_categories_x[(color, marker)].append(x)
        by_categories_y[(color, marker)].append(y)

    for (color, marker), xs in by_categories_x.items():
        ys = by_categories_y[(color, marker)]
        plot.scatter(xs, ys, c=color, marker=marker)
    plot.show()
