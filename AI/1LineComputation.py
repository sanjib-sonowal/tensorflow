from numpy import array


def compute_error_for_line_given_points(b, m, points):
    total_error = 0;
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m*x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, m_current, points):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - b_gradient
    new_m = m_current - m_gradient
    return [new_b, new_m]


def runner(points, starting_b, starting_m, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points))
    return b, m

if __name__ == '__main__':
    compute_error_for_line_given_points(10, 10, 10)

