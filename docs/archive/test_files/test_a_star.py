from a_star import *

if __name__ == '__main__':
    obstacle = [(2.1, 2.3), (3.5, 4.7), (9, 9)]
    env = Env(100, 100, obstacle)

    start_point = (-20.5, -20.6)
    goal_point = (20, 20)

    a_star = AStar(start_point, goal_point, env)

    path, _ = a_star.searching()

