import multiprocessing
import time


def hello(taskq, flag):
    for i in range(1000):
        taskq.put(i)
        if i % 100 == 0:
            time.sleep(2)
    flag.get()


def main():
    taskq = multiprocessing.Queue()
    flag = multiprocessing.Queue()
    flag.put(True)
    p = multiprocessing.Process(target=hello, args=(taskq, flag))
    p.start()

    while not flag.empty() or not taskq.empty():
        t1 = time.perf_counter()
        name = taskq.get(True)
        t2 = time.perf_counter()
        if name % 100 == 0:
            time.sleep(1)
        print(t2 - t1, name)

    p.join()


if __name__ == '__main__':
    main()
