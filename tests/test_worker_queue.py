import threading

from cryoswath import misc


def test_request_workers_marks_failed_tasks_done():
    calls = []

    def failing_task(value):
        calls.append(value)
        raise RuntimeError("remote unavailable")

    task_queue = misc.request_workers(failing_task, 1)
    task_queue.put(("track",))

    joined = threading.Event()
    join_thread = threading.Thread(
        target=lambda: (task_queue.join(), joined.set()),
        daemon=True,
    )
    join_thread.start()

    assert joined.wait(timeout=2)
    assert calls == ["track"]
    assert len(task_queue.worker_errors) == 1
    assert isinstance(task_queue.worker_errors[0][0], RuntimeError)

    task_queue.put(None)
    task_queue.join()
